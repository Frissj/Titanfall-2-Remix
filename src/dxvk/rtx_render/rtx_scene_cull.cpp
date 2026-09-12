#include "rtx_scene_cull.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

#include "dxvk_device.h"
#include "rtx_render/rtx_shader_manager.h"
#include "dxvk_scoped_annotation.h"
#include "dxvk_context.h"
#include "rtx_context.h"
#include "rtx_scene_manager.h"
#include "rtx_camera_manager.h"
#include "rtx_options.h"
#include "../../util/log/log.h"
#include "../../util/util_once.h"
#include "../../util/util_string.h"

#include <rtx_shaders/scene_cull.h>

namespace dxvk {

  namespace {
    class SceneCullShader : public ManagedShader {
      SHADER_SOURCE(SceneCullShader, VK_SHADER_STAGE_COMPUTE_BIT, scene_cull)

      BEGIN_PARAMETER()
        CONSTANT_BUFFER(SCENE_CULL_BINDING_CONSTANTS)
        STRUCTURED_BUFFER(SCENE_CULL_BINDING_RECORDS)
        STRUCTURED_BUFFER(SCENE_CULL_BINDING_SOURCE)
        RW_STRUCTURED_BUFFER(SCENE_CULL_BINDING_CULLED)
        STRUCTURED_BUFFER(SCENE_CULL_BINDING_LIGHTS)
        RW_STRUCTURED_BUFFER(SCENE_CULL_BINDING_STATS)
        RW_STRUCTURED_BUFFER(SCENE_CULL_BINDING_VERDICTS)
      END_PARAMETER()
    };

    constexpr uint32_t kInstanceSize = uint32_t(sizeof(VkAccelerationStructureInstanceKHR));
    constexpr VkDeviceSize kStatsBytes = VkDeviceSize(SCENE_CULL_STATS_COUNT) * sizeof(uint32_t);

    // The verify's float error allowance, relative to the magnitude of the terms
    // behind each compared quantity. A 4-term single-precision dot product
    // rounds to within ~4 ulp (~2.4e-7) of those terms; 1e-5 is ~40x that,
    // which covers FMA contraction on either side and the GPU's division in
    // the slab test.
    constexpr float kSlack = 1e-5f;

    Rc<DxvkBuffer> createDeviceBuffer(DxvkDevice* device, VkDeviceSize size, VkBufferUsageFlags usage,
                                      DxvkMemoryStats::Category category, const char* name) {
      DxvkBufferCreateInfo info;
      info.usage  = usage;
      info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      info.access = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      info.size   = align(size, 256);
      return device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, category, name);
    }

    Rc<DxvkBuffer> createReadback(DxvkDevice* device, VkDeviceSize size, const char* name) {
      DxvkBufferCreateInfo info;
      info.usage  = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
      info.access = VK_ACCESS_TRANSFER_WRITE_BIT;
      info.size   = size;
      // CACHED: the harvest reads every byte on the CPU.
      return device->createBuffer(info,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
        DxvkMemoryStats::Category::RTXBuffer, name);
    }

    // Grow-only with headroom, so a scene that grows by a few entries per frame
    // does not reallocate every frame.
    VkDeviceSize grownSize(VkDeviceSize need) {
      return need + need / 2 + 256;
    }

    bool finiteF(float x) {
      return std::isfinite(x);
    }

    // scene_cull.slangh sceneCullBoxDistSq, same compare-and-select form.
    float boxDistSq(const Vector3& p, const float lo[3], const float hi[3]) {
      float dx = lo[0] - p.x; if (p.x - hi[0] > dx) { dx = p.x - hi[0]; } if (!(dx > 0.0f)) { dx = 0.0f; }
      float dy = lo[1] - p.y; if (p.y - hi[1] > dy) { dy = p.y - hi[1]; } if (!(dy > 0.0f)) { dy = 0.0f; }
      float dz = lo[2] - p.z; if (p.z - hi[2] > dz) { dz = p.z - hi[2]; } if (!(dz > 0.0f)) { dz = 0.0f; }
      return dx * dx + dy * dy + dz * dz;
    }

    // scene_cull.slangh sceneCullSegHitsBox.
    bool segHitsBox(const Vector4& a, const Vector4& b, float pad, const float lo[3], const float hi[3]) {
      float t0 = 0.0f;
      float t1 = 1.0f;
      for (uint32_t ax = 0; ax < 3; ++ax) {
        const float sLo = lo[ax] - pad;
        const float sHi = hi[ax] + pad;
        const float d = b[ax] - a[ax];
        if (std::abs(d) < 1e-6f) {
          if (a[ax] < sLo || a[ax] > sHi) {
            return false;
          }
          continue;
        }
        float ta = (sLo - a[ax]) / d;
        float tb = (sHi - a[ax]) / d;
        if (ta > tb) {
          const float tmp = ta;
          ta = tb;
          tb = tmp;
        }
        if (ta > t0) { t0 = ta; }
        if (tb < t1) { t1 = tb; }
        if (t0 > t1) {
          return false;
        }
      }
      return true;
    }

    const char* verdictName(uint32_t v) {
      switch (v) {
      case SCENE_CULL_VERDICT_UNTESTED:     return "untested";
      case SCENE_CULL_VERDICT_KEPT_FRUSTUM: return "keptFrustum";
      case SCENE_CULL_VERDICT_KEPT_RADIUS:  return "keptRadius";
      case SCENE_CULL_VERDICT_KEPT_LIGHT:   return "keptLight";
      case SCENE_CULL_VERDICT_KEPT_SKINNED: return "keptSkinned";
      case SCENE_CULL_VERDICT_CULLED:       return "culled";
      case SCENE_CULL_VERDICT_CULLED_SMALL: return "culledSmall";
      default:                              return "invalid";
      }
    }
  }

  bool SceneCullPass::isSkinned(const BlasEntry& blas) {
    return blas.input.getSkinningState().numBones > 0
        || blas.input.getGeometryData().numBonesPerVertex > 0;
  }

  // ==========================================================================
  // Frame setup -- the CPU loop's hoisted setup, moved here unchanged in
  // meaning. See RtxOptions::SceneCull for what each term is for.
  // ==========================================================================
  void SceneCullPass::beginFrame(Rc<DxvkContext> ctx, const CameraManager& cameraManager, uint32_t frameId) {
    harvest(frameId);
    logStats(frameId);

    m_enabled = false;
    m_used = false;
    m_constants = SceneCullConstants {};
    m_lights.clear();
    m_lightTableSize = 0;

    if (!RtxOptions::SceneCull::enable()) {
      return;
    }
    const RtCamera& cam = cameraManager.getMainCamera();
    // A camera that was never updated this session has no usable matrices; its
    // worldToProj would cull the entire scene. isValid() is the same gate the
    // rest of the frame uses before trusting a camera.
    if (!cam.isValid(frameId)) {
      return;
    }

    const float radius = RtxOptions::SceneCull::radius();
    const bool radiusOn  = radius > 0.0f;
    const bool frustumOn = RtxOptions::SceneCull::frustumEnable();
    const bool lightOn   = RtxOptions::SceneCull::lightInfluenceEnable();
    // No keep enabled would make the union empty and cull the whole scene;
    // treat it as disabled instead. The solid-angle reject rides along only
    // when at least one keep term is on (it prunes the KEPT set).
    if (!radiusOn && !frustumOn && !lightOn) {
      return;
    }

    uint32_t flags = SCENE_CULL_FLAG_ACTIVE;
    if (frustumOn) { flags |= SCENE_CULL_FLAG_FRUSTUM; }
    if (radiusOn)  { flags |= SCENE_CULL_FLAG_RADIUS; }
    if (lightOn)   { flags |= SCENE_CULL_FLAG_LIGHT; }

    const Vector3 camPos = cam.getPosition();
    const Matrix4 worldToProj = Matrix4(cam.getViewToProjection() * cam.getWorldToView());
    // Matrix4 is column-major (m[c] is column c): row r is (m[0][r] .. m[3][r]).
    m_constants.worldToProjRow0 = vec4(worldToProj[0][0], worldToProj[1][0], worldToProj[2][0], worldToProj[3][0]);
    m_constants.worldToProjRow1 = vec4(worldToProj[0][1], worldToProj[1][1], worldToProj[2][1], worldToProj[3][1]);
    m_constants.worldToProjRow2 = vec4(worldToProj[0][2], worldToProj[1][2], worldToProj[2][2], worldToProj[3][2]);
    m_constants.worldToProjRow3 = vec4(worldToProj[0][3], worldToProj[1][3], worldToProj[2][3], worldToProj[3][3]);
    m_constants.camPos = camPos;
    m_constants.radiusSq = radius * radius;
    m_constants.sideScale = 1.0f + std::max(0.0f, RtxOptions::SceneCull::frustumMargin());

    if (lightOn) {
      float visR = RtxOptions::SceneCull::visibleRange();
      if (visR <= 0.0f) {
        visR = radiusOn ? radius : 50000.0f;
      }
      // Half-diagonal of the visible-bounds cube: the capsule pad that makes
      // the segment sweep contain the cube sweep.
      const float visPad = 1.7320508f * visR;
      const float sunLen = std::max(visR, RtxOptions::SceneCull::lightInfluenceSunLength());
      // Live table, no snapshot, no lag: m_lights is filled by addLight()
      // during draw submission, which precedes this. Measured 2026-08-06:
      // [EngineLights.census] resident=0 active=0 on every frame of the BT
      // mission -- TF2 lights that map entirely with the sky/dome environment.
      const auto& table = ctx->getCommonObjects()->getSceneManager().getLightManager().getLightTable();
      m_lightTableSize = uint32_t(table.size());
      std::vector<Vector4> points;
      uint32_t segs = 0;
      for (const auto& [lightHash, light] : table) {
        if (light.getType() == RtLightType::Distant) {
          // Direction sign convention unverified against the shader side, so
          // sweep BOTH ways along the axis: a wrong-signed one-way sweep would
          // cull real sun occluders, both ways only over-keeps.
          const Vector3 dir = light.getDirection();
          const Vector3 a = camPos - dir * sunLen;
          const Vector3 b = camPos + dir * sunLen;
          m_lights.push_back(Vector4(a.x, a.y, a.z, visPad));
          m_lights.push_back(Vector4(b.x, b.y, b.z, 0.0f));
        } else {
          // Position-carrying lights. Only the sphere exposes a radius; the
          // areal lights' extents are negligible against visPad.
          const float lr = (light.getType() == RtLightType::Sphere) ? light.getSphereLight().getRadius() : 0.0f;
          const Vector3 p = light.getPosition();
          m_lights.push_back(Vector4(camPos.x, camPos.y, camPos.z, visPad + lr));
          m_lights.push_back(Vector4(p.x, p.y, p.z, 0.0f));
          points.push_back(Vector4(p.x, p.y, p.z, 0.0f));
        }
        ++segs;
      }
      m_lights.insert(m_lights.end(), points.begin(), points.end());
      m_constants.lightSegCount = segs;
      m_constants.lightPointCount = uint32_t(points.size());
      // No lights in the table => the scene is lit by the sky/dome, whose light
      // arrives from every direction; no extrusion bounds its occluders, so the
      // only sound light keep is everything. The lever on such maps is the
      // solid-angle reject.
      if (segs == 0u) {
        flags |= SCENE_CULL_FLAG_LIGHT_ALL;
        ONCE(Logger::warn("[SceneCull] light table empty (sky/dome-lit scene) -- "
                          "light keep covers everything; solid-angle is the active cull"));
      }
    }

    // sec 2.2 solid-angle: magnitude, not orientation, so it is sound by sec 1's
    // rule and it is the term that actually culls on sky/dome-lit maps.
    const float saMin = RtxOptions::SceneCull::solidAngleMin();
    if (RtxOptions::SceneCull::solidAngleCull() && saMin > 0.0f) {
      flags |= SCENE_CULL_FLAG_SOLID_ANGLE;
    }
    m_constants.solidAngleMinSq = saMin * saMin;
    const float exempt = RtxOptions::SceneCull::solidAngleLightExemptRadius();
    m_constants.lightExemptDistSq = exempt * exempt;
    m_constants.flags = flags;
    m_enabled = true;
  }

  // ==========================================================================
  // Records
  // ==========================================================================
  void SceneCullPass::beginRecords(DxvkDevice* device, uint32_t count) {
    const VkDeviceSize need = std::max<VkDeviceSize>(VkDeviceSize(count), 1u) * sizeof(SceneCullRecord);
    if (m_recordBuffer == nullptr || m_recordBuffer->info().size < need) {
      m_recordBuffer = createDeviceBuffer(device, grownSize(need),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        DxvkMemoryStats::Category::RTXBuffer, "SceneCull records");
      m_recordBufferReplaced = true;
    }
    m_records.beginFrame(count, m_recordBufferReplaced);
    m_recordBufferReplaced = false;
    m_recordsCpu.resize(count);
  }

  void SceneCullPass::commitRecord(uint32_t index, const SceneCullRecord& record) {
    std::memcpy(m_records.scratch(), &record, sizeof(record));
    m_records.commit(index);
    m_recordsCpu[index] = record;
  }

  void SceneCullPass::invalidate() {
    m_records.invalidateAll();
  }

  // ==========================================================================
  // GPU
  // ==========================================================================
  void SceneCullPass::ensureBindingBuffers(Rc<DxvkContext> ctx, DxvkDevice* device) {
    if (m_cb == nullptr) {
      DxvkBufferCreateInfo info;
      info.usage  = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      info.access = VK_ACCESS_TRANSFER_WRITE_BIT;
      info.size   = sizeof(SceneCullConstants);
      m_cb = device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                  DxvkMemoryStats::Category::RTXBuffer, "SceneCull constants");
      m_cbHoldsActive = true;   // unwritten: force the first inactive write
    }
    if (m_lightBuffer == nullptr) {
      m_lightBuffer = createDeviceBuffer(device, 16 * sizeof(Vector4),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        DxvkMemoryStats::Category::RTXBuffer, "SceneCull lights");
    }
    if (m_stats == nullptr) {
      m_stats = createDeviceBuffer(device, kStatsBytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        DxvkMemoryStats::Category::RTXBuffer, "SceneCull stats");
    }
    if (m_verdicts == nullptr) {
      m_verdicts = createDeviceBuffer(device, 256,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        DxvkMemoryStats::Category::RTXBuffer, "SceneCull verdicts");
    }
  }

  void SceneCullPass::writeConstants(Rc<DxvkContext> ctx) {
    const DxvkBufferSliceHandle slice = m_cb->allocSlice();
    ctx->invalidateBuffer(m_cb, slice);
    ctx->writeToBuffer(m_cb, 0, sizeof(SceneCullConstants), &m_constants);
    m_cbHoldsActive = true;
  }

  void SceneCullPass::dispatch(Rc<DxvkContext> ctx, DxvkDevice* device, const Rc<DxvkBuffer>& instanceTable,
                               const uint32_t (&typeFirstRecord)[Tlas::Count],
                               const uint32_t (&typeBaseElement)[Tlas::Count],
                               uint32_t pointInstancerInstances, uint32_t frameId) {
    if (!m_enabled || instanceTable == nullptr) {
      return;
    }
    ScopedGpuProfileZone(ctx, "SceneCull");
    ensureBindingBuffers(ctx, device);

    const uint32_t entryCount = uint32_t(m_recordsCpu.size());

    // The TLAS input: written in full every frame (CPU entries here, PI entries
    // by the PI pass), so a new buffer holds nothing that needs carrying over.
    const VkDeviceSize tableBytes = instanceTable->info().size;
    if (m_culled == nullptr || m_culled->info().size < tableBytes) {
      DxvkBufferCreateInfo info;
      info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                 | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                 | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                 | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
      info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      info.access = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      info.size = tableBytes;
      m_culled = device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                      DxvkMemoryStats::Category::RTXAccelerationStructure, "SceneCull TLAS input");
    }

    const uint32_t verdictCount = entryCount + pointInstancerInstances;
    const VkDeviceSize verdictBytes = std::max<VkDeviceSize>(VkDeviceSize(verdictCount), 1u) * sizeof(uint32_t);
    if (m_verdicts->info().size < verdictBytes) {
      m_verdicts = createDeviceBuffer(device, grownSize(verdictBytes),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        DxvkMemoryStats::Category::RTXBuffer, "SceneCull verdicts");
    }

    if (!m_lights.empty()) {
      const VkDeviceSize lightBytes = VkDeviceSize(m_lights.size()) * sizeof(Vector4);
      if (m_lightBuffer->info().size < lightBytes) {
        m_lightBuffer = createDeviceBuffer(device, grownSize(lightBytes),
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          DxvkMemoryStats::Category::RTXBuffer, "SceneCull lights");
      }
      ctx->writeToBuffer(m_lightBuffer, 0, lightBytes, m_lights.data());
    }

    m_records.upload(ctx.ptr(), m_recordBuffer);

    for (uint32_t t = 0; t < Tlas::Count; ++t) {
      m_typeFirstRecord[t] = typeFirstRecord[t];
      m_typeBaseElement[t] = typeBaseElement[t];
    }
    m_constants.entryCount = entryCount;
    m_constants.verdictCount = verdictCount;
    m_constants.typeFirstRecord = uvec4 { typeFirstRecord[0], typeFirstRecord[1], typeFirstRecord[2], entryCount };
    m_constants.typeBaseElement = uvec4 { typeBaseElement[0], typeBaseElement[1], typeBaseElement[2], 0u };
    writeConstants(ctx);

    // Both producers (this pass and the PI pass) add into the stats.
    ctx->clearBuffer(m_stats, 0, kStatsBytes, 0u);

    if (entryCount > 0u) {
      ctx->bindResourceBuffer(SCENE_CULL_BINDING_CONSTANTS, DxvkBufferSlice(m_cb));
      ctx->bindResourceBuffer(SCENE_CULL_BINDING_RECORDS, DxvkBufferSlice(m_recordBuffer));
      ctx->bindResourceBuffer(SCENE_CULL_BINDING_SOURCE, DxvkBufferSlice(instanceTable));
      ctx->bindResourceBuffer(SCENE_CULL_BINDING_CULLED, DxvkBufferSlice(m_culled));
      ctx->bindResourceBuffer(SCENE_CULL_BINDING_LIGHTS, DxvkBufferSlice(m_lightBuffer));
      ctx->bindResourceBuffer(SCENE_CULL_BINDING_STATS, DxvkBufferSlice(m_stats));
      ctx->bindResourceBuffer(SCENE_CULL_BINDING_VERDICTS, DxvkBufferSlice(m_verdicts));
      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, SceneCullShader::getShader());
      const VkExtent3D workgroups = util::computeBlockCount(VkExtent3D { entryCount, 1, 1 }, VkExtent3D { 64, 1, 1 });
      ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
    }
    m_used = true;

    // The record table is a delta mirror like the others: same readback gate.
    if (RtxOptions::GpuScene::verify()) {
      m_records.harvestVerify(frameId);
      if ((frameId % std::max(1u, RtxOptions::GpuScene::verifyInterval())) == 0u) {
        m_records.scheduleVerify(ctx.ptr(), device, m_recordBuffer, frameId);
      }
    }
  }

  PointInstancerSceneCullBindings SceneCullPass::pointInstancerBindings(Rc<DxvkContext> ctx, DxvkDevice* device) {
    ensureBindingBuffers(ctx, device);
    if (!m_used && m_cbHoldsActive) {
      // Not used this frame: the PI verdict must read UNTESTED, whatever the
      // buffer held last. Written once per transition, not per frame.
      const SceneCullConstants inactive {};
      const DxvkBufferSliceHandle slice = m_cb->allocSlice();
      ctx->invalidateBuffer(m_cb, slice);
      ctx->writeToBuffer(m_cb, 0, sizeof(SceneCullConstants), &inactive);
      m_cbHoldsActive = false;
    }
    PointInstancerSceneCullBindings b;
    b.constants = DxvkBufferSlice(m_cb);
    b.lights    = DxvkBufferSlice(m_lightBuffer);
    b.stats     = DxvkBufferSlice(m_stats);
    b.verdicts  = DxvkBufferSlice(m_verdicts);
    return b;
  }

  // ==========================================================================
  // Readbacks
  // ==========================================================================
  void SceneCullPass::recordReadbacks(Rc<DxvkContext> ctx, DxvkDevice* device,
                                      const std::vector<VkAccelerationStructureInstanceKHR> (&mergedInstances)[Tlas::Count],
                                      const std::vector<PointInstancerBatch>& batches,
                                      uint32_t totalElements, uint32_t frameId) {
    if (!m_used) {
      return;
    }
    ctx->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);

    StatsSlot& s = m_statsRing[m_statsNext];
    if (!s.pending) {
      if (s.staging == nullptr) {
        s.staging = createReadback(device, kStatsBytes, "SceneCull stats readback");
      }
      ctx->copyBuffer(s.staging, 0, m_stats, 0, kStatsBytes);
      s.pending = true;
      s.frame = frameId;
      m_statsNext = (m_statsNext + 1u) % uint32_t(m_statsRing.size());
    }

    if (!RtxOptions::SceneCull::verify()
        || (frameId % std::max(1u, RtxOptions::GpuScene::verifyInterval())) != 0u) {
      return;
    }
    VerifySlot* slot = nullptr;
    for (VerifySlot& v : m_verify) {
      if (!v.pending) {
        slot = &v;
        break;
      }
    }
    if (slot == nullptr) {
      return;  // both readbacks still in flight; skip rather than stall
    }

    size_t mergedTotal = 0;
    for (uint32_t t = 0; t < Tlas::Count; ++t) {
      mergedTotal += mergedInstances[t].size();
    }
    if (mergedTotal != m_recordsCpu.size()) {
      // Records and entries disagree: the verify would compare the wrong pairs.
      ONCE(Logger::err(str::format("[SceneCull] verify skipped: ", m_recordsCpu.size(),
                                   " records for ", mergedTotal, " CPU-owned TLAS entries")));
      return;
    }

    const VkDeviceSize verdictBytes = VkDeviceSize(m_constants.verdictCount) * sizeof(uint32_t);
    const VkDeviceSize culledBytes = VkDeviceSize(totalElements) * kInstanceSize;
    if (verdictBytes == 0 || culledBytes == 0
        || verdictBytes > m_verdicts->info().size || culledBytes > m_culled->info().size) {
      return;
    }
    if (slot->verdictStaging == nullptr || slot->verdictStaging->info().size < verdictBytes) {
      slot->verdictStaging = createReadback(device, grownSize(verdictBytes), "SceneCull verdict readback");
    }
    if (slot->culledStaging == nullptr || slot->culledStaging->info().size < culledBytes) {
      slot->culledStaging = createReadback(device, grownSize(culledBytes), "SceneCull TLAS input readback");
    }
    ctx->copyBuffer(slot->verdictStaging, 0, m_verdicts, 0, verdictBytes);
    ctx->copyBuffer(slot->culledStaging, 0, m_culled, 0, culledBytes);

    slot->constants = m_constants;
    slot->lights = m_lights;
    slot->records = m_recordsCpu;
    slot->entries.clear();
    slot->entryElement.clear();
    slot->entries.reserve(mergedTotal);
    slot->entryElement.reserve(mergedTotal);
    for (uint32_t t = 0; t < Tlas::Count; ++t) {
      for (uint32_t i = 0; i < uint32_t(mergedInstances[t].size()); ++i) {
        slot->entries.push_back(mergedInstances[t][i]);
        slot->entryElement.push_back(m_typeBaseElement[t] + i);
      }
    }
    slot->pi.clear();
    for (const PointInstancerBatch& b : batches) {
      PiSnapshot p;
      p.objectToWorld = b.objectToWorld;
      p.transforms = b.transforms;
      p.boxMin = b.cullBoxMin;
      p.boxMax = b.cullBoxMax;
      p.recordFlags = b.cullRecordFlags;
      p.mask = b.instanceMask;
      p.verdictBase = b.cullVerdictBase;
      p.count = b.instanceCount;
      p.firstElement = b.instanceBufferByteOffset / kInstanceSize;
      p.blasReference = b.blasReference;
      slot->pi.push_back(std::move(p));
    }
    slot->verdictCount = m_constants.verdictCount;
    slot->totalElements = totalElements;
    slot->frame = frameId;
    slot->pending = true;
  }

  void SceneCullPass::harvest(uint32_t frameId) {
    for (StatsSlot& s : m_statsRing) {
      if (!s.pending || s.staging == nullptr || s.staging->isInUse()) {
        continue;
      }
      s.pending = false;
      const void* data = s.staging->mapPtr(0);
      if (data != nullptr && s.frame >= m_lastStatsFrame) {
        std::memcpy(m_lastStats, data, sizeof(m_lastStats));
        m_lastStatsFrame = s.frame;
      }
    }
    for (VerifySlot& v : m_verify) {
      if (!v.pending || v.verdictStaging == nullptr || v.culledStaging == nullptr
          || v.verdictStaging->isInUse() || v.culledStaging->isInUse()) {
        continue;
      }
      harvestVerify(v, frameId);
    }
  }

  void SceneCullPass::harvestVerify(VerifySlot& slot, uint32_t frameId) {
    slot.pending = false;
    const uint32_t* gpuVerdicts = reinterpret_cast<const uint32_t*>(slot.verdictStaging->mapPtr(0));
    const uint8_t* gpuEntries = reinterpret_cast<const uint8_t*>(slot.culledStaging->mapPtr(0));
    if (gpuVerdicts == nullptr || gpuEntries == nullptr) {
      return;
    }

    uint32_t mismatch = 0, edge = 0, fail = 0, copyFail = 0;
    uint64_t entries = 0, piInstances = 0;
    std::string detail;
    uint32_t details = 0;
    constexpr uint32_t kMaxDetails = 8;

    auto judge = [&](const VerdictInput& in, uint32_t gpu, const char* kind, uint32_t index) {
      const uint32_t cpu = verdictCpu(slot.constants, slot.lights, in, 0);
      if (cpu == gpu) {
        return;
      }
      ++mismatch;
      const uint32_t keepTilt = verdictCpu(slot.constants, slot.lights, in, +1);
      const uint32_t cullTilt = verdictCpu(slot.constants, slot.lights, in, -1);
      if (keepTilt != cullTilt) {
        ++edge;
        return;
      }
      ++fail;
      if (details < kMaxDetails) {
        ++details;
        detail += str::format(" [", kind, " ", index,
          " gpu=", verdictName(gpu), " cpu=", verdictName(cpu),
          " box=(", in.boxMin.x, ",", in.boxMin.y, ",", in.boxMin.z, ")-(",
                    in.boxMax.x, ",", in.boxMax.y, ",", in.boxMax.z, ")",
          " t=(", in.xr[0][3], ",", in.xr[1][3], ",", in.xr[2][3], ")",
          " rec=0x", std::hex, in.recordFlags, " mask=0x", in.mask, std::dec, "]");
      }
    };

    // CPU-owned TLAS entries.
    for (uint32_t r = 0; r < uint32_t(slot.records.size()); ++r) {
      const VkAccelerationStructureInstanceKHR& e = slot.entries[r];
      const SceneCullRecord& rec = slot.records[r];
      VerdictInput in;
      for (uint32_t row = 0; row < 3; ++row) {
        for (uint32_t col = 0; col < 4; ++col) {
          in.xr[row][col] = e.transform.matrix[row][col];
          in.xm[row][col] = std::abs(e.transform.matrix[row][col]);
        }
      }
      in.boxMin = Vector3(rec.boxMin.x, rec.boxMin.y, rec.boxMin.z);
      in.boxMax = Vector3(rec.boxMax.x, rec.boxMax.y, rec.boxMax.z);
      in.recordFlags = rec.flags;
      in.mask = e.mask;
      const uint32_t gpu = (r < slot.verdictCount) ? gpuVerdicts[r] : ~0u;
      ++entries;
      judge(in, gpu, "entry", r);

      // The TLAS input entry must be the table's entry with exactly the
      // verdict's mask: any other difference is a copy defect, and a stale
      // entry (not rewritten this frame) shows up here as well.
      VkAccelerationStructureInstanceKHR expect = e;
      if (culls(gpu)) {
        expect.mask = 0;
      }
      const uint32_t elem = slot.entryElement[r];
      if (elem >= slot.totalElements
          || std::memcmp(gpuEntries + size_t(elem) * kInstanceSize, &expect, kInstanceSize) != 0) {
        ++copyFail;
      }
    }

    // PointInstancer instances: F = O * I exactly as the PI shader forms it
    // (point_instancer_culling.comp.slang), element (r,c) of a column-major
    // Matrix4 being m[c][r].
    for (const PiSnapshot& b : slot.pi) {
      if (b.transforms == nullptr) {
        continue;
      }
      const Matrix4& o2w = b.objectToWorld;
      const uint32_t n = std::min<uint32_t>(b.count, uint32_t(b.transforms->size()));
      for (uint32_t i = 0; i < n; ++i) {
        const Matrix4& i2o = (*b.transforms)[i];
        VerdictInput in;
        for (uint32_t row = 0; row < 3; ++row) {
          for (uint32_t col = 0; col < 4; ++col) {
            in.xr[row][col] = o2w[0][row] * i2o[col][0] + o2w[1][row] * i2o[col][1]
                            + o2w[2][row] * i2o[col][2] + o2w[3][row] * i2o[col][3];
            in.xm[row][col] = std::abs(o2w[0][row] * i2o[col][0]) + std::abs(o2w[1][row] * i2o[col][1])
                            + std::abs(o2w[2][row] * i2o[col][2]) + std::abs(o2w[3][row] * i2o[col][3]);
          }
        }
        in.boxMin = b.boxMin;
        in.boxMax = b.boxMax;
        in.recordFlags = b.recordFlags;
        in.mask = b.mask;
        const uint32_t vIdx = b.verdictBase + i;
        const uint32_t gpu = (vIdx < slot.verdictCount) ? gpuVerdicts[vIdx] : ~0u;
        ++piInstances;
        judge(in, gpu, "pi", vIdx);

        // A culled PI instance must have left the TLAS, and the entry must
        // still reference the batch's BLAS (the radius cull, force-disabled in
        // the PI system, is the only other writer of the mask).
        const uint32_t elem = b.firstElement + i;
        if (elem >= slot.totalElements) {
          ++copyFail;
          continue;
        }
        const uint8_t* pe = gpuEntries + size_t(elem) * kInstanceSize;
        uint32_t customIdxMask = 0;
        uint64_t blasRef = 0;
        std::memcpy(&customIdxMask, pe + 48, 4);
        std::memcpy(&blasRef, pe + 56, 8);
        if ((culls(gpu) && (customIdxMask >> 24) != 0u) || blasRef != b.blasReference) {
          ++copyFail;
        }
      }
    }

    ++m_verifyStats.readbacks;
    m_verifyStats.entries += entries;
    m_verifyStats.piInstances += piInstances;
    m_verifyStats.mismatch += mismatch;
    m_verifyStats.edge += edge;
    m_verifyStats.fail += fail;
    m_verifyStats.copyFail += copyFail;

    if (fail != 0u || copyFail != 0u) {
      // Unthrottled and ungated: a non-zero count is the defect.
      Logger::warn(str::format(
        "[SceneCull] VERIFY-FAIL f=", slot.frame, " harvested=", frameId,
        " entries=", entries, " pi=", piInstances,
        " mismatch=", mismatch, " edge=", edge, " FAIL=", fail, " copyFail=", copyFail,
        "  <- FAIL: the GPU verdict disagrees with the CPU reference away from any threshold;"
        " copyFail: the TLAS input is not the instance table with the verdict's mask.",
        detail));
    }
  }

  void SceneCullPass::logStats(uint32_t frameId) {
    if (!m_enabled || !RtxOptions::SceneCull::logStats()) {
      return;
    }
    static auto s_last = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - s_last).count() < 1000) {
      return;
    }
    s_last = now;

    const uint32_t* st = m_lastStats;
    const uint32_t* pi = m_lastStats + SCENE_CULL_STATS_PI_BASE;
    uint32_t tested = 0, piTested = 0;
    for (uint32_t v = SCENE_CULL_VERDICT_KEPT_FRUSTUM; v < SCENE_CULL_VERDICT_COUNT; ++v) {
      tested += st[v];
      piTested += pi[v];
    }
    const DeltaUploadTable::Stats& rs = m_records.stats();
    // tested/kept*/culled*: TLAS entries, by the FIRST keep term that covered
    // them (frustum before radius before light). pi{}: PointInstancer
    // instances, same verdict. rec{}: the record table's delta upload -- up
    // must be ~0 on a held scene. verify{}: FAIL and copyFail must read 0.
    Logger::warn(str::format("[SceneCull] f=", frameId, " statsF=", m_lastStatsFrame,
      " tested=", tested,
      " keptFrustum=", st[SCENE_CULL_VERDICT_KEPT_FRUSTUM],
      " keptRadius=", st[SCENE_CULL_VERDICT_KEPT_RADIUS],
      " keptLight=", st[SCENE_CULL_VERDICT_KEPT_LIGHT],
      " keptSkinned=", st[SCENE_CULL_VERDICT_KEPT_SKINNED],
      " culled=", st[SCENE_CULL_VERDICT_CULLED],
      " culledSmall=", st[SCENE_CULL_VERDICT_CULLED_SMALL],
      " untested=", st[SCENE_CULL_VERDICT_UNTESTED],
      " pi{tested=", piTested,
      " culled=", pi[SCENE_CULL_VERDICT_CULLED],
      " culledSmall=", pi[SCENE_CULL_VERDICT_CULLED_SMALL],
      " keptSkinned=", pi[SCENE_CULL_VERDICT_KEPT_SKINNED], "}",
      " lights=", m_lightTableSize,
      " lightAllKeep=", ((m_constants.flags & SCENE_CULL_FLAG_LIGHT_ALL) != 0u ? 1 : 0),
      " radius=", RtxOptions::SceneCull::radius(),
      " margin=", RtxOptions::SceneCull::frustumMargin(),
      " rec{n=", m_recordsCpu.size(),
      " chg=", rs.changed, "/", rs.elements,
      " up=", rs.uploadBytes, " full=", rs.fullBytes,
      " rbFail=", rs.verifyFail, "}",
      " verify{rb=", m_verifyStats.readbacks,
      " entries=", m_verifyStats.entries,
      " pi=", m_verifyStats.piInstances,
      " mismatch=", m_verifyStats.mismatch,
      " edge=", m_verifyStats.edge,
      " FAIL=", m_verifyStats.fail,
      " copyFail=", m_verifyStats.copyFail, "}"));
    m_records.resetStats();
  }

  // ==========================================================================
  // THE CPU REFERENCE. Keep in lockstep with scene_cull.slangh.
  // ==========================================================================
  uint32_t SceneCullPass::verdictCpu(const SceneCullConstants& sc, const std::vector<Vector4>& lights,
                                     const VerdictInput& in, int sigma) {
    const uint32_t f = sc.flags;
    if ((f & SCENE_CULL_FLAG_ACTIVE) == 0u || (in.recordFlags & SCENE_CULL_RECORD_TESTED) == 0u || in.mask == 0u) {
      return SCENE_CULL_VERDICT_UNTESTED;
    }

    const float rows[4][4] = {
      { sc.worldToProjRow0.x, sc.worldToProjRow0.y, sc.worldToProjRow0.z, sc.worldToProjRow0.w },
      { sc.worldToProjRow1.x, sc.worldToProjRow1.y, sc.worldToProjRow1.z, sc.worldToProjRow1.w },
      { sc.worldToProjRow2.x, sc.worldToProjRow2.y, sc.worldToProjRow2.z, sc.worldToProjRow2.w },
      { sc.worldToProjRow3.x, sc.worldToProjRow3.y, sc.worldToProjRow3.z, sc.worldToProjRow3.w },
    };
    const bool frustumOn = (f & SCENE_CULL_FLAG_FRUSTUM) != 0u;
    const float sg = float(sigma);
    // sigma > 0 tilts toward keeping, < 0 toward culling; 0 is exact.
    const float grow   = (sigma == 0) ? 1.0f : 1.0f + sg * kSlack;
    const float shrink = (sigma == 0) ? 1.0f : 1.0f - sg * kSlack;

    float lo[3] = {  3.402823466e38f,  3.402823466e38f,  3.402823466e38f };
    float hi[3] = { -3.402823466e38f, -3.402823466e38f, -3.402823466e38f };
    uint32_t outcodeAnd = 0x1Fu;
    for (uint32_t c = 0; c < 8u; ++c) {
      const float p[3] = {
        ((c & 1u) != 0u) ? in.boxMax.x : in.boxMin.x,
        ((c & 2u) != 0u) ? in.boxMax.y : in.boxMin.y,
        ((c & 4u) != 0u) ? in.boxMax.z : in.boxMin.z,
      };
      float w[3];
      float ew[3] = { 0.0f, 0.0f, 0.0f };
      for (uint32_t r = 0; r < 3; ++r) {
        w[r] = in.xr[r][0] * p[0] + in.xr[r][1] * p[1] + in.xr[r][2] * p[2] + in.xr[r][3];
        if (sigma != 0) {
          ew[r] = kSlack * (in.xm[r][0] * std::abs(p[0]) + in.xm[r][1] * std::abs(p[1])
                          + in.xm[r][2] * std::abs(p[2]) + in.xm[r][3]);
        }
        const float wl = (sigma == 0) ? w[r] : w[r] - sg * ew[r];
        const float wh = (sigma == 0) ? w[r] : w[r] + sg * ew[r];
        lo[r] = (wl < lo[r]) ? wl : lo[r];
        hi[r] = (hi[r] < wh) ? wh : hi[r];
      }
      if (frustumOn) {
        float clip[4];
        float ec[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        for (uint32_t k = 0; k < 4; ++k) {
          clip[k] = rows[k][0] * w[0] + rows[k][1] * w[1] + rows[k][2] * w[2] + rows[k][3];
          if (sigma != 0) {
            ec[k] = kSlack * (std::abs(rows[k][0] * w[0]) + std::abs(rows[k][1] * w[1])
                            + std::abs(rows[k][2] * w[2]) + std::abs(rows[k][3]))
                  + std::abs(rows[k][0]) * ew[0] + std::abs(rows[k][1]) * ew[1] + std::abs(rows[k][2]) * ew[2];
          }
        }
        const float cx = clip[0], cy = clip[1], cz = clip[2], cw = clip[3];
        uint32_t oc = 0u;
        // Tilted: a corner only counts as in front when its w clears the
        // allowance (keep tilt) / when it might (cull tilt).
        const bool wPositive = (sigma == 0) ? (cw > 0.0f) : (cw > sg * ec[3]);
        if (wPositive) {
          const float lim = cw * sc.sideScale;
          const float tx = (sigma == 0) ? 0.0f : sg * (ec[0] + sc.sideScale * ec[3]);
          const float ty = (sigma == 0) ? 0.0f : sg * (ec[1] + sc.sideScale * ec[3]);
          if (cx < -lim - tx) { oc |= 0x01u; }
          if (cx >  lim + tx) { oc |= 0x02u; }
          if (cy < -lim - ty) { oc |= 0x04u; }
          if (cy >  lim + ty) { oc |= 0x08u; }
        }
        const float tz = (sigma == 0) ? 0.0f : sg * ec[2];
        if (cz < 0.0f - tz) { oc |= 0x10u; }
        outcodeAnd &= oc;
      }
    }

    bool kept = false;
    bool frustumKept = false;
    uint32_t verdict = SCENE_CULL_VERDICT_UNTESTED;
    if (frustumOn && outcodeAnd == 0u) {
      kept = true;
      frustumKept = true;
      verdict = SCENE_CULL_VERDICT_KEPT_FRUSTUM;
    }
    const Vector3 camPos(sc.camPos.x, sc.camPos.y, sc.camPos.z);
    if (!kept && (f & SCENE_CULL_FLAG_RADIUS) != 0u) {
      const float d2 = boxDistSq(camPos, lo, hi);
      if (!finiteF(d2) || d2 <= sc.radiusSq * grow) {
        kept = true;
        verdict = SCENE_CULL_VERDICT_KEPT_RADIUS;
      }
    }
    if (!kept && (f & SCENE_CULL_FLAG_LIGHT) != 0u) {
      if ((f & SCENE_CULL_FLAG_LIGHT_ALL) != 0u) {
        kept = true;
        verdict = SCENE_CULL_VERDICT_KEPT_LIGHT;
      } else {
        for (uint32_t s = 0; s < sc.lightSegCount && 2u * s + 1u < lights.size(); ++s) {
          const Vector4& a = lights[2u * s];
          const Vector4& b = lights[2u * s + 1u];
          if (segHitsBox(a, b, a.w * grow, lo, hi)) {
            kept = true;
            verdict = SCENE_CULL_VERDICT_KEPT_LIGHT;
            break;
          }
        }
      }
    }

    if ((in.recordFlags & SCENE_CULL_RECORD_SKINNED) != 0u) {
      return kept ? verdict : SCENE_CULL_VERDICT_KEPT_SKINNED;
    }
    if (!kept) {
      return SCENE_CULL_VERDICT_CULLED;
    }
    if ((f & SCENE_CULL_FLAG_SOLID_ANGLE) != 0u && !frustumKept) {
      float maxExtent = hi[0] - lo[0];
      if (hi[1] - lo[1] > maxExtent) { maxExtent = hi[1] - lo[1]; }
      if (hi[2] - lo[2] > maxExtent) { maxExtent = hi[2] - lo[2]; }
      const float distSq = boxDistSq(camPos, lo, hi);
      if (finiteF(distSq) && distSq > 0.0f && finiteF(maxExtent)
          && maxExtent * maxExtent < sc.solidAngleMinSq * distSq * shrink) {
        bool nearLight = false;
        const uint32_t pointBase = 2u * sc.lightSegCount;
        for (uint32_t p = 0; p < sc.lightPointCount && pointBase + p < lights.size(); ++p) {
          const Vector4& lp4 = lights[pointBase + p];
          const Vector3 lp(lp4.x, lp4.y, lp4.z);
          if (boxDistSq(lp, lo, hi) <= sc.lightExemptDistSq * grow) {
            nearLight = true;
            break;
          }
        }
        if (!nearLight) {
          return SCENE_CULL_VERDICT_CULLED_SMALL;
        }
      }
    }
    return verdict;
  }

}
