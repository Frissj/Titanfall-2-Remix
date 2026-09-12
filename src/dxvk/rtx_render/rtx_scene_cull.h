#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "../dxvk_buffer.h"
#include "../../util/util_matrix.h"
#include "../../util/util_vector.h"
#include "rtx_types.h"
#include "rtx_gpu_scene.h"
#include "rtx_point_instancer_system.h"
#include "rtx/pass/instance_culling/scene_cull_binding_indices.h"

namespace dxvk {

  class DxvkContext;
  class DxvkDevice;
  class CameraManager;
  struct BlasEntry;

  // ==========================================================================
  // THE SCENE CULL ON THE GPU -- ARCHITECTURE_OVERHAUL.md slice 9 (sec 5.3).
  //
  // WHAT THIS REPLACES. RtxOptions::SceneCull was a serial CPU loop inside
  // AccelManager::mergeInstancesIntoBlas: per RtInstance, 8 corner transforms,
  // a clip-space outcode, the keep union and the solid-angle reject, deciding
  // mask=0 BEFORE bucketing -- the same job the PointInstancer culling shader
  // already did on the GPU for PI instances. Once residency restores the
  // off-screen population that loop is O(level) on the CS thread every frame,
  // and because it cut instances out of their buckets, every camera turn
  // changed bucket membership and rebuilt merged BLASes.
  //
  // THE TWO-STAGE MODEL IT LEAVES (sec 5.3):
  //   CPU    which objects exist?                -> the TLAS entries
  //   GPU    which of them matter to this view?  -> mask = 0
  // The BLAS set no longer depends on the view at all.
  //
  // THE UNIT IS THE TLAS ENTRY, which is the unit a TLAS can drop. A dynamic
  // BLAS entry is one instance and its verdict is the old one. A merged bucket
  // is one entry whose geometry is baked world-space, so it is judged by the
  // union of its members' world boxes and kept whole if any part of it is kept
  // -- over-keeping, the safe direction every keep term already errs in. A
  // PointInstancer instance is judged by the PI shader with the SAME verdict
  // function (scene_cull.slangh) under its own F, where the CPU loop judged a
  // whole batch by the template box under the batch transform.
  //
  // DATA FLOW, per frame:
  //   beginFrame      options + main camera + light table -> SceneCullConstants
  //   records         one SceneCullRecord per CPU-owned TLAS entry, through a
  //                   DeltaUploadTable: a dynamic entry's record is its BLAS's
  //                   object box (static), so a held scene uploads ~nothing
  //   dispatch        scene_cull.comp: copies every CPU-owned entry from the
  //                   instance table into culledBuffer() with the verdict's mask
  //   PI pass         point_instancer_culling.comp writes its entries into the
  //                   same culledBuffer(), applying the same verdict
  //   buildTlas       builds from culledBuffer()
  //
  // WHY A SECOND INSTANCE BUFFER. The instance table is a DeltaUploadTable
  // mirror (G5): its bytes must equal the CPU's, or the delta skips uploads it
  // needs and its readback verify fails. Writing masks into it in place would
  // break both. The TLAS input is therefore a separate buffer written in full
  // every frame (CPU entries by this pass, PI entries by the PI pass), and only
  // the mask byte can differ from the table -- so G5's refit stays legal:
  // accelerationStructureReference, flags and count are copied untouched.
  //
  // THE GATE (rtx.sceneCull.verify). Every rtx.gpuScene.verifyInterval frames
  // the per-entry GPU verdicts and the TLAS input are read back (no stall) and
  // compared against verdictCpu() -- the shader's function line for line, run
  // on the same records, transforms, constants and lights. A mismatch is
  // classified by re-running verdictCpu with every comparison tilted by a float
  // error allowance both ways: if the two tilted verdicts agree, the GPU
  // disagreed with a verdict that is not near any threshold and that is a
  // FAIL; if they differ, the entry sits on a threshold and GPU/CPU float
  // contraction legitimately decides it (edge). FAIL must read 0.
  //
  // THREADING. CS thread only, like the rest of AccelManager.
  // ==========================================================================
  class SceneCullPass {
  public:
    // Once per frame, before anything else here: harvests finished readbacks,
    // logs, then reads the options, the main camera and the light table.
    void beginFrame(Rc<DxvkContext> ctx, const CameraManager& cameraManager, uint32_t frameId);

    // Options on, camera valid, at least one keep term enabled.
    bool enabled() const { return m_enabled; }
    // dispatch() ran this frame: the TLAS input is culledBuffer().
    bool used() const { return m_used; }
    const Rc<DxvkBuffer>& culledBuffer() const { return m_culled; }
    uint32_t entryCount() const { return m_constants.entryCount; }

    // Records: one per CPU-owned TLAS entry, in instance-table order.
    void beginRecords(DxvkDevice* device, uint32_t count);
    void commitRecord(uint32_t index, const SceneCullRecord& record);

    // After the instance table's upload. typeFirstRecord / typeBaseElement:
    // per TLAS type, the first record and the instance-table element it maps to.
    void dispatch(Rc<DxvkContext> ctx, DxvkDevice* device, const Rc<DxvkBuffer>& instanceTable,
                  const uint32_t (&typeFirstRecord)[Tlas::Count], const uint32_t (&typeBaseElement)[Tlas::Count],
                  uint32_t pointInstancerInstances, uint32_t frameId);

    // What the PointInstancer pass binds. Always valid: when the pass is not
    // used this frame the constants carry no ACTIVE bit and the PI verdict
    // reads UNTESTED.
    PointInstancerSceneCullBindings pointInstancerBindings(Rc<DxvkContext> ctx, DxvkDevice* device);

    // From buildTlas, after both producers ran: the stats readback every frame
    // and, on verify frames, the verdict + TLAS-input readback with the CPU
    // snapshot the verify compares against.
    void recordReadbacks(Rc<DxvkContext> ctx, DxvkDevice* device,
                         const std::vector<VkAccelerationStructureInstanceKHR> (&mergedInstances)[Tlas::Count],
                         const std::vector<PointInstancerBatch>& batches, uint32_t totalElements, uint32_t frameId);

    // Scene clear: the record mirror no longer vouches for anything.
    void invalidate();

    // The skinned exemption's test, shared by the record packer and the PI batch.
    static bool isSkinned(const BlasEntry& blas);
    static bool culls(uint32_t verdict) {
      return verdict == SCENE_CULL_VERDICT_CULLED || verdict == SCENE_CULL_VERDICT_CULLED_SMALL;
    }

    // THE CPU REFERENCE -- scene_cull.slangh's sceneCullVerdict, line for line.
    // sigma = 0 is the reference; +1 / -1 tilt every comparison toward keeping
    // / culling by a float error allowance built from xm (per transform row,
    // the magnitudes of the terms that produced it), for edge classification.
    struct VerdictInput {
      float xr[3][4];
      float xm[3][4];
      Vector3 boxMin;
      Vector3 boxMax;
      uint32_t recordFlags;
      uint32_t mask;
    };
    static uint32_t verdictCpu(const SceneCullConstants& sc, const std::vector<Vector4>& lights,
                               const VerdictInput& in, int sigma);

  private:
    void harvest(uint32_t frameId);
    void logStats(uint32_t frameId);
    void ensureBindingBuffers(Rc<DxvkContext> ctx, DxvkDevice* device);
    void writeConstants(Rc<DxvkContext> ctx);

    bool m_enabled = false;
    bool m_used = false;
    bool m_cbHoldsActive = true;   // forces the first inactive write
    SceneCullConstants m_constants {};
    std::vector<Vector4> m_lights;  // segs as (a,pad),(b,0) pairs, then points (p,0)
    uint32_t m_lightTableSize = 0;  // lights in the table this frame (log only)

    DeltaUploadTable m_records { uint32_t(sizeof(SceneCullRecord)), "sceneCullRecord" };
    std::vector<SceneCullRecord> m_recordsCpu;   // the verify's snapshot source
    bool m_recordBufferReplaced = false;

    Rc<DxvkBuffer> m_cb;
    Rc<DxvkBuffer> m_recordBuffer;
    Rc<DxvkBuffer> m_culled;
    Rc<DxvkBuffer> m_lightBuffer;
    Rc<DxvkBuffer> m_stats;
    Rc<DxvkBuffer> m_verdicts;
    uint32_t m_typeFirstRecord[Tlas::Count] = {};
    uint32_t m_typeBaseElement[Tlas::Count] = {};

    // Stats readback ring: written by both producers, harvested without a stall.
    struct StatsSlot {
      Rc<DxvkBuffer> staging;
      uint32_t frame = 0;
      bool pending = false;
    };
    std::array<StatsSlot, 3> m_statsRing;
    uint32_t m_statsNext = 0;
    uint32_t m_lastStats[SCENE_CULL_STATS_COUNT] = {};
    uint32_t m_lastStatsFrame = 0;

    // Verify: the GPU's verdicts and TLAS input, plus everything verdictCpu
    // needs to recompute them, captured in the same frame.
    struct PiSnapshot {
      Matrix4 objectToWorld;
      std::shared_ptr<const std::vector<Matrix4>> transforms;
      Vector3 boxMin;
      Vector3 boxMax;
      uint32_t recordFlags = 0;
      uint32_t mask = 0;
      uint32_t verdictBase = 0;
      uint32_t count = 0;
      uint32_t firstElement = 0;
      uint64_t blasReference = 0;
    };
    struct VerifySlot {
      Rc<DxvkBuffer> verdictStaging;
      Rc<DxvkBuffer> culledStaging;
      SceneCullConstants constants {};
      std::vector<Vector4> lights;
      std::vector<SceneCullRecord> records;
      std::vector<VkAccelerationStructureInstanceKHR> entries;   // record order
      std::vector<uint32_t> entryElement;                        // record -> instance-table element
      std::vector<PiSnapshot> pi;
      uint32_t verdictCount = 0;
      uint32_t totalElements = 0;
      uint32_t frame = 0;
      bool pending = false;
    };
    std::array<VerifySlot, 2> m_verify;
    void harvestVerify(VerifySlot& slot, uint32_t frameId);

    struct VerifyStats {
      uint32_t readbacks = 0;
      uint64_t entries = 0;       // TLAS entries compared
      uint64_t piInstances = 0;   // PointInstancer instances compared
      uint32_t mismatch = 0;      // GPU verdict != CPU verdict
      uint32_t edge = 0;          //   of which on a threshold (float contraction decides)
      uint32_t fail = 0;          //   of which NOT on a threshold -- MUST stay 0
      uint32_t copyFail = 0;      // TLAS input entry != table entry with the verdict's mask -- MUST stay 0
    };
    VerifyStats m_verifyStats;    // cumulative
  };

}
