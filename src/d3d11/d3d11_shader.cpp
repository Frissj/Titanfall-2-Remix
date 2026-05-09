#include "d3d11_device.h"
#include "d3d11_shader.h"

#include <atomic>
#include <cstring>
#include <fstream>
#include <mutex>
#include <unordered_set>

#include <windows.h>
#include <d3d11shader.h>

#include "../dxvk/rtx_render/rtx_options.h"
#include "../util/util_filesys.h"

namespace dxvk {
  
  D3D11CommonShader:: D3D11CommonShader() { }
  D3D11CommonShader::~D3D11CommonShader() { }
  
  
  D3D11CommonShader::D3D11CommonShader(
          D3D11Device*    pDevice,
    const DxvkShaderKey*  pShaderKey,
    const DxbcModuleInfo* pDxbcModuleInfo,
    const void*           pShaderBytecode,
          size_t          BytecodeLength) {
    const std::string name = pShaderKey->toString();
    Logger::debug(str::format("Compiling shader ", name));
    
    DxbcReader reader(
      reinterpret_cast<const char*>(pShaderBytecode),
      BytecodeLength);
    
    DxbcModule module(reader);
    
    // If requested by the user, dump both the raw DXBC
    // shader and the compiled SPIR-V module to a file.
    // NV-DXVK: also force-dump to a hardcoded fallback path when the
    // env var is not set, so shaders that matter for debug work are
    // always on disk for fxc /dumpbin even if the user forgot to set
    // DXVK_SHADER_DUMP_PATH.
    std::string dumpPath = env::getEnvVar("DXVK_SHADER_DUMP_PATH");
    if (dumpPath.empty()) {
      dumpPath = "shader_dumps";
    }
    reader.store(std::ofstream(str::tows(str::format(dumpPath, "/", name, ".dxbc").c_str()).c_str(),
      std::ios_base::binary | std::ios_base::trunc));
    
    // Decide whether we need to create a pass-through
    // geometry shader for vertex shader stream output
    bool passthroughShader = pDxbcModuleInfo->xfb != nullptr
      && (module.programInfo().type() == DxbcProgramType::VertexShader
       || module.programInfo().type() == DxbcProgramType::DomainShader);

    if (module.programInfo().shaderStage() != pShaderKey->type() && !passthroughShader)
      throw DxvkError("Mismatching shader type.");

    m_shader = passthroughShader
      ? module.compilePassthroughShader(*pDxbcModuleInfo, name)
      : module.compile                 (*pDxbcModuleInfo, name);
    m_shader->setShaderKey(*pShaderKey);
    // NV-DXVK: log the (m_hash → SHA1-name) mapping so logs that reference
    // shaders by getHash() (uint64 fold) can be correlated with the SHA1-named
    // .dxbc files dumped by DXVK_SHADER_DUMP_PATH.
    Logger::info(str::format("[ShaderHashMap] ", name, " getHash=0x", std::hex, m_shader->getHash(), std::dec));
    
    if (dumpPath.size() != 0) {
      std::ofstream dumpStream(
        str::tows(str::format(dumpPath, "/", name, ".spv").c_str()).c_str(),
        std::ios_base::binary | std::ios_base::trunc);
      
      m_shader->dump(dumpStream);
    }
    
    // Create shader constant buffer if necessary
    if (m_shader->shaderConstants().data() != nullptr) {
      DxvkBufferCreateInfo info;
      info.size   = m_shader->shaderConstants().sizeInBytes();
      info.usage  = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
      info.stages = util::pipelineStages(m_shader->stage());
      info.access = VK_ACCESS_UNIFORM_READ_BIT;
      
      VkMemoryPropertyFlags memFlags
        = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
        | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
      
      m_buffer = pDevice->GetDXVKDevice()->createBuffer(info, memFlags, DxvkMemoryStats::Category::AppBuffer, "d3d11 shader constants");

      std::memcpy(m_buffer->mapPtr(0),
        m_shader->shaderConstants().data(),
        m_shader->shaderConstants().sizeInBytes());
    }

    pDevice->GetDXVKDevice()->registerShader(m_shader);

    // NV-DXVK: parse RDEF chunk so ExtractTransforms can look up cbuffer
    // bind slots + field offsets deterministically instead of guessing.
    parseRdef(pShaderBytecode, BytecodeLength);
    // NV-DXVK: stamp each cbuffer field's `used` flag using D3DReflect
    // (ground-truth via D3D_SVF_USED). Needed so we don't apply UV
    // transforms sourced from cbuffer fields the shader never touches
    // (stale data from a previous draw).
    populateFieldUsage(pShaderBytecode, BytecodeLength);
    // NV-DXVK: detect whether the PS declares any color output. "no Output"
    // PSes are depth/alpha-cutout passes; their bound albedoTexture is not
    // authoritative material colour.
    parseOsgn(pShaderBytecode, BytecodeLength);
    // NV-DXVK: read ISGN for per-input-semantic component type. Lets the
    // BLAS path detect packed-uint TEXCOORD encoding (TF2 BSP world VSes
    // pack a uint into TEXCOORD0 and bit-decode it to a tile UV in the VS;
    // a generic Surface flag means we don't have to maintain a per-VS hash
    // allowlist for a feature that's discoverable from the shader itself).
    parseIsgn(pShaderBytecode, BytecodeLength);
    // NV-DXVK [EngineLightsCapture]: write disassembly to disk for any
    // shader that binds s_globalLights so we can read the actual byte
    // offsets it loads from. parseRdef has already filled m_resourceSlots
    // by this point.
    dumpShaderAsmIfRelevant(pShaderBytecode, BytecodeLength);
  }

  // NV-DXVK: minimal DXBC OSGN parser. Sets m_hasColorOutput based on the
  // element count at OSGN body offset 0 — zero elements means the shader
  // writes nothing (depth-only / alpha-cutout pass).
  void D3D11CommonShader::parseOsgn(const void* pShaderBytecode, size_t BytecodeLength) {
    if (pShaderBytecode == nullptr || BytecodeLength < 32) return;
    const uint8_t* base = reinterpret_cast<const uint8_t*>(pShaderBytecode);
    if (std::memcmp(base, "DXBC", 4) != 0) return;
    auto rdU32 = [&](size_t off) -> uint32_t {
      uint32_t v; std::memcpy(&v, base + off, sizeof(v)); return v;
    };
    const uint32_t chunkCount = rdU32(28);
    if (chunkCount == 0 || 32 + chunkCount * 4 > BytecodeLength) return;
    // Try OSG5 first (SM5+ with stream), fall back to OSGN (SM4). Both
    // encode the element count at body offset 0 the same way.
    for (uint32_t i = 0; i < chunkCount; ++i) {
      const uint32_t chunkOff = rdU32(32 + i * 4);
      if (chunkOff + 8 > BytecodeLength) continue;
      const bool isOutputSig =
        std::memcmp(base + chunkOff, "OSGN", 4) == 0 ||
        std::memcmp(base + chunkOff, "OSG1", 4) == 0 ||
        std::memcmp(base + chunkOff, "OSG5", 4) == 0;
      if (!isOutputSig) continue;
      const uint32_t chunkSize = rdU32(chunkOff + 4);
      if (chunkSize < 4 || chunkOff + 8 + chunkSize > BytecodeLength) return;
      const uint32_t elemCount = rdU32(chunkOff + 8);
      m_hasColorOutput = (elemCount > 0);
      return;
    }
    // No OSGN chunk at all — treat as no color output (safer default).
    m_hasColorOutput = false;
  }

  // NV-DXVK: minimal DXBC ISGN parser. Records each input semantic's
  // D3D_REGISTER_COMPONENT_TYPE so the BLAS path can know whether the VS
  // expects float or uint at TEXCOORD0/1 etc. (TF2 BSP world VSes declare
  // TEXCOORD0/1 as uint and apply a bit-decode in the VS body — Remix's
  // BLAS reads them as float and gets garbage UVs with magnitudes in the
  // hundreds, producing per-pixel gradients > 1 → mip 8+ → 1×1 sample →
  // flat walls).
  //
  // ISGN element layout (24 bytes for ISGN/SM4, 28 bytes for ISG1/SM5.1):
  //   off  0: nameOffset (relative to chunk body start = chunkOff+8)
  //   off  4: semanticIndex
  //   off  8: systemValueType
  //   off 12: componentType  (1=uint, 2=sint, 3=float)
  //   off 16: register
  //   off 20: mask, readWriteMask, 2B pad
  //   [off 24: minPrecision  — only present in ISG1]
  void D3D11CommonShader::parseIsgn(const void* pShaderBytecode, size_t BytecodeLength) {
    if (pShaderBytecode == nullptr || BytecodeLength < 32) return;
    const uint8_t* base = reinterpret_cast<const uint8_t*>(pShaderBytecode);
    if (std::memcmp(base, "DXBC", 4) != 0) return;
    auto rdU32 = [&](size_t off) -> uint32_t {
      uint32_t v; std::memcpy(&v, base + off, sizeof(v)); return v;
    };
    const uint32_t chunkCount = rdU32(28);
    if (chunkCount == 0 || 32 + chunkCount * 4 > BytecodeLength) return;
    for (uint32_t i = 0; i < chunkCount; ++i) {
      const uint32_t chunkOff = rdU32(32 + i * 4);
      if (chunkOff + 8 > BytecodeLength) continue;
      const bool isInputSig =
        std::memcmp(base + chunkOff, "ISGN", 4) == 0 ||
        std::memcmp(base + chunkOff, "ISG1", 4) == 0;
      if (!isInputSig) continue;
      const bool isV1 = std::memcmp(base + chunkOff, "ISG1", 4) == 0;
      const uint32_t entrySize = isV1 ? 28u : 24u;
      const uint32_t chunkSize = rdU32(chunkOff + 4);
      if (chunkSize < 8 || chunkOff + 8 + chunkSize > BytecodeLength) return;
      const size_t bodyOff = chunkOff + 8;
      const uint32_t elemCount = rdU32(bodyOff);
      // Skip 4B element count + 4B dataOffset header.
      const size_t firstEntryOff = bodyOff + 8;
      for (uint32_t e = 0; e < elemCount; ++e) {
        const size_t off = firstEntryOff + size_t(e) * entrySize;
        if (off + entrySize > BytecodeLength) break;
        const uint32_t nameOff = rdU32(off + 0);
        const uint32_t semIdx  = rdU32(off + 4);
        const uint32_t compType = rdU32(off + 12);
        // Read ASCIIZ name from bodyOff + nameOff (string offset is relative to body).
        std::string nameStr;
        const size_t nameAbs = bodyOff + nameOff;
        if (nameAbs < BytecodeLength) {
          for (size_t p = nameAbs; p < BytecodeLength && base[p] != 0 && (p - nameAbs) < 64; ++p) {
            nameStr.push_back(static_cast<char>(base[p]));
          }
        }
        if (nameStr.empty()) continue;
        InputCompType ct = InputCompType_Unknown;
        if      (compType == 1u) ct = InputCompType_Uint;
        else if (compType == 2u) ct = InputCompType_Sint;
        else if (compType == 3u) ct = InputCompType_Float;
        m_inputCompTypes.emplace(SemKey{ nameStr, semIdx }, ct);
      }
      return;
    }
  }

  // NV-DXVK: D3DReflect-based "which cbuffer fields does the shader actually
  // sample?" pass. Dynamically loads d3dcompiler_47.dll (falls back to _46)
  // once per process, calls D3DReflect on the DXBC bytecode, and for each
  // declared cbuffer variable copies the D3D_SVF_USED flag onto the
  // matching m_cbuffers[cb].fields[field].used slot. Replaces the
  // hand-rolled SHEX walker which under-reported reads for some operand
  // layouts (e.g. FS_ac8c6ae6651c58b8 reads cb0[0] per DXBC disasm but the
  // walker reported 0 reads).
  //
  // Failure modes (all silent, leaves `used=false` on every field — the
  // downstream gate will treat that as "don't apply the transform"):
  //   - d3dcompiler_47.dll not on the DLL search path
  //   - D3DReflect symbol missing (ancient d3dcompiler)
  //   - DXBC stripped of its RDEF chunk (D3DReflect returns E_FAIL)
  void D3D11CommonShader::populateFieldUsage(const void* pShaderBytecode, size_t BytecodeLength) {
    if (pShaderBytecode == nullptr || BytecodeLength < 32) return;

    typedef HRESULT (WINAPI *PFN_D3DReflect)(LPCVOID, SIZE_T, REFIID, void**);
    static std::once_flag       s_once;
    static PFN_D3DReflect       s_D3DReflect = nullptr;
    static bool                 s_loadFailed = false;

    std::call_once(s_once, []() {
      HMODULE h = LoadLibraryA("d3dcompiler_47.dll");
      if (!h) h = LoadLibraryA("d3dcompiler_46.dll");
      if (!h) {
        Logger::warn("[Reflect] LoadLibrary(d3dcompiler_47.dll) failed; "
                     "cbuffer-field usage detection disabled — UV transforms "
                     "will not be applied");
        s_loadFailed = true;
        return;
      }
      s_D3DReflect = reinterpret_cast<PFN_D3DReflect>(
        GetProcAddress(h, "D3DReflect"));
      if (!s_D3DReflect) {
        Logger::warn("[Reflect] d3dcompiler has no D3DReflect export; "
                     "cbuffer-field usage detection disabled");
        s_loadFailed = true;
      }
    });

    if (!s_D3DReflect) { (void)s_loadFailed; return; }

    // IID of ID3D11ShaderReflection — hard-coded so we work the same under
    // MinGW and MSVC regardless of how the SDK macro expands __uuidof.
    // {8d536ca1-0cca-4956-a837-786963755584}
    static const GUID kIID_ID3D11ShaderReflection = {
      0x8d536ca1, 0x0cca, 0x4956,
      { 0xa8, 0x37, 0x78, 0x69, 0x63, 0x75, 0x55, 0x84 }
    };

    ID3D11ShaderReflection* refl = nullptr;
    HRESULT hr = s_D3DReflect(pShaderBytecode, BytecodeLength,
                              kIID_ID3D11ShaderReflection,
                              reinterpret_cast<void**>(&refl));
    if (FAILED(hr) || refl == nullptr) {
      Logger::warn(str::format("[Reflect] D3DReflect failed hr=0x",
        std::hex, static_cast<uint32_t>(hr), std::dec,
        " shader=",
        (m_shader != nullptr ? m_shader->debugName() : std::string{})));
      return;
    }

    D3D11_SHADER_DESC sd = {};
    if (FAILED(refl->GetDesc(&sd))) {
      refl->Release();
      return;
    }

    size_t usedCount = 0;
    size_t totalFields = 0;
    // NV-DXVK: log which cbuffers D3DReflect reports for the first few
    // shaders so we can see if its names match parseRdef's keys. If they
    // don't match, every field look-up in the inner loop fails and every
    // shader reports usedFields=0/0.
    static std::atomic<uint32_t> sReflDiag{0};
    sReflDiag.fetch_add(1);
    // Diagnose every shader that either (a) reports >0 cbuffers via
    // D3DReflect, or (b) has parseRdef state worth inspecting. Keeps
    // log reasonable while catching every gameplay PS with cbuffers.
    const bool diagThis = (sd.ConstantBuffers > 0) || !m_cbuffers.empty();
    if (diagThis) {
      std::string stored;
      for (const auto& kv : m_cbuffers) { stored += kv.first; stored += " "; }
      Logger::info(str::format("[Reflect.diag] shader=",
        (m_shader != nullptr ? m_shader->debugName() : std::string{}),
        " parseRdef-stored-cbs={", stored, "} reflect-reports ",
        sd.ConstantBuffers, " cbs"));
    }
    for (UINT i = 0; i < sd.ConstantBuffers; i++) {
      ID3D11ShaderReflectionConstantBuffer* rcb =
        refl->GetConstantBufferByIndex(i);
      if (!rcb) continue;
      D3D11_SHADER_BUFFER_DESC cbd = {};
      if (FAILED(rcb->GetDesc(&cbd))) continue;
      const std::string cbName = cbd.Name ? cbd.Name : "";
      if (cbName.empty()) continue;

      if (diagThis) {
        Logger::info(str::format("[Reflect.diag]   reflect-cb=\"", cbName,
          "\" vars=", cbd.Variables,
          " match=", (m_cbuffers.count(cbName) ? "YES" : "NO")));
      }
      auto cbIt = m_cbuffers.find(cbName);
      if (cbIt == m_cbuffers.end()) continue;

      // NV-DXVK: when diagnosing, also dump first few field names from
      // both sides so we can see WHY the field-name match fails inside.
      if (diagThis && cbName == "CBufUberStatic") {
        std::string reflectNames;
        for (UINT v2 = 0; v2 < cbd.Variables && v2 < 6; v2++) {
          ID3D11ShaderReflectionVariable* var2 = rcb->GetVariableByIndex(v2);
          D3D11_SHADER_VARIABLE_DESC vd2 = {};
          if (var2 && SUCCEEDED(var2->GetDesc(&vd2)) && vd2.Name) {
            reflectNames += "\""; reflectNames += vd2.Name; reflectNames += "\" ";
          }
        }
        std::string storedNames;
        size_t n = 0;
        for (const auto& kv : cbIt->second.fields) {
          if (n++ >= 6) break;
          storedNames += "\""; storedNames += kv.first; storedNames += "\" ";
        }
        Logger::info(str::format("[Reflect.diag]     reflect-field-names=[ ",
          reflectNames, "]"));
        Logger::info(str::format("[Reflect.diag]     parseRdef-field-names=[ ",
          storedNames, "]"));
      }

      // NV-DXVK: rebuild the fields map from D3DReflect ground truth rather
      // than trusting parseRdef — the hand-rolled SM5 RDEF variable walk in
      // parseRdef uses the wrong per-entry stride on some shaders and
      // produces scrambled name→offset pairings. FindCBField then returns
      // correct names with wrong byte offsets, and rtx's cbuffer read pulls
      // unrelated data → character/weapon UV poisoning (user's current
      // regression). D3DReflect reads the same DXBC chunk but uses
      // Microsoft's own parser with correct stride handling for every SM.
      // This replaces whatever parseRdef stored for this cbuffer.
      cbIt->second.fields.clear();
      for (UINT v = 0; v < cbd.Variables; v++) {
        ID3D11ShaderReflectionVariable* var = rcb->GetVariableByIndex(v);
        if (!var) continue;
        D3D11_SHADER_VARIABLE_DESC vd = {};
        if (FAILED(var->GetDesc(&vd))) continue;
        const std::string vName = vd.Name ? vd.Name : "";
        if (vName.empty()) continue;

        const bool used = (vd.uFlags & D3D_SVF_USED) != 0;
        D3D11CbufferField f{};
        f.offset = vd.StartOffset;
        f.size   = vd.Size;
        f.used   = used;
        cbIt->second.fields[vName] = f;
        ++totalFields;
        if (used) ++usedCount;
      }
    }

    refl->Release();

    Logger::info(str::format("[Reflect] ",
      (m_shader != nullptr ? m_shader->debugName() : std::string{}),
      " usedFields=", usedCount, "/", totalFields));
  }

  // NV-DXVK: minimal DXBC RDEF parser. Extracts:
  //   - Each declared cbuffer's name, byte size, field list (name + offset)
  //   - Each cbuffer's Vulkan/D3D bind register (via the Resource Bindings table)
  // Only handles SM 4.0/4.1 variable layout (24-byte entries) — SM 5.0 RD11
  // subheader adds 16 more bytes which we skip over. Titanfall 2 ships vs_4_0.
  //
  // DXBC file format (Microsoft-documented):
  //   "DXBC" | 16B hash | u32 version | u32 totalSize | u32 chunkCount | u32 chunkOffsets[chunkCount]
  //   Each chunk: 4B tag | u32 size | data...
  // RDEF chunk body begins right after the 8B chunk header. All internal offsets
  // inside RDEF are relative to that body start (not the file start).
  void D3D11CommonShader::parseRdef(const void* pShaderBytecode, size_t BytecodeLength) {
    if (pShaderBytecode == nullptr || BytecodeLength < 32) return;
    const uint8_t* base = reinterpret_cast<const uint8_t*>(pShaderBytecode);

    auto rdU32 = [&](size_t off) -> uint32_t {
      uint32_t v;
      std::memcpy(&v, base + off, sizeof(v));
      return v;
    };
    auto rdStr = [&](const uint8_t* chunkBody, size_t bodySize, uint32_t off) -> std::string {
      if (off >= bodySize) return {};
      const char* p = reinterpret_cast<const char*>(chunkBody + off);
      size_t maxLen = bodySize - off;
      size_t len = strnlen(p, maxLen);
      return std::string(p, len);
    };

    // DXBC header sanity
    if (std::memcmp(base, "DXBC", 4) != 0) return;
    uint32_t chunkCount = rdU32(28);
    if (chunkCount == 0 || 32 + chunkCount * 4 > BytecodeLength) return;

    // Find the RDEF chunk.
    const uint8_t* rdefBody = nullptr;
    size_t         rdefSize = 0;
    for (uint32_t i = 0; i < chunkCount; i++) {
      uint32_t chunkOff = rdU32(32 + i * 4);
      if (chunkOff + 8 > BytecodeLength) continue;
      if (std::memcmp(base + chunkOff, "RDEF", 4) == 0) {
        uint32_t csize = rdU32(chunkOff + 4);
        if (chunkOff + 8 + csize > BytecodeLength) return;
        rdefBody = base + chunkOff + 8;
        rdefSize = csize;
        break;
      }
    }
    if (!rdefBody) return;  // No RDEF (fxc stripped it, or pass-through shader).

    auto bodyU32 = [&](size_t off) -> uint32_t {
      if (off + 4 > rdefSize) return 0;
      uint32_t v;
      std::memcpy(&v, rdefBody + off, sizeof(v));
      return v;
    };

    // RDEF header (28 bytes): cbCount, cbOff, resBindCount, resBindOff, shaderVer, flags, creatorOff
    if (rdefSize < 28) return;
    uint32_t cbCount       = bodyU32(0);
    uint32_t cbTableOff    = bodyU32(4);
    uint32_t resBindCount  = bodyU32(8);
    uint32_t resBindOff    = bodyU32(12);

    // Detect SM5+ by looking for the RD11 subheader at body offset 28 — this is
    // the actual on-disk indicator of the extended layout. In SM5+, variable
    // entries are 40B instead of 24B (extra fields appended). Reading them with
    // the wrong stride scrambles every field offset including c_cameraOrigin's.
    bool isSm5Plus = false;
    if (rdefSize >= 28 + 4) {
      uint32_t maybeMagic = bodyU32(28);
      // 'RD11' = 0x31314452 little-endian
      if (maybeMagic == 0x31314452u) isSm5Plus = true;
    }

    // First pass: every resource binding -> name -> slot. This covers cbuffers,
    // SRVs (textures, structured buffers), UAVs, and samplers. The slot value
    // is the physical register index (cN/tN/uN/sN) the shader actually reads.
    std::unordered_map<std::string, uint32_t> nameToSlot;
    constexpr uint32_t D3D_SIT_CBUFFER = 0;
    const size_t resBindEntrySize = 32;  // 8 * u32
    // Resource binding entry on-disk layout (32 bytes):
    //   off  0: nameOffset
    //   off  4: type (D3D_SHADER_INPUT_TYPE)
    //   off  8: returnType
    //   off 12: dimension
    //   off 16: numSamples
    //   off 20: bindPoint   <-- physical register (cN/tN/uN/sN)
    //   off 24: bindCount   <-- consecutive registers used (>=1)
    //   off 28: flags
    for (uint32_t i = 0; i < resBindCount; i++) {
      size_t off = size_t(resBindOff) + size_t(i) * resBindEntrySize;
      if (off + resBindEntrySize > rdefSize) break;
      uint32_t nameOff = bodyU32(off + 0);
      uint32_t type    = bodyU32(off + 4);
      uint32_t bindPt  = bodyU32(off + 20);
      std::string n = rdStr(rdefBody, rdefSize, nameOff);
      if (n.empty()) continue;
      if (type == D3D_SIT_CBUFFER) {
        nameToSlot[n] = bindPt;  // for cbuffer match-up below
      }
      // Mirror every named binding into the public lookup so callers can find
      // SRV/UAV slots by HLSL name (e.g. "g_modelInst" -> 31).
      m_resourceSlots[n] = bindPt;
    }

    // Second pass: cbuffer table -> name, size, field list.
    const size_t cbEntrySize    = 24;  // 6 * u32
    const size_t varEntrySize   = isSm5Plus ? 40 : 24;
    for (uint32_t i = 0; i < cbCount; i++) {
      size_t off = size_t(cbTableOff) + size_t(i) * cbEntrySize;
      if (off + cbEntrySize > rdefSize) break;
      uint32_t nameOff   = bodyU32(off + 0);
      uint32_t varCount  = bodyU32(off + 4);
      uint32_t varOff    = bodyU32(off + 8);
      uint32_t cbSize    = bodyU32(off + 12);
      // off+16 flags, off+20 type — we only care about regular cbuffers but
      // the resource-binding type filter above already scoped us correctly.

      std::string cbName = rdStr(rdefBody, rdefSize, nameOff);
      if (cbName.empty()) continue;

      D3D11CbufferInfo info;
      info.size = cbSize;
      auto slotIt = nameToSlot.find(cbName);
      if (slotIt != nameToSlot.end()) info.bindSlot = slotIt->second;

      for (uint32_t v = 0; v < varCount; v++) {
        size_t voff = size_t(varOff) + size_t(v) * varEntrySize;
        if (voff + varEntrySize > rdefSize) break;
        uint32_t vNameOff = bodyU32(voff + 0);
        uint32_t vStart   = bodyU32(voff + 4);
        uint32_t vSize    = bodyU32(voff + 8);
        std::string vName = rdStr(rdefBody, rdefSize, vNameOff);
        if (!vName.empty())
          info.fields[vName] = { vStart, vSize };
      }

      m_cbuffers[std::move(cbName)] = std::move(info);
    }
  }


  D3D11ShaderModuleSet:: D3D11ShaderModuleSet() { }
  D3D11ShaderModuleSet::~D3D11ShaderModuleSet() { }
  
  
  HRESULT D3D11ShaderModuleSet::GetShaderModule(
          D3D11Device*        pDevice,
    const DxvkShaderKey*      pShaderKey,
    const DxbcModuleInfo*     pDxbcModuleInfo,
    const void*               pShaderBytecode,
          size_t              BytecodeLength,
          D3D11CommonShader*  pShader) {
    // Use the shader's unique key for the lookup
    { std::unique_lock<dxvk::mutex> lock(m_mutex);
      
      auto entry = m_modules.find(*pShaderKey);
      if (entry != m_modules.end()) {
        *pShader = entry->second;
        return S_OK;
      }
    }
    
    // This shader has not been compiled yet, so we have to create a
    // new module. This takes a while, so we won't lock the structure.
    D3D11CommonShader module;
    
    try {
      module = D3D11CommonShader(pDevice, pShaderKey,
        pDxbcModuleInfo, pShaderBytecode, BytecodeLength);
    } catch (const DxvkError& e) {
      Logger::err(e.message());
      return E_INVALIDARG;
    }
    
    // Insert the new module into the lookup table. If another thread
    // has compiled the same shader in the meantime, we should return
    // that object instead and discard the newly created module.
    { std::unique_lock<dxvk::mutex> lock(m_mutex);
      
      auto status = m_modules.insert({ *pShaderKey, module });
      if (!status.second) {
        *pShader = status.first->second;
        return S_OK;
      }
    }
    
    *pShader = std::move(module);
    return S_OK;
  }

  // NV-DXVK [EngineLightsCapture]: dump DXBC asm to disk for shaders that
  // bind s_globalLights. Reads ld_structured offsets in the asm reveal
  // the per-type field layout exactly as the engine intended. Throttled
  // to a small set of unique shaders so we don't fill the logs dir.
  void D3D11CommonShader::dumpShaderAsmIfRelevant(const void* pShaderBytecode, size_t BytecodeLength) {
    if (!RtxOptions::dumpEngineLightShaderAsm()) return;
    if (pShaderBytecode == nullptr || BytecodeLength < 32) return;

    // Only dump shaders that actually declare s_globalLights - everything
    // else is irrelevant for Tier 2 light layout decoding.
    auto it = m_resourceSlots.find("s_globalLights");
    if (it == m_resourceSlots.end()) return;
    const uint32_t lightSrvSlot = it->second;

    // Cap the number of dumped shaders so a TF2 session with thousands
    // of shaders doesn't fill the disk. 16 is plenty to see all light-
    // type-handling permutations.
    static std::mutex                     sDumpMutex;
    static std::unordered_set<std::string> sDumpedHashes;
    static constexpr size_t                kMaxDumps = 16;

    const std::string dbgName = (m_shader != nullptr)
      ? m_shader->debugName() : std::string("unknown");
    {
      std::lock_guard<std::mutex> lk(sDumpMutex);
      if (sDumpedHashes.size() >= kMaxDumps) return;
      if (sDumpedHashes.count(dbgName) > 0) return;
      sDumpedHashes.insert(dbgName);
    }

    // Dynamic-load D3DDisassemble from d3dcompiler_47.dll - same DLL
    // populateFieldUsage uses for D3DReflect. We resolve once per
    // process and cache the function pointer.
    typedef HRESULT (WINAPI* PFN_D3DDisassemble)(
      LPCVOID pSrcData, SIZE_T SrcDataSize, UINT Flags,
      LPCSTR szComments, ID3DBlob** ppDisassembly);
    static std::once_flag        sOnceDisasm;
    static PFN_D3DDisassemble    sD3DDisassemble = nullptr;
    std::call_once(sOnceDisasm, []() {
      HMODULE h = LoadLibraryA("d3dcompiler_47.dll");
      if (!h) h = LoadLibraryA("d3dcompiler_46.dll");
      if (!h) {
        Logger::warn("[EngineLights.asm] LoadLibrary(d3dcompiler_47.dll) failed");
        return;
      }
      sD3DDisassemble = reinterpret_cast<PFN_D3DDisassemble>(
        GetProcAddress(h, "D3DDisassemble"));
      if (!sD3DDisassemble) {
        Logger::warn("[EngineLights.asm] D3DDisassemble symbol missing");
      }
    });
    if (!sD3DDisassemble) return;

    constexpr UINT kFlags = 0;
    ID3DBlob* blob = nullptr;
    HRESULT hr = sD3DDisassemble(pShaderBytecode, BytecodeLength,
                                 kFlags, nullptr, &blob);
    if (FAILED(hr) || blob == nullptr) {
      Logger::warn(str::format("[EngineLights.asm] disassemble failed hr=0x",
        std::hex, static_cast<uint32_t>(hr), std::dec,
        " shader=", dbgName));
      return;
    }

    // Build path: <logs_dir>/tf2_shader_<dbgName>.asm
    auto outDir = util::RtxFileSys::path(util::RtxFileSys::Logs);
    std::string fileName = "tf2_shader_" + dbgName + ".asm";
    std::string fullPath;
    if (!outDir.empty()) {
      auto path = outDir;
      path /= fileName;
      fullPath = path.string();
    } else {
      fullPath = fileName;
    }

    const char* asmStart = static_cast<const char*>(blob->GetBufferPointer());
    const size_t asmSize = blob->GetBufferSize();

    std::ofstream out(fullPath, std::ios::binary);
    if (out.is_open()) {
      // Header banner so the user knows what to look at.
      out << "// NV-DXVK [EngineLights.asm] disassembly\n"
          << "// shader=" << dbgName << "\n"
          << "// s_globalLights bound at slot t" << lightSrvSlot << "\n"
          << "// look for: ld_structured rN.xyzw, indexN, l(BYTE_OFFSET),"
             " t" << lightSrvSlot << ".xyzw\n"
          << "// each unique BYTE_OFFSET is a field the shader actually reads.\n"
          << "//\n";
      out.write(asmStart, static_cast<std::streamsize>(asmSize));
      Logger::info(str::format(
        "[EngineLights.asm] wrote ", fullPath,
        " (", asmSize, " bytes, t", lightSrvSlot, ")"));
    } else {
      Logger::warn(str::format(
        "[EngineLights.asm] failed to open ", fullPath, " for writing"));
    }

    // Log a digest of every ld_structured / ld_structured_indexable line
    // that touches s_globalLights' slot. This tells us the byte offsets
    // the shader reads from without opening the asm file. The "t<slot>"
    // token also appears in resource decls; we filter to lines that
    // start with "ld_" to avoid noise.
    const std::string slotTok = "t" + std::to_string(lightSrvSlot);
    size_t pos = 0;
    uint32_t hits = 0;
    while (pos < asmSize) {
      size_t lineEnd = pos;
      while (lineEnd < asmSize && asmStart[lineEnd] != '\n') ++lineEnd;
      const size_t lineLen = lineEnd - pos;
      if (lineLen > 0 && lineLen < 512) {
        const std::string line(asmStart + pos, lineLen);
        // Look for ld_structured (and the indexable variant) referencing
        // s_globalLights' slot. Skip resource decl lines (start with
        // "dcl_resource_structured").
        const bool isLd = (line.find("ld_structured") != std::string::npos
                        || line.find("ld_raw")        != std::string::npos);
        if (isLd && line.find(slotTok) != std::string::npos) {
          // Trim leading whitespace.
          size_t s = 0;
          while (s < line.size() && (line[s] == ' ' || line[s] == '\t')) ++s;
          Logger::info(str::format(
            "[EngineLights.asm] ", dbgName, ": ",
            line.substr(s)));
          ++hits;
          if (hits >= 32) break;  // cap per shader
        }
      }
      pos = lineEnd + 1;
    }

    blob->Release();
  }

}
