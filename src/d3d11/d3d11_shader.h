#pragma once

#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../dxbc/dxbc_module.h"
#include "../dxvk/dxvk_device.h"

#include "../util/sha1/sha1_util.h"

#include "../util/util_env.h"

#include "d3d11_device_child.h"
#include "d3d11_interfaces.h"

namespace dxvk {
  
  class D3D11Device;

  // NV-DXVK: RDEF-derived cbuffer metadata. Used so per-shader ExtractTransforms
  // looks up slots/offsets deterministically from the shader's own declarations,
  // instead of guessing with size/content heuristics. See parseRdef() in cpp.
  // `used` is populated by populateFieldUsage() via D3DReflect (D3D_SVF_USED)
  // and is the authoritative "the shader actually samples this field" signal.
  struct D3D11CbufferField {
    uint32_t offset;   // byte offset within the cbuffer
    uint32_t size;     // bytes
    bool     used = false;
  };
  struct D3D11CbufferInfo {
    uint32_t bindSlot = UINT32_MAX;   // the cbN register the cbuffer is bound to
    uint32_t size     = 0;            // cbuffer size in bytes
    // fieldName -> {offset, size}
    std::unordered_map<std::string, D3D11CbufferField> fields;
  };

  /**
   * \brief Common shader object
   *
   * Stores the compiled SPIR-V shader and the SHA-1
   * hash of the original DXBC shader, which can be
   * used to identify the shader.
   */
  class D3D11CommonShader {

  public:

    D3D11CommonShader();
    D3D11CommonShader(
            D3D11Device*    pDevice,
      const DxvkShaderKey*  pShaderKey,
      const DxbcModuleInfo* pDxbcModuleInfo,
      const void*           pShaderBytecode,
            size_t          BytecodeLength);
    ~D3D11CommonShader();

    Rc<DxvkShader> GetShader() const {
      return m_shader;
    }

    Rc<DxvkBuffer> GetIcb() const {
      return m_buffer;
    }

    std::string GetName() const {
      return m_shader->debugName();
    }

    // NV-DXVK: lookup a cbuffer by name (the HLSL-declared name, e.g.
    // "CBufCommonPerCamera"). Returns nullptr if the shader doesn't bind it.
    const D3D11CbufferInfo* FindCBuffer(const std::string& name) const {
      auto it = m_cbuffers.find(name);
      return it != m_cbuffers.end() ? &it->second : nullptr;
    }
    // NV-DXVK: returns the bind slot (txx / cxx / uxx) the shader uses for a
    // resource by HLSL name, e.g. "g_modelInst" -> 31 if the VS reads t31.
    // Returns UINT32_MAX if the shader does not declare that resource.
    uint32_t FindResourceSlot(const std::string& name) const {
      auto it = m_resourceSlots.find(name);
      return it != m_resourceSlots.end() ? it->second : UINT32_MAX;
    }
    // Convenience: return {slot, offset, size} for a field, or std::nullopt.
    struct CBFieldLoc { uint32_t slot, offset, size; };
    std::optional<CBFieldLoc> FindCBField(const std::string& cbName,
                                          const std::string& fieldName) const {
      auto cb = FindCBuffer(cbName);
      if (!cb) return std::nullopt;
      auto it = cb->fields.find(fieldName);
      if (it == cb->fields.end()) return std::nullopt;
      return CBFieldLoc{ cb->bindSlot, it->second.offset, it->second.size };
    }
    // NV-DXVK: RDEF diagnostic — list all cbuffer names the shader declares,
    // plus their bind slot. Used to identify merged-bucket VS variants where
    // the objectToCameraRelative cbuffer uses a non-default HLSL name.
    std::vector<std::pair<std::string, uint32_t>> GetCBufferNamesAndSlots() const {
      std::vector<std::pair<std::string, uint32_t>> out;
      out.reserve(m_cbuffers.size());
      for (const auto& kv : m_cbuffers) out.emplace_back(kv.first, kv.second.bindSlot);
      return out;
    }
    // NV-DXVK: RDEF diagnostic — list every named resource (SRV/UAV/sampler)
    // the shader declares, with its bind slot. Used to identify unclassified
    // shaders whose albedo/normal/etc. SRVs use non-standard names.
    std::vector<std::pair<std::string, uint32_t>> GetResourceNamesAndSlots() const {
      std::vector<std::pair<std::string, uint32_t>> out;
      out.reserve(m_resourceSlots.size());
      for (const auto& kv : m_resourceSlots) out.emplace_back(kv.first, kv.second);
      return out;
    }
    // NV-DXVK: "is this cbuffer field actually read by the shader?" query,
    // sourced from D3DReflect's D3D_SVF_USED flag. Needed because RDEF lists
    // every declared field regardless of usage — reading per-draw cbuffer
    // values for fields the shader doesn't read picks up stale data from
    // a previous draw that DID use the field (see TF2 FS_7a6e4c5725a53e07:
    // declares CBufUberStatic but fxc marks c_uv1RotScaleX/Y/Translate
    // [unused]; applying the previous BSP draw's 6× scale to character UVs
    // destroys their rendering). Returns false if the shader doesn't
    // declare the cbuffer or field, or if reflection is unavailable.
    bool ReadsCBField(const std::string& cbName,
                      const std::string& fieldName) const {
      auto cb = FindCBuffer(cbName);
      if (!cb) return false;
      auto it = cb->fields.find(fieldName);
      if (it == cb->fields.end()) return false;
      return it->second.used;
    }

    // NV-DXVK: does this shader's OSGN declare any color output element?
    // A false here means the PS is a depth/stencil/alpha-cutout pass that
    // writes nothing to the render target — Remix should not treat its
    // bound albedoTexture as authoritative material colour.
    bool HasColorOutput() const { return m_hasColorOutput; }

    // NV-DXVK: does this shader's OSGN declare a non-system "COLOR"
    // semantic output (i.e. per-vertex color modulation passed from VS
    // to PS, distinct from SV_Target which uses semantic name
    // "SV_TARGET")? Used by the SubViewSkybox classifier in d3d11_rtx
    // to discriminate genuine 3D-skybox dome / mountain shaders (whose
    // colour comes purely from the texture sample and which do NOT
    // emit COLOR0) from generic main-world prop shaders that pass a
    // `diffuseModulation` constant out as COLOR0 to the PS. Confirmed
    // structural via fxc /dumpbin of VS_eda5e (dome, no COLOR),
    // VS_2f543cd7 (sub-view mountains, no COLOR), and VS_95da0b01
    // (false-positive UI/prop, HAS COLOR0).
    bool WritesNonSystemColor() const { return m_writesNonSystemColor; }

    // NV-DXVK: does this PS write SV_Coverage (the programmable MSAA
    // sample mask, oMask)? When true the shader is implementing
    // sub-pixel dithered alpha — at rasterization time the rasterizer
    // drops a fraction of MSAA samples per pixel to fake smooth
    // transparency. In Remix's path tracer there are no MSAA samples,
    // so oMask is silently ignored and the shader's full RGBA writes
    // to every pixel — producing visible BOXY hard-edged corruption.
    // Used by FillMaterialData to flag the surface for hiding (we
    // can't reconstruct the rasterizer's sample-masking math at ray-
    // trace time, so don't render the surface). Detected via OSGN
    // walk: systemValueType == D3D_NAME_COVERAGE (66).
    bool WritesCoverageMask() const { return m_writesCoverageMask; }

    // NV-DXVK [TF2SkyShader-diag]: dump every cbuffer-name -> field-names
    // pair the shader's reflection actually contains.
    std::string DumpCBufferFieldsForDiag() const {
      std::string out;
      for (const auto& cb : m_cbuffers) {
        out += "[" + cb.first + ":cb"
             + std::to_string(cb.second.bindSlot) + "]{";
        bool first = true;
        for (const auto& f : cb.second.fields) {
          if (!first) out += ",";
          first = false;
          out += f.first;
          if (f.second.used) out += "*";
        }
        out += "} ";
      }
      return out;
    }


    // NV-DXVK: D3D_REGISTER_COMPONENT_TYPE values for an input semantic the
    // shader declares. Lets the BLAS path know whether the VS reads its
    // TEXCOORD inputs as float (pass-through), uint, or sint — which TF2's
    // BSP world VSes do (TEXCOORD0/1 are uint32 packed UVs that the VS
    // bit-shifts and converts to float; the ISGN format is the only
    // shader-agnostic signal that flags this case).
    //   0 = unknown / not declared, 1 = uint, 2 = sint, 3 = float
    enum InputCompType : uint8_t {
      InputCompType_Unknown = 0,
      InputCompType_Uint    = 1,
      InputCompType_Sint    = 2,
      InputCompType_Float   = 3,
    };
    InputCompType GetInputSemanticComponentType(
        const std::string& name, uint32_t index) const {
      auto it = m_inputCompTypes.find({ name, index });
      return it != m_inputCompTypes.end() ? it->second : InputCompType_Unknown;
    }

  private:

    void parseRdef(const void* pShaderBytecode, size_t BytecodeLength);
    // NV-DXVK: run D3DReflect (dynamic-loaded from d3dcompiler_47.dll) over
    // the bytecode and copy each variable's D3D_SVF_USED bit into the matching
    // m_cbuffers[cb].fields[field].used slot. Replaces the old hand-rolled
    // SHEX walker, which under-reported reads for some operand layouts.
    void populateFieldUsage(const void* pShaderBytecode, size_t BytecodeLength);
    // NV-DXVK: read the OSGN chunk to detect whether the shader writes any
    // color output. "no Output" PSes are depth/alpha-cutout passes whose
    // bound albedoTexture is not the authoritative surface colour.
    void parseOsgn(const void* pShaderBytecode, size_t BytecodeLength);
    // NV-DXVK: read the ISGN chunk to record each input semantic's declared
    // component type (uint/sint/float). Used by the BLAS path to detect
    // packed-uint TEXCOORD encoding (TF2 BSP world VSes) vs plain-float.
    void parseIsgn(const void* pShaderBytecode, size_t BytecodeLength);
    // NV-DXVK [EngineLightsCapture]: if rtx.lights.dumpEngineLightShaderAsm
    // is on AND this shader declares "s_globalLights", disassemble the
    // DXBC via D3DDisassemble and write the asm to disk. Used to read
    // the per-type s_globalLights struct layout from the shader's own
    // ld_structured offsets (ground truth).
    void dumpShaderAsmIfRelevant(const void* pShaderBytecode, size_t BytecodeLength);

    struct SemKey {
      std::string name;
      uint32_t    index;
      bool operator==(const SemKey& o) const { return index == o.index && name == o.name; }
    };
    struct SemKeyHash {
      size_t operator()(const SemKey& k) const {
        return std::hash<std::string>()(k.name) ^ (size_t(k.index) * 0x9E3779B97F4A7C15ull);
      }
    };

    Rc<DxvkShader> m_shader;
    Rc<DxvkBuffer> m_buffer;

    // NV-DXVK: cbName -> info. Populated from DXBC RDEF chunk at ctor time.
    std::unordered_map<std::string, D3D11CbufferInfo> m_cbuffers;
    // NV-DXVK: resourceName -> bind slot (covers SRVs, UAVs, samplers).
    // Used by Remix to identify which physical slot a named structured buffer
    // (g_modelInst at t31, g_boneMatrix at t30, etc.) is read from per shader.
    std::unordered_map<std::string, uint32_t> m_resourceSlots;
    // NV-DXVK: true iff OSGN declares ≥ 1 output element (colour writes).
    bool m_hasColorOutput = false;
    // NV-DXVK [WritesNonSystemColor]: see getter doc. Set in parseOsgn
    // when an output entry's semantic name string is exactly "COLOR".
    bool m_writesNonSystemColor = false;
    // NV-DXVK: true iff OSGN declares any output element with
    // systemValueType == D3D_NAME_COVERAGE (SV_Coverage / oMask).
    // See WritesCoverageMask().
    bool m_writesCoverageMask = false;
    // NV-DXVK: per-input-semantic D3D_REGISTER_COMPONENT_TYPE from ISGN.
    std::unordered_map<SemKey, InputCompType, SemKeyHash> m_inputCompTypes;
  };
  
  
  /**
   * \brief Common shader interface
   * 
   * Implements methods for all D3D11*Shader
   * interfaces and stores the actual shader
   * module object.
   */
  template<typename D3D11Interface>
  class D3D11Shader : public D3D11DeviceChild<D3D11Interface> {

  public:
    
    D3D11Shader(D3D11Device* device, const D3D11CommonShader& shader)
    : D3D11DeviceChild<D3D11Interface>(device),
      m_shader(shader) { }
    
    ~D3D11Shader() { }
    
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppvObject) final {
      *ppvObject = nullptr;
      
      if (riid == __uuidof(IUnknown)
       || riid == __uuidof(ID3D11DeviceChild)
       || riid == __uuidof(D3D11Interface)) {
        *ppvObject = ref(this);
        return S_OK;
      }
      
      Logger::warn("D3D11Shader::QueryInterface: Unknown interface query");
      return E_NOINTERFACE;
    }
    
    const D3D11CommonShader* GetCommonShader() const {
      return &m_shader;
    }

  private:
    
    D3D11CommonShader m_shader;
    
  };
  
  using D3D11VertexShader   = D3D11Shader<ID3D11VertexShader>;
  using D3D11HullShader     = D3D11Shader<ID3D11HullShader>;
  using D3D11DomainShader   = D3D11Shader<ID3D11DomainShader>;
  using D3D11GeometryShader = D3D11Shader<ID3D11GeometryShader>;
  using D3D11PixelShader    = D3D11Shader<ID3D11PixelShader>;
  using D3D11ComputeShader  = D3D11Shader<ID3D11ComputeShader>;
  
  
  /**
   * \brief Shader module set
   * 
   * Some applications may compile the same shader multiple
   * times, so we should cache the resulting shader modules
   * and reuse them rather than creating new ones. This
   * class is thread-safe.
   */
  class D3D11ShaderModuleSet {
    
  public:
    
    D3D11ShaderModuleSet();
    ~D3D11ShaderModuleSet();
    
    HRESULT GetShaderModule(
            D3D11Device*        pDevice,
      const DxvkShaderKey*      pShaderKey,
      const DxbcModuleInfo*     pDxbcModuleInfo,
      const void*               pShaderBytecode,
            size_t              BytecodeLength,
            D3D11CommonShader*  pShader);
    
  private:
    
    dxvk::mutex m_mutex;
    
    std::unordered_map<
      DxvkShaderKey,
      D3D11CommonShader,
      DxvkHash, DxvkEq> m_modules;
    
  };
  
}
