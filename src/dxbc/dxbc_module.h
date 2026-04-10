#pragma once

#include <atomic>

#include "../dxvk/dxvk_shader.h"

#include "dxbc_chunk_isgn.h"
#include "dxbc_chunk_shex.h"
#include "dxbc_header.h"
#include "dxbc_modinfo.h"
#include "dxbc_reader.h"

// References used for figuring out DXBC:
// - https://github.com/tgjones/slimshader-cpp
// - Wine

namespace dxvk {

  class DxbcAnalyzer;
  class DxbcCompiler;
  // NV-DXVK: RAII guard (defined in dxbc_module.cpp) that bumps the
  // translation-in-flight / -completed counters on DxbcModule.  Friend-
  // declared so it can touch the private static atomics without making
  // the counters themselves public.
  struct DxbcTranslationGuard;
  
  /**
   * \brief DXBC shader module
   * 
   * Reads the DXBC byte code and extracts information
   * about the resource bindings and the instruction
   * stream. A module can then be compiled to SPIR-V.
   */
  class DxbcModule {
    // NV-DXVK: let the RAII guard poke our private counters.
    friend struct DxbcTranslationGuard;

  public:
    
    DxbcModule(DxbcReader& reader);
    ~DxbcModule();
    
    /**
     * \brief Shader type
     * \returns Shader type
     */
    DxbcProgramInfo programInfo() const {
      return m_shexChunk->programInfo();
    }
    
    /**
     * \brief Input and output signature chunks
     * 
     * Parts of the D3D11 API need access to the
     * input or output signature of the shader.
     */
    Rc<DxbcIsgn> isgn() const { return m_isgnChunk; }
    Rc<DxbcIsgn> osgn() const { return m_osgnChunk; }
    
    /**
     * \brief Compiles DXBC shader to SPIR-V module
     * 
     * \param [in] moduleInfo DXBC module info
     * \param [in] fileName File name, will be added to
     *        the compiled SPIR-V for debugging purposes.
     * \returns The compiled shader object
     */
    Rc<DxvkShader> compile(
      const DxbcModuleInfo& moduleInfo,
      const std::string&    fileName) const;
    
    /**
     * \brief Compiles a pass-through geometry shader
     *
     * Applications can pass a vertex shader to create
     * a geometry shader with stream output. In this
     * case, we have to create a passthrough geometry
     * shader, which operates in point to point mode.
     * \param [in] moduleInfo DXBC module info
     * \param [in] fileName SPIR-V shader name
     */
    Rc<DxvkShader> compilePassthroughShader(
      const DxbcModuleInfo& moduleInfo,
      const std::string&    fileName) const;

    // NV-DXVK: DXBC -> SPIR-V translation progress counters, used by the
    // bottom-left HUD indicator in ImGUI::showHudMessages().  These are
    // the *front-end* counters (parse DXBC, emit SPIR-V) — the back-end
    // SPIR-V -> VkPipeline counter lives on DxvkPipelineManager instead.
    // On a first-run DXVK session over a Source-engine game, almost all
    // of the multi-minute "loading screen" stall is in the front-end path
    // invoked via CreateVertexShader/CreatePixelShader, so the HUD needs
    // visibility into THIS stage specifically or it will appear to be
    // idle while the game is in fact grinding through shader translation.
    static uint32_t getDxbcTranslationsInFlight() {
      return s_translationsInFlight.load(std::memory_order_relaxed);
    }
    static uint64_t getDxbcTranslationsCompleted() {
      return s_translationsCompleted.load(std::memory_order_relaxed);
    }

  private:

    // NV-DXVK: Atomic counters for the HUD, bumped by DxbcTranslationGuard
    // in dxbc_module.cpp.  These MUST be inline static (C++17) rather than
    // regular statics defined in dxbc_module.cpp: the DXBC compiler lives
    // in libdxbc.a, but the HUD code that reads these counters lives in
    // libdxvk.a (via imgui_dxvk_imgui.cpp), and dxgi.dll links libdxvk.a
    // WITHOUT linking libdxbc.a.  An out-of-line definition in dxbc_module.cpp
    // would therefore produce an unresolved external when linking dxgi.dll.
    // Inline static gives every including TU its own definition, which the
    // linker then merges across all TUs, dodging the library-level dependency
    // entirely and letting the DXBC and HUD sides stay decoupled.
    inline static std::atomic<uint32_t> s_translationsInFlight{0};
    inline static std::atomic<uint64_t> s_translationsCompleted{0};
    
    DxbcHeader   m_header;
    
    Rc<DxbcIsgn> m_isgnChunk;
    Rc<DxbcIsgn> m_osgnChunk;
    Rc<DxbcIsgn> m_psgnChunk;
    Rc<DxbcShex> m_shexChunk;
    
    void runAnalyzer(
            DxbcAnalyzer&       analyzer,
            DxbcCodeSlice       slice) const;
    
    void runCompiler(
            DxbcCompiler&       compiler,
            DxbcCodeSlice       slice) const;
    
  };
  
}
