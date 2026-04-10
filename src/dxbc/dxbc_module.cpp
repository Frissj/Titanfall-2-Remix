#include "dxbc_analysis.h"
#include "dxbc_compiler.h"
#include "dxbc_module.h"

namespace dxvk {

  // NV-DXVK: The counter storage lives directly in the header as C++17
  // inline static members of DxbcModule.  That is intentional — see the
  // long comment near the declarations in dxbc_module.h for why an
  // out-of-line definition here would break linking of dxgi.dll.

  // NV-DXVK: RAII guard that increments the "in flight" counter at
  // construction and decrements it + bumps the "completed" counter at
  // destruction.  Using a guard (rather than paired manual increments)
  // guarantees exception safety — DxbcCompiler throws DxvkError on
  // unsupported opcodes and we must not leak the in-flight count on those
  // paths.  Declared as a friend of DxbcModule in the header so it can
  // read the private static atomics.  Lives in the dxvk namespace (not an
  // anonymous namespace) so the forward declaration in dxbc_module.h can
  // refer to this exact same type.
  struct DxbcTranslationGuard {
    DxbcTranslationGuard() {
      DxbcModule::s_translationsInFlight.fetch_add(1, std::memory_order_relaxed);
    }
    ~DxbcTranslationGuard() {
      DxbcModule::s_translationsInFlight.fetch_sub(1, std::memory_order_relaxed);
      DxbcModule::s_translationsCompleted.fetch_add(1, std::memory_order_relaxed);
    }
    DxbcTranslationGuard(const DxbcTranslationGuard&) = delete;
    DxbcTranslationGuard& operator=(const DxbcTranslationGuard&) = delete;
  };

  DxbcModule::DxbcModule(DxbcReader& reader)
  : m_header(reader) {
    for (uint32_t i = 0; i < m_header.numChunks(); i++) {
      
      // The chunk tag is stored at the beginning of each chunk
      auto chunkReader = reader.clone(m_header.chunkOffset(i));
      auto tag         = chunkReader.readTag();
      
      // The chunk size follows right after the four-character
      // code. This does not include the eight bytes that are
      // consumed by the FourCC and chunk length entry.
      auto chunkLength = chunkReader.readu32();
      
      chunkReader = chunkReader.clone(8);
      chunkReader = chunkReader.resize(chunkLength);
      
      if ((tag == "SHDR") || (tag == "SHEX"))
        m_shexChunk = new DxbcShex(chunkReader);
      
      if ((tag == "ISGN") || (tag == "ISG1"))
        m_isgnChunk = new DxbcIsgn(chunkReader, tag);
      
      if ((tag == "OSGN") || (tag == "OSG5") || (tag == "OSG1"))
        m_osgnChunk = new DxbcIsgn(chunkReader, tag);
      
      if ((tag == "PCSG") || (tag == "PSG1"))
        m_psgnChunk = new DxbcIsgn(chunkReader, tag);
    }
  }
  
  
  DxbcModule::~DxbcModule() {
    
  }
  
  
  Rc<DxvkShader> DxbcModule::compile(
    const DxbcModuleInfo& moduleInfo,
    const std::string&    fileName) const {
    if (m_shexChunk == nullptr)
      throw DxvkError("DxbcModule::compile: No SHDR/SHEX chunk");

    // NV-DXVK: bump HUD counters for the duration of this translation.
    DxbcTranslationGuard translationGuard;

    DxbcAnalysisInfo analysisInfo;
    
    DxbcAnalyzer analyzer(moduleInfo,
      m_shexChunk->programInfo(),
      m_isgnChunk, m_osgnChunk,
      m_psgnChunk, analysisInfo);
    
    this->runAnalyzer(analyzer, m_shexChunk->slice());
    
    DxbcCompiler compiler(
      fileName, moduleInfo,
      m_shexChunk->programInfo(),
      m_isgnChunk, m_osgnChunk,
      m_psgnChunk, analysisInfo);
    
    this->runCompiler(compiler, m_shexChunk->slice());
    
    return compiler.finalize();
  }
  
  
  Rc<DxvkShader> DxbcModule::compilePassthroughShader(
    const DxbcModuleInfo& moduleInfo,
    const std::string&    fileName) const {
    if (m_shexChunk == nullptr)
      throw DxvkError("DxbcModule::compile: No SHDR/SHEX chunk");

    // NV-DXVK: bump HUD counters for the duration of this translation.
    DxbcTranslationGuard translationGuard;

    DxbcAnalysisInfo analysisInfo;

    DxbcCompiler compiler(
      fileName, moduleInfo,
      DxbcProgramType::GeometryShader,
      m_osgnChunk, m_osgnChunk,
      m_psgnChunk, analysisInfo);
    
    compiler.processXfbPassthrough();
    return compiler.finalize();
  }


  void DxbcModule::runAnalyzer(
          DxbcAnalyzer&       analyzer,
          DxbcCodeSlice       slice) const {
    DxbcDecodeContext decoder;
    
    while (!slice.atEnd()) {
      decoder.decodeInstruction(slice);
      
      analyzer.processInstruction(
        decoder.getInstruction());
    }
  }
  
  
  void DxbcModule::runCompiler(
          DxbcCompiler&       compiler,
          DxbcCodeSlice       slice) const {
    DxbcDecodeContext decoder;
    
    while (!slice.atEnd()) {
      decoder.decodeInstruction(slice);
      
      compiler.processInstruction(
        decoder.getInstruction());
    }
  }
  
}
