#include "dxbc_analysis.h"
#include "dxbc_compiler.h"
#include "dxbc_module.h"
#include "dxbc_spirv_cache.h"

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

    // NV-DXVK: Translation guard is created BELOW, AFTER the cache check.
    // It must NOT be here because cache HITS would still bump the
    // "translations completed" counter in the guard destructor, making
    // the HUD say "Translating D3D11 shaders (N done)" even when N
    // shaders were actually loaded from cache in microseconds.

    // NV-DXVK: Try the on-disk SPIR-V cache first.  This is the primary
    // mechanism that makes subsequent launches much faster -- if a shader
    // with identical DXBC bytecode + identical compile options has been
    // translated before, we can skip the entire DxbcAnalyzer + DxbcCompiler
    // path and reconstruct a ready-to-use DxvkShader from the cached
    // SPIR-V + metadata.
    //
    // The cache key is a hash that combines:
    //   1. The raw SHEX chunk bytes (the DXBC bytecode itself, which is
    //      byte-stable for the same HLSL source + fxc flags), XORed with
    //   2. A hash of DxbcModuleInfo::options as raw bytes (DxbcOptions is
    //      a POD-ish struct of bools + one VkDeviceSize that captures the
    //      compile-time flags affecting codegen, e.g. demote-to-helper,
    //      depth clip workaround, NaN fixup, etc), XORed with
    //   3. Hashes of optional tess/xfb info if present.
    //
    // Folding compile options into the key means a Vulkan driver upgrade
    // that flips a feature flag automatically invalidates the cache, and
    // the next launch transparently re-translates and overwrites the
    // affected entries.  No manual cache wipe needed.
    //
    // On a cache miss we fall through to the real compile path and feed
    // the result back to the cache for next time.
    auto& shaderCache = DxbcSpirvCache::get();
    XXH64_hash_t cacheKey = 0;
    bool cacheKeyValid = false;
    if (shaderCache.isEnabled()) {
      // NV-DXVK: All cache I/O is wrapped in try/catch so a corrupt or
      // version-mismatched entry can NEVER take down the game.  Any
      // exception thrown by the cache (file system error, malformed
      // file, DxvkShader constructor blowing up on garbage SPIR-V, etc.)
      // is logged once and we fall through to the real DxbcCompiler path.
      // The cache is purely an optimization; correctness must never
      // depend on it.
      try {
        const auto slice = m_shexChunk->slice();
        cacheKey = DxbcSpirvCache::hashBytecode(slice.ptr(), slice.sizeInBytes());
        // Fold the compile options into the key.  XOR-combine via XXH64's
        // seed parameter so different option combinations produce
        // independent keys without ever colliding with the bytecode hash
        // alone.
        cacheKey ^= XXH64(&moduleInfo.options, sizeof(moduleInfo.options),
                          0x9a8c7b6c5d4e3f2cULL);
        if (moduleInfo.tess != nullptr) {
          cacheKey ^= XXH64(moduleInfo.tess, sizeof(*moduleInfo.tess),
                            0x1122334455667788ULL);
        }
        if (moduleInfo.xfb != nullptr) {
          cacheKey ^= XXH64(moduleInfo.xfb, sizeof(*moduleInfo.xfb),
                            0xaabbccddeeff0011ULL);
        }
        cacheKeyValid = true;
        Rc<DxvkShader> cached = shaderCache.lookup(cacheKey);
        if (cached != nullptr)
          return cached;
      } catch (const std::exception& e) {
        static std::atomic<bool> s_loggedLookupFail{ false };
        if (!s_loggedLookupFail.exchange(true)) {
          Logger::warn(str::format(
              "DxbcSpirvCache: lookup threw '", e.what(),
              "' -- falling back to slow path. Set DXVK_DISABLE_SPIRV_CACHE=1 ",
              "to disable the cache entirely if this keeps happening."));
        }
        cacheKeyValid = false;
      } catch (...) {
        static std::atomic<bool> s_loggedLookupFail{ false };
        if (!s_loggedLookupFail.exchange(true)) {
          Logger::warn(
              "DxbcSpirvCache: lookup threw an unknown exception -- "
              "falling back to slow path.");
        }
        cacheKeyValid = false;
      }
    }

    // NV-DXVK: bump HUD counters ONLY for real translations (cache miss).
    // On a cache hit we returned early above, so this guard only fires
    // when the DxbcCompiler actually runs.
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

    Rc<DxvkShader> result = compiler.finalize();

    // NV-DXVK: Periodically log cache hit/miss totals so we can verify
    // from the log alone that the cache is actually working (or NOT
    // working) over the course of a session.  Logged every 256 fresh
    // translations as a power-of-two cadence: cheap to compute, cheap to
    // emit, and frequent enough to spot a sudden cliff in hit rate.
    {
      const uint64_t completed = s_translationsCompleted.load(std::memory_order_relaxed) + 1;
      if (shaderCache.isEnabled() && (completed & 0xFF) == 0) {
        const uint64_t attempts = shaderCache.getLookupAttempts();
        const uint64_t hits     = shaderCache.getLookupHits();
        const uint64_t noFile   = shaderCache.getLookupMissesNoFile();
        const uint64_t corrupt  = shaderCache.getLookupMissesCorrupt();
        Logger::info(str::format(
            "DxbcSpirvCache: progress -- ",
            completed, " shaders translated, ",
            attempts, " cache lookups (",
            hits, " hits, ",
            noFile, " no-file misses, ",
            corrupt, " corrupt-file misses)"));
      }
    }

    // NV-DXVK: Write the freshly-translated shader to the disk cache so
    // the next launch can skip this entire function via the lookup above.
    // store() is tolerant of failures and silently skips any I/O error.
    // cacheKeyValid guards against the (rare) case where the lookup-time
    // hash computation threw an exception, in which case the key in
    // 'cacheKey' is uninitialized garbage and we must NOT write a file
    // under it.  Wrapped in try/catch for the same fail-safe reasons as
    // the lookup path above.
    if (shaderCache.isEnabled() && result != nullptr && cacheKeyValid) {
      try {
        shaderCache.store(cacheKey, result);
      } catch (const std::exception& e) {
        static std::atomic<bool> s_loggedStoreFail{ false };
        if (!s_loggedStoreFail.exchange(true)) {
          Logger::warn(str::format(
              "DxbcSpirvCache: store threw '", e.what(),
              "' -- cache writes will be skipped this session."));
        }
      } catch (...) {
        static std::atomic<bool> s_loggedStoreFail{ false };
        if (!s_loggedStoreFail.exchange(true)) {
          Logger::warn(
              "DxbcSpirvCache: store threw an unknown exception -- "
              "cache writes will be skipped this session.");
        }
      }
    }

    return result;
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
