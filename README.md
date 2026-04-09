# dxvk-remix — Direct3D 11 Runtime

A fork of [NVIDIA's dxvk-remix](https://github.com/NVIDIAGameWorks/dxvk-remix) that replaces the Direct3D 9 frontend with a **Direct3D 11** frontend, enabling RTX Remix path-traced rendering for D3D11 games and emulators.

## What this does

The upstream RTX Remix runtime intercepts D3D9 draw calls and feeds geometry, materials, and transforms into NVIDIA's Vulkan-based path tracer. This fork does the same thing for **Direct3D 11**:

- Intercepts D3D11 draw calls via a custom `d3d11.dll` + `dxgi.dll` drop-in replacement
- Extracts projection, view, and world transforms automatically from constant buffers — no per-game configuration needed
- Feeds vertex buffers, index buffers, textures, and materials to the Remix RT pipeline
- Produces the same RTX Remix output: path-traced lighting, USD scene export, runtime material editing via Alt+X

## Compatibility

**Engine-agnostic.** Works with any game or application that uses Direct3D 11, including:
- Unreal Engine 4 games
- Unity games
- Custom engine titles
- **Emulators** (Dolphin, pcsx2 etc.) that render via D3D11

**GPU-agnostic.** No vendor-specific code — works on any GPU that supports RTX Remix's Vulkan requirements.

**Self-contained deployment.** The DLLs use `/DEPENDENTLOADFLAG` to resolve their dependencies from their own directory, not the exe directory. You can place the Remix runtime in a folder separate from the game/emulator exe and it will find all its sibling DLLs.

## Key features

| Feature | Details |
|---|---|
| Automatic transform detection | Two-pass cbuffer scan (row-major then column-major) finds projection, view, and world matrices |
| Axis convention auto-detection | Detects Y-flip, left/right-hand, Z-up and configures Remix accordingly |
| Deferred context safety | Guards against null workers when D3D11 deferred contexts are used |
| Buffer-pinned geometry hashing | Pins vertex/index buffers during async hash computation to prevent use-after-free on dynamic buffers |
| Render target filtering | Skips shadow/depth-only passes and dummy textures automatically |
| TAA jitter stripping | Removes sub-pixel jitter from projection matrices for stable RT |
| Delay-loaded USD plugins | `RemixParticleSystem.dll` is delay-loaded so it stays in `usd/plugins/` where it belongs |

## Build instructions

### Requirements:
1. Windows 10 or 11
2. [Git](https://git-scm.com/download/win)
3. [Visual Studio ](https://visualstudio.microsoft.com/vs/older-downloads/)
    - VS 2019 is tested
    - VS 2022 may also work, but it is not actively tested
    - Note that our build system will always use the most recent version available on the system
4. [Windows SDK](https://developer.microsoft.com/en-us/windows/downloads/sdk-archive/)
    - 10.0.19041.0 is tested
5. [Meson](https://mesonbuild.com/)
    - 1.8.2 has been tested
    - Follow [instructions](https://mesonbuild.com/SimpleStart.html#installing-meson) on how to install and reboot the PC before moving on (Meson will indicate as much)
6. [Vulkan SDK](https://vulkan.lunarg.com/sdk/home#windows)
    - 1.4.313.2 or newer
    - You may need to uninstall previous SDK if you have an old version
7. [Python](https://www.python.org/downloads/)
    - 3.9 or newer
    - Ensure you are using python installed from the link above and not from the Microsoft Store
8. [DirectX Runtime](https://www.microsoft.com/en-us/download/details.aspx?id=35)
    - Latest version should work

#### Additional notes:
- If any dependency paths change (i.e. new Vulkan library), run `meson --reconfigure` in _Compiler64 directory via a command prompt. This may revert some custom VS project settings

### Generate and build dxvk-remix Visual Studio project 
1. Clone the repository with all submodules:
        - `git clone --recursive https://github.com/Murray2k6/dxvk-remix-DX11.git`
	If the clone was made non-recursively and the submodules are missing, clone them separately:
	- `git submodule update --init --recursive`

2. Install all the [requirements](#requirements) before proceeding further

3. Make sure PowerShell scripts are enabled
    - One-time system setup: run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned` in an elevated PowerShell prompt, then close and reopen any existing PowerShell prompts
	
4. To generate and build dxvk-remix project:
    - Right Click on `dxvk-remix\build_dxvk_all_ninja.ps1` and select "Run with Powershell"
    - If that fails or has problems, run the build manually in a way you can read the errors:
        - open a windows file explorer to the `dxvk-remix` folder
        - remove artifacts from the previous attempt by deleting all folders that start with `_`, i.e. `_vs/` and `_Comp64Debug`
        - type `cmd` in the address bar to open a command line window in that folder.
        - copy and paste `powershell -command "& .\build_dxvk_all_ninja.ps1"` into the command line, then press enter
    - This will build all 3 configurations of dxvk-remix project inside subdirectories of the build tree: 
        - **_Comp64Debug** - full debug instrumentation, runtime speed may be slow
        - **_Comp64DebugOptimized** - partial debug instrumentation (i.e. asserts), runtime speed is generally comparable to that of release configuration
        - **_Comp64Release** - fastest runtime 
    - This will generate a project in the **_vs** subdirectory
    - Only x64 build targets are supported

5. Open **_vs/dxvk-remix.sln** in Visual Studio (2019+). 
    - Do not convert the solution on load if prompted when using a newer version of Visual Studio 
    - Once generated, the project can be built via Visual Studio or via powershell scripts
    - A build will copy generated DXVK DLLs to any target project as specified in **gametargets.conf** (see its [setup section](#deploy-built-binaries-to-a-game))

### Deploy built binaries to a game 
1. First time only: copy **gametargets.example.conf** to **gametargets.conf** in the project root

2. Update paths in the **gametargets.conf** for your game. Follow example in the **gametargets.example.conf**. Make sure to remove "#" from the start of all three lines

3. Open and, simply, re-save top-level **meson.build** file (i.e. via notepad) to update its time stamp, and rerun the build. This will trigger a full meson script run which will generate a project within the Visual Studio solution file and deploy built binaries into games' directories specified in **gametargets.conf**

### Profiling Remix
Remix has support for profiling using the [Tracy](https://github.com/wolfpld/tracy) tool, specifically the [v0.8 release](https://github.com/wolfpld/tracy/releases/download/v0.8/Tracy-0.8.7z)

To enable Tracy profiling:
1. Open a command line window in a build folder (i.e. `dxvk-remix/_Comp64Release/`)
2. Run `meson --reconfigure -D enable_tracy=true`
3. Rebuild dxvk-remix-nv

To profile:
1. Launch tracy.exe
2. Launch the game and reach the section you wish to profile
3. When ready, hit `Connect` in Tracy to begin profiling.
4. It's best to collect at least 500 frames worth of data, so you can average out the results.

### Remix API

Remix API can be used to programmatically pass game data to the Remix Renderer. [Click for more info.](/documentation/RemixSDK.md)

## Deployment

### Standard (game exe and Remix DLLs in the same directory)

1. Build the project (see above)
2. Copy `d3d11.dll`, `dxgi.dll` from the build output to the game directory alongside the exe
3. Copy the full RTX Remix runtime (NRD, DLSS, USD, Vulkan RT, etc.) to the same directory
4. The `usd/plugins/` subfolder should contain `RemixParticleSystem` — do **not** copy it next to the exe

### Emulator / separated directory layout

For emulators (Dolphin, RPCS3, Cemu, etc.) where you cannot or do not want to place DLLs next to the emulator exe:

1. Place all Remix DLLs (`d3d11.dll`, `dxgi.dll`, `NRD.dll`, `usd.dll`, `vulkan-1.dll`, etc.) in a single directory
2. The DLLs use `/DEPENDENTLOADFLAG:0x1100` — they find each other from their own directory, not the exe directory
3. Configure the emulator to load the custom `d3d11.dll`/`dxgi.dll` (method varies by emulator)

## How it works

The D3D11 bridge (`d3d11_rtx.cpp`) hooks into the D3D11 immediate context draw path:

1. **Pre-filter** — skips non-triangle topologies, draws without pixel shaders, draws without render targets, and small draws (< 3 vertices)
2. **Transform extraction** — scans all bound constant buffers for 4×4 matrices that look like perspective projections, then finds the corresponding view and world matrices. Caches the projection cbuffer slot/offset across frames for efficiency
3. **Convention detection** — auto-detects row-major vs column-major layout, Y-flip, left/right-hand coordinate system, and Z-up convention from the matrices themselves
4. **Geometry submission** — reads vertex/index buffers, maps input layout semantics to Remix vertex attributes, and submits to `GeometryProcessor`
5. **Material extraction** — reads bound SRVs (skipping render targets and tiny dummy textures), submits as albedo/normal/roughness

All detection is automatic — no per-game/per-engine configuration required.

## Project Documentation

- [Anti-Culling System](/documentation/AntiCullingSystem.md)
- [Contributing Guide](/CONTRIBUTING.md)
- [Foliage System](/documentation/FoliageSystem.md)
- [GPU Print](/documentation/GpuPrint.md)
- [Opacity Micromap](/documentation/OpacityMicromap.md)
- [Remix API](/documentation/RemixSDK.md)
- [Rtx Options](/RtxOptions.md)
- [Terrain System](/documentation/TerrainSystem.md)
- [Unit Test](/documentation/UnitTest.md)
