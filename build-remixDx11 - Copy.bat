@echo off
setlocal EnableDelayedExpansion

echo #############################################################
echo # Setting up Visual Studio 2022 x64 Build Environment...    #
echo #############################################################
echo.

set "VS_SETUP_SCRIPT=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"

if not exist "%VS_SETUP_SCRIPT%" (
    echo ERROR: Visual Studio setup script not found at:
    echo %VS_SETUP_SCRIPT%
    echo Please verify your Visual Studio 2022 Community installation path.
    goto :error_exit
)

call "%VS_SETUP_SCRIPT%" x64
if errorlevel 1 (
    echo ERROR: Failed to initialize the Visual Studio 2022 command prompt environment.
    goto :error_exit
)

echo.
echo #############################################################
echo # Environment configured. Navigating to project directory...#
echo #############################################################
echo.

set "PROJECT_DIR=C:\Users\Friss\Documents\RemixDX11\dxvk-remix-DX11"
set "GAME_DIR=C:\Users\Friss\Downloads\Compressed\Titanfall-2-Digital-Deluxe-Edition-AnkerGames\Titanfall2"
set "GAME_RUNTIME_SUBDIR=bin\x64_retail"
set "GAME_RUNTIME_DIR=%GAME_DIR%\%GAME_RUNTIME_SUBDIR%"
set "GAME_SHADER_DIR=%GAME_RUNTIME_DIR%\rtx_shaders"
set "GAME_LOG_DIR=%GAME_DIR%\rtx-remix\logs"
if not exist "%PROJECT_DIR%" (
    echo ERROR: Project directory not found: %PROJECT_DIR%
    goto :error_exit
)
pushd "%PROJECT_DIR%"

echo.
echo #############################################################
echo # Unlocking potentially locked files...                     #
echo #############################################################
echo.

rem Kill any processes that might lock build files
echo Checking for running game processes...
taskkill /F /IM "Titanfall2.exe" >nul 2>&1
taskkill /F /IM "Titanfall2_trial.exe" >nul 2>&1

rem Clear read-only attributes on build directories
echo Clearing read-only attributes on build output...
if exist "nv-private\hdremix\bin\debug" (
    attrib -R "nv-private\hdremix\bin\debug\*.*" /S /D >nul 2>&1
)
if exist "_Comp64Debug" (
    attrib -R "_Comp64Debug\*.*" /S /D >nul 2>&1
)

rem Force unlock any file handles (best effort)
echo Attempting to unlock file handles...
rem Wait a moment for file system to settle
timeout /t 1 /nobreak >nul 2>&1

echo.
echo #############################################################
echo # Starting/Updating the Remix Runtime build...              #
echo #############################################################
echo.

rem Kept for compatibility with references below — no longer used for any
rem timestamp logic, ninja handles shader dependency tracking on its own.
set "SHADER_OUT_DIR=%PROJECT_DIR%\_Comp64Debug\src\dxvk\rtx_shaders"

rem NV-DXVK: The previous block in this file manually time-compared every
rem *.h / *.hlsli / *.slangh include under src\dxvk\shaders\rtxmg and
rem rtx_megageo against the mtime of one compiled output (fill_clusters.h),
rem and on any mismatch it `copy /b`-touched the main .slang sources to
rem force a rebuild.  That comparison was done as a Windows batch STRING
rem compare of `%%~tF` output — which is locale-formatted ("MM/DD/YYYY
rem HH:MM AM/PM") — so it lied in at least three ways:
rem   * 12:XX PM lexically > 01:XX..11:XX PM (because "12" > "01"),
rem     making any file modified near noon permanently "newer"
rem   * MM/DD sort breaks across month/year boundaries
rem   * git checkouts land every file on the same timestamp, tripping it
rem     for different reasons the first time it ran
rem The combined effect: every single invocation of this script would
rem falsely detect "changed includes", then touch the .slang source files,
rem which made them genuinely newer next run, which made the next run
rem also falsely trigger -- a self-perpetuating full rebuild of every
rem Remix RTX shader on every build, adding ~5 minutes per iteration.
rem Meson/ninja already track .slang dependencies via the generated
rem build.ninja + .ninja_deps file, so the entire block was redundant
rem as well as buggy.  Removed.  If a .slangh include really changes and
rem ninja somehow misses the dependency, delete _Comp64Debug/src/dxvk/
rem rtx_shaders/ (or run `build-remixDx11.bat clean`) to force a reset.

rem enable_dxgi=true is REQUIRED for Titanfall 2: materialsystem_dx11 calls
rem IDXGIFactory::CreateSwapChain directly, which only works if Remix ships its
rem own dxgi.dll wrapper.  Without it, the game ends up on Microsoft's real DXGI
rem with a Vulkan-backed device it can't handle and crashes.
call meson setup --buildtype=debug --backend=ninja -Denable_dxgi=true _Comp64Debug
if errorlevel 1 (
    echo ERROR: Meson setup failed.
    goto :error_build
)

ninja -j6 -C _Comp64Debug
if errorlevel 1 (
    echo ERROR: The build process failed.
    goto :error_build
)

echo.
echo #############################################################
echo # Installing build artifacts...                             #
echo #############################################################
echo.

meson install -C _Comp64Debug
if errorlevel 1 (
    echo ERROR: The Meson install process failed due to locked files.
    echo Please close any applications that may have files locked and try again.
    goto :error_build
)

echo.
echo #############################################################
echo # Copying all build artifacts to _output directory...       #
echo #############################################################
echo.

rem --- Define source and destination directories ---
set "BUILD_DIR=_Comp64Debug"
set "OUTPUT_DIR=%PROJECT_DIR%\_output"
set "SOURCE_DIR=%PROJECT_DIR%\%BUILD_DIR%\tests\rtx\unit"
set "SHADER_BUILD_DIR=%PROJECT_DIR%\%BUILD_DIR%\src\dxvk\rtx_shaders"
set "BUILD_LOG_DIR=%PROJECT_DIR%\%BUILD_DIR%\meson-logs"

echo Cleaning and creating output directory: "%OUTPUT_DIR%"
if exist "%OUTPUT_DIR%" rd /s /q "%OUTPUT_DIR%"
mkdir "%OUTPUT_DIR%"
echo.

if not exist "%SOURCE_DIR%" (
    echo ERROR: Build output directory not found at "%SOURCE_DIR%"
    goto :error_copy
)

echo Copying all files and folders from "%SOURCE_DIR%" to "%OUTPUT_DIR%"...
xcopy "%SOURCE_DIR%" "%OUTPUT_DIR%" /E /I /Y /Q
echo.

rem Ensure the freshly built d3d11.dll is in _output (it lives in src\d3d11 after the build)
set "D3D11_BUILD_DIR=%PROJECT_DIR%\%BUILD_DIR%\src\d3d11"
if exist "%D3D11_BUILD_DIR%\d3d11.dll" (
    echo Copying d3d11.dll from "%D3D11_BUILD_DIR%" to "%OUTPUT_DIR%"...
    copy /Y "%D3D11_BUILD_DIR%\d3d11.dll" "%OUTPUT_DIR%\d3d11.dll" >nul
    if exist "%D3D11_BUILD_DIR%\d3d11.pdb" copy /Y "%D3D11_BUILD_DIR%\d3d11.pdb" "%OUTPUT_DIR%\d3d11.pdb" >nul
) else (
    echo WARNING: d3d11.dll not found at "%D3D11_BUILD_DIR%" - deployment may be incomplete.
)

rem Optional: copy dxgi.dll if enable_dxgi was turned on at meson setup
set "DXGI_BUILD_DIR=%PROJECT_DIR%\%BUILD_DIR%\src\dxgi"
if exist "%DXGI_BUILD_DIR%\dxgi.dll" (
    echo Copying dxgi.dll from "%DXGI_BUILD_DIR%" to "%OUTPUT_DIR%"...
    copy /Y "%DXGI_BUILD_DIR%\dxgi.dll" "%OUTPUT_DIR%\dxgi.dll" >nul
    if exist "%DXGI_BUILD_DIR%\dxgi.pdb" copy /Y "%DXGI_BUILD_DIR%\dxgi.pdb" "%OUTPUT_DIR%\dxgi.pdb" >nul
)

if not exist "%SHADER_BUILD_DIR%" goto :skip_shader_copy
echo Copying RTX shader binaries to "%OUTPUT_DIR%\rtx_shaders"...
mkdir "%OUTPUT_DIR%\rtx_shaders" >nul
robocopy "%SHADER_BUILD_DIR%" "%OUTPUT_DIR%\rtx_shaders" *.spv /NFL /NDL /NJH /NJS /NC /NS /NP >nul
set "ROBOCOPY_EXIT=%ERRORLEVEL%"
if !ROBOCOPY_EXIT! GEQ 8 (
    echo ERROR: Failed to copy RTX shader binaries.
    goto :error_copy
)
goto :shader_copy_done
:skip_shader_copy
echo WARNING: Compiled shader directory not found.
:shader_copy_done
echo.

echo Collecting build logs...
mkdir "%OUTPUT_DIR%\logs" >nul
if not exist "%BUILD_LOG_DIR%" goto :skip_log_copy
mkdir "%OUTPUT_DIR%\logs\build" >nul
robocopy "%BUILD_LOG_DIR%" "%OUTPUT_DIR%\logs\build" *.* /E /NFL /NDL /NJH /NJS /NC /NS /NP >nul
set "ROBOCOPY_EXIT=%ERRORLEVEL%"
if !ROBOCOPY_EXIT! GEQ 8 (
    echo ERROR: Failed to copy Meson build logs.
    goto :error_copy
)
goto :log_copy_done
:skip_log_copy
echo WARNING: Meson log directory not found.
:log_copy_done
if exist "%PROJECT_DIR%\%BUILD_DIR%\.ninja_log" copy "%PROJECT_DIR%\%BUILD_DIR%\.ninja_log" "%OUTPUT_DIR%\logs\.ninja_log" >nul
if exist "%PROJECT_DIR%\%BUILD_DIR%\.ninja_deps" copy "%PROJECT_DIR%\%BUILD_DIR%\.ninja_deps" "%OUTPUT_DIR%\logs\.ninja_deps" >nul
set "README_LINE_1=Build logs copied from !BUILD_LOG_DIR!."
echo !README_LINE_1! > "%OUTPUT_DIR%\logs\README.txt"
set "README_LINE_2=To gather runtime DXVK / Remix logs, set the environment variable DXVK_LOG_PATH to a writable folder before launching the game."
echo !README_LINE_2! >> "%OUTPUT_DIR%\logs\README.txt"


echo.
echo #############################################################
echo # Deploying artifacts to game directory...                  #
echo #############################################################
echo.

if not exist "%GAME_DIR%" (
    echo ERROR: Game directory not found at "%GAME_DIR%".
    goto :error_copy
)

>"%GAME_DIR%\__remix_write_test.tmp" echo.
if errorlevel 1 (
    echo ERROR: Unable to write to "%GAME_DIR%". Please run as Administrator.
    goto :error_copy
)
del "%GAME_DIR%\__remix_write_test.tmp" >nul

rem DXVK state caches are shader-hash keyed, so entries auto-invalidate when
rem their inputs change.  Keeping the cache across DLL rebuilds saves the
rem multi-minute first-run pipeline compile on every iteration.  Pass
rem "clean" (or "cleancache") to this script to force a full cache wipe
rem when you suspect the cache itself is corrupt.
if /i "%1"=="clean" goto :wipe_dxvk_cache
if /i "%1"=="cleancache" goto :wipe_dxvk_cache
echo Preserving DXVK shader caches (pass "clean" to wipe).
goto :skip_cache_wipe
:wipe_dxvk_cache
echo Clearing DXVK shader caches...
del "%GAME_DIR%\*.dxvk-cache" 2>nul
del "%GAME_RUNTIME_DIR%\*.dxvk-cache" 2>nul
:skip_cache_wipe

echo Copying runtime package to "%GAME_RUNTIME_DIR%"...
if not exist "%GAME_RUNTIME_DIR%" (
    mkdir "%GAME_RUNTIME_DIR%" >nul
)
rem This command copies the entire _output folder, including d3d11.dll, into bin\x64_retail
rem next to materialsystem_dx11.dll so Source's loader picks up the Remix DX11 bridge.
robocopy "%OUTPUT_DIR%" "%GAME_RUNTIME_DIR%" *.* /E /IS /R:2 /W:2 /NFL /NDL /NJH /NJS /NC /NS /NP >nul
set "ROBOCOPY_EXIT=%ERRORLEVEL%"
if !ROBOCOPY_EXIT! GEQ 8 (
    echo ERROR: Failed to deploy runtime files.
    goto :error_copy
)

if exist "%OUTPUT_DIR%\rtx_shaders" (
    echo Syncing shader binaries to "%GAME_SHADER_DIR%"...
    if not exist "%GAME_SHADER_DIR%" (
        mkdir "%GAME_SHADER_DIR%" >nul
    )
    robocopy "%OUTPUT_DIR%\rtx_shaders" "%GAME_SHADER_DIR%" *.spv /E /IS /R:2 /W:2 /NFL /NDL /NJH /NJS /NC /NS /NP >nul
    set "ROBOCOPY_EXIT=%ERRORLEVEL%"
    if !ROBOCOPY_EXIT! GEQ 8 (
        echo ERROR: Failed to update shader binaries.
        goto :error_copy
    )
)

rem Always (re)point DXVK_LOG_PATH at THIS game's log directory, regardless of
rem whether it already exists.  Without this, a persistent DXVK_LOG_PATH left
rem over from a previous game (e.g. LEGO Batman 2) will silently redirect
rem remix-dxvk.log to the wrong folder and hide the real crash output.
if not exist "%GAME_LOG_DIR%" mkdir "%GAME_LOG_DIR%" >nul 2>&1
echo Pointing DXVK_LOG_PATH at "%GAME_LOG_DIR%"...
setx DXVK_LOG_PATH "%GAME_LOG_DIR%" >nul
if errorlevel 1 (
    echo WARNING: Failed to configure DXVK_LOG_PATH automatically.
) else (
    rem setx only affects NEW processes; update the current shell too so any
    rem follow-up commands in this session see the new value.
    set "DXVK_LOG_PATH=%GAME_LOG_DIR%"
)

echo.
echo Done copying artifacts.
goto :success


:error_build
echo.
echo AN ERROR OCCURRED during the build process.
goto :error_exit

:error_copy
echo.
echo AN ERROR OCCURRED during the copy process.
goto :error_exit

:error_exit
echo.
echo SCRIPT FAILED.
popd
pause
exit /b 1

:success
echo.
echo #############################################################
echo # Build process finished successfully.                      #
echo #############################################################
echo.
popd
pause
exit /b 0
