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
echo # Checking for shader changes...                            #
echo #############################################################
echo.

rem Check if any cluster builder shaders have changed
rem Source shaders are in src/dxvk/shaders/rtx/pass/rtx_megageo/cluster_builder/*.slang
rem Compiled headers are in _Comp64Debug/src/dxvk/rtx_shaders/*.h

set "SHADER_SRC_DIR=%PROJECT_DIR%\src\dxvk\shaders\rtx\pass\rtx_megageo\cluster_builder"
set "SHADER_OUT_DIR=%PROJECT_DIR%\_Comp64Debug\src\dxvk\rtx_shaders"
set "FORCE_SHADER_REBUILD=0"

rem Check each cluster builder shader
for %%S in (compute_cluster_tiling fill_clusters copy_cluster_offset fill_blas_from_clas_args fill_instantiate_template_args patch_cluster_blas_addresses) do (
    if exist "%SHADER_SRC_DIR%\%%S.comp.slang" (
        if exist "%SHADER_OUT_DIR%\%%S.h" (
            rem Compare timestamps - if source is newer than output, force rebuild
            for %%F in ("%SHADER_SRC_DIR%\%%S.comp.slang") do set "SRC_TIME=%%~tF"
            for %%F in ("%SHADER_OUT_DIR%\%%S.h") do set "OUT_TIME=%%~tF"
            rem Note: This comparison is approximate but works for detecting recent changes
            echo Checking %%S: src=!SRC_TIME! out=!OUT_TIME!
        ) else (
            echo Shader output not found for %%S - will be built
        )
    )
)

rem Also check HIZ shaders
set "HIZ_SRC_DIR=%PROJECT_DIR%\src\dxvk\shaders\rtx\pass\rtx_megageo\hiz"
for %%S in (hiz_pass1 hiz_pass2 hiz_display zbuffer_minmax zbuffer_display) do (
    if exist "%HIZ_SRC_DIR%\%%S.comp.slang" (
        if exist "%SHADER_OUT_DIR%\%%S.h" (
            for %%F in ("%HIZ_SRC_DIR%\%%S.comp.slang") do set "SRC_TIME=%%~tF"
            for %%F in ("%SHADER_OUT_DIR%\%%S.h") do set "OUT_TIME=%%~tF"
            echo Checking %%S: src=!SRC_TIME! out=!OUT_TIME!
        ) else (
            echo Shader output not found for %%S - will be built
        )
    )
)

rem Check ALL shader include files - if any header is newer than compiled output, force rebuild
rem This catches changes to subdivision_plan_hlsl.h, subdivision_eval.hlsli, params, etc.
set "FORCE_SHADER_REBUILD=0"

rem Get the newest fill_clusters output timestamp as reference
set "REF_SPV="
if exist "%SHADER_OUT_DIR%\fill_clusters.h" (
    for %%F in ("%SHADER_OUT_DIR%\fill_clusters.h") do set "REF_SPV=%%~tF"
)
if not defined REF_SPV (
    echo No compiled shader output found - will be built fresh
    set "FORCE_SHADER_REBUILD=1"
)

if "!FORCE_SHADER_REBUILD!"=="0" (
    rem Scan all shader include directories for files newer than compiled output
    for %%D in (
        "%PROJECT_DIR%\src\dxvk\shaders\rtxmg\subdivision"
        "%PROJECT_DIR%\src\dxvk\shaders\rtxmg\cluster_builder"
        "%PROJECT_DIR%\src\dxvk\shaders\rtxmg\utils"
        "%PROJECT_DIR%\src\dxvk\shaders\rtx\pass\rtx_megageo\cluster_builder"
        "%PROJECT_DIR%\src\dxvk\shaders\rtx\pass\rtx_megageo\utils"
        "%PROJECT_DIR%\src\dxvk\shaders\rtxmg\subdivision\osd_ports\tmr"
    ) do (
        if exist %%D (
            for %%F in (%%~D\*.h %%~D\*.hlsli %%~D\*.slangh) do (
                set "INC_TIME=%%~tF"
                if "!INC_TIME!" GTR "!REF_SPV!" (
                    echo Include file newer than output: %%~nxF [!INC_TIME! ^> !REF_SPV!]
                    set "FORCE_SHADER_REBUILD=1"
                )
            )
        )
    )
)

if "!FORCE_SHADER_REBUILD!"=="1" (
    echo Shader includes changed - touching .slang files to force recompilation...
    rem Touch the main slang files so meson/ninja sees them as dirty
    copy /b "%SHADER_SRC_DIR%\fill_clusters.comp.slang"+,, "%SHADER_SRC_DIR%\fill_clusters.comp.slang" >nul 2>&1
    copy /b "%SHADER_SRC_DIR%\compute_cluster_tiling.comp.slang"+,, "%SHADER_SRC_DIR%\compute_cluster_tiling.comp.slang" >nul 2>&1
    copy /b "%SHADER_SRC_DIR%\fill_clusters_texcoords.comp.slang"+,, "%SHADER_SRC_DIR%\fill_clusters_texcoords.comp.slang" >nul 2>&1
)

echo.
echo #############################################################
echo # Starting/Updating the Remix Runtime build...              #
echo #############################################################
echo.

rem Force clean rebuild of rtx_shaders if output directory doesn't exist
if not exist "%SHADER_OUT_DIR%" (
    echo First build - shaders will be compiled
)

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

rem Delete DXVK shader caches in game directory to force shader recompilation
echo Clearing DXVK shader caches...
del "%GAME_DIR%\*.dxvk-cache" 2>nul
del "%GAME_RUNTIME_DIR%\*.dxvk-cache" 2>nul

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
