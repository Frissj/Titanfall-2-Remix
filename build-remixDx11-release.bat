@echo off
setlocal EnableDelayedExpansion

rem ===========================================================================
rem  Remix DX11 (Titanfall 2) - RELEASE build, deploy and first-run setup
rem
rem  Arguments (any order, all optional):
rem    config / reconfigure  re-ask for the project and game folders
rem    clean / cleancache    wipe the DXVK shader caches in the game folder
rem
rem  First run asks for the game folder and stores it in the project folder as
rem  remix-build.cfg; the project folder itself is remembered under
rem  %LOCALAPPDATA%\RemixBuild so this script keeps working from the Desktop
rem  even if the source tree is moved or renamed.
rem ===========================================================================

set "FORCE_CONFIG="
set "WIPE_CACHE="
for %%a in (%*) do (
    if /i "%%~a"=="config" set "FORCE_CONFIG=1"
    if /i "%%~a"=="reconfigure" set "FORCE_CONFIG=1"
    if /i "%%~a"=="clean" set "WIPE_CACHE=1"
    if /i "%%~a"=="cleancache" set "WIPE_CACHE=1"
)

set "STATE_DIR=%LOCALAPPDATA%\RemixBuild"
set "PROJECT_POINTER=%STATE_DIR%\project-dir.txt"

echo #############################################################
echo # Locating source tree and game install...                  #
echo #############################################################
echo.

rem --- Project directory -----------------------------------------------------
rem A valid source tree is identified by meson.build sitting at its root.
set "PROJECT_DIR="

rem 1. This script may itself live inside the source tree.
set "SCRIPT_DIR=%~dp0"
if "!SCRIPT_DIR:~-1!"=="\" set "SCRIPT_DIR=!SCRIPT_DIR:~0,-1!"
if exist "!SCRIPT_DIR!\meson.build" set "PROJECT_DIR=!SCRIPT_DIR!"

rem 2. Remembered from a previous run.
if not defined FORCE_CONFIG if not defined PROJECT_DIR if exist "%PROJECT_POINTER%" (
    for /f "usebackq delims=" %%p in ("%PROJECT_POINTER%") do (
        if exist "%%~p\meson.build" set "PROJECT_DIR=%%~p"
    )
)

rem 3. Well-known spots, so a fresh checkout usually needs no prompt at all.
if not defined PROJECT_DIR (
    for %%c in (
        "%USERPROFILE%\Documents\Titanfall-2-Remix"
        "%USERPROFILE%\Documents\RemixDX11\dxvk-remix-DX11"
        "%USERPROFILE%\Documents\dxvk-remix-DX11"
        "%USERPROFILE%\source\repos\dxvk-remix-DX11"
        "C:\dxvk-remix-DX11"
    ) do (
        if not defined PROJECT_DIR if exist "%%~c\meson.build" set "PROJECT_DIR=%%~c"
    )
)

rem 4. Ask.
:ask_project
if defined PROJECT_DIR goto :project_resolved
echo Could not find the dxvk-remix DX11 source tree.
echo Enter the full path to the folder that contains meson.build
echo (leave blank to abort):
set "INPUT="
set /p "INPUT=  Project folder: "
if not defined INPUT (
    echo ERROR: No project folder given.
    goto :error_exit
)
set INPUT=%INPUT:"=%
if "!INPUT:~-1!"=="\" set "INPUT=!INPUT:~0,-1!"
if not exist "!INPUT!\meson.build" (
    echo   "!INPUT!" does not contain meson.build - that is not the source tree.
    echo.
    goto :ask_project
)
set "PROJECT_DIR=!INPUT!"

:project_resolved
if not exist "%STATE_DIR%" mkdir "%STATE_DIR%" >nul 2>&1
>"%PROJECT_POINTER%" echo !PROJECT_DIR!
echo Project folder: !PROJECT_DIR!

rem --- Game directory --------------------------------------------------------
rem Stored inside the project tree so the setting travels with the source.
set "CONFIG_FILE=!PROJECT_DIR!\remix-build.cfg"
set "GAME_DIR="

if not defined FORCE_CONFIG if exist "!CONFIG_FILE!" (
    for /f "usebackq eol=# tokens=1,* delims==" %%k in ("!CONFIG_FILE!") do (
        if /i "%%~k"=="GAME_DIR" set "GAME_DIR=%%~l"
    )
)
if defined GAME_DIR if not exist "!GAME_DIR!" (
    echo Saved game folder "!GAME_DIR!" no longer exists - asking again.
    set "GAME_DIR="
)

rem Try the usual store layouts before bothering the user.
if not defined GAME_DIR (
    for %%c in (
        "%ProgramFiles(x86)%\Steam\steamapps\common\Titanfall2"
        "%ProgramFiles%\Steam\steamapps\common\Titanfall2"
        "%ProgramFiles(x86)%\Origin Games\Titanfall2"
        "%ProgramFiles%\EA Games\Titanfall2"
        "%ProgramFiles(x86)%\EA Games\Titanfall2"
    ) do (
        if not defined GAME_DIR if exist "%%~c\bin\x64_retail" set "GAME_DIR=%%~c"
    )
)

:ask_game
if defined GAME_DIR goto :game_resolved
echo.
echo Enter the full path to your Titanfall 2 install folder.
echo That is the folder containing Titanfall2.exe and the bin\x64_retail
echo subfolder (leave blank to abort):
set "INPUT="
set /p "INPUT=  Game folder: "
if not defined INPUT (
    echo ERROR: No game folder given.
    goto :error_exit
)
set INPUT=%INPUT:"=%
if "!INPUT:~-1!"=="\" set "INPUT=!INPUT:~0,-1!"
rem Accept a dropped Titanfall2.exe as well as the folder itself.
if /i "!INPUT:~-15!"=="\Titanfall2.exe" for %%f in ("!INPUT!") do set "INPUT=%%~dpf"
if "!INPUT:~-1!"=="\" set "INPUT=!INPUT:~0,-1!"
if not exist "!INPUT!" (
    echo   "!INPUT!" does not exist.
    echo.
    goto :ask_game
)
if not exist "!INPUT!\bin\x64_retail" (
    if not exist "!INPUT!\Titanfall2.exe" (
        echo   "!INPUT!" has no Titanfall2.exe and no bin\x64_retail subfolder.
        echo   That does not look like a Titanfall 2 install.
        echo.
        goto :ask_game
    )
)
set "GAME_DIR=!INPUT!"

rem Persist it. Rewritten from scratch so re-running with "config" updates it.
>"!CONFIG_FILE!" echo # Remix build settings. Delete this file or pass "config" to change it.
>>"!CONFIG_FILE!" echo GAME_DIR=!GAME_DIR!
echo Saved game folder to "!CONFIG_FILE!".

:game_resolved
echo Game folder:    !GAME_DIR!
echo.

set "GAME_RUNTIME_SUBDIR=bin\x64_retail"
set "GAME_RUNTIME_DIR=!GAME_DIR!\!GAME_RUNTIME_SUBDIR!"
set "GAME_SHADER_DIR=!GAME_RUNTIME_DIR!\rtx_shaders"
set "GAME_LOG_DIR=!GAME_DIR!\rtx-remix\logs"

echo #############################################################
echo # Setting up Visual Studio x64 Build Environment...         #
echo #############################################################
echo.

rem Locate vcvarsall.bat instead of hardcoding a VS 2022 path.  vswhere.exe
rem ships with every VS installer since 2017 and always lives at this fixed
rem location, so asking it for the newest install with the C++ toolset keeps
rem this script working across VS upgrades (this machine is on VS 18, which is
rem why the old hardcoded "...\2022\Community\..." path stopped resolving).
set "VS_SETUP_SCRIPT="
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" (
    for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
        if exist "%%i\VC\Auxiliary\Build\vcvarsall.bat" set "VS_SETUP_SCRIPT=%%i\VC\Auxiliary\Build\vcvarsall.bat"
    )
)

rem Fallback for the (unlikely) case where vswhere is missing or reports an
rem install without the C++ workload: probe the standard layouts directly.
if not defined VS_SETUP_SCRIPT (
    for %%v in (18 2022 2019) do (
        for %%e in (Enterprise Professional Community BuildTools) do (
            if not defined VS_SETUP_SCRIPT if exist "%ProgramFiles%\Microsoft Visual Studio\%%v\%%e\VC\Auxiliary\Build\vcvarsall.bat" set "VS_SETUP_SCRIPT=%ProgramFiles%\Microsoft Visual Studio\%%v\%%e\VC\Auxiliary\Build\vcvarsall.bat"
            if not defined VS_SETUP_SCRIPT if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\%%v\%%e\VC\Auxiliary\Build\vcvarsall.bat" set "VS_SETUP_SCRIPT=%ProgramFiles(x86)%\Microsoft Visual Studio\%%v\%%e\VC\Auxiliary\Build\vcvarsall.bat"
        )
    )
)

if not defined VS_SETUP_SCRIPT (
    echo ERROR: Could not locate vcvarsall.bat for any Visual Studio installation.
    echo Make sure the "Desktop development with C++" workload is installed, or
    echo set VS_SETUP_SCRIPT by hand near the top of this script.
    goto :error_exit
)

echo Using: %VS_SETUP_SCRIPT%
call "%VS_SETUP_SCRIPT%" x64
if errorlevel 1 (
    echo ERROR: Failed to initialize the Visual Studio x64 command prompt environment.
    goto :error_exit
)

echo.
echo #############################################################
echo # Checking build tools (python / meson / ninja)...          #
echo #############################################################
echo.

call :ensure_python
if errorlevel 1 goto :error_exit
call :ensure_meson
if errorlevel 1 goto :error_exit
call :ensure_ninja
if errorlevel 1 goto :error_exit
call :ensure_submodules

echo.
echo #############################################################
echo # Environment configured. Navigating to project directory...#
echo #############################################################
echo.

pushd "!PROJECT_DIR!"

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
if exist "nv-private\hdremix\bin\release" (
    attrib -R "nv-private\hdremix\bin\release\*.*" /S /D >nul 2>&1
)
if exist "_Comp64Release" (
    attrib -R "_Comp64Release\*.*" /S /D >nul 2>&1
)

rem Force unlock any file handles (best effort)
echo Attempting to unlock file handles...
rem Wait a moment for file system to settle
timeout /t 1 /nobreak >nul 2>&1

echo.
echo #############################################################
echo # Starting/Updating the Remix Runtime build (RELEASE)...    #
echo #############################################################
echo.

rem Kept for compatibility with references below - no longer used for any
rem timestamp logic, ninja handles shader dependency tracking on its own.
set "SHADER_OUT_DIR=!PROJECT_DIR!\_Comp64Release\src\dxvk\rtx_shaders"

rem NV-DXVK: A block here used to manually time-compare every *.h / *.hlsli /
rem *.slangh include under src\dxvk\shaders against one compiled output, and
rem `copy /b`-touch the .slang sources on any mismatch.  The comparison was a
rem batch STRING compare of locale-formatted timestamps, so 12:XX PM sorted
rem above 01:XX PM, MM/DD broke across month boundaries, and a fresh git
rem checkout tripped it outright.  Every run therefore "detected" changed
rem includes, touched the sources, and guaranteed the next run would too - a
rem self-perpetuating full shader rebuild costing ~5 minutes per iteration.
rem Meson/ninja already track .slang dependencies via build.ninja + .ninja_deps,
rem so the block was redundant as well as wrong.  Removed.  If an include
rem really changes and ninja misses it, delete _Comp64Release\src\dxvk\
rem rtx_shaders\ to force a reset.

rem enable_dxgi=true is REQUIRED for Titanfall 2: materialsystem_dx11 calls
rem IDXGIFactory::CreateSwapChain directly, which only works if Remix ships its
rem own dxgi.dll wrapper.  Without it, the game ends up on Microsoft's real DXGI
rem with a Vulkan-backed device it can't handle and crashes.
rem
rem meson setup only accepts a fresh build dir; on an already-configured one it
rem errors out, so reconfigure in place and only fall back to a wipe if even
rem that fails.  Note the first configure also pulls the packman "external"
rem dependencies (slangc, glslangValidator, spirv-val), which needs a network
rem connection and takes a few minutes.
if exist "_Comp64Release\build.ninja" (
    echo Build directory already configured - reconfiguring for release...
    call meson configure --buildtype=release -Denable_dxgi=true _Comp64Release >nul
    if errorlevel 1 (
        echo Reconfigure failed - wiping _Comp64Release and starting clean.
        rd /s /q "_Comp64Release"
    )
)
if not exist "_Comp64Release\build.ninja" (
    call meson setup --buildtype=release --backend=ninja -Denable_dxgi=true _Comp64Release
    if errorlevel 1 (
        echo ERROR: Meson setup failed.
        goto :error_build
    )
)

ninja -j6 -C _Comp64Release
if errorlevel 1 (
    echo ERROR: The build process failed.
    goto :error_build
)

echo.
echo #############################################################
echo # Installing build artifacts...                             #
echo #############################################################
echo.

meson install -C _Comp64Release
if errorlevel 1 (
    echo ERROR: The Meson install process failed due to locked files.
    echo Please close any applications that may have files locked and try again.
    goto :error_build
)

echo.
echo #############################################################
echo # Copying all build artifacts to output directory...        #
echo #############################################################
echo.

rem --- Define source and destination directories ---
set "BUILD_DIR=_Comp64Release"
set "OUTPUT_DIR=!PROJECT_DIR!\_output_release"
set "SOURCE_DIR=!PROJECT_DIR!\!BUILD_DIR!\tests\rtx\unit"
set "SHADER_BUILD_DIR=!PROJECT_DIR!\!BUILD_DIR!\src\dxvk\rtx_shaders"
set "BUILD_LOG_DIR=!PROJECT_DIR!\!BUILD_DIR!\meson-logs"

echo Cleaning and creating output directory: "!OUTPUT_DIR!"
if exist "!OUTPUT_DIR!" rd /s /q "!OUTPUT_DIR!"
mkdir "!OUTPUT_DIR!"
echo.

if not exist "!SOURCE_DIR!" (
    echo ERROR: Build output directory not found at "!SOURCE_DIR!"
    goto :error_copy
)

echo Copying all files and folders from "!SOURCE_DIR!" to "!OUTPUT_DIR!"...
xcopy "!SOURCE_DIR!" "!OUTPUT_DIR!" /E /I /Y /Q
echo.

rem Ensure the freshly built d3d11.dll is in the output dir (it lives in src\d3d11 after the build)
set "D3D11_BUILD_DIR=!PROJECT_DIR!\!BUILD_DIR!\src\d3d11"
if exist "!D3D11_BUILD_DIR!\d3d11.dll" (
    echo Copying d3d11.dll from "!D3D11_BUILD_DIR!" to "!OUTPUT_DIR!"...
    copy /Y "!D3D11_BUILD_DIR!\d3d11.dll" "!OUTPUT_DIR!\d3d11.dll" >nul
    if exist "!D3D11_BUILD_DIR!\d3d11.pdb" copy /Y "!D3D11_BUILD_DIR!\d3d11.pdb" "!OUTPUT_DIR!\d3d11.pdb" >nul
) else (
    echo WARNING: d3d11.dll not found at "!D3D11_BUILD_DIR!" - deployment may be incomplete.
)

rem Optional: copy dxgi.dll if enable_dxgi was turned on at meson setup
set "DXGI_BUILD_DIR=!PROJECT_DIR!\!BUILD_DIR!\src\dxgi"
if exist "!DXGI_BUILD_DIR!\dxgi.dll" (
    echo Copying dxgi.dll from "!DXGI_BUILD_DIR!" to "!OUTPUT_DIR!"...
    copy /Y "!DXGI_BUILD_DIR!\dxgi.dll" "!OUTPUT_DIR!\dxgi.dll" >nul
    if exist "!DXGI_BUILD_DIR!\dxgi.pdb" copy /Y "!DXGI_BUILD_DIR!\dxgi.pdb" "!OUTPUT_DIR!\dxgi.pdb" >nul
)

if not exist "!SHADER_BUILD_DIR!" goto :skip_shader_copy
echo Copying RTX shader binaries to "!OUTPUT_DIR!\rtx_shaders"...
mkdir "!OUTPUT_DIR!\rtx_shaders" >nul
robocopy "!SHADER_BUILD_DIR!" "!OUTPUT_DIR!\rtx_shaders" *.spv /NFL /NDL /NJH /NJS /NC /NS /NP >nul
set "ROBOCOPY_EXIT=!ERRORLEVEL!"
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
mkdir "!OUTPUT_DIR!\logs" >nul
if not exist "!BUILD_LOG_DIR!" goto :skip_log_copy
mkdir "!OUTPUT_DIR!\logs\build" >nul
robocopy "!BUILD_LOG_DIR!" "!OUTPUT_DIR!\logs\build" *.* /E /NFL /NDL /NJH /NJS /NC /NS /NP >nul
set "ROBOCOPY_EXIT=!ERRORLEVEL!"
if !ROBOCOPY_EXIT! GEQ 8 (
    echo ERROR: Failed to copy Meson build logs.
    goto :error_copy
)
goto :log_copy_done
:skip_log_copy
echo WARNING: Meson log directory not found.
:log_copy_done
if exist "!PROJECT_DIR!\!BUILD_DIR!\.ninja_log" copy "!PROJECT_DIR!\!BUILD_DIR!\.ninja_log" "!OUTPUT_DIR!\logs\.ninja_log" >nul
if exist "!PROJECT_DIR!\!BUILD_DIR!\.ninja_deps" copy "!PROJECT_DIR!\!BUILD_DIR!\.ninja_deps" "!OUTPUT_DIR!\logs\.ninja_deps" >nul
set "README_LINE_1=Release build logs copied from !BUILD_LOG_DIR!."
echo !README_LINE_1! > "!OUTPUT_DIR!\logs\README.txt"
set "README_LINE_2=To gather runtime DXVK / Remix logs, set the environment variable DXVK_LOG_PATH to a writable folder before launching the game."
echo !README_LINE_2! >> "!OUTPUT_DIR!\logs\README.txt"


echo.
echo #############################################################
echo # Deploying artifacts to game directory...                  #
echo #############################################################
echo.

if not exist "!GAME_DIR!" (
    echo ERROR: Game directory not found at "!GAME_DIR!".
    echo Re-run this script with the "config" argument to point it elsewhere.
    goto :error_copy
)

>"!GAME_DIR!\__remix_write_test.tmp" echo.
if errorlevel 1 (
    echo ERROR: Unable to write to "!GAME_DIR!". Please run as Administrator.
    goto :error_copy
)
del "!GAME_DIR!\__remix_write_test.tmp" >nul

rem DXVK state caches are shader-hash keyed, so entries auto-invalidate when
rem their inputs change.  Keeping the cache across DLL rebuilds saves the
rem multi-minute first-run pipeline compile on every iteration.  Pass
rem "clean" (or "cleancache") to this script to force a full cache wipe
rem when you suspect the cache itself is corrupt.
if defined WIPE_CACHE (
    echo Clearing DXVK shader caches...
    del "!GAME_DIR!\*.dxvk-cache" 2>nul
    del "!GAME_RUNTIME_DIR!\*.dxvk-cache" 2>nul
) else (
    echo Preserving DXVK shader caches ^(pass "clean" to wipe^).
)

echo Copying runtime package to "!GAME_RUNTIME_DIR!"...
if not exist "!GAME_RUNTIME_DIR!" (
    mkdir "!GAME_RUNTIME_DIR!" >nul
)
rem This command copies the entire output folder, including d3d11.dll, into bin\x64_retail
rem next to materialsystem_dx11.dll so Source's loader picks up the Remix DX11 bridge.
robocopy "!OUTPUT_DIR!" "!GAME_RUNTIME_DIR!" *.* /E /IS /R:2 /W:2 /NFL /NDL /NJH /NJS /NC /NS /NP >nul
set "ROBOCOPY_EXIT=!ERRORLEVEL!"
if !ROBOCOPY_EXIT! GEQ 8 (
    echo ERROR: Failed to deploy runtime files.
    goto :error_copy
)

if exist "!OUTPUT_DIR!\rtx_shaders" (
    echo Syncing shader binaries to "!GAME_SHADER_DIR!"...
    if not exist "!GAME_SHADER_DIR!" (
        mkdir "!GAME_SHADER_DIR!" >nul
    )
    robocopy "!OUTPUT_DIR!\rtx_shaders" "!GAME_SHADER_DIR!" *.spv /E /IS /R:2 /W:2 /NFL /NDL /NJH /NJS /NC /NS /NP >nul
    set "ROBOCOPY_EXIT=!ERRORLEVEL!"
    if !ROBOCOPY_EXIT! GEQ 8 (
        echo ERROR: Failed to update shader binaries.
        goto :error_copy
    )
)

rem Always (re)point DXVK_LOG_PATH at THIS game's log directory, regardless of
rem whether it already exists.  Without this, a persistent DXVK_LOG_PATH left
rem over from a previous game (e.g. LEGO Batman 2) will silently redirect
rem remix-dxvk.log to the wrong folder and hide the real crash output.
if not exist "!GAME_LOG_DIR!" mkdir "!GAME_LOG_DIR!" >nul 2>&1
echo Pointing DXVK_LOG_PATH at "!GAME_LOG_DIR!"...
setx DXVK_LOG_PATH "!GAME_LOG_DIR!" >nul
if errorlevel 1 (
    echo WARNING: Failed to configure DXVK_LOG_PATH automatically.
) else (
    rem setx only affects NEW processes; update the current shell too so any
    rem follow-up commands in this session see the new value.
    set "DXVK_LOG_PATH=!GAME_LOG_DIR!"
)

echo.
echo Done copying artifacts.
goto :success


rem ===========================================================================
rem  Dependency bootstrap helpers
rem ===========================================================================

:ensure_python
rem Meson is a Python program and the Remix meson.build also shells out to
rem python3 during configure, so a real interpreter is non-negotiable.  Note
rem that C:\...\WindowsApps\python.exe may exist as an App Execution Alias stub
rem that only opens the Microsoft Store - hence probing by RUNNING it rather
rem than trusting `where python`.
set "PY_EXE="
call :probe_python
if defined PY_EXE goto :ensure_python_have
for %%d in (
    "%LOCALAPPDATA%\Programs\Python\Python313"
    "%LOCALAPPDATA%\Programs\Python\Python312"
    "%LOCALAPPDATA%\Programs\Python\Python311"
    "%ProgramFiles%\Python313"
    "%ProgramFiles%\Python312"
    "%ProgramFiles%\Python311"
) do (
    if not defined PY_EXE if exist "%%~d\python.exe" (
        set "PATH=%%~d;%%~d\Scripts;!PATH!"
        call :probe_python
    )
)
if defined PY_EXE goto :ensure_python_have

echo Python 3 was not found on this machine.
where winget >nul 2>&1
if errorlevel 1 (
    echo ERROR: winget is unavailable, so Python cannot be installed automatically.
    echo Install Python 3 from https://www.python.org/downloads/ ^(tick "Add
    echo python.exe to PATH"^) and re-run this script.
    exit /b 1
)
echo Installing Python 3.13 for the current user via winget...
winget install -e --id Python.Python.3.13 --scope user --silent --accept-package-agreements --accept-source-agreements
for %%d in (
    "%LOCALAPPDATA%\Programs\Python\Python313"
    "%LOCALAPPDATA%\Programs\Python\Python312"
) do (
    if not defined PY_EXE if exist "%%~d\python.exe" (
        set "PATH=%%~d;%%~d\Scripts;!PATH!"
        call :probe_python
    )
)
if not defined PY_EXE (
    echo ERROR: Python still not usable after installation.
    echo Close this window, open a new one and re-run the script so the updated
    echo PATH is picked up.
    exit /b 1
)

:ensure_python_have
echo Python:  !PY_EXE!
rem pip is what installs meson; a stripped-down interpreter can be missing it.
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo Bootstrapping pip...
    python -m ensurepip --default-pip >nul 2>&1
    python -m pip --version >nul 2>&1
    if errorlevel 1 (
        echo ERROR: pip is unavailable for "!PY_EXE!" and could not be bootstrapped.
        exit /b 1
    )
)
rem `pip install --user` drops its .exe wrappers in the "nt_user" scripts
rem directory, which is not on PATH by default.  That directory is
rem %APPDATA%\Python\PythonXY\Scripts - NOT site.USER_BASE\Scripts - so ask
rem sysconfig instead of guessing, and prepend it now so meson and ninja
rem resolve as soon as pip creates them.
set "PY_SCRIPTS="
for /f "usebackq delims=" %%s in (`python -c "import sysconfig;print(sysconfig.get_path('scripts','nt_user'))" 2^>nul`) do set "PY_SCRIPTS=%%s"
if defined PY_SCRIPTS set "PATH=!PY_SCRIPTS!;!PATH!"
exit /b 0

:probe_python
rem Tests whatever `python` currently resolves to on PATH and, if it really is
rem Python 3, records the full path in PY_EXE.  Everything downstream invokes
rem the bare name: `for /f` runs its command through cmd, which mis-parses a
rem command line that STARTS with a quoted path, so "C:\...\python.exe" -c ...
rem silently produces no output there.  Callers prepend the interpreter's
rem folder to PATH before calling this, which sidesteps the whole problem.
set "PROBE_RESULT="
for /f "usebackq delims=" %%v in (`python -c "import sys;print(sys.version_info[0])" 2^>nul`) do set "PROBE_RESULT=%%v"
if not "!PROBE_RESULT!"=="3" exit /b 0
for /f "usebackq delims=" %%p in (`where python 2^>nul`) do if not defined PY_EXE set "PY_EXE=%%p"
if not defined PY_EXE set "PY_EXE=python"
exit /b 0

:ensure_meson
where meson >nul 2>&1
if not errorlevel 1 goto :ensure_meson_have
echo Meson not found - installing it with pip...
python -m pip install --user --upgrade meson
if errorlevel 1 (
    echo ERROR: Failed to install meson.
    exit /b 1
)
if defined PY_SCRIPTS set "PATH=!PY_SCRIPTS!;!PATH!"
where meson >nul 2>&1
if errorlevel 1 (
    echo ERROR: meson installed but is still not on PATH.
    exit /b 1
)
:ensure_meson_have
set "MESON_VERSION="
for /f "usebackq delims=" %%v in (`meson --version 2^>nul`) do set "MESON_VERSION=%%v"
echo Meson:   !MESON_VERSION!
exit /b 0

:ensure_ninja
where ninja >nul 2>&1
if not errorlevel 1 goto :ensure_ninja_have
rem Visual Studio's CMake component ships a ninja binary; prefer it over a
rem download when it is already sitting there.
if exist "%VSWHERE%" (
    for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -property installationPath 2^>nul`) do (
        if exist "%%i\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe" set "PATH=%%i\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;!PATH!"
    )
)
where ninja >nul 2>&1
if not errorlevel 1 goto :ensure_ninja_have
echo Ninja not found - installing it with pip...
python -m pip install --user --upgrade ninja
if errorlevel 1 (
    echo ERROR: Failed to install ninja.
    exit /b 1
)
if defined PY_SCRIPTS set "PATH=!PY_SCRIPTS!;!PATH!"
where ninja >nul 2>&1
if errorlevel 1 (
    echo ERROR: ninja installed but is still not on PATH.
    exit /b 1
)
:ensure_ninja_have
set "NINJA_VERSION="
for /f "usebackq delims=" %%v in (`ninja --version 2^>nul`) do set "NINJA_VERSION=%%v"
echo Ninja:   !NINJA_VERSION!
exit /b 0

:ensure_submodules
rem A `git clone` without --recursive leaves submodules\* empty, which later
rem fails with confusing missing-header errors instead of an obvious "you
rem forgot the submodules" message.
if not exist "!PROJECT_DIR!\.git" exit /b 0
where git >nul 2>&1
if errorlevel 1 exit /b 0
set "SUBMODULE_FILES=0"
for /f %%n in ('dir /b "!PROJECT_DIR!\submodules\rtxdi" 2^>nul ^| find /c /v ""') do set "SUBMODULE_FILES=%%n"
if !SUBMODULE_FILES! GTR 0 (
    echo Submodules: present
    exit /b 0
)
echo Git submodules missing - fetching them ^(this can take a while^)...
git -C "!PROJECT_DIR!" submodule update --init --recursive
if errorlevel 1 echo WARNING: git submodule update reported an error; the build may fail.
exit /b 0


rem ===========================================================================
rem  Exit paths
rem ===========================================================================

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
popd 2>nul
pause
exit /b 1

:success
echo.
echo #############################################################
echo # Release build finished successfully.                      #
echo #############################################################
echo.
popd
pause
exit /b 0
