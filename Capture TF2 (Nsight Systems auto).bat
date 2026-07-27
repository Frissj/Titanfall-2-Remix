@echo off
REM ===================================================================
REM  Unattended Nsight Systems capture. Double-click and walk away.
REM
REM  Launches the game under nsys ARMED BUT NOT COLLECTING, so level load
REM  is never in the trace. The GAME then decides when to capture: it
REM  counts rtx.nsysAutoCaptureSettleSeconds of ACTUAL GAMEPLAY (menus and
REM  loading screens do not count) and prints CAPTURE-BEGIN, then
REM  CAPTURE-END rtx.nsysAutoCaptureSeconds later. This script presses F11
REM  on each - that keypress is what nsys is waiting on - and nsys then
REM  writes the report and closes the game.
REM
REM  So the window is measured FROM GAMEPLAY, not from launch. Just play;
REM  the key is pressed for you. If the console ever says a press was
REM  skipped because the game did not have focus, press F11 yourself.
REM
REM  (The keypress is standing in for --capture-range=nvtx on the game's
REM  own NVTX range, which never fired - see
REM  HANDOFF_NSYS_AND_SCENE_REBUILD.md sec 6. If the hotkey turns out to
REM  be dead too, -Trigger delay falls back to a fixed wall-clock window,
REM  which then needs -DelaySeconds 70 to clear the 56-63 s load.)
REM
REM  rtx.conf must have:
REM    rtx.nsysAutoCapture = True
REM    rtx.nsysAutoCaptureExitOnFinish = False
REM
REM  Report lands in:  %USERPROFILE%\Desktop\tf2_nsys\tf2_<stamp>.nsys-rep
REM
REM  Tune without editing anything:
REM    "Capture TF2 (Nsight Systems auto).bat" -HotkeyCapture F9
REM    "Capture TF2 (Nsight Systems auto).bat" -Trigger delay -DelaySeconds 70
REM    "Capture TF2 (Nsight Systems auto).bat" -KeepGameOpen
REM  or call the .ps1 directly for -Trace / -OutDir.
REM ===================================================================

cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Capture TF2 (Nsight Systems auto).ps1" %*
echo.
echo Exit code: %ERRORLEVEL%
pause
