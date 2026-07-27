<#
===============================================================================
 Unattended Nsight Systems capture of Titanfall 2 + Remix.

 Launch it and walk away. The capture window is GAMEPLAY-RELATIVE: it starts
 rtx.nsysAutoCaptureSettleSeconds after you actually reach the level, not
 after some fixed wall-clock delay from launch. The run is:

   1. one-shot 'nsys profile --capture-range=hotkey' - collection is
      configured up front but ARMED, waiting for a keypress. Level load is
      therefore never in the trace.
   2. the GAME decides when. The [NsysAuto] director counts
      rtx.nsysAutoCaptureSettleSeconds of ACTUAL GAMEPLAY - frames with a
      non-empty ordered-instance list, so menus and loading screens freeze
      the clock rather than advance it - and prints [NsysAuto] CAPTURE-BEGIN,
      then CAPTURE-END one rtx.nsysAutoCaptureSeconds later.
   3. this script watches remix-dxvk.log for those two markers and presses
      the hotkey on each: first press starts collection, second ends the
      capture range. That is the same split as before - the game owns the
      TIMING because only it knows what is gameplay - with the trigger swapped
      to one that works.
   4. --capture-range-end=stop-shutdown ends the session on the second press,
      nsys writes the report and --kill=true closes the game.

 WHY THE HOTKEY AND NOT NVTX. Same job, different mechanism, because the NVTX
 one is dead: --capture-range=nvtx on the director's range never started
 collection, never honoured capture-range-end, never killed, and printed
 "No reports were generated" - despite nvtxRangeStartA returning a live id
 in-process. Nine variants were tested and all failed; see
 HANDOFF_NSYS_AND_SCENE_REBUILD.md sec 6 and do not retry them. --capture-range
 =hotkey is a different path entirely (an input hook in the injected library,
 documented as "works for graphic apps only"), and it reuses the one-shot
 'nsys profile' form that IS proven on this game - the same form produced a
 7.4 MB report on the first attempt.

 IF THE HOTKEY ALSO TURNS OUT TO BE DEAD, -Trigger delay is the fallback:
 plain --delay/--duration, wall clock from launch, no in-game trigger of any
 kind. It cannot tell gameplay from a loading screen (load measures 56-63 s
 here, so -DelaySeconds needs to be ~70), which is exactly why it is not the
 default. Both modes print the same COVERAGE verdict at the end, computed
 from the director's markers, so a bad window is never a silent one.

 REQUIRED in rtx.conf:
   rtx.nsysAutoCapture = True
   rtx.nsysAutoCaptureSettleSeconds = 10     <- GAMEPLAY seconds before capture
   rtx.nsysAutoCaptureSeconds = 10           <- length of the capture
   rtx.nsysAutoCaptureExitOnFinish = False   <- leave False, nsys owns the exit

 DXVK_PERF_EVENTS=1 is set for the child process only (never setx it - a
 persistent env var has skewed every run on this machine once before), so
 Remix's own pass names appear as Vulkan debug ranges in the timeline.
 +r_no_stalls 1 is passed too, or the trace is dominated by the engine's
 busy-wait fence instead of real work.
===============================================================================
#>

[CmdletBinding()]
param(
  # Newest install wins if this is left empty.
  [string] $Nsys = "",
  [string] $GameExe = "",
  [string] $GameArgs = "+r_no_stalls 1",
  [string] $RemixLog = "",
  [string] $OutDir = "$env:USERPROFILE\Desktop\tf2_nsys",

  # WHAT STARTS THE CAPTURE.
  #
  #   hotkey  (default) GAMEPLAY-RELATIVE. nsys sits armed on
  #           --capture-range=hotkey; this script presses the key when the game
  #           reports it has been in gameplay for the settle, and again when
  #           the game reports the capture is over. The window is therefore
  #           measured from the level, not from launch, and load times can vary
  #           by as much as they like without moving it.
  #
  #   delay   FALLBACK ONLY, for if the hotkey turns out to be as dead as the
  #           NVTX trigger was. Plain --delay/--duration, wall clock from
  #           process launch, nothing in-game involved. Blunt but proven.
  [ValidateSet('hotkey', 'delay')]
  [string] $Trigger = 'hotkey',

  # Which key. nsys allows F1-F12 except F10, and defaults to F12 - avoided
  # here because F12 is the Steam overlay screenshot key and a screenshot
  # firing on the same press is a needless variable. F11 is not bound by the
  # game.
  [ValidateSet('F1','F2','F3','F4','F5','F6','F7','F8','F9','F11','F12')]
  [string] $HotkeyCapture = 'F11',

  # The capture window, for -Trigger delay only. Both default to -1, meaning
  # "read it out of rtx.conf" so there is ONE place to change it:
  # rtx.nsysAutoCaptureSettleSeconds feeds --delay and rtx.nsysAutoCaptureSeconds
  # feeds --duration. Pass either explicitly to override for a single run.
  #
  # BEWARE THE UNIT CHANGE. Those two conf values are GAMEPLAY seconds to the
  # game, but --delay is WALL CLOCK FROM LAUNCH and nsys has no idea what a
  # loading screen is. Load measures 56-63 s on this machine, so under
  # -Trigger delay:
  #   delay <  ~65   -> the trace is the loading screen, and --kill=true then
  #                     closes the game before the level is even reached
  #   delay >= ~70   -> the trace is gameplay, roughly (delay - 60) seconds in
  # Under the default -Trigger hotkey the conf values keep their real meaning
  # and none of this applies.
  [double] $DelaySeconds = -1,
  [double] $DurationSeconds = -1,

  # Where to read those two defaults from.
  [string] $RtxConf = "",

  # API tracing - this belongs to 'nsys launch'.
  #
  # vulkan-annotations is what turns Remix's ScopedGpuProfileZone labels into
  # named ranges. Without it the timeline has no pass names and the capture is
  # nearly useless for attribution.
  #
  # NOTE: 'osrt' is a LINUX-ONLY trace value - nsys rejects it on Windows and the
  # launch dies instantly. OS thread scheduling on Windows comes from -CpuCtxSw
  # below, which is a 'nsys start' option, not a trace value.
  #
  # 'wddm' (Windows GPU scheduler packets) is NOT in the default, deliberately -
  # see the -Wddm switch below.
  # 'nvtx' is listed EXPLICITLY, not left to nsys to add.
  #
  # Without it every run printed "WARNING: NVTX tracing is required for NVTX
  # profiler API support. Turning it on by default." - i.e. the command line was
  # incomplete and nsys patched it after the fact. The docs are explicit that
  # "you must enable CUDA or NVTX tracing of the target application for
  # '-c nvtx' to work", and it is not established that the late auto-enable
  # reaches the injected library with the same effect as asking for it up front.
  [string] $Trace = "vulkan,vulkan-annotations,nvtx",

  # Add wddm GPU-scheduler tracing. OFF by default because it is the one
  # ETW-based option that is fixed at LAUNCH time and therefore cannot be
  # retried away when 'nsys start' throws ETWSession::Start - by then the
  # gameplay settle has already been paid and the whole run is lost.
  #
  # Measured 2026-07-27: launch/start/stop with wddm + cpuctxsw succeeds on a
  # trivial app on this machine, elevated, so the flags and the workflow are
  # both fine. It failed only with the game attached, and that was not isolated
  # further. If you want wddm, the first thing to try is closing the NVIDIA App
  # overlay / FrameView - GraphicsPerfMonitorSession and three NVIDIA-NVTOPPS-*
  # ETW sessions are running on this machine and are the obvious candidates for
  # a conflict over the graphics providers.
  [switch] $Wddm,

  # Collection options - these belong to 'nsys start', not 'launch'.
  #
  # cpuctxsw shows a thread BLOCKED rather than running - the fenceWaitMs 64 ms
  # appearing as a real wait on the timeline. It is OFF by default because it is
  # the option that was actually breaking the capture:
  #
  #   MEASURED 2026-07-27. Two runs, both elevated, both threw
  #   'ETWSession::Start ... SystemException' from 'nsys start'. The second run
  #   had wddm already removed and still threw, so wddm was never the cause -
  #   cpuctxsw is the ETW consumer here. The throw takes ~13 s to surface, during
  #   which the game keeps rendering, so it superficially looks like collection
  #   started.
  #
  #   It is NOT elevation (both runs elevated; an elevated .bat propagates its
  #   token through powershell.exe to nsys) and NOT the flag itself - the same
  #   launch/start/stop sequence with cpuctxsw=process-tree AND wddm succeeds on
  #   a trivial app on this machine. It fails only with the game running.
  #
  #   Best remaining explanation, NOT proven: opening an ETW kernel session
  #   collides with NVIDIA's own graphics ETW sessions, which are active while a
  #   game runs - 'logman query -ets' shows GraphicsPerfMonitorSession plus three
  #   NVIDIA-NVTOPPS-* sessions. To test that, close the NVIDIA App overlay and
  #   FrameView, then re-run with -CpuCtxSw process-tree.
  #
  # None of this affects per-pass GPU attribution: Vulkan API tracing and the
  # vulkan-annotations pass names come from the injected layer, not from ETW.
  [ValidateSet('process-tree', 'system-wide', 'none')]
  [string] $CpuCtxSw = "none",

  # CPU IP/backtrace sampling. 'process-tree' and 'system-wide' REQUIRE
  # administrative privileges - left off so a non-elevated run does not fail
  # after the game has already loaded.
  [ValidateSet('process-tree', 'system-wide', 'none')]
  [string] $Sample = "none",

  # Backstop, counted ON TOP of the delay+duration window. nsys ends the session
  # itself under --duration, so this is only reached by a run that has hung.
  [double] $ArmTimeoutSec = 900,

  # Seconds to wait for the report to finish writing after the game is force
  # closed. Only used on the hung path above.
  [double] $ReportTimeoutSec = 600,

  # Leave the game running after a successful capture. Passes --kill=false to
  # nsys, which may then sit waiting for the application to exit rather than
  # returning when the duration elapses - in which case the ArmTimeoutSec
  # backstop above eventually force-closes the game anyway. So this buys you
  # roughly ArmTimeoutSec of play after the capture, not an indefinite session.
  [switch] $KeepGameOpen
)

$ErrorActionPreference = 'Stop'
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

function Write-Step([string] $msg) {
  Write-Host ("[{0:HH:mm:ss}] {1}" -f (Get-Date), $msg)
}
function Write-Bad([string] $msg) {
  Write-Host ("[{0:HH:mm:ss}] {1}" -f (Get-Date), $msg) -ForegroundColor Red
}

# ---------------------------------------------------------------- resolve paths
if ([string]::IsNullOrWhiteSpace($GameExe)) {
  $GameExe = Join-Path $scriptDir "Titanfall2.exe"
}
if ([string]::IsNullOrWhiteSpace($RemixLog)) {
  $RemixLog = Join-Path $scriptDir "rtx-remix\logs\remix-dxvk.log"
}
if ([string]::IsNullOrWhiteSpace($RtxConf)) {
  $RtxConf = Join-Path $scriptDir "rtx.conf"
}
if ([string]::IsNullOrWhiteSpace($Nsys)) {
  $candidates = Get-ChildItem "C:\Program Files\NVIDIA Corporation" -Directory -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -like "Nsight Systems*" } |
                Sort-Object Name -Descending
  foreach ($c in $candidates) {
    $probe = Join-Path $c.FullName "target-windows-x64\nsys.exe"
    if (Test-Path $probe) { $Nsys = $probe; break }
  }
}

if (-not (Test-Path $Nsys))    { Write-Bad "nsys.exe not found. Pass -Nsys <path>."; exit 2 }
if (-not (Test-Path $GameExe)) { Write-Bad "Game not found: $GameExe"; exit 2 }

if (Get-Process -Name "Titanfall2" -ErrorAction SilentlyContinue) {
  Write-Bad "Titanfall2 is already running. Close it first - nsys must own the process it profiles."
  exit 2
}

if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir -Force | Out-Null }

# ------------------------------------------------- capture window from rtx.conf
# One source of truth. The conf already carries both numbers, with the comment
# block that explains them sitting right above; re-typing them on the command
# line would only create a second place to be wrong.
function Get-ConfNumber([string] $path, [string] $key) {
  if (-not (Test-Path $path)) { return $null }
  # Last uncommented assignment wins, matching how the option parser reads the
  # file - a value appended at the end of the conf overrides an earlier one.
  # '#' lines cannot match because the pattern is anchored at the key.
  $hit = $null
  foreach ($line in (Get-Content -LiteralPath $path)) {
    if ($line -match ('^\s*' + [regex]::Escape($key) + '\s*=\s*([0-9]*\.?[0-9]+)')) {
      $hit = [double] $matches[1]
    }
  }
  return $hit
}

if ($DelaySeconds -lt 0) {
  $v = Get-ConfNumber $RtxConf 'rtx.nsysAutoCaptureSettleSeconds'
  if ($null -eq $v) { Write-Bad "rtx.nsysAutoCaptureSettleSeconds not found in $RtxConf - pass -DelaySeconds instead."; exit 2 }
  $DelaySeconds = $v
}
if ($DurationSeconds -lt 0) {
  $v = Get-ConfNumber $RtxConf 'rtx.nsysAutoCaptureSeconds'
  if ($null -eq $v) { Write-Bad "rtx.nsysAutoCaptureSeconds not found in $RtxConf - pass -DurationSeconds instead."; exit 2 }
  $DurationSeconds = $v
}

# nsys takes whole seconds for both.
$delayArg    = [int] [Math]::Round($DelaySeconds)
$durationArg = [int] [Math]::Round($DurationSeconds)
if ($durationArg -lt 1) { $durationArg = 1 }

# Measured 2026-07-27 across seven runs: 56-63 s from launch to [NsysAuto] ARMED.
# Only meaningful under -Trigger delay; the hotkey path is measured from the
# level and does not care how long the load took.
#
# Warn rather than override - the value came from the conf and it is not this
# script's place to silently disagree with it, but a run that profiles the
# loading screen and then kills the game should not be a surprise afterwards.
$kMeasuredLoadSec = 56
if ($Trigger -eq 'delay' -and $delayArg -lt $kMeasuredLoadSec) {
  Write-Host ""
  Write-Host ("  WARNING: -Trigger delay with --delay={0}, shorter than the measured 56-63 s load." -f $delayArg) -ForegroundColor Yellow
  Write-Host  "  In this mode nsys counts WALL CLOCK FROM LAUNCH, not gameplay seconds, so the" -ForegroundColor Yellow
  Write-Host  "  window is very likely to cover the loading screen - and --kill=true will then" -ForegroundColor Yellow
  Write-Host  "  close the game while it is still loading." -ForegroundColor Yellow
  Write-Host  "  Either drop -Trigger delay (the default hotkey mode is gameplay-relative and" -ForegroundColor Yellow
  Write-Host  "  needs no such number), or pass -DelaySeconds 70." -ForegroundColor Yellow
  Write-Host  "  The run continues; the verdict at the end says which one you actually got." -ForegroundColor Yellow
  Write-Host ""
}

# Validate the trace list BEFORE launching. nsys rejects an unknown value with a
# usage dump and exits immediately, which - once the game is in the picture -
# looks exactly like "the game crashed on startup". Catching it here names the
# actual problem. ('osrt' is the trap: valid on Linux, not on Windows.)
$validTrace = @(
  'cuda','cuda-hw','nvtx','cublas','cublas-verbose','cuDNN','cuDNN-verbose',
  'cusolver','cusolver-verbose','cusparse','cusparse-verbose','nvvideo',
  'opengl','opengl-annotations','vulkan','vulkan-annotations',
  'dx11','dx11-annotations','dx12','dx12-annotations','openxr',
  'openxr-annotations','wddm','none'
)
foreach ($t in ($Trace -split ',')) {
  $tt = $t.Trim()
  if ($tt -eq '') { continue }
  if ($validTrace -notcontains $tt) {
    Write-Bad "Invalid -Trace value '$tt' (Windows nsys). Valid: $($validTrace -join ', ')"
    exit 2
  }
}

if ($Wddm) {
  $Trace = ($Trace.TrimEnd(',') + ",wddm")
  Write-Host ("[{0:HH:mm:ss}] wddm tracing ON by request - if 'nsys start' throws ETWSession::Start, this is the first thing to remove." -f (Get-Date)) -ForegroundColor Yellow
}

# ------------------------------------------------------------ ETW / elevation
# Context-switch tracing, CPU sampling and wddm all collect through ETW (Event
# Tracing for Windows), and per nsys's own help the 'process-tree'/'system-wide'
# modes require Administrator. So drop them when unelevated rather than let
# 'nsys start' throw a minute into the run.
#
# This is NOT the explanation for the 17:49 failure on this machine - that run
# WAS elevated (an elevated .bat passes its token to powershell.exe and on to
# nsys, so it does propagate). Elevation, the individual flags, and the
# launch/start/stop workflow were each tested and each work. Only the run with
# the game attached failed, and that is still unexplained. This block remains
# because the requirement is real, not because it was the cause.
#
# None of these are needed for per-pass GPU attribution - that is
# vulkan-annotations, which is not ETW-based.
$isAdmin = ([Security.Principal.WindowsPrincipal] `
            [Security.Principal.WindowsIdentity]::GetCurrent() `
           ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
  $dropped = @()
  $traceList = @($Trace -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne '' })
  if ($traceList -contains 'wddm') {
    $traceList = $traceList | Where-Object { $_ -ne 'wddm' }
    $Trace = ($traceList -join ',')
    $dropped += 'wddm trace'
  }
  if ($CpuCtxSw -ne 'none') { $CpuCtxSw = 'none'; $dropped += 'context switches' }
  if ($Sample   -ne 'none') { $Sample   = 'none'; $dropped += 'CPU sampling' }

  if ($dropped.Count -gt 0) {
    Write-Host ("[{0:HH:mm:ss}] NOT ELEVATED - dropping ETW-based collection: {1}." -f (Get-Date), ($dropped -join ', ')) -ForegroundColor Yellow
    Write-Host ("           Remix pass names and GPU workload timings are unaffected (vulkan-annotations is not ETW).") -ForegroundColor Yellow
    Write-Host ("           Right-click the .bat -> Run as administrator to keep them.") -ForegroundColor Yellow
  }
}

$stamp   = Get-Date -Format "yyyyMMdd_HHmmss"
$report  = Join-Path $OutDir "tf2_$stamp"     # nsys appends .nsys-rep
$repFile = "$report.nsys-rep"

Write-Step "nsys     : $Nsys"
Write-Step "game     : $GameExe $GameArgs"
Write-Step "log      : $RemixLog"
Write-Step "conf     : $RtxConf"
if ($Trigger -eq 'hotkey') {
  Write-Step ("trigger  : hotkey {0}, pressed by this script on the game's own gameplay markers" -f $HotkeyCapture)
  Write-Step ("window   : {0}s of gameplay after the level starts, then {1}s of capture (from rtx.conf)" -f $delayArg, $durationArg)
} else {
  Write-Step  "trigger  : none - fixed wall-clock window (-Trigger delay)"
  Write-Step ("window   : collect from t+{0}s to t+{1}s, wall clock from launch" -f $delayArg, ($delayArg + $durationArg))
}
Write-Step "report   : $repFile"

# --------------------------------------------------- log tailing (shared read)
# The game holds remix-dxvk.log open for writing, so it must be opened with
# FileShare ReadWrite|Delete or every read throws. Offset is tracked manually and
# reset if the file shrinks, which is what happens when the game recreates the
# log at startup - without that reset the markers from a PREVIOUS run would be
# matched immediately and the capture would fire before the game even loaded.
$script:logOffset = 0
if (Test-Path $RemixLog) { $script:logOffset = (Get-Item $RemixLog).Length }

function Read-NewLogText {
  if (-not (Test-Path $RemixLog)) { return "" }
  $fs = $null
  try {
    $fs = New-Object System.IO.FileStream($RemixLog,
            [System.IO.FileMode]::Open,
            [System.IO.FileAccess]::Read,
            ([System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete))
    if ($fs.Length -lt $script:logOffset) { $script:logOffset = 0 }   # recreated
    if ($fs.Length -eq $script:logOffset) { return "" }
    [void]$fs.Seek($script:logOffset, [System.IO.SeekOrigin]::Begin)
    $sr   = New-Object System.IO.StreamReader($fs)
    $text = $sr.ReadToEnd()
    $script:logOffset = $fs.Position
    $sr.Dispose()
    return $text
  } catch {
    return ""
  } finally {
    if ($fs -ne $null) { $fs.Dispose() }
  }
}

function Invoke-Nsys([string[]] $nsysArgs) {
  $p = Start-Process -FilePath $Nsys -ArgumentList $nsysArgs -NoNewWindow -PassThru -Wait
  return $p.ExitCode
}

# ------------------------------------------------------------- hotkey delivery
# nsys's hotkey trigger lives in the library injected INTO THE GAME, and it is
# documented as working for graphic apps only - i.e. it watches the game's own
# input, not this console. So the press has to be delivered as real system input
# while the game has focus.
#
# keybd_event, not SendKeys: SendKeys targets the foreground window through the
# managed forms layer and is unreliable against a game that reads input its own
# way, whereas keybd_event synthesises at the driver level and is seen by
# low-level keyboard hooks - which is what an injected profiler hook is.
#
# It deliberately does NOT steal focus. If the game is not in front, the press
# would land in whatever is, so the script says so and asks for a manual press
# instead of silently pressing the key into the wrong window.
Add-Type -Namespace Win32 -Name Kbd -MemberDefinition @'
  [DllImport("user32.dll")]
  public static extern void keybd_event(byte bVk, byte bScan, uint dwFlags, UIntPtr dwExtraInfo);
  [DllImport("user32.dll")]
  public static extern IntPtr GetForegroundWindow();
  [DllImport("user32.dll")]
  public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint pid);
'@

function Test-GameHasFocus {
  $hwnd = [Win32.Kbd]::GetForegroundWindow()
  if ($hwnd -eq [IntPtr]::Zero) { return $false }
  $pid32 = 0
  [void][Win32.Kbd]::GetWindowThreadProcessId($hwnd, [ref] $pid32)
  if ($pid32 -eq 0) { return $false }
  $proc = Get-Process -Id $pid32 -ErrorAction SilentlyContinue
  return ($null -ne $proc -and $proc.ProcessName -eq 'Titanfall2')
}

function Send-CaptureHotkey([string] $what) {
  # F1..F9 are VK 0x70..0x78, F11 is 0x7A, F12 is 0x7B (F10 is 0x79 and nsys
  # does not allow it on Windows, so it is not in the ValidateSet).
  $vk = 0x70 + ([int] $HotkeyCapture.Substring(1)) - 1
  $KEYEVENTF_KEYUP = 0x2

  if (-not (Test-GameHasFocus)) {
    Write-Bad ("{0}: the game does not have focus, so the {1} press would go to another window." -f $what, $HotkeyCapture)
    Write-Bad ("  Click back into the game and press {0} yourself - nsys is listening for it." -f $HotkeyCapture)
    return $false
  }

  [Win32.Kbd]::keybd_event([byte] $vk, 0, 0, [UIntPtr]::Zero)
  Start-Sleep -Milliseconds 60
  [Win32.Kbd]::keybd_event([byte] $vk, 0, $KEYEVENTF_KEYUP, [UIntPtr]::Zero)
  Write-Step ("{0}: sent {1} to the game." -f $what, $HotkeyCapture)
  return $true
}

function Stop-Game {
  $g = Get-Process -Name "Titanfall2" -ErrorAction SilentlyContinue
  if ($g) {
    # Force-kill deliberately. A clean quit crashes in this fork (a cached
    # client.dll function pointer is called after the engine unloads it), and
    # the report is already finalised by this point, so there is nothing to
    # lose by skipping shutdown.
    Write-Step "Closing the game."
    $g | Stop-Process -Force -ErrorAction SilentlyContinue
  }
}

# ----------------------------------------------------------------------- launch
$env:DXVK_PERF_EVENTS = "1"     # process-scoped ONLY; never setx this

# ONE-SHOT 'nsys profile', NOT 'launch' + 'start' + 'stop'.
#
# The interactive session workflow was tried first and does not work on this
# game: 'nsys start' threw 'ETWSession::Start ... SystemException' after ~13 s
# on three consecutive runs, including one with cpuctxsw, sample and wddm all
# disabled - so it was never those options. One-shot 'nsys profile' on the same
# game with the same trace options produced a 7.4 MB report on the first try.
# The fault is attaching collection mid-run, not the configuration.
#
# The WINDOW is still decided by the GAME, because only the game knows what is
# gameplay. What changed is how that decision reaches nsys.
#
# It used to reach it in-process, through an NVTX range the [NsysAuto] director
# opened after its gameplay settle:
#
#   --capture-range=nvtx -p RemixCapture     start collecting when that range
#                                            begins, stop when it ends
#   --capture-range-end=stop-shutdown        end the session at range end
#
# THAT NEVER TRIGGERED. Collection never started, capture-range-end never
# fired, --kill=true never killed, nsys waited for the app and then printed
# "No reports were generated". Not for want of the range: nvtxRangeStartA
# returned a live id in-process, and the domain form, push/pop vs start/end,
# and explicit '-t nvtx' were each tested separately and each failed. See
# HANDOFF_NSYS_AND_SCENE_REBUILD.md sec 6 before spending a run re-testing any
# of them.
#
# It now reaches nsys as INPUT instead, on the same markers:
#
#   --capture-range=hotkey                   start collecting on the keypress
#   --hotkey-capture=F11                     which key
#   --capture-range-end=stop-shutdown        end the session at range end
#   --kill=true                              close the game once profiling ends
#
# The director is unchanged and still authoritative - it counts GAMEPLAY frames
# and prints CAPTURE-BEGIN / CAPTURE-END exactly as before. The only difference
# is that this script now converts those two markers into two keypresses rather
# than nsys reading the NVTX range itself. Same timing, working trigger.
#
# 'nvtx' stays in -t regardless: the RemixCapture range is then drawn in the
# timeline alongside the collected region, so the window can be checked from
# inside the trace and not only from this console. (That range is named by
# kNsysCaptureRangeName in rtx_context.cpp; nothing here has to match it, since
# it is no longer the trigger.)

# No '--' separator: nsys takes the application positionally, and a version that
# does not recognise '--' would treat it AS the application name.
# Elements are not pre-quoted either - Start-Process quotes what needs it, and
# hand-quoting on top produces doubled quotes that nsys reads as a literal path.
$profileArgs = @(
  "profile",
  "-t", $Trace,
  "--vulkan-gpu-workload=individual",
  "--cpuctxsw=$CpuCtxSw",
  "--sample=$Sample",
  # -KeepGameOpen was previously declared and never wired to anything, so the
  # switch the .bat advertises did nothing. It has to be honoured HERE: with
  # --kill=true nsys terminates the game as soon as profiling ends, and no
  # amount of not-calling-Stop-Game afterwards can undo that.
  ("--kill=" + $(if ($KeepGameOpen) { "false" } else { "true" })),
  "-o", $report,
  "--force-overwrite=true"
)

if ($Trigger -eq 'hotkey') {
  # NO --duration here, deliberately. Whether nsys measures --duration from the
  # start of COLLECTION or from process launch is not documented, and if it is
  # the latter the session would shut down before the level even loaded - which
  # is precisely the failure this mode exists to avoid. The second hotkey press
  # ends the range instead, which keeps the window exactly as long as the game
  # says it is (rtx.nsysAutoCaptureSeconds) rather than however long nsys
  # thinks. If the second press is somehow not honoured, the backstop below
  # closes the game and nsys writes what it collected.
  $profileArgs += @(
    "--capture-range=hotkey",
    "--hotkey-capture=$HotkeyCapture",
    "--capture-range-end=stop-shutdown"
  )
} else {
  # Fallback mode. --capture-range is absent entirely, so --capture-range-end
  # would be meaningless (it is documented as applicable only alongside
  # --capture-range) and is omitted with it.
  $profileArgs += @(
    "--delay=$delayArg",
    "--duration=$durationArg"
  )
}

# The application goes LAST: nsys takes it positionally, and every option after
# it would be passed to the game instead of to nsys.
$profileArgs += $GameExe
if (-not [string]::IsNullOrWhiteSpace($GameArgs)) {
  $profileArgs += $GameArgs.Split(' ')
}

if ($Trigger -eq 'hotkey') {
  Write-Step ("nsys profile - armed on hotkey {0}. Nothing is collected until the game reaches gameplay." -f $HotkeyCapture)
} else {
  Write-Step ("nsys profile - fixed window, collecting t+{0}s to t+{1}s." -f $delayArg, ($delayArg + $durationArg))
}
$launchStart = Get-Date
$launchProc = Start-Process -FilePath $Nsys -ArgumentList $profileArgs -NoNewWindow -PassThru

# ------------------------------------------------------------------ marker loop
# Under -Trigger hotkey this loop DRIVES the capture: the game's two markers
# become the two keypresses that open and close the range. Under -Trigger delay
# nsys is on its own clock and the loop only observes.
#
# Either way every marker is stamped in seconds since launch, which is the same
# origin --delay counts from, so the verdict at the end is arithmetic rather
# than a guess.
$armedAt = $null
$beginAt = $null
$endAt   = $null
$noArmedWarned = $false
$pressedBegin  = $false
$pressedEnd    = $false

# Backstop only. It has to cover the whole window plus the write-out, not just
# the arming, or a long capture would be torn down while it was still collecting.
$deadline = (Get-Date).AddSeconds($delayArg + $durationArg + $ArmTimeoutSec)

if ($Trigger -eq 'hotkey') {
  Write-Step "Waiting for the game to reach gameplay. Play normally - the key is pressed for you."
  Write-Step ("If a press is ever reported as skipped, just press {0} yourself; nsys is listening for it." -f $HotkeyCapture)
} else {
  Write-Step "Running. Watching remix-dxvk.log to timestamp what the window caught..."
}

while (-not $launchProc.HasExited -and (Get-Date) -lt $deadline) {
  Start-Sleep -Milliseconds 250
  $now = ((Get-Date) - $launchStart).TotalSeconds

  $chunk = Read-NewLogText
  if ([string]::IsNullOrEmpty($chunk)) { continue }

  foreach ($line in $chunk -split "`r?`n") {
    if ($line -match '\[NsysAuto\] ARMED' -and $null -eq $armedAt) {
      $armedAt = $now
      Write-Step ("t+{0,6:N1}s  GAMEPLAY REACHED (ARMED) - {1}" -f $armedAt, $line.Trim())
    }
    elseif ($line -match '\[NsysAuto\] CAPTURE-BEGIN' -and $null -eq $beginAt) {
      $beginAt = $now
      Write-Step ("t+{0,6:N1}s  CAPTURE-BEGIN - {1}s of gameplay elapsed" -f $beginAt, $delayArg)
      if ($Trigger -eq 'hotkey') { $pressedBegin = Send-CaptureHotkey "CAPTURE-BEGIN" }
    }
    elseif ($line -match '\[NsysAuto\] CAPTURE-END' -and $null -eq $endAt) {
      $endAt = $now
      Write-Step ("t+{0,6:N1}s  CAPTURE-END" -f $endAt)
      if ($Trigger -eq 'hotkey') {
        $pressedEnd = Send-CaptureHotkey "CAPTURE-END"
        Write-Step "nsys should now stop collecting, write the report and close the game."
      }
    }
  }

  # A run that never prints ARMED is almost always rtx.nsysAutoCapture=False or
  # a binary without the director. Under -Trigger hotkey that is fatal - there
  # is nothing to press on - so say so early rather than at the timeout.
  if ($null -eq $armedAt -and -not $noArmedWarned -and $now -gt 120) {
    $noArmedWarned = $true
    if ($Trigger -eq 'hotkey') {
      Write-Bad ("No [NsysAuto] ARMED after 120 s - the director is not running, so nothing will ever press {0}. Check rtx.nsysAutoCapture = True in rtx.conf. You can still press {0} by hand to capture." -f $HotkeyCapture)
    } else {
      Write-Bad "No [NsysAuto] ARMED after 120 s - the director is not running (check rtx.nsysAutoCapture = True in rtx.conf). The capture still ran; it just cannot be cross-referenced."
    }
  }
}

# Refresh() is required: on a Start-Process -PassThru object ExitCode stays empty
# until the process record is re-read, which is why an earlier run printed
# "nsys exited with code ." with nothing after it.
$launchProc.Refresh()

if ($launchProc.HasExited) {
  $elapsed = ((Get-Date) - $launchStart).TotalSeconds

  # "Too early to have collected anything" means different things per mode: the
  # fixed window has a known end, whereas the hotkey window ends whenever the
  # game says so, and the only thing that can be called premature there is
  # exiting before the capture was even triggered.
  $diedEarly = if ($Trigger -eq 'hotkey') { -not $pressedEnd } `
               else { $elapsed -lt ($delayArg + $durationArg - 2) }

  if ($diedEarly) {
    # Three causes, all needing different fixes, so name which one it was.
    # nsys rejecting an option dies in seconds with a usage dump above.
    if ($launchProc.ExitCode -ne 0 -and $elapsed -lt 15) {
      Write-Bad ("nsys profile itself failed (exit {0} after {1:N1}s) - the game was never started. Read the nsys output above; it names the rejected option." -f $launchProc.ExitCode, $elapsed)
    } elseif ($Trigger -eq 'hotkey' -and $pressedBegin) {
      Write-Bad ("The run ended {0:N1}s in, after the capture started but before it was closed (nsys exit {1}). Whatever was collected is below." -f $elapsed, $launchProc.ExitCode)
    } elseif ($Trigger -eq 'hotkey') {
      Write-Bad ("The game exited {0:N1}s in, before the capture was ever triggered (nsys exit {1})." -f $elapsed, $launchProc.ExitCode)
    } else {
      Write-Bad ("The game exited {0:N1}s in, before the t+{1}s end of the window (nsys exit {2})." -f $elapsed, ($delayArg + $durationArg), $launchProc.ExitCode)
    }
    # Only bail outright if nothing was collected. If the first press landed,
    # there may well be a usable report, so fall through and let the size check
    # and the verdict speak.
    if (-not ($Trigger -eq 'hotkey' -and $pressedBegin)) {
      Stop-Game
      exit 3
    }
  }
} else {
  # nsys past its own deadline. Both modes are supposed to shut the session down
  # by themselves - --duration on its clock, capture-range-end on the second
  # press - so if nsys is still up it is waiting on the game. Closing the game
  # releases it and salvages what was collected, which is what had to be done by
  # hand on the first NVTX run.
  Write-Bad ("nsys still running at t+{0:N0}s, past the point it should have shut down. Closing the game to release it." -f ((Get-Date) - $launchStart).TotalSeconds)
  if ($Trigger -eq 'hotkey' -and -not $pressedBegin) {
    Write-Bad ("The capture was never triggered: {0} was never pressed (no CAPTURE-BEGIN marker), so expect no report." -f $HotkeyCapture)
  } elseif ($Trigger -eq 'hotkey' -and -not $pressedEnd) {
    Write-Bad ("Collection started but the closing {0} press never went out, so the session was never shut down." -f $HotkeyCapture)
  }
  Stop-Game

  $repDeadline = (Get-Date).AddSeconds($ReportTimeoutSec)
  while (-not $launchProc.HasExited -and (Get-Date) -lt $repDeadline) {
    Start-Sleep -Seconds 1
    [void](Read-NewLogText)     # keep draining so the game never blocks on the log
  }
  $launchProc.Refresh()
  if (-not $launchProc.HasExited) {
    Write-Bad "nsys did not exit within $ReportTimeoutSec s."
    exit 7
  }
}

Write-Step "nsys exited with code $($launchProc.ExitCode)."
if (-not $KeepGameOpen) {
  Stop-Game   # backstop; normally --kill=true has already done this
}

if (-not (Test-Path $repFile)) {
  Write-Bad "No report at $repFile."
  if ($Trigger -eq 'hotkey' -and $pressedBegin) {
    # This is the NVTX failure signature exactly: armed, triggered, nothing
    # collected, "No reports were generated". If it repeats, the hotkey path is
    # as dead as the NVTX one and the gameplay-relative trigger is a dead end on
    # this machine - fall back rather than spend more runs on it.
    Write-Bad ("{0} was pressed but no report came out. That is the same signature the NVTX trigger had." -f $HotkeyCapture)
    Write-Bad  "  1. Re-run and press the key BY HAND at CAPTURE-BEGIN, to separate 'nsys ignores the hotkey' from 'the synthetic press did not reach the game'."
    Write-Bad  "  2. If a manual press does not work either, --capture-range is broken here for every trigger type. Use -Trigger delay -DelaySeconds 70 and accept the wall-clock window."
  } elseif ($Trigger -eq 'hotkey') {
    Write-Bad ("{0} was never pressed - the capture was never triggered. If the game did reach gameplay, the director is not printing markers: check rtx.nsysAutoCapture = True in rtx.conf." -f $HotkeyCapture)
  } else {
    Write-Bad "Read the nsys output above. One-shot --delay/--duration is the form that is PROVEN to work on this game, so a failure in this mode is an option or an environment problem, not the trigger - do not go back to --capture-range=nvtx, it was tested nine ways and never fired."
  }
  exit 6
}

$finalSize = (Get-Item $repFile).Length

# A stub report is the signature of a session that started and collected
# nothing: the failed interactive runs produced exactly 37,713 bytes. A real
# 10 s capture of this game is megabytes, so treat anything tiny as a failure
# rather than reporting success on an empty file.
if ($finalSize -lt 1MB) {
  Write-Bad ("Report is only {0:N0} bytes - that is an empty capture, not a short one. Collection never attached." -f $finalSize)
  exit 8
}

Write-Step ("Report written: {0} ({1:N1} MB)" -f $repFile, ($finalSize / 1MB))

# ------------------------------------------------------------------- the verdict
# A report is not the same thing as a USEFUL report, so state what the window
# landed on instead of leaving it to be discovered once the trace is open.
$verdictExit = 0
Write-Host ""

if ($Trigger -eq 'hotkey') {
  # Nothing to compute. The window was DEFINED by the game's own gameplay
  # markers, so if both presses went out it is gameplay by construction - the
  # whole reason for triggering this way rather than on a clock.
  if ($pressedBegin -and $pressedEnd) {
    Write-Step ("COVERAGE OK BY CONSTRUCTION. Capture ran from the game's CAPTURE-BEGIN to CAPTURE-END, i.e. {0}s of gameplay after the level started, for {1}s." -f $delayArg, $durationArg)
    if ($null -ne $armedAt -and $null -ne $beginAt) {
      Write-Step ("  gameplay began t+{0:N1}s, capture t+{1:N1}s..t+{2:N1}s (wall clock, for lining up against the trace)" -f $armedAt, $beginAt, $endAt)
    }
    # Texture streaming was still ramping ~20 s past load on this machine
    # (textures 497 -> 997, frame time 79.8 ms against 64.3 ms settled), so an
    # early window measures the ramp rather than steady play.
    if ($delayArg -lt 20) {
      Write-Host ("  NOTE: the settle is only {0}s of gameplay - texture streaming is still ramping this early and inflates frame time. Raise rtx.nsysAutoCaptureSettleSeconds to 45-60 for a settled-scene trace." -f $delayArg) -ForegroundColor Yellow
    }
  } elseif ($pressedBegin) {
    Write-Bad "PARTIAL. The capture was started on the game's marker but the closing press did not go out, so the trace runs to wherever the session was torn down rather than to CAPTURE-END."
  } else {
    Write-Bad ("COVERAGE UNKNOWN. {0} was never sent by the script, so whatever is in this report was not triggered by the gameplay markers." -f $HotkeyCapture)
  }
  Write-Host ""
  Write-Step "Done."
  exit $verdictExit
}

# -Trigger delay from here down. ARMED is the first frame the director accepted
# as gameplay, on the same t+ origin --delay counts from, so the coverage
# question is a straight comparison against the fixed window.
$winStart = $delayArg
$winEnd   = $delayArg + $durationArg

if ($null -eq $armedAt) {
  # Two very different reasons the marker is missing, and at a short delay the
  # SECOND one is by far the more likely - so do not blame the director for what
  # is really --kill=true firing before the level ever loaded.
  if ($winEnd -lt $kMeasuredLoadSec) {
    Write-Bad ("LOADING SCREEN, NOT GAMEPLAY. Collected t+{0}s..t+{1}s and nsys then killed the game - but load takes 56-63 s, so it never reached gameplay at all. [NsysAuto] ARMED was never printed because the game did not live long enough to print it." -f $winStart, $winEnd)
    Write-Bad ("Fix: -DelaySeconds 70  (or set rtx.nsysAutoCaptureSettleSeconds = 70 in rtx.conf)")
    $verdictExit = 9
  } else {
    Write-Bad ("COVERAGE UNKNOWN. The window was t+{0}s..t+{1}s, long enough to have reached gameplay, but the game never printed [NsysAuto] ARMED - so there is nothing to compare against. Check rtx.nsysAutoCapture = True in rtx.conf." -f $winStart, $winEnd)
  }
} elseif ($armedAt -ge $winEnd) {
  Write-Bad ("LOADING SCREEN, NOT GAMEPLAY. Collected t+{0}s..t+{1}s; gameplay did not start until t+{2:N1}s. The trace is level load." -f $winStart, $winEnd, $armedAt)
  Write-Bad ("Fix: -DelaySeconds {0}  (or set rtx.nsysAutoCaptureSettleSeconds = {0} in rtx.conf)" -f ([int][Math]::Ceiling($armedAt) + 10))
  $verdictExit = 9
} elseif ($armedAt -le $winStart) {
  Write-Step ("COVERAGE OK. Gameplay began t+{0:N1}s; the window is entirely inside it, {1:N1}s to {2:N1}s of gameplay." -f $armedAt, ($winStart - $armedAt), ($winEnd - $armedAt))
  # Texture streaming was still ramping ~20 s past load on this machine
  # (textures 497 -> 997, frame time 79.8 ms against 64.3 ms settled), so an
  # early window measures the ramp rather than steady play.
  if (($winStart - $armedAt) -lt 20) {
    Write-Host ("  NOTE: only {0:N1}s into gameplay - texture streaming is still ramping this early and inflates frame time. Add ~30 s to -DelaySeconds for a settled-scene trace." -f ($winStart - $armedAt)) -ForegroundColor Yellow
  }
} else {
  Write-Bad ("PARTIAL. Collected t+{0}s..t+{1}s but gameplay only began at t+{2:N1}s, so the first {3:N1}s of the trace is loading screen." -f $winStart, $winEnd, $armedAt, ($armedAt - $winStart))
}
Write-Host ""

# Exit 9 = the report is valid but it is not gameplay. Distinct from 6/8 (no
# report / empty report) so an unattended run can tell "profiler broke" from
# "you profiled the loading screen", which need completely different fixes.
Write-Step "Done."
exit $verdictExit
