# NV-DXVK: live shader-compile counter for build-remixDx11.bat.
#
# WHY THIS IS OUT-OF-BAND, and not a progress print inside compile_shaders.py:
# every shader variant is compiled by ONE ninja edge (compile_shaders.py -parallel
# fans out internally), and ninja BUFFERS a command's stdout until that command
# finishes. A counter printed from inside the script would therefore all arrive
# at once, after the last shader - which is precisely when it stops being useful.
# Watching the output directory and the live slangc processes is the only way to
# see progress while it is happening.
#
# WHY A SEPARATE .ps1 rather than an inline -Command in the .bat: the variant
# regex has to match "//!variant", and build-remixDx11.bat runs under
# EnableDelayedExpansion, where a bare "!" inside a quoted string is silently
# consumed. That class of quoting bug already cost this build script a
# self-perpetuating full-rebuild loop (see its own comments about the
# locale-string mtime compare). Keeping the script in a real file means the
# regex is just a regex.

param(
    [Parameter(Mandatory = $true)][string] $ShaderSrcRoot,
    [Parameter(Mandatory = $true)][string] $SpvDir,
    [Parameter(Mandatory = $true)][string] $StopFile,
    # Anything written at or after this instant counts as "rebuilt this run".
    # Passed in rather than taken as "now" here, so the .bat controls the epoch
    # and a slow monitor start cannot make freshly built shaders look stale.
    [Parameter(Mandatory = $true)][string] $StartTimeTicks
)

$ErrorActionPreference = 'SilentlyContinue'

$startTime = [datetime]::new([long]$StartTimeTicks)

# Expected variant count. compile_shaders.py emits one .spv per "//!variant"
# declaration, and exactly one for a .slang with no variant block at all (the
# parser falls back to a single implicit variant). Counting the declarations
# reproduces that rule, and it was checked against a complete build tree:
# 211 declarations, 211 .spv on disk.
#
# The "." in "//.variant" stands in for "!" so this file stays copy-pasteable
# into an inline batch command if it ever needs to be; it costs nothing here.
# Extension filtered explicitly rather than with -Include: -Include is silently
# IGNORED when the path is given as -LiteralPath, which made this count every
# file under shaders/ and floor each at 1 - reporting 482 units instead of 211.
$total = 0
$slangFiles = @(Get-ChildItem -Recurse -LiteralPath $ShaderSrcRoot -File |
                Where-Object { $_.Extension -eq '.slang' })
foreach ($f in $slangFiles) {
    $text = Get-Content -LiteralPath $f.FullName -Raw
    if ($null -eq $text) { continue }
    $n = ([regex]::Matches($text, '(?m)^\s*//.variant\b')).Count
    $total += [math]::Max($n, 1)
}
if ($total -le 0) { $total = 1 }

$peakRunning = 0
$lastLine = ''

while (-not (Test-Path -LiteralPath $StopFile)) {
    $spvFiles = @(Get-ChildItem -LiteralPath $SpvDir -Filter *.spv -File)
    $present  = $spvFiles.Count
    $rebuilt  = @($spvFiles | Where-Object { $_.LastWriteTime -ge $startTime }).Count
    # slangc is the actual per-variant compiler process; its live count is the
    # true "in progress" number, not an estimate derived from job slots.
    $running  = @(Get-Process slangc).Count
    if ($running -gt $peakRunning) { $peakRunning = $running }

    $line = '[shaders] compiled {0}/{1}  rebuilt-this-run {2}  in-progress {3}    ' -f `
        $present, $total, $rebuilt, $running

    # Only redraw on change: a carriage return every 400 ms with identical text
    # fights ninja for the cursor for no benefit.
    if ($line -ne $lastLine) {
        Write-Host -NoNewline ([char]13 + $line)
        $lastLine = $line
    }

    Start-Sleep -Milliseconds 400
}

# Final state, on its own line so the build log keeps it.
$spvFiles = @(Get-ChildItem -LiteralPath $SpvDir -Filter *.spv -File)
$present  = $spvFiles.Count
$rebuilt  = @($spvFiles | Where-Object { $_.LastWriteTime -ge $startTime }).Count
Write-Host ([char]13 + ('[shaders] final: {0}/{1} present, {2} rebuilt this run (peak {3} parallel)      ' -f `
    $present, $total, $rebuilt, $peakRunning))
