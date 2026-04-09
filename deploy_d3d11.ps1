# Build and deploy d3d11 bridge artifacts to build/
# Usage: .\deploy_d3d11.ps1

$vcvars = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
$buildDir = "_build_d3d11_release"
$dstDir   = "build\DX11 DXVK Remix WIP"

Write-Host "Building..." -ForegroundColor Cyan
cmd /c "call ""$vcvars"" && cd /d ""$PSScriptRoot"" && ninja -C $buildDir 2>&1" |
    Select-Object -Last 15

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build FAILED" -ForegroundColor Red
    exit 1
}

Write-Host "Deploying artifacts..." -ForegroundColor Cyan

$targets = @(
    @{ sub = "d3d11";  names = @("d3d11");                          dst = $dstDir },
    @{ sub = "dxgi";   names = @("dxgi");                           dst = $dstDir },
    @{ sub = "d3d10";  names = @("d3d10", "d3d10core", "d3d10_1"); dst = "$dstDir\optional" }
)

foreach ($target in $targets) {
    foreach ($name in $target.names) {
        foreach ($ext in "dll", "lib", "exp", "pdb") {
            $src = "$PSScriptRoot\$buildDir\src\$($target.sub)\$name.$ext"
            $dst = "$PSScriptRoot\$($target.dst)\$name.$ext"
            if (Test-Path $src) {
                Copy-Item $src $dst -Force
                $item = Get-Item $dst
                Write-Host ("  $name.$ext  {0:yyyy-MM-dd HH:mm:ss}  {1} bytes" -f $item.LastWriteTime, $item.Length)
            } else {
                Write-Host "  $name.$ext  NOT FOUND (skipped)" -ForegroundColor Yellow
            }
        }
    }
}

Write-Host "Done." -ForegroundColor Green
