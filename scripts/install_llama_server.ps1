# PowerShell helper — guide tai llama-server.exe ve bin/llama-cpp/.
# Khong tu dong tai vi binary phai khop GPU/CPU runtime cua user.

$ErrorActionPreference = "Stop"

$root      = Split-Path -Parent $PSScriptRoot
$binDir    = Join-Path $root "bin\llama-cpp"
$target    = Join-Path $binDir "llama-server.exe"

Write-Host ""
Write-Host "================ EXACT 2026 — llama-server installer ================" -ForegroundColor Cyan
Write-Host ""

if (Test-Path $target) {
    Write-Host "[OK] Da co: $target" -ForegroundColor Green
    Write-Host "Bo qua buoc tai. Neu muon update, xoa file roi chay lai script."
    exit 0
}

Write-Host "[INFO] Chua tim thay llama-server.exe tai $target" -ForegroundColor Yellow
Write-Host ""
Write-Host "Vui long tai pre-built binary tu trang releases chinh thuc:"
Write-Host "  https://github.com/ggerganov/llama.cpp/releases/latest" -ForegroundColor Cyan
Write-Host ""
Write-Host "Chon dung asset cho he thong:"
Write-Host "  - CPU x64 (mac dinh):        llama-XXXX-bin-win-cpu-x64.zip"
Write-Host "  - GPU NVIDIA (CUDA 12):      llama-XXXX-bin-win-cuda-12.x-x64.zip"
Write-Host "  - GPU AMD (Vulkan):          llama-XXXX-bin-win-vulkan-x64.zip"
Write-Host ""
Write-Host "Sau khi tai:"
Write-Host "  1. Giai nen file zip."
Write-Host "  2. Copy llama-server.exe (cung tat ca .dll di kem) vao:"
Write-Host "       $binDir" -ForegroundColor Cyan
Write-Host "  3. Tao folder neu chua co (script da tao luon ben duoi)."
Write-Host ""

if (-not (Test-Path $binDir)) {
    New-Item -ItemType Directory -Path $binDir -Force | Out-Null
    Write-Host "[OK] Tao folder: $binDir" -ForegroundColor Green
}

Write-Host ""
Write-Host "Sau khi copy xong, kiem tra bang:"
Write-Host "  & '$target' --help" -ForegroundColor Cyan
Write-Host ""
Write-Host "=====================================================================" -ForegroundColor Cyan
