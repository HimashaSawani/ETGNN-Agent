$ErrorActionPreference = "Stop"
Write-Host "=== Starting Live FastAPI Server on 127.0.0.1:8000 ==="

$backend = Start-Process -FilePath ".venv\Scripts\python.exe" -ArgumentList "-m","uvicorn","src.web.app:app","--host","127.0.0.1","--port","8000" -PassThru -WindowStyle Hidden -RedirectStandardOutput "scratch\live_backend_out.log" -RedirectStandardError "scratch\live_backend_err.log"

$ready = $false
for ($i = 0; $i -lt 30; $i++) {
  Start-Sleep -Seconds 2
  try {
    $res = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/health" -UseBasicParsing -TimeoutSec 3
    if ($res.StatusCode -eq 200) {
      $ready = $true
      Write-Host "FastAPI server ready in $(($i+1)*2)s with HTTP 200"
      break
    }
  } catch {}
}

if (-not $ready) {
  Write-Error "FastAPI server failed to become ready on port 8000 within 60s"
  taskkill /PID $backend.Id /T /F | Out-Null
  exit 1
}

Write-Host "=== Executing Complete Playwright E2E Suite (13 Mocked + 1 Real Backend) ==="
Set-Location web\frontend

try {
  $exitCode = 0
  & cmd.exe /c "npx playwright test --project=chromium"
  $exitCode = $LASTEXITCODE
} finally {
  Set-Location ..\..
  Write-Host "=== Stopping Live FastAPI Server ==="
  taskkill /PID $backend.Id /T /F | Out-Null
}

Write-Host "Playwright Complete Suite Exit Code: $exitCode"
exit $exitCode
