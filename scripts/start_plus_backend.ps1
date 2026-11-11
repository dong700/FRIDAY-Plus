# Start the FRIDAY Plus backend (Plus-owned FastAPI app under backend/).
# ASCII only on purpose: Windows PowerShell 5 reads BOM-less UTF-8 .ps1 as ANSI.
# Logging is handled inside run_server.py (writes D:\FRIDAY\logs\plus_backend.log),
# so no shell redirection here - redirecting a long-running python from .ps1
# intermittently fails with "Access to the path is denied" on this machine.

# Idempotent: start_pet.vbs and start_plus.vbs both look at port 18080.
# If it is already listening, exit instead of spawning a second backend that
# dies with [Errno 10048] and leaves half-initialized state behind.
$backendPort = 18080
if ($env:FRIDAY_PLUS_BACKEND_PORT) { $backendPort = [int]$env:FRIDAY_PLUS_BACKEND_PORT }
$listening = Get-NetTCPConnection -LocalPort $backendPort -State Listen -ErrorAction SilentlyContinue
if ($listening) { exit 0 }

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
Set-Location "C:\Users\aodon\Desktop\Active_AI_Plus\backend"
& "C:\ProgramData\miniconda3\python.exe" -u run_server.py
