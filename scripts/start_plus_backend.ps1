# Start the FRIDAY Plus backend (Plus-owned FastAPI app under backend/).
# ASCII only on purpose: Windows PowerShell 5 reads BOM-less UTF-8 .ps1 as ANSI.
# Logging is handled inside run_server.py (writes D:\FRIDAY\logs\plus_backend.log),
# so no shell redirection here - redirecting a long-running python from .ps1
# intermittently fails with "Access to the path is denied" on this machine.
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
Set-Location "C:\Users\aodon\Desktop\Active_AI_Plus\backend"
& "C:\ProgramData\miniconda3\python.exe" -u run_server.py
