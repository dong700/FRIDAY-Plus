' FRIDAY Plus pet launcher (ASCII only - UTF-8 Chinese breaks VBScript)
' Opens the desktop pet window (Electron, Plus-owned) against the Plus backend,
' and makes sure the Plus console page is up as well.
Dim sh, rc, exe, appDir, appRoot, py, url, cmd, psBackend

Set sh = CreateObject("WScript.Shell")
appRoot = "C:\Users\aodon\Desktop\Active_AI_Plus"
appDir = appRoot & "\electron"
sh.CurrentDirectory = appRoot

exe = appDir & "\node_modules\electron\dist\electron.exe"
py = "C:\ProgramData\miniconda3\python.exe"
url = "http://127.0.0.1:18080/?pet=1&model=deepseek"
psBackend = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & appRoot & "\scripts\start_plus_backend.ps1"""

' 1) backend 18080 FIRST - start_plus.vbs also checks this port, so bring it up
'    before calling it; otherwise both scripts race and spawn two backends.
'    Wait until it is actually listening (fixed 8s is not enough on cold start:
'    RAG restore alone takes ~10s), so the pet page never opens too early.
rc = sh.Run("cmd /c netstat -ano | find ""127.0.0.1:18080"" | find ""LISTENING"" >nul 2>&1", 0, True)
If rc <> 0 Then
  sh.Run psBackend, 0, False
  WaitPort 18080, 45000
End If

' 2) console page 18090 (start_plus sees 18080 already listening and skips it)
sh.Run """wscript.exe"" """ & appRoot & "\start_plus.vbs""", 0, False
WScript.Sleep 3000

' 3) pet window (Electron single-instance lock focuses an existing one)
cmd = """" & exe & """ """ & appDir & """ --url=" & url & " --port=18080 --py=""" & py & """ --app=""" & appRoot & """"
sh.Run cmd, 0, False

' Poll a local port until LISTENING or timeout (ms). Keeps cold start reliable.
Sub WaitPort(port, timeoutMs)
  Dim t0, rc2
  t0 = Timer
  Do
    rc2 = sh.Run("cmd /c netstat -ano | find ""127.0.0.1:" & port & """ | find ""LISTENING"" >nul 2>&1", 0, True)
    If rc2 = 0 Then Exit Sub
    WScript.Sleep 1000
  Loop While (Timer - t0) * 1000 < timeoutMs
End Sub
