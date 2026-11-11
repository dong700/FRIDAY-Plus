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

' 1) console page 18090 (the vbs also starts backend if it is not listening)
sh.Run """wscript.exe"" """ & appRoot & "\start_plus.vbs""", 0, False

' 2) backend 18080
rc = sh.Run("cmd /c netstat -ano | find ""127.0.0.1:18080"" | find ""LISTENING"" >nul 2>&1", 0, True)
If rc <> 0 Then
  sh.Run psBackend, 0, False
  WScript.Sleep 8000
End If

' 3) pet window (Electron single-instance lock focuses an existing one)
cmd = """" & exe & """ """ & appDir & """ --url=" & url & " --port=18080 --py=""" & py & """ --app=""" & appRoot & """"
sh.Run cmd, 0, False
