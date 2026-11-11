' FRIDAY Plus launcher (ASCII only - UTF-8 Chinese breaks VBScript)
' Starts the Plus-owned backend (18080) and the Plus page/proxy server (18090),
' then opens the console as its own Electron window (bundled in Plus/electron).
Dim sh, rc, exe, appFile, url, fso, psBackend
Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
sh.CurrentDirectory = "C:\Users\aodon\Desktop\Active_AI_Plus"

exe = "C:\Users\aodon\Desktop\Active_AI_Plus\electron\node_modules\electron\dist\electron.exe"
appFile = "C:\Users\aodon\Desktop\Active_AI_Plus\electron\console.js"
url = "http://127.0.0.1:18090/"
psBackend = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File ""C:\Users\aodon\Desktop\Active_AI_Plus\scripts\start_plus_backend.ps1"""

' 0) OpenViking semantic store (18095). It takes ~90s to load indexes, so start
'    it detached and NEVER wait: the pet/console must come up immediately.
'    Backend degrades gracefully (semantic recall returns empty) until it is up.
rc = sh.Run("cmd /c netstat -ano | find ""127.0.0.1:18095"" | find ""LISTENING"" >nul 2>&1", 0, True)
If rc <> 0 Then
  sh.Run "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File ""C:\Users\aodon\Desktop\Active_AI_Plus\scripts\start_openviking.ps1""", 0, False
End If

' 1) backend 18080
rc = sh.Run("cmd /c netstat -ano | find ""127.0.0.1:18080"" | find ""LISTENING"" >nul 2>&1", 0, True)
If rc <> 0 Then
  sh.Run psBackend, 0, False
  WScript.Sleep 6000
End If

' 2) plus page server 18090
rc = sh.Run("cmd /c netstat -ano | find ""127.0.0.1:18090"" | find ""LISTENING"" >nul 2>&1", 0, True)
If rc <> 0 Then
  sh.Run """C:\ProgramData\miniconda3\python.exe"" -u server.py", 0, False
  WScript.Sleep 4000
End If

If fso.FileExists(exe) Then
  sh.Run """" & exe & """ """ & appFile & """ --url=" & url, 1, False
Else
  sh.Run url, 1, False
End If
