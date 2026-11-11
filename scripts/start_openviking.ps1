# 启动本地 OpenViking 服务。
# 密钥不落盘：运行时从 Plus 后端的 config/config.json 读取 ark_api_key，
# 注入环境变量后再启动服务（ov.conf 里用 ${OPENVIKING_ARK_API_KEY} 占位）。
$ErrorActionPreference = "Stop"

$fridayCfg = "C:\Users\aodon\Desktop\Active_AI_Plus\backend\config\config.json"
if (Test-Path $fridayCfg) {
    $cfg = Get-Content $fridayCfg -Raw | ConvertFrom-Json
    $ark = $cfg.voice_chat_v2.ark_api_key
    if ($ark) {
        $env:OPENVIKING_ARK_API_KEY = $ark
        $env:OPENVIKING_VLM_API_KEY = $ark
    }
}

$scripts = "C:\Users\aodon\AppData\Roaming\Python\Python312\Scripts"
$env:Path = $scripts + ";" + $env:Path
$exe = Join-Path $scripts "openviking-server.exe"
$conf = "C:\Users\aodon\.openviking\ov.conf"

New-Item -ItemType Directory -Force -Path "D:\OpenViking\data" | Out-Null
New-Item -ItemType Directory -Force -Path "D:\OpenViking\logs" | Out-Null

& $exe --config $conf --host 127.0.0.1 --port 18095 *> "D:\OpenViking\logs\server.log"
