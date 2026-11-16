/**
 * FRIDAY 桌宠窗口（Electron 版）
 *
 * 为什么不用 pywebview：
 *   Windows 上 pywebview(WebView2) 既不支持 per-pixel 透明窗口，也不吃
 *   WS_EX_LAYERED + LWA_COLORKEY 色键（实测整窗变纯洋红）。GitHub 上做桌宠的
 *   主流方案就是 Electron：transparent:true + frame:false + alwaysOnTop，
 *   DWM 直接合成带 alpha 的窗口表面，模型边缘是真透明。
 *
 * 交互：
 *   - 默认 setIgnoreMouseEvents(true, {forward:true})，鼠标事件穿透到桌面
 *   - 渲染层在鼠标进入角色范围时通过 IPC 关掉穿透 → 角色可点、可拖
 *   - 角色范围外重新打开穿透，桌面图标照常能点
 */
const { app, BrowserWindow, ipcMain, screen, Menu, session } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const ARG_URL = (process.argv.find(a => a.startsWith('--url=')) || '').slice('--url='.length)
  || 'http://127.0.0.1:18080/?pet=1';
// 默认不再自己随机溜达：走位交给 Agent（pet_move 工具 → pet:move-to）。
// 调试探路时才加 --wander 打开旧的随机漫步。
const WANDER = process.argv.includes('--wander');
// 调试用：--shot=<路径> 在页面加载后延时截图（截图带 alpha，不受桌面合成干扰）
const SHOT_PATH = (process.argv.find(a => a.startsWith('--shot=')) || '').slice('--shot='.length);
const SHOT_DELAY = Number((process.argv.find(a => a.startsWith('--shot-delay=')) || '').slice('--shot-delay='.length)) || 6000;
// 调试用：--log=<路径> 把渲染进程的 console 落盘
const LOG_PATH = (process.argv.find(a => a.startsWith('--log=')) || '').slice('--log='.length);
// 调试用：--pos=center 把窗口放到屏幕中央（默认右下角）
const POS_CENTER = process.argv.includes('--pos=center');
// 调试用：--dump 延时后把渲染层的画布/舞台状态写进 log
const DUMP = process.argv.includes('--dump');
// 调试用：--opaque 用不透明窗口跑（区分「渲染不对」还是「透明合成不对」）
const OPAQUE = process.argv.includes('--opaque');

const WIN_W = 420;
const WIN_H = 560;
// 顶部留一条透明带给气泡：气泡从头顶冒出来，不挡脸（角色本体仍然占 420×560）
const BUBBLE_BAND = 220;
// 底部菜单占的高度（固定 100%，不跟人物缩放）——单独预留，别压在人物身上
const HUD_H = 86;
const SMALL_SCALE = 0.62;                       // 大小变：整窗 + 角色一起缩到 62%
let small = process.argv.includes('--small');
// 用户自己调的大小（滚轮/菜单），0.35~1.6
let userScale = 1.0;
// 虚拟形象是否隐藏（隐藏时窗口里只剩一个 48px 的唤醒按钮）
let hiddenAvatar = false;
// 供「启动服务」用：由 run_friday.py 传进来
const PORT_ARG = Number((process.argv.find(a => a.startsWith('--port=')) || '').slice('--port='.length)) || 18080;
const PY_EXE = (process.argv.find(a => a.startsWith('--py=')) || '').slice('--py='.length) || 'python';
const APP_DIR = (process.argv.find(a => a.startsWith('--app=')) || '').slice('--app='.length) || process.cwd();
let backendProc = null;

// Windows 上透明窗口 + WebGL 画布偶发「整窗变白」：
// 打开透明视觉合成，并让窗口晚一点创建，等 DWM 的透明表面就绪。
app.commandLine.appendSwitch('enable-transparent-visuals');

let win = null;

// 桌宠只允许一个窗口：桌面上再点一次快捷方式时，把已有的窗口唤到前面，
// 而不是又开一个一模一样的（旧启动器用 wmic 判断，新版 Windows 已经没有 wmic 了）。
if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (win) {
      try { win.show(); win.focus(); } catch { /* ignore */ }
    }
  });
}
let ignoring = true;
let dragState = null;   // { winX, winY, sx, sy }
let dragTimer = null;
let hitTimer = null;
let barOpen = false;    // 输入条是否展开（展开时底部一条也要能点）
let wanderTimer = null;

function winSize() {
  const s = (small ? SMALL_SCALE : 1) * userScale;
  // 宽度至少留 420：底部菜单是固定 100% 尺寸，不跟着人物缩
  return [
    Math.max(WIN_W, Math.round(WIN_W * s)),
    BUBBLE_BAND + Math.round(WIN_H * s) + HUD_H,
  ];
}

/** 角色本体在窗口里的位置（窗口比角色宽时，角色居中） */
function charBox(w, h) {
  const s = (small ? SMALL_SCALE : 1) * userScale;
  const mw = WIN_W * s;
  const left = Math.round((w - mw) / 2);
  return {
    left: left + mw * 0.18,
    right: left + mw * 0.82,
    top: BUBBLE_BAND,
    bottom: BUBBLE_BAND + Math.round(WIN_H * s),
  };
}

/**
 * Windows 11 会给每个窗口描一圈 2px 的系统边框（就是用户说的「白边」）。
 * Electron 没有对应 API，这里用一次 PowerShell + P/Invoke 调 DwmSetWindowAttribute：
 *   34 = DWMWA_BORDER_COLOR，-2（0xFFFFFFFE）= DWMWA_COLOR_NONE，去掉边框
 *   33 = DWMWA_WINDOW_CORNER_PREFERENCE，1 = DONOTROUND，去掉圆角
 * 只做一次，窗口隐藏启动，不会被看到。
 */
function applyDwmTweaks() {
  if (OPAQUE) return;
  try {
    const buf = win.getNativeWindowHandle();
    const hwnd = buf.length >= 8 ? buf.readBigUInt64LE(0) : BigInt(buf.readUInt32LE(0));
    const script = `
$sig = @'
using System;
using System.Runtime.InteropServices;
public class DwmFw {
  [DllImport("dwmapi.dll")] public static extern int DwmSetWindowAttribute(IntPtr hwnd, int attr, ref int val, int size);
}
'@
Add-Type -TypeDefinition $sig | Out-Null
$h = [IntPtr]${hwnd}
$none = -2
[DwmFw]::DwmSetWindowAttribute($h, 34, [ref]$none, 4) | Out-Null
$noRound = 1
[DwmFw]::DwmSetWindowAttribute($h, 33, [ref]$noRound, 4) | Out-Null
`;
    spawn('powershell', ['-NoProfile', '-NonInteractive', '-Command', script],
      { windowsHide: true, detached: true, stdio: 'ignore' }).unref();
  } catch (e) {
    console.log('[pet] DWM 边框处理失败:', e && e.message);
  }
}

function createWindow() {
  const area = screen.getPrimaryDisplay().workArea;
  const [W, H] = winSize();
  const wx = POS_CENTER
    ? Math.round(area.x + (area.width - W) / 2)
    : Math.round(area.x + area.width - W - 60);
  const wy = POS_CENTER
    ? Math.round(area.y + (area.height - H) / 2)
    : Math.round(area.y + area.height - H - 10);

  win = new BrowserWindow({
    width: W,
    height: H,
    x: wx,
    y: wy,
    frame: false,
    transparent: !OPAQUE,
    backgroundColor: OPAQUE ? '#ffffff' : '#00000000',
    hasShadow: false,
    // Windows 会给每个窗口描一圈 1px 边框、Win11 还会加圆角阴影；
    // 透明窗上这圈边框就是用户看到的「白边」。关掉 thickFrame 和圆角一并去掉。
    thickFrame: false,
    roundedCorners: false,
    resizable: false,
    maximizable: false,
    fullscreenable: false,
    skipTaskbar: true,
    alwaysOnTop: true,
    show: false,
    // 任务栏/Alt-Tab 里用角色头像，而不是 Electron 的原子图标
    icon: path.join(__dirname, 'friday.ico'),
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      backgroundThrottling: false,
    },
  });

  // 压在普通窗口之上，但不要盖住任务栏
  win.setAlwaysOnTop(true, 'screen-saver');
  win.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });
  win.setIgnoreMouseEvents(true, { forward: true });

  win.once('ready-to-show', () => {
    win.show();
    // 某些 GPU 合成路径下 transparent 会退化成白底，show 之后再钉一次
    if (!OPAQUE) win.setBackgroundColor('#00000000');
    applyDwmTweaks();
  });
  win.loadURL(ARG_URL);
  win.webContents.once('did-finish-load', () => {
    // 不再改页面 zoom：窗口尺寸就是角色尺寸，画布按真实像素绘制，
    // 放大时画布跟着变大 → 清晰；用 zoom 是把旧位图拉伸 → 糊。
    if (win && !win.isDestroyed()) {
      try { win.webContents.setZoomFactor(1); } catch { /* ignore */ }
    }
  });

  if (LOG_PATH) {
    win.webContents.on('console-message', (_e, level, message) => {
      try { fs.appendFileSync(LOG_PATH, `${new Date().toISOString()} [${level}] ${message}\n`); } catch { /* ignore */ }
    });
    win.webContents.on('render-process-gone', (_e, detail) => {
      try { fs.appendFileSync(LOG_PATH, `render-process-gone ${JSON.stringify(detail)}\n`); } catch { /* ignore */ }
    });
  }

  if (DUMP) {
    win.webContents.once('did-finish-load', () => {
      setTimeout(async () => {
        const out = (line) => {
          try { LOG_PATH ? fs.appendFileSync(LOG_PATH, line + '\n') : console.log(line); } catch { /* ignore */ }
        };
        try {
          const r = await win.webContents.executeJavaScript(`(() => {
            const canvases = [...document.querySelectorAll('canvas')].map(c => {
              const b = c.getBoundingClientRect();
              return { attr: [c.width, c.height], css: [c.style.width, c.style.height],
                       client: [c.clientWidth, c.clientHeight], rect: [b.x, b.y, b.width, b.height] };
            });
            const chain = [];
            let el = document.querySelector('canvas');
            while (el && chain.length < 12) {
              const s = getComputedStyle(el);
              chain.push({ tag: el.tagName, id: el.id, cls: (el.className || '').toString().slice(0, 40),
                           bg: s.backgroundColor, bgImg: s.backgroundImage, op: s.opacity,
                           filter: s.filter, backdrop: s.backdropFilter });
              el = el.parentElement;
            }
            const rootStyle = getComputedStyle(document.getElementById('root')).backgroundColor;
            return {
              dpr: window.devicePixelRatio,
              inner: [window.innerWidth, window.innerHeight],
              bodyBg: getComputedStyle(document.body).backgroundColor,
              htmlBg: getComputedStyle(document.documentElement).backgroundColor,
              canvases,
              chain,
              rootStyle,
              petDebug: window.__petDebug || null,
            };
          })()`);
          out('[pet] dump ' + JSON.stringify(r, null, 0));
        } catch (e) {
          out('[pet] dump failed ' + String(e));
        }
      }, SHOT_DELAY);
    });
  }

  if (SHOT_PATH) {
    win.webContents.once('did-finish-load', () => {
      setTimeout(async () => {
        try {
          const img = await win.webContents.capturePage();
          fs.writeFileSync(SHOT_PATH, img.toPNG());
          console.log('[pet] shot saved', SHOT_PATH);
        } catch (e) {
          console.log('[pet] shot failed', e);
        }
      }, SHOT_DELAY);
    });
  }

  win.webContents.on('did-fail-load', (_e, code, desc) => {
    console.log(`[pet] 页面加载失败 ${code} ${desc}，3 秒后重试`);
    setTimeout(() => win && win.loadURL(ARG_URL), 3000);
  });

  if (WANDER) startWander();
}

/** 在桌面底部缓慢溜达（不是瞬移，是走过去） */
function startWander() {
  let x = null;
  let targetX = null;
  const step = 3;

  wanderTimer = setInterval(() => {
    if (!win || win.isDestroyed()) return;
    const area = screen.getPrimaryDisplay().workArea;
    const [W, H] = winSize();
    const y = Math.round(area.y + area.height - H - 10);

    if (targetX === null) {
      targetX = Math.round(area.x + Math.random() * Math.max(0, area.width - W));
      if (x === null) x = win.getPosition()[0];
    }

    x = x === null ? win.getPosition()[0] : x;
    if (Math.abs(targetX - x) <= step) {
      x = targetX;
      targetX = null;
      // 到了之后歇一会儿再选下一个目标
      const pause = 4000 + Math.random() * 8000;
      setTimeout(() => { targetX = null; }, pause);
    } else {
      x += Math.sign(targetX - x) * step;
    }
    win.setPosition(Math.round(x), y);
  }, 40);
}

// ── Agent 驾驶：走到工作区的归一化坐标 ──
let moveTimer = null;

function stopMove() {
  if (moveTimer) { clearInterval(moveTimer); moveTimer = null; }
}

/**
 * @param nx 0=工作区最左，1=最右
 * @param ny 0=最上，1=最下（贴任务栏）
 * @param seconds 走过去用多少秒；<=0 时按距离自动
 */
function moveTo(nx, ny, seconds) {
  if (!win || win.isDestroyed()) return;
  if (!Number.isFinite(nx) || !Number.isFinite(ny)) return;

  const area = screen.getPrimaryDisplay().workArea;
  const [W, H] = winSize();
  const left = area.x + Math.max(0, Math.min(1, nx)) * Math.max(0, area.width - W);
  const top = area.y + Math.max(0, Math.min(1, ny)) * Math.max(0, area.height - H);
  const tx = Math.round(left);
  const ty = Math.round(top);

  stopMove();
  const [sx, sy] = win.getPosition();
  const dist = Math.hypot(tx - sx, ty - sy);
  const dur = seconds > 0 ? seconds : Math.min(8, Math.max(0.35, dist / 260));
  const t0 = Date.now();

  moveTimer = setInterval(() => {
    if (!win || win.isDestroyed()) { stopMove(); return; }
    const k = Math.min(1, (Date.now() - t0) / (dur * 1000));
    // easeInOutQuad：起步/落脚都慢，看起来像走过去而不是瞬移
    const e = k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
    win.setPosition(Math.round(sx + (tx - sx) * e), Math.round(sy + (ty - sy) * e));
    if (k >= 1) stopMove();
  }, 16);
}

/** 反复 ping 后端，直到它应答或超时 */
async function waitBackend(timeoutMs) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    try {
      const r = await fetch(`http://127.0.0.1:${PORT_ARG}/api/brain/status`, {
        signal: AbortSignal.timeout(1500),
      });
      if (r.ok) return true;
    } catch { /* 还没起来 */ }
    await new Promise(res => setTimeout(res, 400));
  }
  return false;
}

/** 等端口真正空出来（旧服务收尾时还会短暂应答），避免新进程 bind 失败 */
async function waitPortFree(timeoutMs) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    try {
      await fetch(`http://127.0.0.1:${PORT_ARG}/api/brain/status`, {
        signal: AbortSignal.timeout(800),
      });
    } catch {
      return true;
    }
    await new Promise(res => setTimeout(res, 400));
  }
  return false;
}

function setIgnore(flag) {
  if (!win || win.isDestroyed() || flag === ignoring) return;
  ignoring = flag;
  win.setIgnoreMouseEvents(flag, { forward: true });
}

/** 大小变：窗口跟着角色尺寸走，页面本身不 zoom（zoom 会把画布拉大 → 糊） */
function applySize() {
  if (!win || win.isDestroyed()) return;
  const area = screen.getPrimaryDisplay().workArea;
  const [W, H] = winSize();
  const [x, y] = win.getPosition();
  win.setBounds({
    x: Math.max(area.x, Math.min(x, area.x + area.width - W)),
    y: Math.max(area.y, Math.min(y + H, area.y + area.height - H)),
    width: W,
    height: H,
  });
  // 缩小时再顶一下，免得被别的置顶窗口盖住
  if (small) win.setAlwaysOnTop(true, 'screen-saver');
}

app.whenReady().then(async () => {
  // 开发期改贴图后要能立刻看到效果：先清缓存再创建窗口
  try { await session.defaultSession.clearCache(); } catch { /* ignore */ }
  await new Promise(r => setTimeout(r, 120));
  createWindow();

  ipcMain.on('pet:hit', (_e, hit) => setIgnore(!hit));
  ipcMain.on('pet:bar', (_e, v) => { barOpen = !!v; });
  ipcMain.on('pet:hide', (_e, v) => { hiddenAvatar = !!v; });

  // 命中区判定放在主进程：渲染层的 mousemove 在穿透窗口上并不可靠（实测经常收不到，
  // 于是「点不开输入框 / 拖不动」）。这里按系统光标位置每 80ms 重新决定穿透开关。
  if (hitTimer) clearInterval(hitTimer);
  hitTimer = setInterval(() => {
    if (!win || win.isDestroyed()) return;
    if (dragState) { setIgnore(false); return; }   // 拖拽中保持接收
    const p = screen.getCursorScreenPoint();
    const [wx, wy] = win.getPosition();
    const [W, H] = winSize();
    if (hiddenAvatar) {
      // 隐藏时只留右下角那颗 48px 的唤醒按钮可点
      const inBtn = p.x >= wx + W - 64 && p.x <= wx + W - 16 && p.y >= wy + H - 64 && p.y <= wy + H - 16;
      setIgnore(!inBtn);
      return;
    }
    const box = charBox(W, H);
    const inChar = p.x >= wx + box.left && p.x <= wx + box.right
      && p.y >= wy + box.top && p.y <= wy + box.bottom;
    const inBar = barOpen && p.y >= wy + box.bottom && p.x >= wx && p.x <= wx + W;
    setIgnore(!(inChar || inBar));
  }, 80);

  ipcMain.on('pet:drag-start', () => {
    if (!win) return;
    stopMove();   // 手动拖拽优先，停掉正在进行的自动走位
    const [wx, wy] = win.getPosition();
    const p = screen.getCursorScreenPoint();
    dragState = { winX: wx, winY: wy, sx: p.x, sy: p.y };
    // 不靠渲染层的 mousemove：拖拽时页面经常收不到 mid-drag 的 move 事件（实测窗口纹丝不动）。
    // 主进程直接按系统光标轮询，窗口跟手；mouseup 由渲染层发 drag-end 收尾。
    if (dragTimer) clearInterval(dragTimer);
    dragTimer = setInterval(() => {
      if (!win || win.isDestroyed() || !dragState) { clearInterval(dragTimer); dragTimer = null; return; }
      const q = screen.getCursorScreenPoint();
      win.setPosition(
        Math.round(dragState.winX + (q.x - dragState.sx)),
        Math.round(dragState.winY + (q.y - dragState.sy)),
      );
    }, 16);
  });

  ipcMain.on('pet:drag-move', () => {
    if (!win || !dragState) return;
    const p = screen.getCursorScreenPoint();
    win.setPosition(
      Math.round(dragState.winX + (p.x - dragState.sx)),
      Math.round(dragState.winY + (p.y - dragState.sy)),
    );
  });

  ipcMain.on('pet:drag-end', () => {
    dragState = null;
    if (dragTimer) { clearInterval(dragTimer); dragTimer = null; }
  });

  ipcMain.on('pet:toggle-size', () => {
    small = !small;
    applySize();
  });

  // 调大小：窗口尺寸跟着角色变（页面不 zoom，否则画布被拉大就糊）。
  // 以「底边中心」为锚点，人物站在原地变大变小，不会一边变一边跑出屏幕。
  ipcMain.on('pet:scale', (_e, v) => {
    if (!win || win.isDestroyed()) return;
    const next = Math.min(1.6, Math.max(0.35, Number(v) || 1));
    if (Math.abs(next - userScale) < 0.001) return;
    const [ox, oy] = win.getPosition();
    const [oW, oH] = winSize();
    const cx = ox + oW / 2;
    const bottom = oy + oH;
    userScale = next;
    const [W, H] = winSize();
    const area = screen.getPrimaryDisplay().workArea;
    win.setBounds({
      x: Math.round(Math.max(area.x, Math.min(cx - W / 2, area.x + area.width - W))),
      y: Math.round(Math.max(area.y, Math.min(bottom - H, area.y + area.height - H))),
      width: W,
      height: H,
    });
    win.setAlwaysOnTop(true, 'screen-saver');
  });

  // 「关停服务」= 只退出后端进程，桌宠窗口留着；「启动服务」再拉一个后端回来。
  ipcMain.handle('pet:service-stop', async () => {
    try {
      const r = await fetch(`http://127.0.0.1:${PORT_ARG}/api/service/stop`, { method: 'POST' });
      return await r.json();
    } catch (e) {
      return { ok: false, error: String(e) };
    }
  });

  ipcMain.handle('pet:service-start', async () => {
    if (backendProc) return { ok: true, already: true };
    try {
      // 后端就在 Plus 自己的 backend/ 下，不再依赖任何外部项目目录
      const backendDir = path.join(__dirname, '..', 'backend');
      const script = path.join(backendDir, 'run_server.py');
      if (!fs.existsSync(script)) {
        return { ok: false, error: `找不到 ${script}` };
      }
      // 上一份服务可能还在收尾（端口没释放），先等它退干净，否则新进程 bind 失败
      await waitPortFree(5000);
      const logDir = 'D:\\FRIDAY\\logs';
      let out = null;
      try {
        fs.mkdirSync(logDir, { recursive: true });
        out = fs.openSync(path.join(logDir, 'backend_service.log'), 'a');
      } catch { out = null; }
      backendProc = spawn(PY_EXE,
        [script],
        {
          cwd: backendDir,
          stdio: out === null ? 'ignore' : ['ignore', out, out],
          windowsHide: true,
          detached: false,
          env: {
            ...process.env,
            PYTHONIOENCODING: 'utf-8',
            PYTHONUTF8: '1',
            FRIDAY_PLUS_BACKEND_PORT: String(PORT_ARG),
          },
        });
      backendProc.on('exit', () => { backendProc = null; });
      // 等后端真的起来再报成功：uvicorn 是先跑完 startup（恢复 TFGR/RAG 记忆）才 bind 端口，
      // 记忆多的时候要 30 秒以上，20 秒会把正常启动误判成失败、连带把进程杀掉。
      const ok = await waitBackend(60000);
      if (!ok) {
        try { backendProc && backendProc.kill(); } catch { /* ignore */ }
        backendProc = null;
        return { ok: false, error: '后端 60 秒内没起来（看 D:/FRIDAY/logs/plus_backend.log）' };
      }
      return { ok: true };
    } catch (e) {
      backendProc = null;
      return { ok: false, error: String(e) };
    }
  });

  ipcMain.on('pet:move-to', (_e, x, y, seconds) => {
    // 大小变只认当前窗口尺寸，移动之后再走一遍 applySize 保证不出屏
    moveTo(Number(x), Number(y), Number(seconds) || 0);
  });

  ipcMain.on('pet:quit', () => app.quit());

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  // 自己拉起来的后端要跟着收掉，别留孤儿进程
  try { if (backendProc) backendProc.kill(); } catch { /* ignore */ }
  app.quit();
});
