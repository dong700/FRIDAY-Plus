/**
 * FRIDAY 控制台窗口（独立进程，不是浏览器标签页）
 *
 * 用法：
 *   electron.exe console.js --url=http://127.0.0.1:18090/
 *
 * 和桌宠窗口（main.js）分开：userData 目录不同，所以两者可以同时存在，
 * 各自有单实例锁；关掉控制台不会影响后端，重新启动会复用已有窗口。
 */
const { app, BrowserWindow, Menu } = require('electron');
const path = require('path');

const APP_TITLE = 'Active_AI_Plus';
const URL = (process.argv.find(a => a.startsWith('--url=')) || '').slice('--url='.length)
  || 'http://127.0.0.1:18090/';

// 单独的用户数据目录：避免和桌宠窗口抢单实例锁
app.setPath('userData', path.join(app.getPath('appData'), 'friday-console'));

let win = null;

if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (!win) return;
    try {
      if (win.isMinimized()) win.restore();
      win.show();
      win.focus();
    } catch { /* ignore */ }
  });

  app.whenReady().then(() => {
    Menu.setApplicationMenu(null);
    win = new BrowserWindow({
      width: 1320,
      height: 880,
      minWidth: 900,
      minHeight: 620,
      title: APP_TITLE,
      backgroundColor: '#e8f1fb',
      autoHideMenuBar: true,
      icon: path.join(__dirname, '..', 'assets', 'friday.ico'),
      webPreferences: {
        contextIsolation: true,
        nodeIntegration: false,
      },
    });
    // 页面自己的 <title> 不再覆盖窗口标题
    win.on('page-title-updated', (e) => e.preventDefault());
    win.loadURL(URL);
    win.webContents.on('did-finish-load', () => win.setTitle(APP_TITLE));
    win.on('closed', () => { win = null; });
    win.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
  });

  app.on('window-all-closed', () => app.quit());
  app.on('activate', () => {
    if (win) win.show();
  });
}
