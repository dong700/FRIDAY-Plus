/**
 * preload.js — 只暴露桌宠需要的那几个能力，不给渲染层 node 权限。
 */
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('petAPI', {
  /** 鼠标是否在角色身上：true 时窗口接收点击，false 时穿透到桌面 */
  setHit: hit => ipcRenderer.send('pet:hit', hit),
  /** 输入条展开状态：展开时底部一条也要能点 */
  setBar: open => ipcRenderer.send('pet:bar', open),
  /** 虚拟形象隐藏：隐藏时窗口里只剩右下角那颗唤醒按钮可点 */
  setHidden: hidden => ipcRenderer.send('pet:hide', hidden),
  dragStart: () => ipcRenderer.send('pet:drag-start'),
  dragMove: () => ipcRenderer.send('pet:drag-move'),
  dragEnd: () => ipcRenderer.send('pet:drag-end'),
  /** 大小变：整窗 + 角色一起缩到 62%（再点一次还原） */
  toggleSize: () => ipcRenderer.send('pet:toggle-size'),
  /** 自己调大小：0.35~1.6，窗口跟着角色尺寸变（页面不 zoom，所以不会糊） */
  scale: v => ipcRenderer.send('pet:scale', v),
  /** Agent 驾驶：走到工作区的归一化坐标 (x, y)，0=最左/最上，1=最右/最下 */
  moveTo: (x, y, seconds) => ipcRenderer.send('pet:move-to', x, y, seconds),
  /** 关停后端（虚拟形象留着） */
  stopService: () => ipcRenderer.invoke('pet:service-stop'),
  /** 再拉起一个后端 */
  startService: () => ipcRenderer.invoke('pet:service-start'),
  quit: () => ipcRenderer.send('pet:quit'),
});
