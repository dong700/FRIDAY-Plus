import { useEffect, useCallback } from 'react';
import { useAppStore } from '../stores/appStore';
import type { BrainStatus, ChatMessage, WorkspaceFile, WorkspaceExecution } from '../types';

let _msgCounter = 0;
function wsUid() { return `${Date.now()}_${++_msgCounter}`; }

// 模块级单例WebSocket
let _ws: WebSocket | null = null;
let _reconnectTimer: ReturnType<typeof setTimeout> | null = null;
let _isConnecting = false;

// 暴露给外部组件获取 WS 实例（用于监听事件）
export function getWsInstance(): WebSocket | null { return _ws; }

/**
 * 订阅单例 WS 的消息（组件挂载时可能 WS 还没建好，这里会自动接上；
 * 断线重连会换一个新实例，也会自动换绑）。
 * 之前直接在 effect 里 getWsInstance() 拿一次，拿不到就再也不绑 → 桌宠气泡收不到
 * mind.delta，观感上就是"一口气全跳出来"。
 */
export function subscribeWs(handler: (data: any) => void): () => void {
  let bound: WebSocket | null = null;
  const onMessage = (ev: MessageEvent) => {
    try { handler(JSON.parse(ev.data)); } catch { /* ignore */ }
  };
  const attach = () => {
    const ws = _ws;
    if (!ws || ws === bound) return;
    if (bound) { try { bound.removeEventListener('message', onMessage); } catch { /* ignore */ } }
    bound = ws;
    ws.addEventListener('message', onMessage);
  };
  attach();
  const timer = setInterval(attach, 300);
  return () => {
    clearInterval(timer);
    if (bound) { try { bound.removeEventListener('message', onMessage); } catch { /* ignore */ } }
    bound = null;
  };
}

function _connect() {
  if (_ws?.readyState === WebSocket.OPEN || _ws?.readyState === WebSocket.CONNECTING || _isConnecting) {
    return;
  }

  _isConnecting = true;
  try {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    _ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

    _ws.onopen = () => {
      _isConnecting = false;
      useAppStore.getState().setConnected(true);
    };

    _ws.onclose = () => {
      _isConnecting = false;
      _ws = null;
      useAppStore.getState().setConnected(false);
      // 自动重连
      if (_reconnectTimer) clearTimeout(_reconnectTimer);
      _reconnectTimer = setTimeout(_connect, 3000);
    };

    _ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        const store = useAppStore.getState();
        switch (data.type) {
          case 'brain.status':
            store.setBrainStatus(data.payload as BrainStatus);
            break;
          case 'brain.mode_changed': {
            const current = store.brainStatus;
            if (current) {
              store.setBrainStatus({ ...current, mode: data.payload.new_mode });
            }
            break;
          }
          case 'chat.message':
            store.addMessage(data.payload as ChatMessage);
            break;
          case 'brain.notification':
            store.addMessage({
              id: wsUid(),
              role: 'brain' as const,
              content: data.payload.content,
              timestamp: new Date().toISOString()
            });
            break;
          case 'memory.search_result':
            store.setMemories(data.payload.memories);
            break;
          case 'brain.reasoning':
            store.addReasoningLog(data.payload);
            break;
        }
      } catch (e) {
        console.error('WS parse error:', e);
      }
    };

    _ws.onerror = () => {
      // onclose will handle reconnection
    };
  } catch (e) {
    _isConnecting = false;
    console.error('WS connect error:', e);
  }
}

function _disconnect() {
  if (_reconnectTimer) {
    clearTimeout(_reconnectTimer);
    _reconnectTimer = null;
  }
  if (_ws) {
    _ws.onclose = null; // 防止触发重连
    _ws.close();
    _ws = null;
  }
  _isConnecting = false;
  useAppStore.getState().setConnected(false);
}

export function useWebSocket() {
  const send = useCallback((type: string, payload: Record<string, unknown>) => {
    if (_ws?.readyState === WebSocket.OPEN) {
      _ws.send(JSON.stringify({ type, payload }));
    }
  }, []);

  useEffect(() => {
    _connect();

    // 用户活动检测 - 通知后端用户在线
    let activityTimer: ReturnType<typeof setTimeout> | null = null;
    const sendActivity = () => {
      if (_ws?.readyState === WebSocket.OPEN) {
        _ws.send(JSON.stringify({ type: "user.activity", payload: {} }));
      }
    };
    const onActivity = () => {
      if (activityTimer) clearTimeout(activityTimer);
      activityTimer = setTimeout(sendActivity, 2000); // 防抖2秒
    };
    window.addEventListener('mousemove', onActivity);
    window.addEventListener('keydown', onActivity);
    window.addEventListener('click', onActivity);

    return () => {
      window.removeEventListener('mousemove', onActivity);
      window.removeEventListener('keydown', onActivity);
      window.removeEventListener('click', onActivity);
      if (activityTimer) clearTimeout(activityTimer);
    };
  }, []);

  return { send };
}

// 页面关闭时断开
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', _disconnect);
}
