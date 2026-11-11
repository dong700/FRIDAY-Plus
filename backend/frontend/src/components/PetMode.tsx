/**
 * PetMode — 桌宠模式（2026-11-03）
 *
 * 把大肥鱼从完整界面里拿出来，只留角色，做成桌面上的一个小家伙：
 *   - 角色只露上半身，窗口大小 = 角色大小（不做页面缩放，放大时画布真的变大，所以清晰）
 *   - 顶部留 220px 透明带：气泡从头顶上方冒，不挡脸；留够高度，长回答不会被裁掉
 *   - 点角色 → 弹出菜单：说话 / 调大小 / 开关视觉 / 关停启动服务 / 隐藏形象
 *   - 底部菜单固定 100%，不跟着人物缩放
 */
import { useEffect, useRef, useState } from 'react';
import type { CSSProperties, WheelEvent } from 'react';
import { AvatarWidget } from './AvatarWidget';
import { useWebSocket, subscribeWs } from '../hooks/useWebSocket';

/** Electron preload 注入的桌宠桥（浏览器里不存在，用可选调用兜底） */
type PetAPI = {
  setHit: (hit: boolean) => void;
  setBar: (open: boolean) => void;
  setHidden: (hidden: boolean) => void;
  dragStart: () => void;
  dragMove: () => void;
  dragEnd: () => void;
  toggleSize: () => void;
  /** 0.35~1.6：窗口跟着角色缩放 */
  scale: (v: number) => void;
  moveTo: (x: number, y: number, seconds?: number) => void;
  stopService: () => Promise<{ ok: boolean; error?: string }>;
  startService: () => Promise<{ ok: boolean; already?: boolean; error?: string }>;
  quit: () => void;
};
const petAPI = (): PetAPI | undefined => (window as any).petAPI;

// 和 electron/main.js 里的常量必须一致
const BASE_W = 420;      // 角色画布基准宽
const BASE_H = 560;      // 角色画布基准高
const TOP_BAND = 220;    // 顶部气泡带（固定，不跟着人物缩）—— 长回答也放得下
const HUD_W = 420;       // 底部菜单宽度（固定 100%）
const HUD_H = 86;        // 底部菜单预留高度（固定 100%，不压在人物身上）

const btnStyle = (bg: string): CSSProperties => ({
  background: bg,
  color: '#fff',
  border: 'none',
  borderRadius: 8,
  padding: '4px 10px',
  fontSize: 12,
  cursor: 'pointer',
});

export function PetMode() {
  const { send } = useWebSocket();
  const [open, setOpen] = useState(false);
  const [text, setText] = useState('');
  const inputRef = useRef<HTMLInputElement | null>(null);
  const draggingRef = useRef(false);
  const movedRef = useRef(false);
  const lastClickRef = useRef(0);
  const [visionOn, setVisionOn] = useState(true);
  const [serviceOn, setServiceOn] = useState(true);
  const [busy, setBusy] = useState(false);
  const [scalePct, setScalePct] = useState(100);
  const [hiddenAvatar, setHiddenAvatar] = useState(false);

  // ?pet=1&model=deepseek|deepseek_q
  // 双击角色 = 在 默认 → 体型1 → 体型2 之间循环（模型自带的变形，不是窗口缩放）
  const modelKey = new URLSearchParams(window.location.search).get('model') || 'deepseek';
  const modelUrl = undefined;
  const initialBody = (() => {
    const raw = new URLSearchParams(window.location.search).get('body');
    const b = raw === null ? -1 : Number(raw);
    if (b === 0 || b === 1 || b === 2) return b;
    return modelKey === 'deepseek_q' ? 2 : 0;
  })();
  const [body, setBody] = useState(initialBody);

  // 角色尺寸跟着 scalePct 走；窗口尺寸由 main.js 用同一套公式算
  const s = scalePct / 100;
  const modelW = Math.round(BASE_W * s);
  const modelH = Math.round(BASE_H * s);
  const winW = Math.max(HUD_W, modelW);
  const left = Math.round((winW - modelW) / 2);
  const charBottom = TOP_BAND + modelH;

  // 让整个页面透明（窗口由 Electron 以 transparent:true 承载）
  useEffect(() => {
    const prevBody = document.body.style.background;
    const prevHtml = document.documentElement.style.background;
    document.body.style.background = 'transparent';
    document.documentElement.style.background = 'transparent';
    document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.background = prevBody;
      document.documentElement.style.background = prevHtml;
    };
  }, []);

  useEffect(() => {
    petAPI()?.setHidden?.(hiddenAvatar);
    if (hiddenAvatar) setOpen(false);
  }, [hiddenAvatar]);

  // 后端在不在：5 秒一问，别让按钮停在错误的状态上（比如后端被外部杀掉）
  useEffect(() => {
    let alive = true;
    const check = async () => {
      try {
        const r = await fetch('/api/brain/status', { signal: AbortSignal.timeout(2000) });
        if (alive) setServiceOn(r.ok);
      } catch {
        if (alive) setServiceOn(false);
      }
    };
    check();
    const t = setInterval(check, 5000);
    return () => { alive = false; clearInterval(t); };
  }, []);

  // Agent 驾驶：后端调 pet_move 工具 → WS 发 pet.move → 传给 Electron 走过去
  useEffect(() => {
    const off = subscribeWs((data) => {
      if (data.type !== 'pet.move') return;
      const { x, y, seconds } = data.payload || {};
      petAPI()?.moveTo?.(Number(x ?? 0.5), Number(y ?? 1), Number(seconds) || 0);
    });
    return off;
  }, []);

  // 鼠标穿透由主进程按光标位置轮询决定（渲染层收 mousemove 并不可靠）；
  // 这里只负责拖拽和点击判定。
  useEffect(() => {
    const api = petAPI();
    if (!api) return;

    const onMove = () => {
      if (draggingRef.current) {
        movedRef.current = true;
        api.dragMove();
      }
    };

    const onDown = (e: MouseEvent) => {
      if (e.button !== 0) return;
      if (hiddenAvatar) return;                      // 隐藏态只点唤醒按钮
      // 菜单区域（角色下方的预留带）不拖窗口，否则点按钮会先被拖拽拦截
      if (open && e.clientY > charBottom) return;
      draggingRef.current = true;
      movedRef.current = false;
      api.dragStart();
    };

    const onUp = () => {
      if (!draggingRef.current) return;
      draggingRef.current = false;
      api.dragEnd();
      if (movedRef.current) return;                  // 拖过就不是点击
      const now = Date.now();
      if (now - lastClickRef.current < 320) {        // 双击 = 换体型
        lastClickRef.current = 0;
        setBody(b => (b + 1) % 3);
        setOpen(false);
        return;
      }
      lastClickRef.current = now;
      setOpen(v => !v);
    };

    window.addEventListener('mousemove', onMove);
    window.addEventListener('mousedown', onDown);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mousedown', onDown);
      window.removeEventListener('mouseup', onUp);
    };
  }, [open, hiddenAvatar, charBottom]);

  useEffect(() => {
    if (open) setTimeout(() => inputRef.current?.focus(), 60);
    petAPI()?.setBar?.(open);
  }, [open]);

  const submit = () => {
    const t = text.trim();
    if (!t) {
      setOpen(false);
      return;
    }
    send('chat.send', { content: t, client_msg_id: `${Date.now()}_${Math.random().toString(36).slice(2, 7)}` });
    setText('');
    setOpen(false);
  };

  const toggleVision = async () => {
    const next = !visionOn;
    try {
      const r = await fetch('/api/vision/toggle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: next }),
      });
      const j = await r.json();
      if (j?.ok) setVisionOn(!!j.enabled);
    } catch { /* 后端没起 */ }
  };

  /** 关停服务 = 只退出后端（虚拟形象留在桌上）；再点就是启动服务 */
  const toggleService = async () => {
    if (busy) return;
    setBusy(true);
    try {
      if (serviceOn) {
        const r = await petAPI()?.stopService?.();
        if (r?.ok) setServiceOn(false);
      } else {
        const r = await petAPI()?.startService?.();
        if (r?.ok) setServiceOn(true);
      }
    } catch { /* ignore */ }
    setBusy(false);
  };

  const applyScale = (pct: number) => {
    const next = Math.max(35, Math.min(160, Math.round(pct / 5) * 5));
    if (next === scalePct) return;
    setScalePct(next);
    petAPI()?.scale?.(next / 100);
  };

  const onWheel = (e: WheelEvent<HTMLDivElement>) => {
    // 必须按住 Ctrl：桌宠窗口盖住桌面一大片，普通滚轮容易误触
    if (!e.ctrlKey && !e.metaKey) return;
    applyScale(scalePct + (e.deltaY < 0 ? 5 : -5));
  };

  // 隐藏态：窗口里只留一颗唤醒按钮
  if (hiddenAvatar) {
    return (
      <div style={{ width: '100vw', height: '100vh', position: 'relative', background: 'transparent' }}>
        <button
          onClick={() => setHiddenAvatar(false)}
          title="唤醒虚拟形象"
          style={{
            position: 'absolute',
            right: 16,
            bottom: 16,
            width: 48,
            height: 48,
            borderRadius: 24,
            border: '1px solid rgba(193,95,60,0.55)',
            background: 'rgba(26,24,21,0.85)',
            color: '#C15F3C',
            fontSize: 20,
            cursor: 'pointer',
            backdropFilter: 'blur(8px)',
          }}
        >
          🐟
        </button>
      </div>
    );
  }

  return (
    <div
      style={{
        width: '100vw',
        height: '100vh',
        overflow: 'hidden',
        position: 'relative',
        background: 'transparent',
      }}
    >
      {/* 角色层：顶部 120 留给气泡，下面按 scalePct 画 */}
      <div
        style={{
          position: 'absolute',
          left,
          top: TOP_BAND,
          width: modelW,
          height: modelH,
          cursor: 'grab',
          animation: 'petFloat 4s ease-in-out infinite',
        }}
        onWheel={onWheel}
      >
        <AvatarWidget
          width={modelW}
          height={modelH}
          modelUrl={modelUrl}
          bodyType={body}
          showBubbles
          bubblePlacement="above"
          bubbleSpace={TOP_BAND - 10}
        />
      </div>

      {/* 底部菜单：固定 100%，不跟人物缩放 */}
      {open && (
        <div
          style={{
            position: 'absolute',
            left: Math.round((winW - HUD_W) / 2) + 12,
            width: HUD_W - 24,
            bottom: 12,
            display: 'flex',
            flexDirection: 'column',
            gap: 6,
            padding: '6px 8px',
            borderRadius: 14,
            background: 'rgba(26,24,21,0.92)',
            border: '1px solid rgba(193,95,60,0.4)',
            backdropFilter: 'blur(8px)',
          }}
        >
          <div style={{ display: 'flex', gap: 6 }}>
            <input
              ref={inputRef}
              value={text}
              onChange={e => setText(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter') submit();
                if (e.key === 'Escape') setOpen(false);
              }}
              placeholder={serviceOn ? '跟她说句话…' : '服务已关，先点「启动服务」'}
              style={{
                flex: 1,
                background: 'transparent',
                border: 'none',
                outline: 'none',
                color: '#e8e6e3',
                fontSize: 13,
                padding: '4px 6px',
              }}
            />
            <button onClick={submit} style={btnStyle('#C15F3C')}>发</button>
          </div>
          <div style={{ display: 'flex', gap: 5, alignItems: 'center' }}>
            <div style={{ display: 'flex', gap: 3, alignItems: 'center' }}>
              <button onClick={() => applyScale(scalePct - 10)} style={btnStyle('rgba(90,90,96,0.85)')}>－</button>
              <span style={{ color: '#cfc8bf', fontSize: 12, width: 36, textAlign: 'center' }}>{scalePct}%</span>
              <button onClick={() => applyScale(scalePct + 10)} style={btnStyle('rgba(90,90,96,0.85)')}>＋</button>
            </div>
            <button onClick={toggleVision} style={btnStyle(visionOn ? 'rgba(90,90,96,0.85)' : '#3E8E6B')}>
              {visionOn ? '关视觉' : '开视觉'}
            </button>
            <button onClick={toggleService} style={btnStyle(serviceOn ? 'rgba(90,90,96,0.85)' : '#3E8E6B')}>
              {serviceOn ? '关服务' : '启动服务'}
            </button>
            <button onClick={() => setHiddenAvatar(true)} style={btnStyle('rgba(90,90,96,0.85)')}>
              隐藏
            </button>
          </div>
        </div>
      )}

      <style>{`
        @keyframes petFloat {
          0%, 100% { transform: translateY(0px); }
          50%      { transform: translateY(-8px); }
        }
      `}</style>
    </div>
  );
}
