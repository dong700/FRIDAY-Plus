/**
 * AvatarWidget - Live2D 虚拟形象组件（可嵌入任意页面）
 *
 * 功能：
 * 1. 渲染 Live2D 模型
 * 2. 从角色嘴边冒出对话气泡
 * 3. 支持自定义尺寸
 * 4. 加载状态 / 错误重试
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { useLive2D } from '../hooks/useLive2D';
import { subscribeWs } from '../hooks/useWebSocket';
import { useAppStore } from '../stores/appStore';
import { RefreshCw } from 'lucide-react';

interface SpeechBubble {
  id: number;
  text: string;
  fromUser: boolean;
  timestamp: number;
}

let bubbleId = 0;

/**
 * 气泡只放口语：把 markdown 记号洗掉、换行压成空格。
 * 之前正文原样塞进气泡，模型一说长列表就是"显示一大堆"。
 */
function toPlainText(text: string, max = 160): string {
  const s = (text || '')
    .replace(/```[\s\S]*?```/g, '')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/^\s*#{1,6}\s*/gm, '')
    .replace(/^\s*[-*+]\s+/gm, '')
    .replace(/^\s*>\s*/gm, '')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/__([^_]+)__/g, '$1')
    .replace(/\[(.+?)\]\((.+?)\)/g, '$1')
    .replace(/\s*\n\s*/g, ' ')
    .replace(/\s{2,}/g, ' ')
    .trim();
  if (s.length <= max) return s;
  return s.slice(0, max).replace(/[，。、；：,.;:\s]+$/, '') + '…';
}

interface AvatarWidgetProps {
  width?: number;
  height?: number;
  modelUrl?: string;
  showBubbles?: boolean;
  /** 气泡放在容器顶部（面板里用）/ 容器上方（桌宠留了顶部透明带） */
  bubblePlacement?: 'top' | 'above';
  /** 'above' 时气泡区可用的高度（桌面桌宠是预留的透明带） */
  bubbleSpace?: number;
  speechState?: 'idle' | 'speaking' | 'listening';
  /** 体型：0 默认 / 1 体型1 / 2 体型2(Q版)，模型自带 Clothes2 变形 */
  bodyType?: number;
}

export function AvatarWidget({
  width = 400,
  height = 500,
  modelUrl,
  showBubbles = true,
  bubblePlacement = 'top',
  bubbleSpace = 116,
  speechState,
  bodyType = 0,
}: AvatarWidgetProps) {
  const [bubbles, setBubbles] = useState<SpeechBubble[]>([]);
  // 心智流式输出：mind.delta 逐字追加，mind.end 收尾（对应后端 brain/mind.py）
  const [liveText, setLiveText] = useState('');
  const liveRef = useRef('');
  const skipBubbleRef = useRef('');
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const messages = useAppStore(s => s.messages);

  const controller = useLive2D({
    canvasRef,
    width,
    height,
    modelUrl,
    bodyType,
  });

  const addBubble = useCallback((text: string, fromUser: boolean) => {
    const displayText = toPlainText(text, 160);
    if (!displayText) return;
    const id = ++bubbleId;
    // 只留最新 2 条：说的话多了会累成一大摞把人挡死
    setBubbles(prev => [...prev.slice(-1), { id, text: displayText, fromUser, timestamp: Date.now() }]);
    // 之前 8 秒太短，读完就没了；20 秒够看完
    setTimeout(() => {
      setBubbles(prev => prev.filter(b => b.id !== id));
    }, 20000);
  }, []);

  // 口型同步（2026-11-02）：FridayPanel 播放 PCM 时按音量派发，这里接给模型
  useEffect(() => {
    const handleLip = (ev: Event) => {
      const v = (ev as CustomEvent<number>).detail;
      controller.setLipSync(typeof v === 'number' ? v : 0);
    };
    window.addEventListener('avatar-lip-sync', handleLip);
    return () => window.removeEventListener('avatar-lip-sync', handleLip);
  }, [controller]);

  // 外部控制说话状态（星期五语音页用）
  const lastSpeechStateRef = useRef<string>('');
  useEffect(() => {
    if (!speechState || speechState === lastSpeechStateRef.current) return;
    lastSpeechStateRef.current = speechState;
    if (speechState === 'speaking') {
      controller.playMotion('speaking');
    } else if (speechState === 'listening') {
      controller.playMotion('listening');
    } else {
      controller.playMotion('idle');
    }
  }, [speechState, controller]);

  // 监听聊天消息 → 触发动作 + 气泡
  const lastMsgIdRef = useRef<string>('');
  useEffect(() => {
    if (messages.length === 0) return;
    const lastMsg = messages[messages.length - 1];
    if (lastMsg.id === lastMsgIdRef.current) return;
    lastMsgIdRef.current = lastMsg.id;

    if (lastMsg.role === 'assistant') {
      // 已经用流式气泡显示过的回复不再重复弹一次
      const skip = skipBubbleRef.current && lastMsg.content.trim() === skipBubbleRef.current;
      if (skip) {
        skipBubbleRef.current = '';
      } else if (showBubbles) {
        addBubble(lastMsg.content, false);
      }
      controller.playMotion('speaking');
      setTimeout(() => controller.playMotion('idle'), 4000);
    } else if (lastMsg.role === 'user') {
      controller.playMotion('listening');
      setTimeout(() => controller.playMotion('idle'), 2000);
    }
  }, [messages, controller, addBubble, showBubbles]);

  // 监听 WebSocket avatar.action 事件
  useEffect(() => {
    const off = subscribeWs((data) => {
      if (data.type === 'avatar.action') {
        const { text, action } = data.payload || {};
        if (text && action !== 'idle' && showBubbles) {
          const fromUser = action === 'listening';
          addBubble(text, fromUser);
        }
      } else if (data.type === 'mind.thinking') {
        liveRef.current = '';
        setLiveText('');
        controller.playMotion('speaking');
      } else if (data.type === 'mind.delta') {
        liveRef.current += (data.payload?.text || '');
        setLiveText(toPlainText(liveRef.current, 600));
      } else if (data.type === 'mind.end') {
        const finalText = (data.payload?.text || '').trim();
        if (finalText) {
          skipBubbleRef.current = finalText;
          if (showBubbles) addBubble(finalText, false);
        }
        liveRef.current = '';
        setLiveText('');
        setTimeout(() => controller.playMotion('idle'), 3000);
      }
    });
    return off;
  }, [addBubble, showBubbles, controller]);

  const handleRetry = () => {
    window.location.reload();
  };

  return (
    <div className="relative" style={{ width, height }}>
      {/* 不给 canvas 设 width/height 属性：尺寸全交给 PIXI（renderer.resize），
          否则 React 每次改属性都会重建绘制缓冲，把渲染状态冲掉 */}
      <canvas ref={canvasRef} style={{ display: 'block', width: '100%', height: '100%' }} />

      {/* 加载中 */}
      {controller.loading && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-black/30 backdrop-blur-sm">
          <div className="w-10 h-10 border-2 border-cyan-400/20 border-t-cyan-400 rounded-full animate-spin" />
          <span className="text-sm text-gray-300">虚拟形象加载中...</span>
        </div>
      )}

      {/* 错误 + 重试 */}
      {controller.error && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-black/50 backdrop-blur-sm">
          <span className="text-red-400 text-sm text-center max-w-[80%]">
            {controller.error}
          </span>
          <button
            onClick={handleRetry}
            className="flex items-center gap-1.5 px-4 py-2 bg-[#C15F3C] text-white rounded-lg text-sm hover:bg-[#A14A2F] transition-colors"
          >
            <RefreshCw size={14} />
            重新加载
          </button>
        </div>
      )}

      {/* 对话气泡层 - 从角色嘴边冒出 */}
      {showBubbles && !controller.loading && !controller.error && (
        <div
          className={`flex flex-col gap-2 pointer-events-none ${bubblePlacement === 'above'
            ? '' : 'absolute top-2 left-2 right-2'}`}
          style={bubblePlacement === 'above'
            ? {
                position: 'absolute',
                left: 0,
                right: 0,
                bottom: '100%',       // 挂在容器上沿：气泡在头顶上方冒，不遮脸
                maxHeight: bubbleSpace,   // 头顶预留的那条透明带有多高，气泡就能长多高
                overflow: 'hidden',
                justifyContent: 'flex-end',
              }
            : undefined}
        >
          {bubbles.map((bubble, idx) => (
            <div
              key={bubble.id}
              className={`px-4 py-2 rounded-2xl text-sm leading-relaxed shadow-xl backdrop-blur-md max-w-[75%]
                         ${bubble.fromUser
                           ? 'bg-[#C15F3C]/80 text-white self-start rounded-bl-sm'
                           : 'bg-cyan-500/70 text-white self-end rounded-br-sm'
                         }`}
              style={{
                animation: 'fadeInSlide 0.3s ease-out',
                opacity: 0.85 + idx * 0.05,
              }}
            >
              {bubble.text}
            </div>
          ))}
          {liveText && (
            <div
              className="px-4 py-2 rounded-2xl text-sm leading-relaxed shadow-xl backdrop-blur-md max-w-[75%]
                         bg-cyan-500/70 text-white self-end rounded-br-sm"
              style={{ animation: 'fadeInSlide 0.3s ease-out', opacity: 0.9 }}
            >
              {liveText}
              {/* 流式光标：让人一眼看出还在生成，而不是"卡住之后一口气全跳出来" */}
              <span style={{ opacity: 0.7 }}>▍</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
