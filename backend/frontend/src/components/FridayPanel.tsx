import { useState, useEffect, useRef, useCallback } from 'react';
import { AvatarWidget } from './AvatarWidget';
import { getWsInstance } from '../hooks/useWebSocket';

interface FridayPanelProps {
  onClose?: () => void;
}

interface Subtitle {
  id: number;
  text: string;
  isUser: boolean;
  timestamp: number;
}

let subtitleId = 0;

export function FridayPanel({ onClose }: FridayPanelProps) {
  const [connected, setConnected] = useState(false);
  const [connecting, setConnecting] = useState(false);
  const [subtitles, setSubtitles] = useState<Subtitle[]>([]);
  const [micOn, setMicOn] = useState(true);
  const [micVolume, setMicVolume] = useState(0);
  const [aiSpeaking, setAiSpeaking] = useState(false);
  const [aiSpeechText, setAiSpeechText] = useState('');

  const wsRef = useRef<WebSocket | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const captureNodeRef = useRef<AudioWorkletNode | null>(null);
  const playbackRef = useRef<{ ctx: AudioContext; node: AudioWorkletNode } | null>(null);
  const aiReplyBufferRef = useRef<string>('');
  // 追踪音频播放进度：记录已发送的样本数和发送时间
  const audioSamplesSentRef = useRef<number>(0);
  const audioFirstSampleTimeRef = useRef<number>(0);
  const ttsEndTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const addSubtitle = useCallback((text: string, isUser: boolean) => {
    const id = ++subtitleId;
    setSubtitles(prev => [...prev.slice(-30), { id, text, isUser, timestamp: Date.now() }]);
  }, []);

  const reconnectCountRef = useRef(0);
  const maxReconnects = 3;
  const intentionalCloseRef = useRef(false);

  // 初始化播放器 (接收 24kHz PCM，输出到默认采样率)
  const initPlayback = useCallback(async () => {
    try {
      const ctx = new AudioContext();
      const workletCode = `
        class PCMPlayer extends AudioWorkletProcessor {
          constructor() {
            super();
            this.queue = [];
            this.port.onmessage = (e) => {
              // ★ 支持 FLUSH 命令：清空播放队列
              if (e.data === 'FLUSH') {
                this.queue = [];
                return;
              }
              this.queue.push(e.data);
            };
          }
          process(inputs, outputs) {
            const out = outputs[0][0];
            let offset = 0;
            while (offset < out.length && this.queue.length > 0) {
              const chunk = this.queue[0];
              const copyLen = Math.min(chunk.length, out.length - offset);
              for (let i = 0; i < copyLen; i++) out[offset + i] = chunk[i];
              offset += copyLen;
              if (copyLen < chunk.length) this.queue[0] = chunk.slice(copyLen);
              else this.queue.shift();
            }
            for (let i = offset; i < out.length; i++) out[i] = 0;
            return true;
          }
        }
        registerProcessor('pcm-player', PCMPlayer);
      `;
      const blob = new Blob([workletCode], { type: 'application/javascript' });
      await ctx.audioWorklet.addModule(URL.createObjectURL(blob));
      const node = new AudioWorkletNode(ctx, 'pcm-player');
      node.connect(ctx.destination);
      playbackRef.current = { ctx, node };
    } catch (e: any) {
      console.error('Friday playback init error:', e.message);
    }
  }, []);

  // 麦克风采集 (16kHz → Int16 PCM)
  const startMicCapture = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      micStreamRef.current = stream;

      const ctx = new AudioContext({ sampleRate: 16000 });
      const source = ctx.createMediaStreamSource(stream);

      const analyser = ctx.createAnalyser();
      analyser.fftSize = 1024;
      analyser.smoothingTimeConstant = 0.3;
      source.connect(analyser);
      const dataArray = new Uint8Array(analyser.fftSize);
      const pollVol = () => {
        if (!micStreamRef.current) return;
        analyser.getByteTimeDomainData(dataArray);
        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) {
          const v = (dataArray[i] - 128) / 128;
          sum += v * v;
        }
        const rms = Math.sqrt(sum / dataArray.length);
        const vol = Math.min(100, Math.round(rms * 300));
        setMicVolume(vol);
        setTimeout(pollVol, 200);
      };
      pollVol();

      const capCode = `
        class PCMCapture extends AudioWorkletProcessor {
          constructor() {
            super();
            this.buffer = [];
            this.chunkSize = 3200;
          }
          process(inputs) {
            const input = inputs[0];
            if (!input || !input[0]) return true;
            const s = input[0];
            const pcm = new Int16Array(s.length);
            for (let i = 0; i < s.length; i++) {
              const v = Math.max(-1, Math.min(1, s[i]));
              pcm[i] = v < 0 ? v * 0x8000 : v * 0x7FFF;
            }
            this.buffer.push(pcm);
            let total = this.buffer.reduce((a, b) => a + b.length, 0);
            if (total >= this.chunkSize) {
              const merged = new Int16Array(this.chunkSize);
              let offset = 0;
              while (offset < this.chunkSize && this.buffer.length > 0) {
                const chunk = this.buffer[0];
                const copyLen = Math.min(chunk.length, this.chunkSize - offset);
                merged.set(chunk.subarray(0, copyLen), offset);
                offset += copyLen;
                if (copyLen < chunk.length) {
                  this.buffer[0] = chunk.subarray(copyLen);
                } else {
                  this.buffer.shift();
                }
              }
              this.port.postMessage(merged.buffer, [merged.buffer]);
            }
            return true;
          }
        }
        registerProcessor('pcm-capture', PCMCapture);
      `;
      const blob = new Blob([capCode], { type: 'application/javascript' });
      await ctx.audioWorklet.addModule(URL.createObjectURL(blob));
      const captureNode = new AudioWorkletNode(ctx, 'pcm-capture');
      source.connect(captureNode);
      captureNode.connect(ctx.destination);
      captureNodeRef.current = captureNode;

      captureNode.port.onmessage = async (e) => {
        const ws = wsRef.current;
        if (ws?.readyState === WebSocket.OPEN && micOn) {
          ws.send(e.data);
        }
      };
    } catch (e: any) {
      console.error('Friday mic capture error:', e.message);
    }
  }, [micOn]);

  // 播放 PCM 音频（24kHz Float32 → 重采样到 AudioContext 采样率）
  const playAudio = useCallback((data: ArrayBuffer) => {
    const pb = playbackRef.current;
    if (!pb) return;
    const pcmBuf = data.slice(0);
    const alignedByteLen = pcmBuf.byteLength - (pcmBuf.byteLength % 4);
    if (alignedByteLen < 4) return;
    const f32_src = new Float32Array(pcmBuf, 0, alignedByteLen / 4);

    // 追踪音频样本数（24kHz 源采样率）
    if (audioSamplesSentRef.current === 0) {
      audioFirstSampleTimeRef.current = Date.now();
    }
    audioSamplesSentRef.current += f32_src.length;

    // 口型同步（2026-11-02）：按这一块 PCM 的实际音量驱动 Live2D 嘴部。
    // 之前 setLipSync 暴露了但没人调用，说话时只有动作、嘴不动。
    // 这里用 24kHz 源数据的 RMS，放大 3 倍是因为正常语音 RMS 多在 0.1~0.3。
    let _sum = 0;
    for (let i = 0; i < f32_src.length; i++) _sum += f32_src[i] * f32_src[i];
    const _rms = Math.sqrt(_sum / f32_src.length);
    const _lip = Math.min(1, _rms * 3);
    window.dispatchEvent(new CustomEvent('avatar-lip-sync', { detail: _lip }));

    const ratio = pb.ctx.sampleRate / 24000;
    if (Math.abs(ratio - 1) < 0.01) {
      pb.node.port.postMessage(f32_src);
    } else {
      const outLen = Math.floor(f32_src.length * ratio);
      const f32 = new Float32Array(outLen);
      for (let i = 0; i < outLen; i++) {
        const srcIdx = i / ratio;
        const i0 = Math.floor(srcIdx);
        const i1 = Math.min(i0 + 1, f32_src.length - 1);
        const frac = srcIdx - i0;
        f32[i] = f32_src[i0] * (1 - frac) + f32_src[i1] * frac;
      }
      pb.node.port.postMessage(f32);
    }
  }, []);

  // ★ 打断：清空音频播放队列，立即停止 TTS 输出
  const flushPlayback = useCallback(() => {
    const pb = playbackRef.current;
    if (pb) {
      pb.node.port.postMessage('FLUSH');
    }
    // 取消 ttse_end 的 setTimeout
    if (ttsEndTimerRef.current) {
      clearTimeout(ttsEndTimerRef.current);
      ttsEndTimerRef.current = null;
    }
    // 重置音频追踪
    audioSamplesSentRef.current = 0;
    audioFirstSampleTimeRef.current = 0;
    // 停止虚拟形象说话
    setAiSpeaking(false);
    window.dispatchEvent(new CustomEvent('avatar-lip-sync', { detail: 0 }));
    setAiSpeechText('');
    aiReplyBufferRef.current = '';
    window.dispatchEvent(new CustomEvent('avatar-speaking-end'));
  }, []);

  // 启动对话
  const startChat = useCallback(async () => {
    setConnecting(true);
    intentionalCloseRef.current = false;
    if (reconnectCountRef.current === 0) setSubtitles([]);

    try {
      await initPlayback();
      await startMicCapture();

      const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${wsProto}//${window.location.host}/api/realtime/ws`;

      const ws = new WebSocket(wsUrl);
      ws.binaryType = 'arraybuffer';
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        setConnecting(false);
        reconnectCountRef.current = 0;
      };

      ws.onmessage = (event) => {
        if (event.data instanceof ArrayBuffer) {
          playAudio(event.data);
          setAiSpeaking(true);
        } else {
          try {
            const msg = JSON.parse(event.data);
            switch (msg.type) {
              case 'asr':
                if (msg.is_final && msg.text) {
                  addSubtitle(msg.text, true);
                }
                break;
              case 'ai_reply':
                aiReplyBufferRef.current += msg.text || '';
                setAiSpeechText(aiReplyBufferRef.current);
                setAiSpeaking(true);
                break;
              case 'ai_reply_end':
                // ai_reply_end 只标记文本结束，不立刻移到右边
                // 等 tts_end（TTS播放完）再移到右侧字幕
                break;
              case 'tts_end':
                // ★ 关键修复：TTS 服务端生成完毕 ≠ 客户端播放完毕
                // 估算剩余播放时长，等音频播完再移字幕
                {
                  const samplesSent = audioSamplesSentRef.current;
                  const firstTime = audioFirstSampleTimeRef.current;
                  // 重置追踪
                  audioSamplesSentRef.current = 0;
                  audioFirstSampleTimeRef.current = 0;

                  if (samplesSent > 0 && firstTime > 0) {
                    // 总音频时长 = 样本数 / 采样率(24kHz)
                    const totalDurationMs = (samplesSent / 24000) * 1000;
                    // 已播放时长 = 当前时间 - 首个样本到达时间
                    const elapsedMs = Date.now() - firstTime;
                    // 剩余播放时长（加 200ms 缓冲避免过早切换）
                    const remainingMs = Math.max(0, totalDurationMs - elapsedMs + 200);
                    console.log(`[TTS] Server done, audio remaining: ${remainingMs.toFixed(0)}ms (total=${totalDurationMs.toFixed(0)}ms elapsed=${elapsedMs}ms)`);

                    // 延迟剩余播放时长后再移字幕 + 结束虚拟形象说话状态
                    ttsEndTimerRef.current = setTimeout(() => {
                      ttsEndTimerRef.current = null;
                      setAiSpeaking(false);
                      if (aiReplyBufferRef.current) {
                        addSubtitle(aiReplyBufferRef.current, false);
                        aiReplyBufferRef.current = '';
                      }
                      setAiSpeechText('');
                      // ★ 音频播完 → 通知虚拟形象结束说话（拿话筒时长 = 字幕时长）
                      window.dispatchEvent(new CustomEvent('avatar-speaking-end'));
                      // ★ 通知后端 TTS 播完 → 重置大脑空闲计时器
                      try {
                        const mainWs = getWsInstance();
                        if (mainWs?.readyState === WebSocket.OPEN) {
                          mainWs.send(JSON.stringify({ type: 'tts_playback_done' }));
                        }
                      } catch { /* ignore */ }
                    }, remainingMs);
                  } else {
                    // 没有音频追踪数据，立即切换（如 SayHello 的 tts_end）
                    setAiSpeaking(false);
                    if (aiReplyBufferRef.current) {
                      addSubtitle(aiReplyBufferRef.current, false);
                      aiReplyBufferRef.current = '';
                    }
                    setAiSpeechText('');
                    // SayHello 等短回复也通知结束说话
                    window.dispatchEvent(new CustomEvent('avatar-speaking-end'));
                    // ★ 通知后端 TTS 播完
                    try {
                      const mainWs = getWsInstance();
                      if (mainWs?.readyState === WebSocket.OPEN) {
                        mainWs.send(JSON.stringify({ type: 'tts_playback_done' }));
                      }
                    } catch { /* ignore */ }
                  }
                }
                break;
              case 'user_speak_end':
                break;
              case 'interrupt':
                // ★ 打断：用户开始说话时，立即停止 AI 的 TTS 播放
                console.log('[Friday] Interrupt received — flushing audio');
                flushPlayback();
                break;
              case 'error':
                addSubtitle(`[错误] ${msg.msg}`, false);
                break;
            }
          } catch {}
        }
      };

      ws.onclose = () => {
        setConnected(false);
        setConnecting(false);
        setAiSpeaking(false);
        setAiSpeechText('');
        if (!intentionalCloseRef.current && reconnectCountRef.current < maxReconnects) {
          reconnectCountRef.current++;
          setTimeout(() => startChat(), 3000);
        }
      };

      ws.onerror = () => {
        setConnecting(false);
      };

    } catch (e: any) {
      addSubtitle(`[启动失败] ${e.message}`, false);
      setConnecting(false);
    }
  }, [addSubtitle, initPlayback, startMicCapture, playAudio]);

  const stopChat = useCallback(() => {
    intentionalCloseRef.current = true;
    const ws = wsRef.current;
    if (ws) ws.close();
    wsRef.current = null;
    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach(t => t.stop());
      micStreamRef.current = null;
    }
    captureNodeRef.current = null;
    if (playbackRef.current) {
      playbackRef.current.ctx.close();
      playbackRef.current = null;
    }
    setConnected(false);
    setAiSpeaking(false);
    setAiSpeechText('');
    setMicVolume(0);
  }, []);

  useEffect(() => () => { stopChat(); }, []);

  return (
    <div className="flex h-full bg-gray-900 text-white overflow-hidden">
      {/* 左侧：虚拟形象 + 控制按钮 */}
      <div className="flex-1 flex flex-col items-center justify-between py-6 relative">
        {/* 顶部标题 */}
        <div className="absolute top-4 left-4 flex items-center gap-2">
          <div className={`w-2.5 h-2.5 rounded-full ${
            connected ? 'bg-green-500 animate-pulse' :
            connecting ? 'bg-yellow-500 animate-pulse' : 'bg-gray-500'
          }`} />
          <span className="text-sm font-bold">FRIDAY</span>
          <span className="text-xs text-gray-400">
            {connected ? '语音通话中' : connecting ? '连接中...' : '离线'}
          </span>
        </div>

        {/* 中间：虚拟形象（大尺寸居中） */}
        <div className="flex-1 flex items-center justify-center w-full">
          <div className="relative">
            {/* 说话时的光晕效果 */}
            {aiSpeaking && (
              <>
                <div className="absolute inset-0 rounded-full bg-cyan-400/10 animate-ping"
                     style={{ transform: 'scale(1.3)', filter: 'blur(20px)' }} />
                <div className="absolute inset-0 rounded-full bg-cyan-400/5 animate-pulse"
                     style={{ transform: 'scale(1.15)', filter: 'blur(15px)' }} />
              </>
            )}
            {/* 麦克风音量环 */}
            {connected && micOn && micVolume > 10 && (
              <div className="absolute inset-0 rounded-full border-2 border-green-400/30 transition-all duration-100"
                   style={{
                     transform: `scale(${1 + micVolume * 0.004})`,
                     opacity: Math.min(1, micVolume / 50),
                   }} />
            )}
            <AvatarWidget
              width={400}
              height={560}
              showBubbles={false}
              speechState={aiSpeaking ? 'speaking' : connected ? 'listening' : 'idle'}
            />
            {/* AI 正在说的话（气泡放到画布右侧，不挡人物） */}
            {aiSpeaking && aiSpeechText && (
              <div className="absolute left-full ml-4 top-[25%] w-64 px-4 py-3 bg-cyan-500/90 text-white text-sm rounded-2xl rounded-bl-none shadow-xl backdrop-blur-md text-left z-10"
                style={{ animation: 'fadeInSlide 0.3s ease-out' }}
              >
                <div className="absolute left-0 top-6 -translate-x-full w-0 h-0 border-t-[8px] border-t-transparent border-r-[10px] border-r-cyan-500/90 border-b-[8px] border-b-transparent" />
                {aiSpeechText}
              </div>
            )}
          </div>
        </div>

        {/* 底部：控制按钮 + 音量条 */}
        <div className="flex flex-col items-center gap-3 w-full">
          {/* 音量条 */}
          {connected && (
            <div className="flex items-center gap-3">
              <span className="text-xs text-gray-400">🎤</span>
              <div className="w-48 h-2 bg-gray-700 rounded-full overflow-hidden">
                <div className="h-full bg-green-500 rounded-full transition-all duration-100"
                     style={{ width: `${micVolume}%` }} />
              </div>
              <span className="text-xs font-mono text-gray-400 w-8">{micVolume}</span>
            </div>
          )}

          {/* 按钮 */}
          <div className="flex items-center gap-6">
            {!connected && !connecting ? (
              <button onClick={startChat}
                      className="px-8 py-3 bg-green-600 hover:bg-green-500 rounded-full font-medium transition text-sm">
                开始语音对话
              </button>
            ) : connecting ? (
              <span className="text-gray-400 animate-pulse">连接中...</span>
            ) : (
              <>
                <button onClick={() => setMicOn(v => !v)}
                        className={`w-14 h-14 rounded-full flex items-center justify-center text-xl relative overflow-hidden transition ${
                          micOn ? 'bg-blue-600 hover:bg-blue-500' : 'bg-gray-700 hover:bg-gray-600'
                        }`}
                        title={micOn ? '关闭麦克风' : '开启麦克风'}>
                  {micOn ? '🎤' : '🔇'}
                  {micOn && micVolume > 5 && (
                    <div className="absolute bottom-0 left-0 right-0 bg-green-400 opacity-60"
                         style={{ height: `${Math.min(micVolume * 0.8, 85)}%`, transition: 'height 0.1s' }} />
                  )}
                </button>
                <button onClick={stopChat}
                        className="w-14 h-14 rounded-full bg-red-600 hover:bg-red-500 flex items-center justify-center text-xl"
                        title="挂断">
                  📞
                </button>
              </>
            )}
          </div>

          {connected && (
            <p className="text-xs text-gray-500">直接说话即可，AI 会自动应答</p>
          )}
        </div>
      </div>

      {/* 右侧：字幕历史 */}
      <div className="w-[320px] border-l border-gray-700 flex flex-col bg-gray-900/50">
        <div className="px-4 py-3 border-b border-gray-700">
          <h3 className="text-sm font-medium text-gray-300">对话记录</h3>
        </div>
        <div className="flex-1 overflow-y-auto px-3 py-3 space-y-2">
          {subtitles.length === 0 ? (
            <div className="text-xs text-gray-600 text-center mt-8">
              {connecting ? '正在连接...' : '暂无对话记录'}
            </div>
          ) : (
            subtitles.map((sub) => (
              <div key={sub.id}
                   className={`flex ${sub.isUser ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[85%] px-3 py-2 rounded-xl text-xs leading-relaxed
                  ${sub.isUser
                    ? 'bg-[#C15F3C] text-white rounded-br-sm'
                    : 'bg-cyan-600/30 text-cyan-100 border border-cyan-500/20 rounded-bl-sm'
                  }`}>
                  {sub.text}
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
