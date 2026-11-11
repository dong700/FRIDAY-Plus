import { useAppStore } from '../stores/appStore';
import { Wifi, WifiOff, Brain, Activity, Database, Clock, Terminal } from 'lucide-react';

export function StatusBar({ onToggleTerminal }: { onToggleTerminal: () => void }) {
  const { connected, brainStatus } = useAppStore();
  const mode = brainStatus?.mode || 'idle';
  const modeColor = mode === 'active' ? '#10b981' : mode === 'reflect' ? '#3b82f6' : '#71717a';

  return (
    <div className="h-7 bg-[#1a1815] border-t border-[#2a2520] flex items-center px-3 text-[10px] font-mono select-none">
      <div className="flex items-center gap-3 flex-1">
        <div className="flex items-center gap-1" style={{ color: connected ? '#10b981' : '#ef4444' }}>
          {connected ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
          <span>{connected ? '已连接' : '未连接'}</span>
        </div>

        <div className="w-px h-3 bg-[#2a2520]" />

        <div className="flex items-center gap-1" style={{ color: modeColor }}>
          <Brain className="w-3 h-3" />
          <span className="uppercase">{mode}</span>
        </div>

        {brainStatus && (
          <>
            <div className="w-px h-3 bg-[#2a2520]" />
            <div className="flex items-center gap-1 text-[#8a8580]">
              <Activity className="w-3 h-3" />
              <span>好奇 {Math.round(brainStatus.curiosity * 100)}%</span>
            </div>

            <div className="w-px h-3 bg-[#2a2520]" />
            <div className="flex items-center gap-1 text-[#8a8580]">
              <Database className="w-3 h-3" />
              <span>缺口 {brainStatus.knowledge_gaps?.length || 0}</span>
            </div>
          </>
        )}

        <div className="w-px h-3 bg-[#2a2520]" />
        <button onClick={onToggleTerminal} className="flex items-center gap-1 text-[#8a8580] hover:text-[#c8c6c3]">
          <Terminal className="w-3 h-3" />
          <span>Shell</span>
        </button>
      </div>

      <div className="flex items-center gap-1 text-[#5a5550]">
        <Clock className="w-3 h-3" />
        <span>{new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })}</span>
      </div>
    </div>
  );
}
