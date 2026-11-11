import { Brain, Activity, BookOpen, AlertTriangle, User, Clock, Zap } from 'lucide-react';
import { useAppStore } from '../stores/appStore';
import { useWebSocket } from '../hooks/useWebSocket';
import type { BrainMode, ReasoningLog } from '../types';

const modeConfig: Record<BrainMode, { label: string; color: string; desc: string }> = {
  idle: { label: 'IDLE', color: '#71717a', desc: '静默监听' },
  active: { label: 'ACTIVE', color: '#10b981', desc: '对话交互' },
  reflect: { label: 'REFLECT', color: '#3b82f6', desc: '反思整理' },
};

const presenceConfig: Record<string, { label: string; color: string; icon: string }> = {
  online: { label: '在线', color: '#10b981', icon: '●' },
  idle: { label: '暂离', color: '#f59e0b', icon: '◐' },
  away: { label: '离开', color: '#ef4444', icon: '○' },
  offline: { label: '离线', color: '#6b7280', icon: '○' },
};

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}秒`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}分钟`;
  return `${(seconds / 3600).toFixed(1)}小时`;
}

function MetricBar({ label, value, color }: { label: string; value: number; color: string }) {
  const pct = Math.round(Math.abs(value) * 100);
  return (
    <div className="flex items-center gap-2">
      <span className="text-[10px] text-[#8a8580] w-14 shrink-0">{label}</span>
      <div className="flex-1 h-2 bg-[#2a2520] rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all duration-1000" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <span className="text-xs font-mono w-10 text-right" style={{ color }}>{pct}%</span>
    </div>
  );
}

function ReasoningEntry({ log }: { log: ReasoningLog }) {
  const isAction = log.decision === 'proactive_dialog';
  const time = new Date(log.tick_time).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  
  return (
    <div className={`px-3 py-2 rounded-lg border text-xs ${
      isAction 
        ? 'bg-[#C15F3C]/10 border-[#C15F3C]/30' 
        : 'bg-[#221f1a] border-[#2a2520]'
    }`}>
      <div className="flex items-center gap-2 mb-1">
        <span className="font-mono text-[10px] text-[#5a5550]">{time}</span>
        <span className={`px-1.5 py-0.5 rounded text-[10px] font-mono ${
          isAction ? 'bg-[#C15F3C]/20 text-[#C15F3C]' : 'bg-[#2a2520] text-[#8a8580]'
        }`}>
          {isAction ? 'ACTION' : 'WAIT'}
        </span>
        <span className="text-[10px] text-[#5a5550]">
          好奇心={log.curiosity?.toFixed(2) ?? '-'} 探索={log.exploration?.toFixed(2) ?? '-'}
        </span>
      </div>
      <div className="text-[#8a8580] leading-relaxed">
        {isAction ? (
          <span>
            <span className="text-[#C15F3C]">主动对话 →</span> {log.action_taken || log.reason}
          </span>
        ) : (
          log.reason
        )}
      </div>
    </div>
  );
}

export function BrainPanel() {
  const { brainStatus, reasoningLogs } = useAppStore();
  const { send } = useWebSocket();

  if (!brainStatus) {
    return (
      <div className="flex-1 flex items-center justify-center bg-[#1a1815]">
        <p className="text-sm text-[#8a8580]">连接大脑中...</p>
      </div>
    );
  }

  const currentMode = brainStatus.mode;
  const presence = presenceConfig[brainStatus.user_presence] || presenceConfig.offline;

  return (
    <div className="flex-1 overflow-y-auto bg-[#1a1815] p-6 space-y-5">
      {/* 模式控制 */}
      <div>
        <h3 className="text-xs font-mono text-[#8a8580] uppercase tracking-wider mb-3">模式控制</h3>
        <div className="grid grid-cols-3 gap-2">
          {(Object.entries(modeConfig) as [BrainMode, typeof modeConfig[BrainMode]][]).map(([mode, config]) => (
            <button
              key={mode}
              onClick={() => send('brain.switch_mode', { mode })}
              className={`py-2.5 px-3 rounded-lg border text-center transition-all duration-200
                ${currentMode === mode
                  ? 'border-current bg-opacity-10'
                  : 'bg-[#221f1a] border-[#2a2520] text-[#8a8580] hover:border-[#3a3530]'
                }`}
              style={currentMode === mode ? { borderColor: config.color, color: config.color, backgroundColor: config.color + '10' } : {}}
            >
              <div className="text-xs font-mono font-medium">{config.label}</div>
              <div className="text-[10px] mt-0.5 opacity-70">{config.desc}</div>
            </button>
          ))}
        </div>
      </div>

      {/* 用户在线状态 */}
      <div>
        <h3 className="text-xs font-mono text-[#8a8580] uppercase tracking-wider mb-3 flex items-center gap-1.5">
          <User className="w-3 h-3" />
          用户状态
        </h3>
        <div className="bg-[#221f1a] rounded-lg p-3 border border-[#2a2520] flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span style={{ color: presence.color }} className="text-lg">{presence.icon}</span>
            <span className="text-sm font-mono" style={{ color: presence.color }}>{presence.label}</span>
          </div>
          <div className="text-right">
            <div className="text-[10px] text-[#5a5550]">空闲时间</div>
            <div className="text-xs font-mono text-[#8a8580]">{formatDuration(brainStatus.user_idle_time)}</div>
          </div>
          <div className="text-right">
            <div className="text-[10px] text-[#5a5550]">无回复次数</div>
            <div className="text-xs font-mono text-[#8a8580]">{brainStatus.consecutive_no_response}</div>
          </div>
        </div>
      </div>

      {/* 内在动机 */}
      <div>
        <h3 className="text-xs font-mono text-[#8a8580] uppercase tracking-wider mb-3 flex items-center gap-1.5">
          <Zap className="w-3 h-3" />
          内在动机
        </h3>
        <div className="bg-[#221f1a] rounded-lg p-3 border border-[#2a2520] space-y-2">
          <MetricBar label="好奇心" value={brainStatus.curiosity} color="#C15F3C" />
          <MetricBar label="探索欲" value={brainStatus.exploration} color="#10b981" />
          <MetricBar label="学习进度" value={Math.abs(brainStatus.learning_progress)} color="#3b82f6" />
          {brainStatus.motivation_reason && (
            <div className="mt-2 pt-2 border-t border-[#2a2520] text-xs text-[#8a8580] leading-relaxed">
              <span className="text-[#5a5550]">大脑思考：</span>
              {brainStatus.motivation_reason}
            </div>
          )}
        </div>
      </div>

      {/* 知识缺口 */}
      <div>
        <h3 className="text-xs font-mono text-[#8a8580] uppercase tracking-wider mb-3 flex items-center gap-1.5">
          <AlertTriangle className="w-3 h-3" />
          知识缺口
        </h3>
        <div className="space-y-1.5">
          {brainStatus.knowledge_gaps.length === 0 ? (
            <div className="text-xs text-[#5a5550] py-2 bg-[#221f1a] rounded-lg p-3 border border-[#2a2520]">暂无知识缺口</div>
          ) : (
            brainStatus.knowledge_gaps.map((gap) => {
              const colors: Record<string, string> = {
                critical: 'border-red-500/30 text-red-400',
                high: 'border-orange-500/30 text-orange-400',
                medium: 'border-yellow-500/30 text-yellow-400',
                low: 'border-green-500/30 text-green-400',
              };
              return (
                <div key={gap.id} className={`px-3 py-2 rounded-lg border bg-[#221f1a] text-xs ${colors[gap.priority] || colors.low}`}>
                  <span className="truncate">{gap.query}</span>
                  <span className="float-right font-mono text-[10px] opacity-60">{gap.priority}</span>
                </div>
              );
            })
          )}
        </div>
      </div>

      {/* 推理日志 - 核心新增 */}
      <div>
        <h3 className="text-xs font-mono text-[#8a8580] uppercase tracking-wider mb-3 flex items-center gap-1.5">
          <Brain className="w-3 h-3" />
          推理过程
        </h3>
        <div className="space-y-1.5 max-h-[400px] overflow-y-auto">
          {reasoningLogs.length === 0 ? (
            <div className="text-xs text-[#5a5550] py-2 bg-[#221f1a] rounded-lg p-3 border border-[#2a2520]">
              等待大脑推理...
            </div>
          ) : (
            [...reasoningLogs].reverse().slice(0, 20).map((log, i) => (
              <ReasoningEntry key={`${log.tick_time}-${i}`} log={log} />
            ))
          )}
        </div>
      </div>

      {/* 系统状态 */}
      <div>
        <h3 className="text-xs font-mono text-[#8a8580] uppercase tracking-wider mb-3 flex items-center gap-1.5">
          <Activity className="w-3 h-3" />
          系统状态
        </h3>
        <div className="grid grid-cols-2 gap-2">
          {[
            { label: '对话记录', value: brainStatus.dialog_history_count, suffix: '条' },
            { label: '热记忆', value: brainStatus.hot_memory_count ?? 0, suffix: '24h' },
            { label: '上下文版本', value: `v${brainStatus.context_summary?.version ?? 0}`, suffix: '' },
            { label: '系统健康', value: brainStatus.system_health_label ?? brainStatus.system_health?.overall ?? '-', suffix: '', 
              highlight: (brainStatus.system_health_label ?? '') !== '正常' },
          ].map(({ label, value, suffix, highlight }) => (
            <div key={label} className={`bg-[#221f1a] rounded-lg p-3 border ${highlight ? 'border-[#C15F3C]/30' : 'border-[#2a2520]'}`}>
              <div className="text-[10px] text-[#8a8580]">{label}</div>
              <div className={`text-base font-mono mt-0.5 ${highlight ? 'text-[#C15F3C]' : 'text-[#e8e6e3]'}`}>
                {value}<span className="text-[10px] ml-1 opacity-50">{suffix}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
