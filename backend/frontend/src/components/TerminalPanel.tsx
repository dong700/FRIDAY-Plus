import { useState, useRef, useEffect, useCallback } from 'react';
import { X, Terminal, Play, FileCode, List, Trash2, Cpu, ChevronRight, Bot, Zap } from 'lucide-react';
import { useWebSocket, getWsInstance } from '../hooks/useWebSocket';
import type { WorkspaceFile, WorkspaceExecution, WorkspaceInfo, ExecutorStatus } from '../types';

interface TerminalLine {
  type: 'input' | 'output' | 'error' | 'brain' | 'script' | 'result' | 'system' | 'executor';
  content: string;
  timestamp: string;
}

type TabType = 'shell' | 'files' | 'executions' | 'executor';

export function TerminalPanel({ onClose }: { onClose: () => void }) {
  const [lines, setLines] = useState<TerminalLine[]>([
    { type: 'system', content: 'Active AI Pro Shell v2.0 — 大脑工作区已就绪', timestamp: new Date().toISOString() },
    { type: 'system', content: '输入 help 查看命令 | 切换标签查看工作区', timestamp: new Date().toISOString() },
  ]);
  const [input, setInput] = useState('');
  const [activeTab, setActiveTab] = useState<TabType>('shell');
  const [workspaceFiles, setWorkspaceFiles] = useState<WorkspaceFile[]>([]);
  const [workspaceExecs, setWorkspaceExecs] = useState<WorkspaceExecution[]>([]);
  const [workspaceInfo, setWorkspaceInfo] = useState<WorkspaceInfo | null>(null);
  const [executorStatus, setExecutorStatus] = useState<ExecutorStatus | null>(null);
  const { send } = useWebSocket();
  const termRef = useRef<HTMLDivElement>(null);

  // 自动滚动
  useEffect(() => {
    if (termRef.current) termRef.current.scrollTop = termRef.current.scrollHeight;
  }, [lines]);

  // 加载工作区数据
  const loadWorkspace = useCallback(() => {
    send('workspace.files', {});
    send('workspace.executions', { limit: 20 });
    fetch('/api/workspace').then(r => r.json()).then(setWorkspaceInfo).catch(() => {});
    fetch('/api/executor').then(r => r.json()).then(setExecutorStatus).catch(() => {});
  }, [send]);

  useEffect(() => {
    if (activeTab !== 'shell') loadWorkspace();
  }, [activeTab, loadWorkspace]);

  // 监听大脑工作区事件
  useEffect(() => {
    const ws = getWsInstance();
    const handler = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);
        const payload = data.payload || {};

        switch (data.type) {
          case 'brain.writing_script':
            setLines(prev => [...prev, {
              type: 'brain',
              content: `🧠 大脑正在写脚本: ${payload.script_name} | 任务: "${payload.task}"`,
              timestamp: new Date().toISOString(),
            }]);
            break;
          case 'brain.task_completed':
            setLines(prev => [...prev,
              { type: 'result', content: `✅ 任务完成: ${payload.script_name} (${payload.duration_ms}ms)`, timestamp: new Date().toISOString() },
              { type: payload.success ? 'output' : 'error',
                content: payload.stdout?.slice(0, 500) || payload.stderr || '(无输出)',
                timestamp: new Date().toISOString() },
            ]);
            break;
          case 'workspace.script_written':
            setLines(prev => [...prev, {
              type: 'script', content: `📝 脚本已写入: ${payload.name} (${payload.size}b)`,
              timestamp: new Date().toISOString(),
            }]);
            if (payload.execution) {
              setLines(prev => [...prev, {
                type: payload.execution.success ? 'output' : 'error',
                content: `执行结果(${payload.execution.duration_ms}ms): ${payload.execution.stdout?.slice(0, 300)}`,
                timestamp: new Date().toISOString(),
              }]);
            }
            break;
          case 'workspace.script_executed':
            setLines(prev => [...prev, {
              type: payload.success ? 'output' : 'error',
              content: `▶ ${payload.script_name} → exit=${payload.exit_code} (${payload.duration_ms}ms)\n${payload.stdout?.slice(0, 500)}`,
              timestamp: new Date().toISOString(),
            }]);
            break;
          case 'workspace.file_list':
            setWorkspaceFiles(payload.files || []);
            break;
          case 'workspace.execution_list':
            setWorkspaceExecs(payload.executions || []);
            break;
          case 'executor.task_completed':
            setLines(prev => [...prev,
              { type: 'executor', content: `🤖 执行器完成: ${payload.description?.slice(0,60)} | ${payload.script_name} (${payload.duration_ms}ms)`, timestamp: new Date().toISOString() },
              { type: payload.success ? 'result' : 'error',
                content: payload.stdout?.slice(0, 500) || '(无输出)',
                timestamp: new Date().toISOString() },
            ]);
            // 刷新执行器状态
            fetch('/api/executor').then(r => r.json()).then(setExecutorStatus).catch(() => {});
            break;
          case 'executor.task_failed':
            setLines(prev => [...prev, {
              type: 'executor', content: `❌ 执行器失败: ${payload.description?.slice(0,60)} | ${payload.error}`,
              timestamp: new Date().toISOString(),
            }]);
            fetch('/api/executor').then(r => r.json()).then(setExecutorStatus).catch(() => {});
            break;
          case 'executor.executing':
            setLines(prev => [...prev, {
              type: 'executor', content: `⚡ 执行器正在生成脚本: ${payload.task_id} | "${payload.description}"`,
              timestamp: new Date().toISOString(),
            }]);
            break;
        }
      } catch {}
    };

    ws.addEventListener('message', handler);
    return () => ws.removeEventListener('message', handler);
  }, []);

  const addLine = (line: TerminalLine) => setLines(prev => [...prev, line]);

  const handleCommand = (cmd: string) => {
    const trimmed = cmd.trim();
    if (!trimmed) return;

    addLine({ type: 'input', content: `$ ${trimmed}`, timestamp: new Date().toISOString() });

    if (trimmed === 'help') {
      addLine({ type: 'output', content: '可用命令:', timestamp: new Date().toISOString() });
      addLine({ type: 'output', content: '  status       - 查看大脑状态', timestamp: new Date().toISOString() });
      addLine({ type: 'output', content: '  mode <m>     - 切换模式 (idle/active/reflect)', timestamp: new Date().toISOString() });
      addLine({ type: 'output', content: '  search <q>   - 搜索记忆', timestamp: new Date().toISOString() });
      addLine({ type: 'output', content: '  say <msg>    - 让AI说话', timestamp: new Date().toISOString() });
      addLine({ type: 'output', content: '  run <code>   - 在工作区运行代码', timestamp: new Date().toISOString() });
      addLine({ type: 'output', content: '  ls           - 列出工作区文件', timestamp: new Date().toISOString() });
      addLine({ type: 'output', content: '  history      - 执行历史', timestamp: new Date().toISOString() });
      addLine({ type: 'output', content: '  clear        - 清屏', timestamp: new Date().toISOString() });
    } else if (trimmed === 'clear') {
      setLines([]);
    } else if (trimmed === 'ls' || trimmed.startsWith('mode ')) {
      if (trimmed === 'ls') {
        send('workspace.files', {});
        setActiveTab('files');
      } else {
        const mode = trimmed.split(' ')[1];
        if (['idle','active','reflect'].includes(mode)) {
          send('brain.switch_mode', { mode });
          addLine({ type: 'output', content: `模式切换: ${mode}`, timestamp: new Date().toISOString() });
        } else {
          addLine({ type: 'error', content: `无效模式: ${mode}`, timestamp: new Date().toISOString() });
        }
      }
    } else if (trimmed === 'history') {
      send('workspace.executions', { limit: 10 });
      setActiveTab('executions');
    } else if (trimmed.startsWith('run ')) {
      const code = trimmed.slice(4);
      send('workspace.execute', { content: code });
      addLine({ type: 'brain', content: `▶ 执行内联脚本...`, timestamp: new Date().toISOString() });
    } else if (trimmed.startsWith('search ')) {
      send('memory.search', { query: trimmed.slice(7), limit: 10 });
      addLine({ type: 'output', content: `搜索记忆: "${trimmed.slice(7)}"`, timestamp: new Date().toISOString() });
    } else if (trimmed.startsWith('say ')) {
      send('chat.send', { content: trimmed.slice(4), client_msg_id: `t_${Date.now()}` });
      addLine({ type: 'output', content: `发送消息: "${trimmed.slice(4)}"`, timestamp: new Date().toISOString() });
    } else if (trimmed === 'status') {
      send('brain.get_status', {});
      addLine({ type: 'output', content: '获取大脑状态...', timestamp: new Date().toISOString() });
    } else {
      addLine({ type: 'error', content: `未知命令: ${trimmed} (输入 help)`, timestamp: new Date().toISOString() });
    }
    setInput('');
  };

  const lineColor = (t: TerminalLine['type']) =>
    t === 'input' ? 'text-[#C15F3C]' :
    t === 'error' ? 'text-red-400' :
    t === 'brain' ? 'text-cyan-400' :
    t === 'script' ? 'text-yellow-400' :
    t === 'result' ? 'text-green-400' :
    t === 'executor' ? 'text-purple-400' :
    t === 'system' ? 'text-[#6a6560]' :
    'text-[#8a8580]';

  const tabs: { key: TabType; label: string; icon: typeof Terminal; badge?: number }[] = [
    { key: 'shell', label: 'Shell', icon: Terminal },
    { key: 'files', label: '文件', icon: FileCode, badge: workspaceFiles.length },
    { key: 'executions', label: '执行记录', icon: List, badge: workspaceExecs.length },
    { key: 'executor', label: '执行器', icon: Bot, badge: executorStatus?.active_count },
  ];

  return (
    <div className="h-80 bg-[#141210] border-t border-[#2a2520] flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-1.5 bg-[#1a1815] border-b border-[#2a2520]">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 text-[10px] text-[#8a8580]">
            <Terminal className="w-3 h-3" />
            <span className="font-mono">WORKSPACE</span>
          </div>
          {/* Tabs */}
          <div className="flex gap-1">
            {tabs.map(tab => (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key)}
                className={`flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono transition-colors ${
                  activeTab === tab.key
                    ? 'bg-[#C15F3C]/20 text-[#C15F3C]'
                    : 'text-[#6a6560] hover:text-[#8a8580]'
                }`}
              >
                <tab.icon className="w-3 h-3" />
                {tab.label}
                {tab.badge != null && tab.badge > 0 && (
                  <span className="text-[9px] text-[#C15F3C]">{tab.badge}</span>
                )}
              </button>
            ))}
          </div>
        </div>
        <button onClick={onClose} className="text-[#8a8580] hover:text-[#c8c6c3]">
          <X className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Content Area */}
      {activeTab === 'shell' ? (
        <>
          <div ref={termRef} className="flex-1 overflow-y-auto p-3 font-mono text-xs space-y-0.5">
            {lines.map((line, i) => (
              <div key={`${i}_${line.timestamp}`} className={lineColor(line.type)}>
                {line.content}
              </div>
            ))}
          </div>
          <div className="flex items-center gap-2 px-3 py-1.5 border-t border-[#2a2520]">
            <span className="text-[#C15F3C] font-mono text-xs">$</span>
            <input
              type="text"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleCommand(input)}
              className="flex-1 bg-transparent text-xs text-[#e8e6e3] outline-none font-mono"
              placeholder="输入命令... (help 查看帮助)"
            />
          </div>
        </>
      ) : activeTab === 'files' ? (
        <div className="flex-1 overflow-y-auto p-3 font-mono text-xs">
          {workspaceInfo && (
            <div className="mb-3 pb-2 border-b border-[#2a2520] text-[#6a6560]">
              <div>路径: {workspaceInfo.root}</div>
              <div>{workspaceInfo.script_count} 个脚本 · {workspaceInfo.total_executions} 次执行</div>
            </div>
          )}
          {workspaceFiles.length === 0 ? (
            <div className="text-[#6a6560]">工作区为空，等待大脑创建脚本...</div>
          ) : (
            workspaceFiles.map(f => (
              <div key={f.name} className="flex items-center gap-2 py-1.5 px-2 rounded hover:bg-[#1a1815] group">
                <FileCode className="w-3 h-3 text-[#C15F3C] shrink-0" />
                <span className="text-[#e8e6e3] flex-1 truncate">{f.name}</span>
                <span className="text-[#6a6560] text-[10px]">{f.script_type}</span>
                <span className="text-[#6a6560] text-[10px]">{f.size}b</span>
              </div>
            ))
          )}
        </div>
      ) : activeTab === 'executions' ? (
        <div className="flex-1 overflow-y-auto p-3 font-mono text-xs">
          {workspaceExecs.length === 0 ? (
            <div className="text-[#6a6560]">暂无执行记录...</div>
          ) : (
            workspaceExecs.map((ex, i) => (
              <div key={i} className={`py-2 px-2 rounded mb-1 ${ex.success ? 'bg-[#1a1815]' : 'bg-red-900/10'}`}>
                <div className="flex items-center gap-2 mb-1">
                  <Play className={`w-3 h-3 ${ex.success ? 'text-green-400' : 'text-red-400'} fill-current`} />
                  <span className="text-[#e8e6e3] font-medium">{ex.script_name}</span>
                  <span className={`text-[10px] ml-auto ${ex.success ? 'text-green-400' : 'text-red-400'}`}>
                    exit={ex.exit_code}
                  </span>
                  <span className="text-[#6a6560] text-[10px]">{ex.duration_ms}ms</span>
                </div>
                {ex.output_preview && (
                  <pre className="text-[#8a8580] text-[11px] whitespace-pre-wrap pl-5 max-h-16 overflow-hidden">
                    {ex.output_preview}
                  </pre>
                )}
              </div>
            ))
          )}
        </div>
      ) : activeTab === 'executor' ? (
        <div className="flex-1 overflow-y-auto p-3 font-mono text-xs">
          {executorStatus ? (
            <>
              {/* 状态概览 */}
              <div className="mb-3 pb-2 border-b border-[#2a2520]">
                <div className="flex items-center gap-3 mb-1">
                  <Zap className={`w-3 h-3 ${executorStatus.running ? 'text-green-400' : 'text-[#6a6560]'}`} />
                  <span className={executorStatus.running ? 'text-green-400' : 'text-[#6a6560]'}>
                    {executorStatus.running ? '运行中' : '已停止'}
                  </span>
                  <span className="text-[#6a6560] ml-auto">
                    队列: {executorStatus.queue_size} | 活跃: {executorStatus.active_count}
                  </span>
                </div>
                <div className="text-[#6a6560] text-[10px] flex gap-4 mt-1">
                  <span>完成: {executorStatus.memory.total_completed}</span>
                  <span>失败: {executorStatus.memory.total_failed}</span>
                  <span>模式: {executorStatus.memory.patterns_learned}</span>
                  <span>文件: {executorStatus.memory.known_files}</span>
                </div>
              </div>

              {/* 活跃任务 */}
              {executorStatus.active_tasks.length > 0 && (
                <div className="mb-3">
                  <div className="text-purple-400 text-[10px] mb-1">活跃任务</div>
                  {executorStatus.active_tasks.map(t => (
                    <div key={t.id} className="py-1.5 px-2 rounded bg-purple-900/10 mb-1 animate-pulse">
                      <div className="flex items-center gap-2">
                        <Bot className="w-3 h-3 text-purple-400" />
                        <span className="text-[#e8e6e3] text-[11px] truncate">{t.description}</span>
                        <span className="text-[#6a6560] text-[10px] ml-auto">{t.status}</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* 历史任务 */}
              <div>
                <div className="text-[#8a8580] text-[10px] mb-1">最近任务</div>
                {executorStatus.recent_tasks.length === 0 ? (
                  <div className="text-[#6a6560]">暂无任务记录...</div>
                ) : (
                  executorStatus.recent_tasks.slice().reverse().map((t, i) => (
                    <div key={`${t.id}-${i}`} className={`py-1.5 px-2 rounded mb-1 ${t.success ? 'bg-[#1a1815]' : 'bg-red-900/10'}`}>
                      <div className="flex items-center gap-2">
                        <Play className={`w-2.5 h-2.5 ${t.success ? 'text-green-400' : 'text-red-400'} fill-current`} />
                        <span className="text-[#e8e6e3] text-[11px] truncate flex-1">{t.description?.slice(0, 50)}</span>
                        <span className="text-[#6a6560] text-[10px]">{t.duration_ms || 0}ms</span>
                      </div>
                      {t.output_preview && (
                        <pre className="text-[#6a6560] text-[10px] whitespace-pre-wrap pl-4 max-h-12 overflow-hidden mt-0.5">
                          {t.output_preview}
                        </pre>
                      )}
                    </div>
                  ))
                )}
              </div>
            </>
          ) : (
            <div className="text-[#6a6560]">加载执行器状态...</div>
          )}
        </div>
      ) : null}
    </div>
  );
}
