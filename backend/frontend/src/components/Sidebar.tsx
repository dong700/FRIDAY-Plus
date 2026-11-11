import { useState, useEffect, useCallback, useRef } from 'react';
import { MessageSquare, Brain, Database, Settings, Zap, Plus, CheckCircle2, Clock, AlertCircle, Trash2, Package, Bot, ChevronRight, ChevronDown, FileCode, Radio } from 'lucide-react';
import { useAppStore } from '../stores/appStore';
import type { ExecutorStatus } from '../types';

/**
 * 清理执行器输出的预览文本
 * - 解码 Unicode 转义 (\uXXXX → 实际字符)
 * - 去除 Python repr 产物（引号、三引号等）
 * - 去除控制字符
 * - 截断到合理长度
 */
function cleanOutputPreview(raw: string | undefined): string {
  if (!raw) return '';
  let text = raw;

  // 解码 \UXXXXXXXX（8位大写Unicode转义，如 \U4EFB\U52A1 → 任务）
  text = text.replace(/\\U([0-9a-fA-F]{8})/g, (_, hex) => {
    const cp = parseInt(hex, 16);
    if (cp > 0xFFFF) {
      const hi = Math.floor((cp - 0x10000) / 0x400) + 0xD800;
      const lo = ((cp - 0x10000) % 0x400) + 0xDC00;
      return String.fromCharCode(hi, lo);
    }
    return String.fromCharCode(cp);
  });
  // 解码 \uXXXX（4位转义）
  text = text.replace(/\\u([0-9a-fA-F]{4})/g, (_, hex) => String.fromCharCode(parseInt(hex, 16)));
  // 去除 \xXX 转义
  text = text.replace(/\\x[0-9a-fA-F]{2}/g, '');
  // 去除 Python 三引号包裹
  text = text.replace(/^['"]{3}(.*)['"]{3}$/s, '$1');
  text = text.replace(/^['"`](.*)['"`]$/s, '$1');
  // 去除控制字符（保留换行）
  text = text.replace(/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/g, '');
  // 清理多余空白行
  text = text.replace(/\n{3,}/g, '\n\n');
  // 截断
  if (text.length > 250) {
    text = text.slice(0, 247) + '...';
  }
  return text.trim();
}

/** 清理任务描述中的Unicode转义 */
function cleanDescription(raw: string | undefined): string {
  if (!raw) return '';
  let text = raw;
  text = text.replace(/\\U([0-9a-fA-F]{8})/g, (_, hex) => {
    const cp = parseInt(hex, 16);
    if (cp > 0xFFFF) {
      const hi = Math.floor((cp - 0x10000) / 0x400) + 0xD800;
      const lo = ((cp - 0x10000) % 0x400) + 0xDC00;
      return String.fromCharCode(hi, lo);
    }
    return String.fromCharCode(cp);
  });
  text = text.replace(/\\u([0-9a-fA-F]{4})/g, (_, hex) => String.fromCharCode(parseInt(hex, 16)));
  text = text.replace(/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/g, '');
  return text.trim();
}

const navItems = [
  { id: 'chat' as const, icon: MessageSquare, label: '对话' },
  { id: 'friday' as const, icon: Radio, label: '星期五' },
  { id: 'brain' as const, icon: Brain, label: '大脑' },
  { id: 'memory' as const, icon: Database, label: '记忆' },
  { id: 'skills' as const, icon: Package, label: '技能' },
  { id: 'settings' as const, icon: Settings, label: '设置' },
];

interface SidebarProps {
  onTaskSelect?: (task: ExecutorStatus['recent_tasks'][0]) => void;
  onTaskDeleted?: (id: string) => void;
  selectedTaskId?: string | null;
}

export function Sidebar({ onTaskSelect, onTaskDeleted, selectedTaskId }: SidebarProps) {
  const { currentPage, setCurrentPage, brainStatus, connected } = useAppStore();
  const [showNewTask, setShowNewTask] = useState(false);
  const [newTaskDesc, setNewTaskDesc] = useState('');
  const [executorStatus, setExecutorStatus] = useState<ExecutorStatus | null>(null);
  const [executorTasks, setExecutorTasks] = useState<ExecutorStatus['recent_tasks']>([]);
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());
  // 追踪正在删除的任务ID，防止刷新后重新出现导致崩溃
  const deletingIds = useRef<Set<string>>(new Set());

  // 定期加载执行器数据
  const loadExecutorData = useCallback(() => {
    fetch('/api/executor').then(r => r.json()).then(setExecutorStatus).catch(() => {});
    fetch('/api/executor/history?limit=15').then(r => r.json()).then(data => {
      const tasks: ExecutorStatus['recent_tasks'] = data.tasks || [];
      // 过滤掉已删除的任务（乐观删除后后端可能还没同步）
      const filtered = tasks.filter((t: any) => t?.id && !deletingIds.current.has(t.id));
      setExecutorTasks(filtered);
      // 自动展开活跃任务
      if (filtered.length > 0) {
        const activeIds = filtered.filter((t: any) => t.status === 'executing' || t.status === 'generating').map((t: any) => t.id);
        if (activeIds.length > 0) {
          setExpandedIds(prev => new Set([...prev, ...activeIds]));
        }
      }
    }).catch(() => {});
  }, []);

  useEffect(() => {
    loadExecutorData();
    const interval = setInterval(loadExecutorData, 5000);
    return () => clearInterval(interval);
  }, [loadExecutorData]);

  const mode = brainStatus?.mode || 'idle';
  const modeColor = mode === 'active' ? '#10b981' : mode === 'reflect' ? '#3b82f6' : '#71717a';
  const modeLabel = mode === 'active' ? 'ACTIVE' : mode === 'reflect' ? 'REFLECT' : 'IDLE';

  const handleCreateTask = () => {
    if (!newTaskDesc.trim()) return;
    fetch('/api/executor/tasks', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ description: newTaskDesc.trim(), priority: 'normal' }),
    }).then(() => {
      setNewTaskDesc('');
      setShowNewTask(false);
      loadExecutorData();
    });
  };

  const toggleExpand = (id: string) => {
    setExpandedIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  const handleClickTask = (task: ExecutorStatus['recent_tasks'][0]) => {
    if (onTaskSelect) onTaskSelect(task);
  };

  const handleDeleteTask = (e: React.MouseEvent, id: string, name?: string) => {
    e.stopPropagation();
    e.preventDefault();

    // ★ 必须先确认，确认之前绝对不修改任何状态
    const confirmed = window.confirm(`确定删除任务 #${id} ${name || ''} 吗？`);
    if (!confirmed) {
      return;
    }

    // 1. 先标记为删除中（防止5秒刷新拉回旧数据）
    deletingIds.current.add(id);

    // 2. 先通知父组件关闭面板（如果当前选中了这个任务）
    if (selectedTaskId === id && onTaskDeleted) {
      onTaskDeleted(id);
    }

    // 3. 乐观更新：立即从本地状态移除
    setExecutorTasks(prev => (prev || []).filter(t => t && t.id !== id));
    setExpandedIds(prev => { const next = new Set(prev); next.delete(id); return next; });

    // 4. 调用后端 API 真正删除（从 executor_state.json 中移除 + 删除脚本文件）
    fetch(`/api/executor/tasks/${id}`, { method: 'DELETE' })
      .then(r => r.json())
      .then(data => {
        if (!data.success) {
          console.warn('[Sidebar] delete API returned:', data);
        }
      })
      .catch(err => {
        console.error('[Sidebar] delete API error:', err);
      });
  };

  // 执行器状态图标
  const execStatusConfig: Record<string, { color: string; icon: typeof Clock }> = {
    pending: { color: '#eab308', icon: Clock },
    generating: { color: '#8b5cf6', icon: Bot },
    executing: { color: '#3b82f6', icon: Zap },
    completed: { color: '#10b981', icon: CheckCircle2 },
    failed: { color: '#ef4444', icon: AlertCircle },
  };

  return (
    <div className="w-56 h-screen bg-[#1a1815] border-r border-[#2a2520] flex flex-col">
      {/* Logo */}
      <div className="p-4 border-b border-[#2a2520]">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-[#C15F3C] to-[#A14A2F] flex items-center justify-center">
            <Zap className="w-4 h-4 text-white" />
          </div>
          <div>
            <div className="text-sm font-medium text-[#e8e6e3]">Active AI</div>
            <div className="flex items-center gap-1.5 mt-0.5">
              <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: connected ? '#10b981' : '#ef4444' }} />
              <span className="text-[10px] text-[#8a8580]">{connected ? '已连接' : '未连接'}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav className="p-2 space-y-0.5">
        {navItems.map(({ id, icon: Icon, label }) => (
          <button key={id} onClick={() => setCurrentPage(id)}
            className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm transition-all duration-150
              ${currentPage === id ? 'bg-[#2a2520] text-[#e8e6e3]' : 'text-[#8a8580] hover:text-[#c8c6c3] hover:bg-[#221f1a]'}`}>
            <Icon className="w-4 h-4" />{label}
          </button>
        ))}
      </nav>

      {/* Task Section */}
      <div className="flex-1 min-h-0 border-t border-[#2a2520] flex flex-col">
        {/* Header + Add */}
        <div className="flex items-center justify-between px-3 py-2">
          <span className="text-[10px] text-[#8a8580] uppercase tracking-wider">任务</span>
          <button onClick={() => setShowNewTask(!showNewTask)} className="text-[#8a8580] hover:text-[#C15F3C]">
            <Plus className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* New task input */}
        {showNewTask && (
          <div className="px-3 pb-2 space-y-2">
            <input type="text" value={newTaskDesc} onChange={(e) => setNewTaskDesc(e.target.value)}
                   onKeyDown={(e) => e.key === 'Enter' && handleCreateTask()}
                   placeholder="让大脑执行任务..."
                   className="w-full bg-[#221f1a] rounded border border-[#2a2520] px-2 py-1 text-xs text-[#e8e6e3] outline-none focus:border-[#C15F3C]/50" />
            <button onClick={handleCreateTask}
                    className="w-full px-2 py-1 rounded text-[10px] bg-[#C15F3C] text-white hover:bg-[#A14A2F]">
              提交给执行器
            </button>
          </div>
        )}

        {/* Executor status bar */}
        {executorStatus && (
          <div className="px-3 pb-1 pt-1 space-y-1">
            <div className="flex items-center justify-between text-[9px] text-[#5a5550]">
              <span>{executorStatus.running ? <span className="text-green-400">运行中</span> : '已停止'}</span>
              <span>队列:{executorStatus.queue_size}</span>
              <span>活跃:{executorStatus.active_count}</span>
            </div>
            <div className="flex items-center gap-2 text-[9px] text-[#5a5550]">
              <span>{executorStatus.memory?.total_completed || 0} 完成</span>
              <span>{executorStatus.memory?.total_failed || 0} 失败</span>
              <span className="ml-auto">{executorStatus.memory?.patterns_learned || 0} 模式</span>
            </div>
          </div>
        )}

        {/* Task list (collapsible + clickable) */}
        <div className="flex-1 overflow-y-auto px-2 space-y-0.5">
          {/* Active tasks (always expanded) */}
          {executorStatus?.active_tasks && executorStatus.active_tasks.length > 0 &&
            executorStatus.active_tasks.filter(t => t?.id && !deletingIds.current.has(t.id)).map(task => {
              const sc = execStatusConfig[task.status] || execStatusConfig.pending;
              return (
                <div key={`act-${task.id}`}
                     className={`rounded transition-colors cursor-pointer group ${
                       selectedTaskId === task.id ? 'bg-purple-900/20 ring-1 ring-purple-500/30' : 'hover:bg-[#221f1a]'
                     }`}
                     onClick={() => handleClickTask({ ...task, success: false })}>
                  <div className="flex items-start gap-1.5 px-2 py-1.5 animate-pulse">
                    <sc.icon className="w-3 h-3 mt-0.5 shrink-0" style={{ color: sc.color }} />
                    <div className="flex-1 min-w-0">
                      <div className="text-[11px] text-purple-300 truncate">{cleanDescription(task.description)}</div>
                      <div className="text-[9px] text-[#5a5550]">{task.status}...</div>
                    </div>
                  </div>
                </div>
              );
            })
          }

          {/* History tasks (click to expand/select) */}
          {(!Array.isArray(executorTasks) || executorTasks.length === 0) && (!executorStatus?.active_tasks || executorStatus.active_tasks.length === 0) ? (
            <div className="text-[10px] text-[#5a5550] px-2 py-2">
              {executorStatus?.running ? '等待大脑下发任务...' : '执行器未启动'}
            </div>
          ) : (
            [...(executorTasks || [])].reverse().filter(t => t?.id && !deletingIds.current.has(t.id)).map((task, i) => {
              const isSelected = selectedTaskId === task.id;
              const isExpanded = expandedIds.has(task.id);
              const iconColor = task.success ? '#10b981' : (task.status === 'failed' ? '#ef4444' : '#5a5550');
              const StatusIcon = task.success ? CheckCircle2 : (task.status === 'failed' ? AlertCircle : Clock);

              return (
                <div key={`${task.id}-${i}`}>
                  {/* Row: click to select/expand */}
                  <div className={`group flex items-start gap-1.5 px-2 py-1.5 rounded transition-colors cursor-pointer ${
                    isSelected ? 'bg-[#2a2520] ring-1 ring-[#C15F3C]/30' : 'hover:bg-[#221f1a]'
                  }`}
                       onClick={() => toggleExpand(task.id)}
                       onDoubleClick={() => handleClickTask(task)}>
                    <button onClick={(e) => { e.stopPropagation(); toggleExpand(task.id); }}
                            className="shrink-0 mt-0.5 text-[#5a5550] hover:text-[#8a8580]">
                      {isExpanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                    </button>
                    <StatusIcon className="w-3 h-3 mt-0.5 shrink-0" style={{ color: iconColor }} />
                    <div className="flex-1 min-w-0">
                      <div className={`text-[11px] truncate ${isSelected ? 'text-[#C15F3C]' : 'text-[#c8c6c3]'}`}>
                        {cleanDescription(task.description)}
                      </div>
                      <div className="text-[9px] text-[#5a5550] flex items-center gap-2">
                        {task.duration_ms ? `${task.duration_ms}ms` : '-'}
                        {task.script_name && (
                          <span className="truncate max-w-[60px] flex items-center gap-0.5">
                            <FileCode className="w-2.5 h-2.5" />{task.script_name}
                          </span>
                        )}
                      </div>
                    </div>
                    {/* Delete button (hover) */}
                    <button onClick={(e) => handleDeleteTask(e, task.id, task.script_name)}
                            className="shrink-0 opacity-0 group-hover:opacity-100 p-0.5 text-[#5a5550] hover:text-red-400 transition-opacity">
                      <Trash2 className="w-3 h-3" />
                    </button>
                  </div>

                  {/* Expanded content preview */}
                  {isExpanded && task.output_preview && (
                    <div className="ml-7 mr-2 mb-1 px-2 py-1 rounded bg-[#12100e] text-[9px] text-[#5a5550]"
                         onClick={(e) => { e.stopPropagation(); handleClickTask(task); }}>
                      <pre className="whitespace-pre-wrap max-h-16 overflow-hidden leading-relaxed">{cleanOutputPreview(task.output_preview)}</pre>
                      <button className="mt-1 text-[#C15F3C] hover:underline">查看代码 &rarr;</button>
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>
      </div>

      {/* Bottom: Mode + Motivation bars */}
      <div className="p-3 border-t border-[#2a2520] space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-[10px] text-[#8a8580] uppercase tracking-wider">模式</span>
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full animate-pulse" style={{ backgroundColor: modeColor }} />
            <span className="text-xs font-mono" style={{ color: modeColor }}>{modeLabel}</span>
          </div>
        </div>

        {brainStatus && (
          <>
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-[10px]">
                <span className="text-[#8a8580]">好奇心</span>
                <span className="text-[#C15F3C] font-mono">{Math.round(brainStatus.curiosity * 100)}%</span>
              </div>
              <div className="h-1 bg-[#2a2520] rounded-full overflow-hidden">
                <div className="h-full bg-[#C15F3C] rounded-full transition-all duration-1000"
                     style={{ width: `${brainStatus.curiosity * 100}%` }} />
              </div>
            </div>
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-[10px]">
                <span className="text-[#8a8580]">探索欲</span>
                <span className="text-[#10b981] font-mono">{Math.round(brainStatus.exploration * 100)}%</span>
              </div>
              <div className="h-1 bg-[#2a2520] rounded-full overflow-hidden">
                <div className="h-full bg-[#10b981] rounded-full transition-all duration-1000"
                     style={{ width: `${brainStatus.exploration * 100}%` }} />
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
