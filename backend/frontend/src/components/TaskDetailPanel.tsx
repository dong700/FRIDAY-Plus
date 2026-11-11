import { useState, useEffect, useCallback, useRef, lazy, Suspense } from 'react';
import { X, Play, Save, Trash2, Copy, CheckCircle2, AlertCircle, ChevronDown, ChevronRight, Clock, Bot, Minimize2, Maximize2, GripVertical, Terminal } from 'lucide-react';

// 懒加载 Monaco Editor（不阻塞首屏）
const Editor = lazy(() => import('@monaco-editor/react').then(m => ({ default: m.Editor })));

/** 清理Unicode转义（与Sidebar共用） */
function cleanDesc(raw: string): string {
  if (!raw) return raw;
  let t = raw;
  t = t.replace(/\\U([0-9a-fA-F]{8})/g, (_, h) => { const c = parseInt(h,16); return c > 0xFFFF ? String.fromCharCode(Math.floor((c-0x10000)/0x400)+0xD800, ((c-0x10000)%0x400)+0xDC00) : String.fromCharCode(c); });
  t = t.replace(/\\u([0-9a-fA-F]{4})/g, (_, h) => String.fromCharCode(parseInt(h,16)));
  return t;
}

interface TaskDetail {
  id: string;
  description: string;
  status: string;
  success: boolean;
  script_name: string;
  script_code: string;
  stdout: string;
  stderr: string;
  duration_ms: number;
  executed_at: string;
  created_at: string;
}

interface TaskDetailPanelProps {
  task: TaskDetail | null;
  onClose: () => void;
  onRun: (code: string) => Promise<{ stdout?: string; stderr?: string; success?: boolean; exit_code?: number }>;
  onSave: (name: string, code: string) => Promise<void>;
  onDelete: (id: string) => void;
}

// Monaco Loading 骨架屏
function EditorSkeleton() {
  return (
    <div className="flex-1 flex items-center justify-center bg-[#0d0c0a]">
      <div className="text-center space-y-3">
        <div className="w-8 h-8 border-2 border-[#C15F3C]/30 border-t-[#C15F3C] rounded-full animate-spin mx-auto" />
        <div className="text-xs text-[#5a5550]">Loading Editor...</div>
        {/* 模拟代码行骨架 */}
        <div className="space-y-1.5 px-8">
          {[...Array(8)].map((_, i) => (
            <div key={i} className="h-3 bg-[#1a1815] rounded animate-pulse"
                 style={{ width: `${60 + Math.random() * 40}%`, animationDelay: `${i * 100}ms` }} />
          ))}
        </div>
      </div>
    </div>
  );
}

export function TaskDetailPanel({ task, onClose, onRun, onSave, onDelete }: TaskDetailPanelProps) {
  const [code, setCode] = useState('');
  const [output, setOutput] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [showMeta, setShowMeta] = useState(true);
  const [copied, setCopied] = useState(false);
  // 可折叠/展开
  const [isMinimized, setIsMinimized] = useState(false);
  // 可拖拽宽度
  const [width, setWidth] = useState(420);
  const [isDragging, setIsDragging] = useState(false);
  // Monaco 加载状态
  const [editorReady, setEditorReady] = useState(false);
  // 终端面板：显示/隐藏 + 自动滚动
  const [showTerminal, setShowTerminal] = useState(true);
  const terminalRef = useRef<HTMLPreElement>(null);
  // 终端输出历史（保留多次运行结果）
  const [terminalHistory, setTerminalHistory] = useState<string[]>([]);

  // 切换任务时重置
  useEffect(() => {
    if (task?.script_code) {
      setCode(task.script_code);
      setOutput(task.stdout || task.stderr || '');
      setEditorReady(false);
      // 如果有历史输出，显示在终端
      const hist = [];
      if (task.stdout) hist.push(task.stdout);
      if (task.stderr) hist.push(task.stderr);
      setTerminalHistory(hist.length > 0 ? hist : []);
    } else {
      setCode('');
      setOutput('');
      setTerminalHistory([]);
    }
    setIsRunning(false);
    setIsMinimized(false);
    setShowTerminal(true);
  }, [task?.id]);

  // 加载脚本内容
  useEffect(() => {
    if (!task?.script_name || task.script_code) return;
    fetch(`/api/workspace/scripts/${encodeURIComponent(task.script_name)}`)
      .then(r => r.json())
      .then(data => { if (data.success && data.content) setCode(data.content); })
      .catch(() => {});
  }, [task?.script_name]);

  const handleRun = async () => {
    if (!code.trim() || isRunning) return;
    setIsRunning(true);
    setShowTerminal(true);

    try {
      const result = await onRun(code);
      console.log('[TaskDetail] execute result:', JSON.stringify(result)?.slice(0, 800));

      // 直接显示原始终端输出，不做任何包装处理
      // 注意：stdout 可能为空字符串 ""（falsy），必须用 typeof 判断
      let rawOutput = '';
      if (typeof result?.stdout === 'string') {
        rawOutput = result.stdout;
      }
      if (typeof result?.stderr === 'string' && result.stderr) {
        if (rawOutput) rawOutput += '\n--- stderr ---\n';
        rawOutput += result.stderr;
      }
      console.log('[TaskDetail] final output length:', rawOutput.length, 'first200:', rawOutput.slice(0, 200));
      setOutput(rawOutput || '(无输出)');

      // 追加到终端历史
      setTerminalHistory(prev => [...prev,
        `\x1b[36m--- ${new Date().toLocaleTimeString()} ---\x1b[0m`,
        rawOutput || '(无输出)'
      ]);
    } catch (e: any) {
      console.error('[TaskDetail] execute error:', e);
      const errMsg = `错误: ${e?.message || e}`;
      setOutput(errMsg);
      setTerminalHistory(prev => [...prev, errMsg]);
    } finally {
      setIsRunning(false);
    }
  };

  const handleSave = async () => {
    if (!task?.script_name || !code.trim()) return;
    setIsSaving(true);
    try { await onSave(task.script_name, code); }
    finally { setIsSaving(false); }
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };

  const handleDelete = () => {
    if (!task) return;
    const confirmed = window.confirm(`确定删除任务 "${cleanDesc(task.description)}" 吗？`);
    if (!confirmed) return;
    // 先关闭面板，再通知删除（避免渲染时 task 已不存在）
    onClose();
    onDelete(task.id);
  };

  // 拖拽调整宽度
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
    const startX = e.clientX;
    const startWidth = width;

    const onMouseMove = (moveEvent: MouseEvent) => {
      const delta = moveEvent.clientX - startX;
      // 左边缘手柄：向右拉鼠标→面板变窄(delta>0, width减小)→左边缘跟随鼠标右移
      // 向左拉鼠标→面板变宽(delta<0, width增大)→左边缘跟随鼠标左移
      const newWidth = Math.max(280, Math.min(800, startWidth - delta));
      setWidth(newWidth);
    };
    const onMouseUp = () => {
      setIsDragging(false);
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
    };
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  }, [width]);

  const statusColor = task?.success ? 'text-green-400' :
                       task?.status === 'failed' ? 'text-red-400' :
                       task?.status === 'executing' ? 'text-blue-400 animate-pulse' : 'text-[#8a8580]';
  const statusIcon = task?.success ? <CheckCircle2 className="w-3 h-3" /> :
                     task?.status === 'failed' ? <AlertCircle className="w-3 h-3" /> :
                     <Bot className="w-3 h-3" />;

  if (!task) return null;

  // 最小化状态：只显示一行标题栏
  if (isMinimized) {
    return (
      <div className="shrink-0 border-l border-[#2a2520] flex flex-col" style={{ width }}>
        {/* Drag handle */}
        <div className="absolute left-0 top-0 bottom-0 w-1 cursor-col-resize z-10"
             onMouseDown={handleMouseDown} />
        <div className="flex items-center justify-between px-3 py-1.5 bg-[#1a1815] border-b border-[#2a2520]"
             onDoubleClick={() => setIsMinimized(false)}>
          <div className="flex items-center gap-2 min-w-0">
            <span style={{ color: statusColor }}>{statusIcon}</span>
            <span className="text-xs text-[#e8e6e3] truncate">{cleanDesc(task.description)}</span>
          </div>
          <div className="flex items-center gap-1 shrink-0 ml-2">
            <button onClick={() => setIsMinimized(false)} title="展开"
                    className="p-1 rounded hover:bg-[#2a2520] text-[#8a8580] hover:text-[#c8c6c3]">
              <Maximize2 className="w-3 h-3" />
            </button>
            <button onClick={onClose} title="关闭"
                    className="p-1 rounded hover:bg-[#2a2520] text-[#8a8580] hover:text-[#c8c6c3]">
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full bg-[#141210] flex flex-col border-l border-[#2a2520] relative shrink-0" style={{ width }}>
      {/* 左侧拖拽条 */}
      <div className={`absolute left-0 top-0 bottom-0 w-1.5 cursor-col-resize z-20 transition-colors ${
        isDragging ? 'bg-[#C15F3C]' : 'hover:bg-[#C15F3C]/40'
      }`} onMouseDown={handleMouseDown}>
        <GripVertical className="w-full h-full text-[#3a3530] opacity-50 hover:opacity-100" />
      </div>

      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 bg-[#1a1815] border-b border-[#2a2520] shrink-0 pl-4">
        <div className="flex items-center gap-2 min-w-0 flex-1">
          <button onClick={() => setShowMeta(!showMeta)} className="shrink-0 text-[#8a8580] hover:text-[#c8c6c3]">
            {showMeta ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
          </button>
          <span style={{ color: statusColor }} className="shrink-0">{statusIcon}</span>
          <span className="text-xs font-medium text-[#e8e6e3] truncate">{cleanDesc(task.description)}</span>
        </div>
        <div className="flex items-center gap-1 ml-2 shrink-0">
          <button onClick={handleCopy} title="复制代码"
                  className="p-1 rounded hover:bg-[#2a2520] text-[#8a8580] hover:text-[#c8c6c3]">
            {copied ? <CheckCircle2 className="w-3.5 h-3.5 text-green-400" /> : <Copy className="w-3.5 h-3.5" />}
          </button>
          <button onClick={() => setIsMinimized(true)} title="最小化"
                  className="p-1 rounded hover:bg-[#2a2520] text-[#8a8580] hover:text-[#c8c6c3]">
            <Minimize2 className="w-3.5 h-3.5" />
          </button>
          <button onClick={onClose} title="关闭"
                  className="p-1 rounded hover:bg-[#2a2520] text-[#8a8580] hover:text-[#c8c6c3]">
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Meta Info */}
      {showMeta && (
        <div className="px-3 py-1.5 bg-[#12100e] border-b border-[#2a2520] shrink-0 flex items-center gap-4 text-[10px] text-[#6a6560] pl-4">
          <span>#{task.id}</span>
          <span>{task.status}</span>
          {task.duration_ms > 0 && <span>{task.duration_ms}ms</span>}
          {task.script_name && (
            <span className="truncate max-w-[120px]" title={task.script_name}>
              📄 {task.script_name}
            </span>
          )}
        </div>
      )}

      {/* Toolbar */}
      <div className="flex items-center justify-between px-3 py-1.5 bg-[#1a1815] border-b border-[#2a2520] shrink-0 pl-4">
        <div className="flex items-center gap-1">
          <button onClick={handleRun} disabled={isRunning}
                  className={`flex items-center gap-1 px-2 py-0.5 rounded text-[10px] transition-colors ${
                    isRunning ? 'bg-blue-900/30 text-blue-400 cursor-not-allowed' : 'bg-green-900/30 text-green-400 hover:bg-green-900/50'
                  }`}>
            <Play className={`w-3 h-3 ${isRunning ? 'animate-pulse' : ''} fill-current`} />
            {isRunning ? '运行中...' : '▶ 运行'}
          </button>
          <button onClick={handleSave} disabled={isSaving}
                  className={`flex items-center gap-1 px-2 py-0.5 rounded text-[10px] ${
                    isSaving ? 'bg-[#2a2520] text-[#6a6560]' : 'bg-[#2a2520] text-[#8a8580] hover:text-[#C15F3C]'
                  }`}>
            <Save className="w-3 h-3" /> 保存
          </button>
          <button onClick={handleDelete}
                  className="flex items-center gap-1 px-2 py-0.5 rounded text-[10px] bg-[#2a2520] text-[#8a8580] hover:text-red-400">
            <Trash2 className="w-3 h-3" /> 删除
          </button>
        </div>
        <span className="text-[9px] text-[#5a5550]">Python</span>
      </div>

      {/* Code Editor */}
      <div className="flex-1 min-h-0 relative pl-2">
        <Suspense fallback={<EditorSkeleton />}>
          <Editor
            height="100%"
            language="python"
            theme="vs-dark"
            value={code}
            onChange={(val) => val !== undefined && setCode(val)}
            options={{
              fontSize: 12,
              fontFamily: "'JetBrains Mono', 'Fira Code', Consolas, monospace",
              lineNumbers: 'on',
              minimap: { enabled: false },
              scrollBeyondLastLine: false,
              wordWrap: 'on',
              tabSize: 4,
              insertSpaces: true,
              automaticLayout: true,
              padding: { top: 8 },
              renderLineHighlight: 'line',
              cursorBlinking: 'smooth',
              smoothScrolling: true,
              bracketPairColorization: { enabled: true },
              suggestOnTriggerCharacters: true,
              quickSuggestions: true,
              folding: true,
              scrollbar: { verticalScrollbarSize: 6, horizontalScrollbarSize: 6 },
            }}
            onMount={() => setEditorReady(true)}
            loading={<EditorSkeleton />}
          />
        </Suspense>
      </div>

      {/* Terminal Output Panel */}
      <div className={`border-t border-[#2a2520] flex flex-col shrink-0 ${showTerminal ? 'min-h-[160px] max-h-[50%]' : ''}`}>
        {/* Terminal header */}
        <div className="flex items-center justify-between px-3 py-1 bg-[#0d0c0a] shrink-0 pl-4">
          <button onClick={() => setShowTerminal(!showTerminal)}
                  className="flex items-center gap-1.5 text-[10px] text-[#8a8580] hover:text-[#c8c6c3]">
            <Terminal className="w-3 h-3" />
            <span>终端输出</span>
            {isRunning && <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />}
            <ChevronDown className={`w-3 h-3 transition-transform ${showTerminal ? '' : '-rotate-90'}`} />
          </button>
          <div className="flex items-center gap-2">
            {terminalHistory.length > 0 && (
              <button onClick={() => setTerminalHistory([])}
                      className="text-[9px] text-[#5a5550] hover:text-[#8a8580]">
                清空
              </button>
            )}
            <span className="text-[9px] text-[#5a5550]">Ctrl+Enter 运行</span>
          </div>
        </div>
        {/* Terminal content */}
        {showTerminal && (
          <pre ref={terminalRef}
               className="flex-1 px-3 py-2 text-[11px] font-mono leading-relaxed overflow-y-auto whitespace-pre-wrap break-all bg-[#0a0908] pl-4"
               style={{ minHeight: '120px' }}>
            {terminalHistory.length > 0 ? terminalHistory.map((line, i) => (
              <span key={i} className="text-[#c8e6c8]">{line}{'\n'}</span>
            )) : (
              <span className="text-[#5a5550]">点击 ▶ 运行 或 Ctrl+Enter 执行代码，输出将显示在这里</span>
            )}
          </pre>
        )}
      </div>
    </div>
  );
}
