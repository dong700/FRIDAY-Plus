import { useState, useEffect, useCallback } from 'react';
import { Sidebar } from './components/Sidebar';
import { ChatPanel } from './components/ChatPanel';
import { BrainPanel } from './components/BrainPanel';
import { MemoryPanel } from './components/MemoryPanel';
import { SettingsPanel } from './components/SettingsPanel';
import { SkillPanel } from './components/SkillPanel';
import { StatusBar } from './components/StatusBar';
import { TerminalPanel } from './components/TerminalPanel';
import { TaskDetailPanel } from './components/TaskDetailPanel';
import { FridayPanel } from './components/FridayPanel';
import { PetMode } from './components/PetMode';
import { useAppStore } from './stores/appStore';
import { useWebSocket } from './hooks/useWebSocket';
import type { ExecutorStatus } from './types';

function AppContent() {
  const { currentPage, loadChatHistory } = useAppStore();
  const [showTerminal, setShowTerminal] = useState(false);
  const [selectedTask, setSelectedTask] = useState<ExecutorStatus['recent_tasks'][0] | null>(null);
  useWebSocket();

  useEffect(() => {
    loadChatHistory();
  }, []);

  // 任务操作回调
  const handleRunCode = useCallback(async (code: string) => {
    const res = await fetch('/api/workspace/execute', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content: code, timeout: 60 }),
    });
    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody?.detail || `HTTP ${res.status}`);
    }
    return res.json();
  }, []);

  const handleSaveCode = useCallback(async (name: string, code: string) => {
    const res = await fetch('/api/workspace/scripts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, content: code, script_type: 'python' }),
    });
    return res.json();
  }, []);

  const handleDeleteTask = useCallback(async (id: string) => {
    setSelectedTask(null);
    // 调用后端 API 真正删除任务记录
    try {
      await fetch(`/api/executor/tasks/${id}`, { method: 'DELETE' });
    } catch {}
  }, []);

  const renderPage = () => {
    switch (currentPage) {
      case 'chat': return <ChatPanel />;
      case 'friday': return <FridayPanel />;
      case 'brain': return <BrainPanel />;
      case 'memory': return <MemoryPanel />;
      case 'skills': return <SkillPanel />;
      case 'settings': return <SettingsPanel />;
    }
  };

  return (
    <div className="flex h-screen bg-[#1a1815] text-[#e8e6e3]">
      <Sidebar
        onTaskSelect={setSelectedTask}
        onTaskDeleted={(id) => { if (selectedTask?.id === id) setSelectedTask(null); }}
        selectedTaskId={selectedTask?.id || null}
      />
      <div className="flex-1 flex flex-col overflow-hidden">
        <div className="flex-1 flex overflow-hidden">
          {/* Main content area */}
          <div className="flex-1 flex flex-col overflow-hidden">
            {renderPage()}
            {showTerminal && <TerminalPanel onClose={() => setShowTerminal(false)} />}
          </div>
          {/* Task detail panel (right side overlay) */}
          {selectedTask && (
            <TaskDetailPanel
              task={{
                id: selectedTask.id,
                description: selectedTask.description,
                status: selectedTask.status || 'completed',
                success: selectedTask.success ?? true,
                script_name: selectedTask.script_name || '',
                script_code: '',  // will be loaded by TaskDetailPanel
                stdout: selectedTask.output_preview || '',
                stderr: '',
                duration_ms: selectedTask.duration_ms || 0,
                executed_at: '',
                created_at: '',
              }}
              onClose={() => setSelectedTask(null)}
              onRun={handleRunCode}
              onSave={handleSaveCode}
              onDelete={handleDeleteTask}
            />
          )}
        </div>
        <StatusBar onToggleTerminal={() => setShowTerminal(!showTerminal)} />
      </div>
    </div>
  );
}

function App() {
  // 桌宠模式：run_friday.py --pet 会加载 /?pet=1，只渲染角色、背景透明
  const isPet = typeof window !== 'undefined'
    && new URLSearchParams(window.location.search).get('pet') === '1';
  if (isPet) return <PetMode />;
  return <AppContent />;
}

export default App;
