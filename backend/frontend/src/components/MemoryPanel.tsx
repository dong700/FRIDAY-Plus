import { useState } from 'react';
import { Search, Trash2, Tag, User, Bot, MessageSquare } from 'lucide-react';
import { useAppStore } from '../stores/appStore';
import { useWebSocket } from '../hooks/useWebSocket';

const roleConfig: Record<string, { label: string; color: string; icon: React.ReactNode }> = {
  user: { label: '用户', color: '#10b981', icon: <User className="w-3 h-3" /> },
  assistant: { label: 'AI', color: '#C15F3C', icon: <Bot className="w-3 h-3" /> },
};

export function MemoryPanel() {
  const [searchQuery, setSearchQuery] = useState('');
  const { memories } = useAppStore();
  const { send } = useWebSocket();

  const handleSearch = () => {
    if (searchQuery.trim()) {
      send('memory.search', { query: searchQuery.trim(), limit: 20 });
    }
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-[#1a1815]">
      {/* 搜索栏 */}
      <div className="p-4 border-b border-[#2a2520]">
        <div className="flex items-center gap-2 bg-[#221f1a] rounded-lg border border-[#2a2520] px-3 py-2 focus-within:border-[#C15F3C]/50 transition-colors">
          <Search className="w-4 h-4 text-[#8a8580] shrink-0" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
            placeholder="搜索对话记忆..."
            className="flex-1 bg-transparent text-sm text-[#e8e6e3] placeholder-[#5a5550] outline-none min-w-0"
          />
          <button onClick={handleSearch} className="text-xs text-[#C15F3C] hover:text-[#A14A2F] shrink-0 whitespace-nowrap">搜索</button>
        </div>
        {memories.length > 0 && (
          <div className="mt-2 text-[10px] text-[#5a5550]">
            找到 {memories.length} 条记录
          </div>
        )}
      </div>

      {/* 记忆列表 */}
      <div className="flex-1 overflow-y-auto p-4 space-y-2">
        {memories.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-[#5a5550] gap-2">
            <MessageSquare className="w-8 h-8 opacity-30" />
            <p className="text-sm">输入关键词搜索对话记录</p>
            <p className="text-xs opacity-60">搜索范围：本地聊天记录 + Viking云端</p>
          </div>
        ) : (
          memories.map((mem) => {
            const role = roleConfig[mem.role] || { label: mem.role || '未知', color: '#71717a', icon: null };
            return (
              <div key={mem.id + (mem.timestamp || '')} className="bg-[#221f1a] rounded-lg p-3 border border-[#2a2520] hover:border-[#3a3530] transition-colors group">
                {/* 角色标签 + 来源 */}
                <div className="flex items-center gap-2 mb-1.5">
                  <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-mono"
                    style={{ backgroundColor: role.color + '15', color: role.color }}>
                    {role.icon}
                    {role.label}
                  </span>
                  {(mem as any).source === 'viking' && (
                    <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-blue-500/10 text-blue-400">
                      云端
                    </span>
                  )}
                  {(mem as any).source === 'local_chat' && (
                    <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-[#2a2520] text-[#5a5550]">
                      本地
                    </span>
                  )}
                </div>

                {/* 内容 */}
                <p className="text-sm text-[#c8c6c3] leading-relaxed">{mem.text}</p>

                {/* 底部信息 */}
                <div className="flex items-center justify-between mt-2">
                  <span className="text-[10px] text-[#5a5550]">
                    {mem.timestamp ? new Date(mem.timestamp).toLocaleString('zh-CN') : ''}
                  </span>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
