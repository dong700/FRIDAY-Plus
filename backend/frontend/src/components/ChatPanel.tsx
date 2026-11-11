import { useState, useRef, useEffect } from 'react';
import { Send, Mic, Terminal, Image as ImageIcon, X } from 'lucide-react';
import { useAppStore } from '../stores/appStore';
import { useWebSocket } from '../hooks/useWebSocket';
import { AvatarWidget } from './AvatarWidget';
import type { ChatMessage } from '../types';

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === 'user';
  const isBrain = message.role === 'brain';
  const isSystem = message.role === 'system';

  if (isSystem) return null;

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-3`}>
      <div className={`max-w-[75%] ${isUser ? 'order-2' : ''}`}>
        <div className="flex items-center gap-2 mb-1">
          {!isUser && (
            <span className={`text-[10px] font-mono uppercase tracking-wider ${isBrain ? 'text-[#3b82f6]' : 'text-[#C15F3C]'}`}>
              {isBrain ? '🧠 大脑' : '🤖 AI'}
            </span>
          )}
          <span className="text-[10px] text-[#5a5550]">
            {new Date(message.timestamp).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })}
          </span>
        </div>
        {/* 图片显示 */}
        {message.image_url && (
          <img
            src={message.image_url}
            alt="用户上传"
            className="max-w-[200px] max-h-[200px] rounded-lg mb-1 border border-[#3a3530] object-cover"
          />
        )}
        {/* 文字内容 */}
        {message.content && (
          <div className={`px-3.5 py-2.5 rounded-xl text-sm leading-relaxed
            ${isUser
              ? 'bg-[#C15F3C] text-white rounded-br-sm'
              : isBrain
              ? 'bg-[#1e2a3a] text-[#93c5fd] border border-[#2a3a4a] rounded-bl-sm'
              : 'bg-[#2a2520] text-[#e8e6e3] border border-[#3a3530] rounded-bl-sm'
            }`}
          >
            {message.content}
          </div>
        )}
      </div>
    </div>
  );
}

let _msgCounter = 0;
function uid() { return `${Date.now()}_${++_msgCounter}`; }

export function ChatPanel() {
  const [input, setInput] = useState('');
  const [pendingImage, setPendingImage] = useState<string | null>(null); // base64 data URL
  const { messages, brainStatus } = useAppStore();
  const { send } = useWebSocket();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // ★ 图片上传处理
  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    // 限制 5MB
    if (file.size > 5 * 1024 * 1024) {
      alert('图片不能超过 5MB');
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      setPendingImage(reader.result as string);
    };
    reader.readAsDataURL(file);
    // 清空 input 以便重复选择同一文件
    e.target.value = '';
  };

  const handleSend = () => {
    const text = input.trim();
    if (!text && !pendingImage) return;

    // 构造用户消息（含图片预览）
    const msg: ChatMessage = {
      id: uid(),
      role: 'user',
      content: text,
      timestamp: new Date().toISOString(),
    };
    if (pendingImage) {
      msg.image_url = pendingImage;
      msg.has_image = true;
    }
    useAppStore.getState().addMessage(msg);

    // 发送到后端：图片走 base64；带上 client_msg_id，服务端广播回来时按 id 去重
    if (pendingImage) {
      // 提取纯 base64（去掉 data:image/xxx;base64, 前缀）
      const base64Data = pendingImage.split(',')[1] || pendingImage;
      send('chat.send', { content: text, image_base64: base64Data, client_msg_id: msg.id });
    } else {
      send('chat.send', { content: text, client_msg_id: msg.id });
    }

    setInput('');
    setPendingImage(null);
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    e.target.style.height = 'auto';
    e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px';
  };

  return (
    <div className="flex-1 flex h-full bg-[#1a1815]">
      {/* 左侧：聊天区 */}
      <div className="flex-1 flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-3 border-b border-[#2a2520]">
        <div className="flex items-center gap-3">
          <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
          <span className="text-sm font-medium text-[#c8c6c3]">星期五</span>
          <span className="text-xs text-[#5a5550]">在线</span>
        </div>
        <div className="flex items-center gap-2 text-[10px] text-[#5a5550]">
          <span className="px-2 py-0.5 rounded bg-[#2a2520]">
            好奇心 {brainStatus?.curiosity ? Math.round(brainStatus.curiosity * 100) : 0}%
          </span>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center">
            <p className="text-sm text-[#5a5550] mb-4">和 Friday 聊点什么吧</p>
            <div className="flex gap-3">
              {['你好', '今天有什么要告诉我的吗？', '你在想什么？'].map((suggestion) => (
                <button
                  key={suggestion}
                  onClick={() => {
                    setInput(suggestion);
                    inputRef.current?.focus();
                  }}
                  className="px-3 py-1.5 rounded-lg bg-[#2a2520] border border-[#3a3530] text-xs text-[#c8c6c3] hover:border-[#C15F3C]/50 transition-colors"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="px-4 pb-4">
        <div className="bg-[#221f1a] rounded-xl border border-[#3a3530] focus-within:border-[#C15F3C]/50 transition-colors">
          {/* 图片预览区 */}
          {pendingImage && (
            <div className="relative inline-block m-2">
              <img
                src={pendingImage}
                alt="待发送"
                className="max-w-[120px] max-h-[120px] rounded-lg border border-[#3a3530]"
              />
              <button
                onClick={() => setPendingImage(null)}
                className="absolute -top-1.5 -right-1.5 w-5 h-5 rounded-full bg-[#C15F3C] text-white flex items-center justify-center hover:bg-[#A14A2F] transition-colors"
              >
                <X className="w-3 h-3" />
              </button>
            </div>
          )}
          <textarea
            ref={inputRef}
            value={input}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            placeholder={pendingImage ? "描述这张图片... (可直接发送)" : "输入消息... (Enter发送, Shift+Enter换行)"}
            rows={1}
            className="w-full bg-transparent px-4 pt-3 pb-1 text-sm text-[#e8e6e3] placeholder-[#5a5550] outline-none resize-none"
          />
          <div className="flex items-center justify-between px-3 pb-2">
            <div className="flex items-center gap-1">
              {/* ★ 图片上传按钮 */}
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleImageSelect}
                className="hidden"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className={`p-1.5 rounded-md transition-colors ${pendingImage ? 'text-[#C15F3C] bg-[#2a2520]' : 'text-[#8a8580] hover:text-[#c8c6c3] hover:bg-[#2a2520]'}`}
                title="上传图片"
              >
                <ImageIcon className="w-4 h-4" />
              </button>
              <button className="p-1.5 rounded-md text-[#8a8580] hover:text-[#c8c6c3] hover:bg-[#2a2520] transition-colors">
                <Terminal className="w-4 h-4" />
              </button>
              <button className="p-1.5 rounded-md text-[#8a8580] hover:text-[#c8c6c3] hover:bg-[#2a2520] transition-colors">
                <Mic className="w-4 h-4" />
              </button>
            </div>
            <button
              onClick={handleSend}
              disabled={!input.trim() && !pendingImage}
              className="px-3 py-1 rounded-lg bg-[#C15F3C] text-white text-xs font-medium hover:bg-[#A14A2F] disabled:opacity-30 disabled:cursor-not-allowed transition-all"
            >
              发送
            </button>
          </div>
        </div>
      </div>
      </div>

      {/* 右侧：虚拟形象 */}
      <div className="w-[280px] border-l border-[#2a2520] bg-[#161411] flex flex-col items-center justify-center">
        <AvatarWidget width={260} height={380} showBubbles={false} />
      </div>
    </div>
  );
}
