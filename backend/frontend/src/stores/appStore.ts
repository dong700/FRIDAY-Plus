import { create } from 'zustand';
import type { BrainStatus, ChatMessage, MemoryItem, Task, Skill, AppConfig, ReasoningLog } from '../types';

interface AppState {
  connected: boolean;
  brainStatus: BrainStatus | null;
  messages: ChatMessage[];
  memories: MemoryItem[];
  tasks: Task[];
  skills: Skill[];
  config: AppConfig | null;
  currentPage: 'chat' | 'friday' | 'brain' | 'memory' | 'skills' | 'settings';
  reasoningLogs: ReasoningLog[];

  setConnected: (connected: boolean) => void;
  setBrainStatus: (status: BrainStatus) => void;
  addMessage: (message: ChatMessage) => void;
  setMessages: (messages: ChatMessage[]) => void;
  loadChatHistory: () => Promise<void>;
  setMemories: (memories: MemoryItem[]) => void;
  setTasks: (tasks: Task[]) => void;
  addTask: (task: Task) => void;
  updateTask: (id: string, updates: Partial<Task>) => void;
  removeTask: (id: string) => void;
  setSkills: (skills: Skill[]) => void;
  addSkill: (skill: Skill) => void;
  removeSkill: (id: string) => void;
  setConfig: (config: AppConfig) => void;
  setCurrentPage: (page: 'chat' | 'friday' | 'brain' | 'memory' | 'skills' | 'settings') => void;
  addReasoningLog: (log: ReasoningLog) => void;
}

export const useAppStore = create<AppState>((set, get) => ({
  connected: false,
  brainStatus: null,
  messages: [],
  memories: [],
  tasks: [],
  skills: [],
  config: null,
  currentPage: 'chat',
  reasoningLogs: [],

  setConnected: (connected) => set({ connected }),
  setBrainStatus: (brainStatus) => set({ brainStatus }),
  addMessage: (message) => set((state) => {
    if (state.messages.some(m => m.id === message.id)) return state;
    return { messages: [...state.messages, message] };
  }),
  setMessages: (messages) => set({ messages }),
  loadChatHistory: async () => {
    try {
      const res = await fetch('/api/chat/history?limit=100');
      const data = await res.json();
      if (data.messages && data.messages.length > 0) {
        set({ messages: data.messages });
      }
    } catch {
      // 静默失败
    }
  },
  setMemories: (memories) => set({ memories }),
  setTasks: (tasks) => set({ tasks }),
  addTask: (task) => set((state) => ({ tasks: [...state.tasks, task] })),
  updateTask: (id, updates) => set((state) => ({
    tasks: state.tasks.map(t => t.id === id ? { ...t, ...updates } : t)
  })),
  removeTask: (id) => set((state) => ({
    tasks: state.tasks.filter(t => t.id !== id)
  })),
  setSkills: (skills) => set({ skills }),
  addSkill: (skill) => set((state) => ({ skills: [...state.skills, skill] })),
  removeSkill: (id) => set((state) => ({ skills: state.skills.filter(s => s.id !== id) })),
  setConfig: (config) => set({ config }),
  setCurrentPage: (currentPage) => set({ currentPage }),
  addReasoningLog: (log) => set((state) => ({
    reasoningLogs: [...state.reasoningLogs.slice(-49), log]  // 保留最近50条
  })),
}));
