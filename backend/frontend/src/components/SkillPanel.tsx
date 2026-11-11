import { useState, useEffect } from 'react';
import { Plus, Trash2, Play, Package } from 'lucide-react';
import type { Skill } from '../types';

export function SkillPanel() {
  const [skills, setSkills] = useState<Skill[]>([]);
  const [showInstall, setShowInstall] = useState(false);
  const [newSkill, setNewSkill] = useState({ name: '', description: '', command: '', version: '1.0.0' });
  const [executing, setExecuting] = useState<string | null>(null);

  useEffect(() => {
    loadSkills();
  }, []);

  const loadSkills = async () => {
    try {
      const res = await fetch('/api/skills');
      const data = await res.json();
      setSkills(data.skills || []);
    } catch {}
  };

  const handleInstall = async () => {
    if (!newSkill.name.trim() || !newSkill.command.trim()) return;
    try {
      await fetch('/api/skills', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newSkill),
      });
      setNewSkill({ name: '', description: '', command: '', version: '1.0.0' });
      setShowInstall(false);
      loadSkills();
    } catch {}
  };

  const handleUninstall = async (id: string) => {
    await fetch(`/api/skills/${id}`, { method: 'DELETE' });
    loadSkills();
  };

  const handleExecute = async (id: string) => {
    setExecuting(id);
    try {
      const res = await fetch(`/api/skills/${id}/execute`, { method: 'POST' });
      const data = await res.json();
      alert(data.success ? `执行成功:\n${data.stdout}` : `执行失败:\n${data.stderr || data.detail}`);
    } catch {
      alert('执行失败');
    } finally {
      setExecuting(null);
    }
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-[#1a1815]">
      <div className="p-4 border-b border-[#2a2520] flex items-center justify-between">
        <h3 className="text-xs font-mono text-[#8a8580] uppercase tracking-wider flex items-center gap-1.5">
          <Package className="w-3.5 h-3.5" />
          技能库
        </h3>
        <button onClick={() => setShowInstall(!showInstall)}
          className="flex items-center gap-1 px-2 py-1 rounded text-[10px] bg-[#C15F3C] text-white hover:bg-[#A14A2F]">
          <Plus className="w-3 h-3" />
          安装技能
        </button>
      </div>

      {showInstall && (
        <div className="p-4 border-b border-[#2a2520] bg-[#221f1a] space-y-2">
          <input type="text" value={newSkill.name} onChange={(e) => setNewSkill({ ...newSkill, name: e.target.value })}
            placeholder="技能名称" className="w-full bg-[#1a1815] rounded border border-[#2a2520] px-2 py-1.5 text-xs text-[#e8e6e3] outline-none focus:border-[#C15F3C]/50" />
          <input type="text" value={newSkill.description} onChange={(e) => setNewSkill({ ...newSkill, description: e.target.value })}
            placeholder="描述" className="w-full bg-[#1a1815] rounded border border-[#2a2520] px-2 py-1.5 text-xs text-[#e8e6e3] outline-none focus:border-[#C15F3C]/50" />
          <input type="text" value={newSkill.command} onChange={(e) => setNewSkill({ ...newSkill, command: e.target.value })}
            placeholder="Shell命令 (如: python script.py)" className="w-full bg-[#1a1815] rounded border border-[#2a2520] px-2 py-1.5 text-xs text-[#e8e6e3] outline-none focus:border-[#C15F3C]/50" />
          <div className="flex items-center gap-2">
            <input type="text" value={newSkill.version} onChange={(e) => setNewSkill({ ...newSkill, version: e.target.value })}
              placeholder="版本" className="w-24 bg-[#1a1815] rounded border border-[#2a2520] px-2 py-1.5 text-xs text-[#e8e6e3] outline-none focus:border-[#C15F3C]/50" />
            <button onClick={handleInstall} className="ml-auto px-3 py-1.5 rounded text-xs bg-[#C15F3C] text-white hover:bg-[#A14A2F]">
              安装
            </button>
          </div>
        </div>
      )}

      <div className="flex-1 overflow-y-auto p-4 space-y-2">
        {skills.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-[#5a5550]">
            <Package className="w-8 h-8 mb-2" />
            <p className="text-sm">暂无技能</p>
            <p className="text-xs mt-1">点击"安装技能"添加Shell命令</p>
          </div>
        ) : (
          skills.map(skill => (
            <div key={skill.id} className="bg-[#221f1a] rounded-lg p-3 border border-[#2a2520] hover:border-[#3a3530] transition-colors group">
              <div className="flex items-start justify-between">
                <div>
                  <div className="text-sm text-[#e8e6e3]">{skill.name}</div>
                  {skill.description && <div className="text-[10px] text-[#8a8580] mt-0.5">{skill.description}</div>}
                </div>
                <span className="text-[9px] text-[#5a5550] font-mono">v{skill.version}</span>
              </div>
              <div className="mt-2 flex items-center justify-between">
                <code className="text-[10px] text-[#C15F3C] bg-[#1a1815] px-1.5 py-0.5 rounded truncate max-w-[60%]">{skill.command}</code>
                <div className="flex items-center gap-1">
                  <button onClick={() => handleExecute(skill.id)} disabled={executing === skill.id}
                    className="p-1 rounded text-[#8a8580] hover:text-[#10b981] disabled:opacity-30">
                    <Play className="w-3.5 h-3.5" />
                  </button>
                  <button onClick={() => handleUninstall(skill.id)}
                    className="p-1 rounded text-[#8a8580] hover:text-[#ef4444] opacity-0 group-hover:opacity-100">
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
