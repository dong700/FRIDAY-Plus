import { useState, useEffect } from 'react';
import { Save, Eye, EyeOff, CheckCircle, AlertCircle, RefreshCw, ChevronDown, ChevronRight } from 'lucide-react';
import type { AppConfig } from '../types';

function ConfigSection({ title, defaultOpen, children }: { title: string; defaultOpen?: boolean; children: React.ReactNode }) {
  const [open, setOpen] = useState(defaultOpen ?? false);
  return (
    <div className="bg-[#221f1a] rounded-xl border border-[#2a2520] overflow-hidden">
      <button onClick={() => setOpen(!open)} className="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-[#2a2520] transition-colors">
        <span className="text-xs font-mono text-[#c8c6c3] uppercase tracking-wider">{title}</span>
        {open ? <ChevronDown className="w-3.5 h-3.5 text-[#8a8580]" /> : <ChevronRight className="w-3.5 h-3.5 text-[#8a8580]" />}
      </button>
      {open && <div className="px-4 pb-4 space-y-3 border-t border-[#2a2520] pt-3">{children}</div>}
    </div>
  );
}

export function SettingsPanel() {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [showKeys, setShowKeys] = useState<Record<string, boolean>>({});

  useEffect(() => { loadConfig(); }, []);

  const loadConfig = async () => {
    try {
      setLoading(true);
      const res = await fetch('/api/config');
      const data = await res.json();
      setConfig(data);
    } catch { setMessage({ type: 'error', text: '加载配置失败' }); }
    finally { setLoading(false); }
  };

  const handleSave = async () => {
    if (!config) return;
    try {
      setSaving(true);
      const res = await fetch('/api/config', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ config }) });
      setMessage({ type: res.ok ? 'success' : 'error', text: res.ok ? '配置已保存' : '保存失败' });
    } catch { setMessage({ type: 'error', text: '保存失败' }); }
    finally { setSaving(false); setTimeout(() => setMessage(null), 3000); }
  };

  const updateConfig = (path: string[], value: string | number) => {
    if (!config) return;
    const newConfig = { ...config };
    let obj: Record<string, unknown> = newConfig as Record<string, unknown>;
    for (let i = 0; i < path.length - 1; i++) {
      obj[path[i]] = { ...(obj[path[i]] as Record<string, unknown>) };
      obj = obj[path[i]] as Record<string, unknown>;
    }
    obj[path[path.length - 1]] = value;
    setConfig(newConfig);
  };

  const Field = ({ label, path, password = false, value }: { label: string; path: string[]; password?: boolean; value?: string }) => {
    const key = path.join('.');
    const show = showKeys[key] ?? false;
    const val = value ?? (path.reduce((o, k) => o?.[k], config as unknown) as string | undefined) ?? '';
    return (
      <div>
        <label className="text-[10px] text-[#8a8580] mb-1 block">{label}</label>
        <div className={password ? "flex items-center gap-2" : ""}>
          <input type={password && !show ? 'password' : 'text'} value={val}
            onChange={(e) => updateConfig(path, e.target.value)}
            className={`w-full bg-[#1a1815] rounded-lg border border-[#2a2520] px-3 py-1.5 text-xs text-[#e8e6e3] outline-none focus:border-[#C15F3C]/50 transition-colors ${password ? '' : ''}`}
          />
          {password && (
            <button onClick={() => setShowKeys(s => ({ ...s, [key]: !s[key] }))} className="text-[#8a8580] hover:text-[#c8c6c3] shrink-0">
              {show ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
            </button>
          )}
        </div>
      </div>
    );
  };

  if (loading) return <div className="flex-1 flex items-center justify-center bg-[#1a1815]"><RefreshCw className="w-5 h-5 text-[#8a8580] animate-spin" /></div>;
  if (!config) return <div className="flex-1 flex items-center justify-center bg-[#1a1815] text-[#8a8580]">无法加载配置</div>;

  return (
    <div className="flex-1 overflow-y-auto bg-[#1a1815] p-5 space-y-3">
      {message && (
        <div className={`fixed top-4 right-4 flex items-center gap-2 px-4 py-2 rounded-lg text-sm z-50 ${message.type === 'success' ? 'bg-[#10b981]/20 text-[#10b981]' : 'bg-[#ef4444]/20 text-[#ef4444]'}`}>
          {message.type === 'success' ? <CheckCircle className="w-4 h-4" /> : <AlertCircle className="w-4 h-4" />} {message.text}
        </div>
      )}

      {/* LLM - 默认展开 */}
      <ConfigSection title="LLM 模型" defaultOpen={true}>
        <Field label="API Key" path={['llm', 'api_key']} password />
        <Field label="Base URL" path={['llm', 'base_url']} />
        <Field label="模型" path={['llm', 'model']} />
      </ConfigSection>

      {/* 实时对话 */}
      <ConfigSection title="实时对话（S2S 端到端语音）" defaultOpen={true}>
        <p className="text-xs text-gray-400 mb-2">火山引擎语音对话控制台 → 应用详情</p>
        <Field label="App ID" path={['voice_chat_v2', 'dialog_app_id']} />
        <Field label="Access Key" path={['voice_chat_v2', 'dialog_access_key']} password />
        <Field label="App Key" path={['voice_chat_v2', 'dialog_app_key']} password />
      </ConfigSection>

      {/* Viking 记忆 - 默认折叠 */}
      <ConfigSection title="Viking 记忆库">
        <Field label="API Key" path={['viking_memory', 'api_key']} password />
        <Field label="Base URL" path={['viking_memory', 'base_url']} />
        <Field label="Collection" path={['viking_memory', 'collection_name']} />
      </ConfigSection>

      {/* 调度 - 默认折叠 */}
      <ConfigSection title="调度器">
        <div className="grid grid-cols-3 gap-2">
          <Field label="记忆间隔(秒)" path={['scheduler', 'memory_interval']} />
          <Field label="目标间隔(秒)" path={['scheduler', 'goal_interval']} />
          <Field label="检查间隔(秒)" path={['scheduler', 'check_interval']} />
        </div>
      </ConfigSection>

      {/* 机械臂 - 默认折叠 */}
      <ConfigSection title="RoArm-M3 机械臂">
        <div className="grid grid-cols-2 gap-2">
          <Field label="通信模式" path={['roarm', 'mode']} />
          <Field label="HTTP IP" path={['roarm', 'http_ip']} />
        </div>
        <div className="grid grid-cols-2 gap-2">
          <Field label="HTTP 端口" path={['roarm', 'http_port']} />
          <Field label="超时(秒)" path={['roarm', 'timeout']} />
        </div>
      </ConfigSection>

      {/* 虚拟形象 - 默认展开 */}
      <ConfigSection title="虚拟形象 (Live2D)" defaultOpen={true}>
        <AvatarModelConfig />
      </ConfigSection>

      <div className="flex items-center gap-3 pt-2">
        <button onClick={handleSave} disabled={saving} className="flex items-center gap-2 px-4 py-2 bg-[#C15F3C] text-white rounded-lg hover:bg-[#A14A2F] disabled:opacity-50 text-sm"><Save className="w-4 h-4" />{saving ? '保存中...' : '保存配置'}</button>
        <button onClick={loadConfig} className="px-4 py-2 bg-[#2a2520] text-[#c8c6c3] rounded-lg hover:bg-[#3a3530] text-sm"><RefreshCw className="w-4 h-4 inline mr-1" />重载</button>
      </div>
    </div>
  );
}

/** 虚拟形象模型配置组件 */
function AvatarModelConfig() {
  const [modelUrl, setModelUrl] = useState(() => {
    try { return localStorage.getItem('avatar_model_url') || '/avatar/deepseek/deepseek.model3.json'; }
    catch { return '/avatar/deepseek/deepseek.model3.json'; }
  });
  const [saved, setSaved] = useState(false);

  const handleSave = () => {
    localStorage.setItem('avatar_model_url', modelUrl);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <div className="space-y-3">
      <div>
        <label className="text-[10px] text-[#8a8580] mb-1 block">模型路径</label>
        <input type="text" value={modelUrl}
          onChange={(e) => setModelUrl(e.target.value)}
          className="w-full bg-[#1a1815] rounded-lg border border-[#2a2520] px-3 py-1.5 text-xs text-[#e8e6e3] outline-none focus:border-[#C15F3C]/50 transition-colors"
          placeholder="/avatar/your-model.model3.json 或完整URL"
        />
      </div>
      <div className="flex items-center gap-2">
        <button onClick={handleSave}
          className="px-3 py-1.5 bg-[#C15F3C] text-white rounded-lg hover:bg-[#A14A2F] text-xs">
          应用
        </button>
        {saved && <span className="text-xs text-[#10b981]">已保存，刷新页面生效</span>}
      </div>
      <div className="text-[10px] text-[#8a8580] space-y-1">
        <p>支持的格式：</p>
        <p>• <code className="text-[#c8c6c3]">.model.json</code> (Cubism 2.1) — 如 shizuku</p>
        <p>• <code className="text-[#c8c6c3]">.model3.json</code> (Cubism 3/4) — 如 Haru, Epsilon</p>
        <p className="mt-2">模型下载地址：</p>
        <p>• <a href="https://www.live2d.com/en/learn/sample/" target="_blank" className="text-cyan-400 underline">Live2D 官方示例模型</a>（Epsilon, Haru, Shizuku 等）</p>
        <p>• <a href="https://shiralive2d.com/live2d-sample-models/" target="_blank" className="text-cyan-400 underline">ShiraLive2D 免费模型</a>（Lisette, Komi 等，动作丰富）</p>
        <p>• <a href="https://github.com/Eikanya/Live2d-model" target="_blank" className="text-cyan-400 underline">GitHub 模型合集</a></p>
        <p className="mt-2">使用方法：下载后把整个模型文件夹放到 <code className="text-[#c8c6c3]">frontend/public/avatar/</code>，然后填写 <code className="text-[#c8c6c3]">/avatar/文件夹名/xxx.model3.json</code></p>
      </div>
    </div>
  );
}
