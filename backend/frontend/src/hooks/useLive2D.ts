/**
 * useLive2D — 基于 pixi-live2d-display（GitHub 主流桌宠方案）
 *
 * 2026-11-03 重构：不再自己维护 Cubism WebGL 管线。
 * 之前自研是为了绕开 PIXI v7/v8 的纹理 GC，但代价是：
 *   - 预乘 alpha / clipping / 混合模式都要自己对齐，容易出白斑
 *   - 表情、物理、口型、滤镜这些都要手写
 * 现在直接用 @jannchie/pixi-live2d-display v1.4（支持 PIXI 8）：
 *   - 模型纹理由库自己管理，不会再被 PIXI 的 GC 误删（该 fork 已处理）
 *   - 表情 / 动作 / 口型 / 鼠标跟随 / PIXI.Filter 全部开箱可用
 */

import { useEffect, useRef, useCallback, useState, useMemo } from 'react';
import { Application, Ticker } from 'pixi.js';
import * as cubism4 from 'pixi-live2d-display/cubism4';
import type { Live2DModel as Live2DModelType } from 'pixi-live2d-display/cubism4';
import { subscribeWs } from './useWebSocket';

const Live2DModel: any = (cubism4 as any).Live2DModel;
// 0.4 需要显式注册 Ticker 才会自动更新模型（它默认去找 window.PIXI.Ticker）
if (Live2DModel && typeof Live2DModel.registerTicker === 'function') {
  Live2DModel.registerTicker(Ticker);
}

// ── 透明窗口「大白底」修复 ──
// Live2D / Cubism 的渲染管线会执行 clearColor(1,1,1,0)（白色但 alpha=0）。
// 「白 + 零 alpha」在 premultiplied 语义里是非法像素，Chromium 把这块 WebGL canvas
// 合成到页面/窗口时只认 RGB，于是整块画布变成不透明白色（桌宠窗口的白边/白底）。
// 只在「默认帧缓冲 = 屏幕画布」、且 alpha=0 但 RGB 非黑时，把清屏色换成透明黑；
// 遮罩用的离屏 FBO 不动（那里的白色是给遮罩采样用的）。
let glClearPatched = false;
function patchTransparentClear() {
  if (glClearPatched || typeof window === 'undefined') return;
  if (new URLSearchParams(window.location.search).get('nopatch') === '1') return;
  glClearPatched = true;
  for (const name of ['WebGLRenderingContext', 'WebGL2RenderingContext']) {
    const proto: any = (window as any)[name]?.prototype;
    if (!proto || proto.__petClearPatch) continue;
    proto.__petClearPatch = true;
    const origClear = proto.clear;
    const origClearColor = proto.clearColor;
    proto.clear = function (mask: number) {
      try {
        if ((mask & this.COLOR_BUFFER_BIT) && this.getParameter(this.FRAMEBUFFER_BINDING) === null) {
          const c = this.getParameter(this.COLOR_CLEAR_VALUE);
          if (c && c[3] === 0 && (c[0] !== 0 || c[1] !== 0 || c[2] !== 0)) {
            origClearColor.call(this, 0, 0, 0, 0);
            const r = origClear.call(this, mask);
            origClearColor.call(this, c[0], c[1], c[2], c[3]);
            return r;
          }
        }
      } catch { /* ignore */ }
      return origClear.call(this, mask);
    };
  }
}
patchTransparentClear();

// ──────────────────────────────────────
// 类型 / 常量
// ──────────────────────────────────────

const MOTION_MAP: Record<string, string> = {
  idle: 'Idle',
  tap_body: 'Gesture',
  speaking: 'Gesture',
  listening: 'Gesture',
  wave: 'Gesture',
  think: 'Gesture',
  tear: 'Gesture',
  cry: 'Gesture',
  sad: 'Gesture',
};

const EXPRESSION_MAP: Record<string, string | string[]> = {
  happy: ['星星眼', '爱心眼', '脸红'],
  excited: ['爱心眼', '星星眼'],
  love: ['爱心眼'],
  angry: ['黑脸'],
  sad: ['泪汪汪', '流泪', '流冷汗'],
  cry: ['眼泪', '流泪'],
  surprised: ['蚊香眼', '圈圈眼'],
  confused: ['歪嘴', '＞＜'],
  playful: ['猫猫嘴', '兔牙'],
  silly: ['舌头伸出', '兔牙'],
  speaking: ['拿话筒'],
  shy: ['脸红'],
  loading: ['加载'],
  default: '',
};

export interface AvatarController {
  playMotion: (action: string, priority?: number) => void;
  playExpression: (name: string) => void;
  /** 同时开多个表情（叠加），传 [] 清空 —— AI 可以用任意组合；holdMs 到点自动淡出 */
  setExpressions: (names: string[], holdMs?: number) => void;
  clearExpressions: () => void;
  /** 模型里所有表情名（不含 .exp3 后缀） */
  getExpressionNames: () => string[];
  /** 换体型：0 默认 / 1 体型1(0.4) / 2 体型2(1.0, Q版)。模型自带 Clothes2 变形 */
  setBodyType: (n: number) => void;
  speak: (audioUrl: string) => void;
  setLipSync: (value: number) => void;
  setHeadTracking: (enabled: boolean) => void;
  getModel: () => any;
  loading: boolean;
  error: string | null;
}

interface UseLive2DOptions {
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  modelUrl?: string;
  width?: number;
  height?: number;
  /** 0=默认体型 1=体型1(0.4) 2=体型2(1.0, Q版)。对应模型里的 Clothes2 表情 */
  bodyType?: number;
}

const DEFAULT_MODEL_URL = '/avatar/deepseek/deepseek.model3.json';
/**
 * 大肥鱼的体型变形 = 模型参数 Clothes2（cdi3 里名字叫「2. 大小变」）。
 * 自带的三个 expression（默认体型 / 体型1 / 体型2）其实只是把 Clothes2 设成 0 / 0.4 / 1.0，
 * 所以这里直接写参数：走 expression 的话，任何一次情绪表情都会把它顶掉（库同时只保留一个 expression）。
 */
const BODY_VALUES = [0, 0.4, 1.0];
const BODY_PARAM = 'Clothes2';
/**
 * Clothes2 越大，角色本体越小（1.0 时只剩 Q 版那么点大，整个画布还按原尺寸）。
 * 这里按经验值补偿舞台缩放，让三种体型在窗口里的「占地面积」差不多：
 * 0→1x、0.4→1.6x、1.0→3.6x（中间线性插值，跟着变形一起平滑变化）。
 */
const BODY_ZOOM = [1, 1.6, 3.6];
/** 画布顶部空白占画布高度的比例（实测：整身与 Q 版的发梢/丝带都从 ~10% 处开始） */
const CANVAS_TOP_PAD = 0.102;

/**
 * 渲染帧率上限。
 * 原来用 Ticker.shared 跟着屏幕刷新率跑：120Hz 屏上就是 120 次/秒的 Live2D 更新 + 重绘，
 * 实测一个桌宠稳定吃掉 ~1.6 个核心（renderer 96% + GPU 71%），整机就变卡。
 * Cubism 的动作本身就是按 30fps 做的，30 已经够顺；要试更顺可以加 ?fps=60。
 */
const MAX_FPS = (() => {
  const raw = Number(new URLSearchParams(window.location.search).get('fps') || 0);
  return Number.isFinite(raw) && raw >= 10 ? Math.min(120, raw) : 30;
})();

function bodyZoom(v: number): number {
  if (v <= 0) return BODY_ZOOM[0];
  if (v <= 0.4) return BODY_ZOOM[0] + (BODY_ZOOM[1] - BODY_ZOOM[0]) * (v / 0.4);
  if (v >= 1) return BODY_ZOOM[2];
  return BODY_ZOOM[1] + (BODY_ZOOM[2] - BODY_ZOOM[1]) * ((v - 0.4) / 0.6);
}

function readModelUrl(): string {
  try {
    const saved = localStorage.getItem('avatar_model_url');
    // 旧版本存过粉色 Doro 的路径，模型目录已删 —— 直接丢掉，回落到大肥鱼
    if (saved && !/doro/i.test(saved)) return saved;
  } catch { /* ignore */ }
  return DEFAULT_MODEL_URL;
}

function readQuery(): URLSearchParams {
  return new URLSearchParams(window.location.search);
}

// ──────────────────────────────────────
// Hook
// ──────────────────────────────────────

export function useLive2D(options: UseLive2DOptions): AvatarController {
  const { canvasRef, modelUrl, width = 400, height = 500, bodyType = 0 } = options;

  const resolvedModelUrl = useMemo(() => readModelUrl(), []);
  const finalUrl = useMemo(() => {
    const raw = modelUrl || resolvedModelUrl;
    const normalized = raw.replace(/\\/g, '/');
    return normalized.startsWith('/') || normalized.startsWith('http')
      ? normalized
      : '/' + normalized;
  }, [modelUrl, resolvedModelUrl]);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const modelRef = useRef<Live2DModelType | null>(null);
  const appRef = useRef<Application | null>(null);
  const disposedRef = useRef(false);
  const mouseRef = useRef({ x: 0, y: 0, active: false });
  /** 目标体型值（Clothes2），每帧朝它插值 → 变形是平滑的而不是瞬间切换 */
  const bodyTargetRef = useRef(BODY_VALUES[Math.max(0, Math.min(2, Math.round(bodyType)))]);
  const bodyCurRef = useRef(BODY_VALUES[Math.max(0, Math.min(2, Math.round(bodyType)))]);
  /** 口型开合（0~1），每帧写进 ParamMouthOpenY */
  const lipRef = useRef(0);
  /** 平滑后的跟随量（度）：头用 AngleX/AngleY，身体用 PositionX/LeanInput */
  const lookRef = useRef({ x: 0, y: 0 });
  // ── 表情（可叠加）──
  // 库自带的 expressionManager 同时只保留一个表情；这里直接读 exp3 里的参数，
  // 每帧在 beforeModelUpdate 里按权重叠加，所以「爱心眼 + 脸红 + 兔牙」可以同时开。
  const exprDefsRef = useRef<{ base: string; map: Record<string, string> }>({ base: '', map: {} });
  const exprParamRef = useRef<Map<string, { Id: string; Value: number; Blend?: string }[]>>(new Map());
  const exprActiveRef = useRef<{ name: string; w: number; target: number }[]>([]);
  /** AI 明确点名的表情（说话结束时不立刻清掉，等 holdMs 到点再淡出） */
  const exprStickyRef = useRef(false);
  /** 表情自动收起计时器：AI 点的表情也会到点消失，不能一直挂着 */
  const exprTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  /** 默认挂多久（毫秒）：AI 没指定 seconds 时用这个 */
  const DEFAULT_EXPR_HOLD_MS = 12000;
  /** 情绪自动表情（speaking/thinking 等）的兜底时长 */
  const AUTO_EXPR_HOLD_MS = 20000;
  // 画布尺寸动态变（桌宠缩放）：不重载模型，只改渲染尺寸并重算适配比例，
  // 这样放大是真的多画像素，而不是把 420 宽的位图拉大（那就是"变糊"的来源）。
  const sizeRef = useRef({ w: width, h: height });
  const fitRef = useRef({ baseScale: 1, localW: 3749, localH: 7697, fitZoom: 1 });
  /** 动作自动回 idle 的计时器：默认时长，AI 给了 seconds 就按它的 */
  const motionTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const DEFAULT_MOTION_HOLD_MS = 5000;

  // ?q=1 / model=deepseek_q → 模型自带「体型2」表情（Clothes2 = 1.0，Q 版）
  const isQMode = useMemo(() => {
    const q = readQuery();
    return q.get('q') === '1' || q.get('model') === 'deepseek_q';
  }, []);

  // ── 动作 ──

  const playMotion = useCallback((action: string, priority?: number) => {
    const model = modelRef.current;
    if (!model) return;
    const group = MOTION_MAP[action] || action;
    try {
      const defs: any = (model.internalModel.motionManager as any).definitions;
      const names: string[] = Array.isArray(defs)
        ? defs.map((g: any) => g?.name).filter(Boolean)
        : Object.keys(defs || {});
      const target = names.includes(group) ? group : names[0];
      if (!target) return;
      model.motion(target, undefined, priority as any);
    } catch (err) {
      console.warn('[Avatar] playMotion error:', err);
    }
  }, []);

  // ── 表情（可叠加） ──

  const bareExpr = (s: string) => String(s || '').replace(/\.exp3$/i, '');

  /** 读一个表达式里要改哪些参数（缓存），失败返回 null */
  const loadExprParams = useCallback(async (name: string) => {
    const bare = bareExpr(name);
    const cached = exprParamRef.current.get(bare);
    if (cached) return cached;
    const file = exprDefsRef.current.map[bare];
    if (!file) return null;
    try {
      const res = await fetch(exprDefsRef.current.base + file);
      const json = await res.json();
      const params = (json?.Parameters || []).filter((p: any) => p && p.Id);
      exprParamRef.current.set(bare, params);
      return params;
    } catch (err) {
      console.warn('[Avatar] expression load failed:', bare, err);
      return null;
    }
  }, []);

  /** 真正的叠加入口：sticky=true 表示这是 AI/外部明确点的表情，说话结束也不会被清掉。
   *  语义是「叠加」：多次调用不会把之前挂的顶掉，只有传空数组才整体清空。 */
  const applyExpr = useCallback((names: string[], sticky: boolean) => {
    const wanted = Array.from(new Set((names || []).map(bareExpr).filter(Boolean)));
    const act = exprActiveRef.current;
    if (!wanted.length) for (const e of act) e.target = 0;
    for (const n of wanted) {
      if (exprDefsRef.current.map[n] === undefined) continue;
      const hit = act.find(e => e.name === n);
      if (hit) hit.target = 1; else act.push({ name: n, w: 0, target: 1 });
      void loadExprParams(n);
    }
    exprStickyRef.current = sticky && (wanted.length > 0 || act.some(e => e.target > 0));
  }, [loadExprParams]);

  /** 同时开若干个表情（传空数组 = 全部关掉）。名字可以带 ".exp3" 也可以不带。
   *  holdMs：挂多久后自动淡出（默认 12 秒；AI 显式给 seconds 时以它为准）
   *  persist：true = 这是「状态」而不是一次性动作，一直挂着直到 AI 自己清掉 */
  const clearExpressions = useCallback(() => {
    if (exprTimerRef.current) {
      clearTimeout(exprTimerRef.current);
      exprTimerRef.current = null;
    }
    for (const e of exprActiveRef.current) e.target = 0;
    exprStickyRef.current = false;
  }, []);

  const setExpressions = useCallback((names: string[], holdMs?: number, persist = false) => {
    applyExpr(names, true);
    if (exprTimerRef.current) {
      clearTimeout(exprTimerRef.current);
      exprTimerRef.current = null;
    }
    const ms = Number(holdMs);
    const hold = Number.isFinite(ms) && ms > 0 ? ms : DEFAULT_EXPR_HOLD_MS;
    if (!persist && (names || []).length) {
      exprTimerRef.current = setTimeout(() => clearExpressions(), hold);
    }
  }, [applyExpr, clearExpressions]);

  const getExpressionNames = useCallback(() => Object.keys(exprDefsRef.current.map), []);

  const playExpression = useCallback((name: string) => {
    if (!name || name === 'default') { clearExpressions(); return; }
    const mapped = EXPRESSION_MAP[name] ?? name;
    const candidates = (Array.isArray(mapped) ? mapped : [mapped]).map(bareExpr);
    const map = exprDefsRef.current.map;
    const hit = candidates.find(c => c && map[c] !== undefined);
    // 不认识的别名（比如说话动作里的「拿话筒」）保持现状，别把 AI 刚组合好的表情清掉
    if (!hit) return;
    applyExpr([hit], false);
    // 兜底：情绪表情万一没等到 speaking_end，也别一直挂着
    if (exprTimerRef.current) clearTimeout(exprTimerRef.current);
    exprTimerRef.current = setTimeout(() => clearExpressions(), AUTO_EXPR_HOLD_MS);
  }, [applyExpr, clearExpressions]);

  /** 换体型（模型自带的 Clothes2「大小变」）：0 默认 / 1 体型1(0.4) / 2 体型2(1.0, Q版) */
  const setBodyType = useCallback((n: number) => {
    const idx = Math.max(0, Math.min(2, Math.round(n)));
    bodyTargetRef.current = BODY_VALUES[idx];
  }, []);

  // bodyType 变化（双击换体型）时同步目标值；不重建模型，只改参数目标
  useEffect(() => {
    bodyTargetRef.current = BODY_VALUES[Math.max(0, Math.min(2, Math.round(bodyType)))];
  }, [bodyType]);

  const speak = useCallback((audioUrl: string) => {
    const model = modelRef.current;
    if (!model) return;
    try {
      (model as any).speak(audioUrl, { volume: 1 });
    } catch (err) {
      console.warn('[Avatar] speak error:', err);
    }
  }, []);

  const setLipSync = useCallback((value: number) => {
    // pixi-live2d-display 0.4 没有 setLipSyncValue / lipSyncValue 接口，
    // 这里只记录目标值，由下面的 im.update 每帧写 ParamMouthOpenY。
    lipRef.current = Math.max(0, Math.min(1, value));
  }, []);

  const setHeadTracking = useCallback((enabled: boolean) => {
    const model = modelRef.current as any;
    if (!model) return;
    try {
      model.internalModel.focusController.targetX = 0;
      model.internalModel.focusController.targetY = 0;
      if (!enabled) {
        model.internalModel.focusController.x = 0;
        model.internalModel.focusController.y = 0;
      }
    } catch { /* ignore */ }
  }, []);

  const getModel = useCallback(() => modelRef.current?.internalModel ?? null, []);

  // ── 鼠标跟随（视线 + 头部） ──

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const onMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width * 2 - 1;
      const y = (e.clientY - rect.top) / rect.height * 2 - 1;
      mouseRef.current = { x, y: -y, active: true };
    };
    const onLeave = () => { mouseRef.current = { x: 0, y: 0, active: false }; };

    canvas.addEventListener('mousemove', onMove);
    canvas.addEventListener('mouseleave', onLeave);
    return () => {
      canvas.removeEventListener('mousemove', onMove);
      canvas.removeEventListener('mouseleave', onLeave);
    };
  }, [canvasRef]);

  useEffect(() => {
    if (!modelRef.current) return;
    // 跟 app 用同一个 ticker：两个 ticker 各挂一次 RAF 等于每帧干两遍活
    const ticker = appRef.current?.ticker;
    if (!ticker) return;
    const fn = () => {
      const model: any = modelRef.current;
      const m = mouseRef.current;
      if (!model || !m.active) return;
      try {
        model.internalModel.focusController.focus(m.x, m.y);
      } catch { /* ignore */ }
    };
    ticker.add(fn);
    return () => { ticker.remove(fn); };
  }, [loading]);

  // ── 初始化 ──

  useEffect(() => {
    disposedRef.current = false;
    setLoading(true);
    setError(null);

    const canvas = canvasRef.current;
    if (!canvas) return;

    let app: Application | null = null;
    let model: Live2DModelType | null = null;

    async function init() {
      try {
        // PIXI 7：Application 构造时直接建 renderer（不需要 await init）
        app = new Application({
          view: canvas as HTMLCanvasElement,
          width: sizeRef.current.w,
          height: sizeRef.current.h,
          backgroundAlpha: 0,
          backgroundColor: 0,
          clearBeforeRender: true,
          antialias: false,
          // 与参考包（PIXI 7 默认 true）保持一致：Cubism 的混合是 ONE / ONE_MINUS_SRC_ALPHA，
          // 产出本来就是预乘 alpha；声明成 false 会让半透明边缘在桌面合成时发灰/发光（尾巴外圈那层光影）。
          // 需要对照可加 ?premul=0
          premultipliedAlpha: readQuery().get('premul') !== '0',
          autoDensity: true,
          // 分辨率跟屏幕 DPI 走（这台是 200%）：CSS 尺寸不变，绘制缓冲按物理像素来，
          // 否则在 2x 屏上一个 CSS 像素只画了 1 个物理像素，人物边缘就是毛的/糊的。
          resolution: (Number(new URLSearchParams(window.location.search).get('res')) || Math.min(2, window.devicePixelRatio || 1)),
        });
        appRef.current = app;
        // 帧率封顶（详见 MAX_FPS）：120Hz 屏不封顶会稳定吃掉一个多核心
        app.ticker.maxFPS = MAX_FPS;
        // 0.4 的模型自动更新挂在 Ticker.shared 上，也要一起封顶
        Ticker.shared.maxFPS = MAX_FPS;

        if (disposedRef.current) return;

        console.log('[Avatar] Loading (pixi):', finalUrl);
        model = await Live2DModel.from(finalUrl, {
          // 0.4 没有 ticker 选项：它固定用 Ticker.shared 自动更新（上面已限帧）
          autoInteract: false,
        } as any);

        if (disposedRef.current) {
          model.destroy();
          return;
        }
        modelRef.current = model;
        app.stage.addChild(model as any);

        // resolution > 1 时库把 WebGL viewport 设成 CSS 尺寸（420×560），
        // 但真实 drawingBuffer 是物理像素（840×1120）→ 模型会被缩成一半并跑到左下角。
        // 每帧在 updateTransform 之前把 viewport 改回物理像素（库是紧接其后执行的）。
        try {
          const im0: any = (model as any).internalModel;
          const origUpdateTransform = im0.updateTransform.bind(im0);
          im0.updateTransform = (mat: any) => {
            im0.viewport[0] = 0;
            im0.viewport[1] = 0;
            im0.viewport[2] = canvas.width;
            im0.viewport[3] = canvas.height;
            origUpdateTransform(mat);
          };
        } catch { /* ignore */ }

        // 缩放定位：按「整只装得下」来适配 —— 宽不超过窗口 95%，高不超过 105%，取小。
        // 之前是写死 (height/localH)*1.35（Q 版 1.85），对画布比例不同的模型会直接放大到
        // 只剩一坨头顶（画布比例和窗口不一致的模型都中招）。用上下限取小就不会溢出。
        const im = model.internalModel as any;
        const localW: number = im.width || 3749;
        const localH: number = im.height || 7697;
        const fitZoom = Number(
          (new URLSearchParams(location.search).get('fit') || '') || 1,
        ) || 1;
        const baseScale = Math.min((sizeRef.current.w * 0.95) / localW, (sizeRef.current.h * 0.98) / localH) * fitZoom;
        fitRef.current = { baseScale, localW, localH, fitZoom };
        // 锚点必须在画布顶部：Clothes2 变形只会把「人」缩小、画布尺寸不变，
        // 用底部锚点的话 Q 版整个人会被推到窗口上方（看不见）。
        model.anchor.set(0.5, 0);
        const bodyK0 = bodyZoom(bodyCurRef.current);
        model.scale.set(baseScale * bodyK0);
        // 画布上方本来有 10% 空白（头发丝带顶到 y≈0.10 画布高）。放大补偿时这段空白
        // 也会被放大，把整个人往下推出窗口 —— 所以按 (k-1) 把这一段量回来。
        model.position.set(sizeRef.current.w / 2, -(CANVAS_TOP_PAD * localH * baseScale) * (bodyK0 - 1));

        console.log('[Avatar] fit ' + JSON.stringify({ localW, localH, baseScale, isQMode }));

        // 体型：每帧写 Clothes2，朝 bodyTargetRef 平滑插值。
        // 必须写在 im.update() 之前 —— Cubism 在 update 里才按参数重算网格，
        // 放在 update 之后写只会改到「下一次」且会跟表情/物理互相覆盖。
        const core = im.coreModel as any;
        bodyCurRef.current = bodyTargetRef.current;
        const origUpdate = (im.update as Function).bind(im);
        im.update = (dt: number, now: number) => {
          const target = bodyTargetRef.current;
          const cur = bodyCurRef.current;
          const next = Math.abs(target - cur) < 0.002 ? target : cur + (target - cur) * 0.18;
          bodyCurRef.current = next;
          try { core.setParameterValueById(BODY_PARAM, next, 1); } catch { /* ignore */ }
          try { core.setParameterValueById('ParamMouthOpenY', lipRef.current, 1); } catch { /* ignore */ }
          // 鼠标跟随（头 + 身体）。注意用模型的「物理输入」而不是最终角度：
          //   AngleX / AngleY  → physics3 的 PhysicsSetting1/2 → ParamAngleX/Y（头）
          //   PositionX        → PhysicsSetting4 → ParamBodyAngleX（胸腔、手臂、头发一起扭）
          //   LeanInput        → PhysicsSetting8 → ParamBodyAngleY（身体前后倾）
          // 直接写 ParamAngleX/ParamBodyAngleX 会被 physics 重新算回去，所以必须走输入。
          try {
            const m = mouseRef.current;
            const tx = m.active ? m.x * 18 : 0;
            const ty = m.active ? m.y * 14 : 0;
            const l = lookRef.current;
            l.x += (tx - l.x) * 0.15;
            l.y += (ty - l.y) * 0.15;
            core.setParameterValueById('AngleX', l.x, 1);
            core.setParameterValueById('AngleY', l.y, 1);
            core.setParameterValueById('PositionX', l.x * 0.6, 1);
            // 前倾(LeanInput) 是物理输入：physics 在 update 内部才跑，
            // 必须在之前把「鼠标基准 + 表情贡献」一起写进去，否则前倾类表情等于没调。
            let lean = l.y * 0.4;
            for (const e of exprActiveRef.current) {
              if (e.w < 0.01) continue;
              const ps = exprParamRef.current.get(e.name);
              if (!ps) continue;
              for (const p of ps) if (p.Id === 'LeanInput') lean += p.Value * e.w;
            }
            core.setParameterValueById('LeanInput', lean, 1);
          } catch { /* ignore */ }
          try {
            const k = bodyZoom(next);
            const f = fitRef.current;
            model!.scale.set(f.baseScale * k);
            model!.position.set(sizeRef.current.w / 2,
              -(CANVAS_TOP_PAD * f.localH * f.baseScale) * (k - 1));
          } catch { /* ignore */ }
          return origUpdate(dt, now);
        };

        console.log('[Avatar] mask buffers ' + JSON.stringify({
          renderTextureCount: (im as any).renderer?._clippingManager?._renderTextureCount,
        }));

        // 表情文件表：name -> File（相对 model3.json 所在目录），供叠加表情用
        try {
          const defs = (im.motionManager?.expressionManager as any)?.definitions || [];
          const base = finalUrl.slice(0, finalUrl.lastIndexOf('/') + 1);
          const map: Record<string, string> = {};
          for (const d of defs) {
            const nm = String(d?.Name || d?.name || '').replace(/\.exp3$/i, '');
            const file = d?.File || d?.file;
            if (nm && file) map[nm] = file;
          }
          exprDefsRef.current = { base, map };
          console.log('[Avatar] expressions ' + JSON.stringify(Object.keys(map)));
        } catch { /* ignore */ }

        // 叠加表情：在「动作/物理都算完、网格还没刷新」的时机写参数，
        // 这样多个表情能同时生效，也不会被动作曲线或下一帧的 loadParameters 冲掉。
        try {
          (im as any).on?.('beforeModelUpdate', () => {
            const act = exprActiveRef.current;
            if (!act.length) return;
            for (let i = act.length - 1; i >= 0; i--) {
              const e = act[i];
              e.w += (e.target - e.w) * 0.25;
              if (e.target === 0 && e.w < 0.02) { act.splice(i, 1); continue; }
              const params = exprParamRef.current.get(e.name);
              if (!params || e.w < 0.01) continue;
              for (const p of params) {
                // LeanInput 是物理输入，已经在 update 之前按「鼠标基准+表情」写过了，
                // 这里再叠加一次只会加两遍（而且此时 physics 已经跑完，写了也不生效）
                if (p.Id === 'LeanInput') continue;
                const cur = core.getParameterValueById(p.Id);
                if (typeof cur !== 'number') continue;
                let v = cur;
                if (p.Blend === 'Multiply') v = cur * (1 + (p.Value - 1) * e.w);
                else if (p.Blend === 'Overwrite') v = cur * (1 - e.w) + p.Value * e.w;
                else v = cur + p.Value * e.w;
                core.setParameterValueById(p.Id, v, 1);
              }
            }
            // 前倾(LeanInput=-30) 会被 LeanInput 的量程卡成 0，物理算不出身体前倾。
            // 这里把表情要的 lean 直接折算成胸腔角度，视觉上才真的会往前栽。
            let leanSum = 0;
            for (const e of act) {
              if (e.w < 0.01) continue;
              const ps = exprParamRef.current.get(e.name);
              if (!ps) continue;
              for (const p of ps) if (p.Id === 'LeanInput') leanSum += p.Value * e.w;
            }
            if (leanSum) {
              const base = Number(core.getParameterValueById('ParamBodyAngleY')) || 0;
              core.setParameterValueById(
                'ParamBodyAngleY', Math.max(-25, Math.min(25, base + leanSum * 0.8)), 1,
              );
            }
          });
        } catch { /* ignore */ }

        // ── 脸部默认外观修正 ──
        // 大肥鱼自带两类「纯色叠加网格」：multiply=0，颜色完全由 screen 决定。
        //   ArtMesh340（Part99 头）：一整块不透明粉 screen(0.95,0.73,0.73)，默认可见，
        //     会把整张脸刷成粉红 —— 这是「脸太红」的来源。
        //   ArtMesh246 / ArtMesh682（Part46 鼻子）：两块不透明近白高光，在鼻梁上形成
        //     一块硬边白方块 —— 这是「鼻子那一块有问题」的来源。
        // 作者预览里这两处都不存在（鼻梁只是细线），所以按网格 ID 默认隐藏；
        // 不动贴图、不改渲染管线，其它网格保持模型原样。
        const hiddenIds = ['ArtMesh340', 'ArtMesh246', 'ArtMesh682'];
        const hiddenSet = new Set<number>();
        const drawableCount: number = core.getDrawableCount?.() || 0;
        for (let i = 0; i < drawableCount; i++) {
          if (hiddenIds.includes(core.getDrawableId?.(i))) hiddenSet.add(i);
        }
        if (hiddenSet.size && typeof core.getDrawableDynamicFlagIsVisible === 'function') {
          const origVisible = core.getDrawableDynamicFlagIsVisible.bind(core);
          core.getDrawableDynamicFlagIsVisible = (i: number) =>
            hiddenSet.has(i) ? false : origVisible(i);
        }

        setLoading(false);
        setError(null);
        // 调试句柄：控制台/截图脚本可以直接改参数、看 mask 配置（不影响正常渲染）
        (window as any).__fridayAvatar = { model, core, app, setExpressions, playExpression, playMotion };
        console.log('[Avatar] Ready (pixi)');
      } catch (err: any) {
        console.error('[Avatar] Init failed:', err);
        if (!disposedRef.current) {
          setError(err?.message || '模型加载失败');
          setLoading(false);
        }
      }
    }

    init();

    return () => {
      disposedRef.current = true;
      try { model?.destroy(); } catch { /* ignore */ }
      try { app?.destroy(true, { children: true } as any); } catch { /* ignore */ }
      modelRef.current = null;
      appRef.current = null;
    };
  }, [finalUrl, isQMode]);

  // ── 尺寸变化：只调渲染分辨率 + 重新适配，不重建模型 ──
  // 桌宠拖大小是常事，每次重载模型既慢又会闪一下；而且真正要提高清晰度必须让
  // 画布像素跟着变大（单纯 zoom 是把旧位图拉大，越拉越糊）。
  useEffect(() => {
    sizeRef.current = { w: width, h: height };
    const app = appRef.current;
    const model = modelRef.current as any;
    if (!app || !model) return;
    try {
      app.renderer.resize(width, height);
      const f = fitRef.current;
      const baseScale = Math.min((width * 0.95) / f.localW, (height * 0.98) / f.localH) * f.fitZoom;
      fitRef.current = { ...f, baseScale };
      const k = bodyZoom(bodyCurRef.current);
      model.scale.set(baseScale * k);
      model.position.set(width / 2, -(CANVAS_TOP_PAD * f.localH * baseScale) * (k - 1));
    } catch (e) {
      console.warn('[Avatar] resize failed:', e);
    }
  }, [width, height]);

  // ── WebSocket 事件 ──

  useEffect(() => {
    // subscribeWs 会自动跟着断线重连换绑；直接在 effect 里拿一次实例的话，
    // 后端重启后 socket 被换成新对象，这里就永远收不到 avatar.* 了（表现为"没反应"）。
    return subscribeWs((data) => {
      try {
        switch (data.type) {
          case 'avatar.speaking':
            // 说话只换表情（拿话筒）。但 AI 明确挂上的表情/状态优先，不能被开口就顶掉。
            if (!exprStickyRef.current) playExpression('speaking');
            break;
          case 'avatar.speaking_end':
            // AI 明确点的表情（组合）保留，只清掉情绪自动表情；动作不回 idle，
            // 由动作自己的计时器决定什么时候结束。
            if (!exprStickyRef.current) playExpression('default');
            break;
          case 'avatar.listening':
            playExpression('playful');
            playMotion('listening');
            break;
          case 'avatar.thinking':
            playExpression('confused');
            playMotion('think');
            break;
          case 'avatar.action': {
            // AI 可以一次点一串表情（叠加），也可以只给单个 expression / 动作。
            // persist=true 表示这是「状态」（哭、生气、拿话筒……），不自动淡出；
            // 默认 false：动作/表情按 seconds 或默认时长自己收掉。
            const { action, expression, expressions, seconds, persist } = data.payload || {};
            const keep = !!persist;
            if (action) {
              playMotion(action);
              // 动作默认 5 秒；AI 给了 seconds 就按它的（可以更长）
              if (motionTimerRef.current) {
                clearTimeout(motionTimerRef.current);
                motionTimerRef.current = null;
              }
              if (action !== 'idle' && !keep) {
                const hold = Number(seconds) > 0 ? Number(seconds) * 1000 : DEFAULT_MOTION_HOLD_MS;
                motionTimerRef.current = setTimeout(() => playMotion('idle'), hold);
              }
            }
            if (Array.isArray(expressions)) {
              const hold = Number(seconds) > 0 ? Number(seconds) * 1000 : undefined;
              setExpressions(expressions, hold, keep);
            } else if (expression) playExpression(expression);
            break;
          }
          default:
            break;
        }
      } catch { /* ignore */ }
    });
  }, [playMotion, playExpression, setExpressions, clearExpressions]);

  return {
    playMotion,
    playExpression,
    setExpressions,
    clearExpressions,
    getExpressionNames,
    setBodyType,
    speak,
    setLipSync,
    setHeadTracking,
    getModel,
    loading,
    error,
  };
}
