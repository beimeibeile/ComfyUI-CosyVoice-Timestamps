// CosyVoice-Timestamps 前端扩展:
// 1) sft 说话人下拉实时同步(注册/删除后无需重启浏览器, 也无需关闭页面);
// 2) 按 voice 模式灰色禁用无关参数(全局参数始终可用)。
//
// 机制均经前端源码 + 真实页面双重验证(ComfyUI 0.34 / frontend 1.49.6):
// - 下拉数据源是 widget.options.values; canvas 与 Vue 两种渲染模式在打开下拉时
//   都会实时读取它, 故采用【原地更新数组】(保持引用, 兼容响应式)。
// - 切换当前值必须 widget.setValue(value, {node, canvas})(直接 w.value 无效, 受控)。
// - widget.disabled 是跨模式可靠的灰色禁用接口(canvas 走 computedDisabled→alpha 0.5,
//   Vue 走 widgetState.disabled)。
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// 统一注册规则(DBW 前缀): 主名带前缀; 旧类名保留为兼容别名, 新旧均匹配
const TTS_CLASSES = ["DBW-CosyVoice-TTSWithTimestamps", "CosyVoiceTTSWithTimestamps"];
const REGISTER_CLASSES = ["DBW-CosyVoice-RegisterSpeaker", "CosyVoiceRegisterSpeaker"];
const DELETE_CLASSES = ["DBW-CosyVoice-DeleteSpeaker", "CosyVoiceDeleteSpeaker"];

/* ================= 1. sft 说话人下拉刷新 ================= */

const getTTSNodes = () => {
    const graph = app.graph;
    if (!graph || !graph._nodes) return [];
    return graph._nodes.filter((n) => TTS_CLASSES.includes(n.comfyClass));
};

// 拉取服务端真实列表并原地更新下拉; 服务未运行/异常时静默返回(由轮询/事件下次重试)
const refreshSpks = async (node) => {
    if (!node || !node.widgets) return;
    const urlW = node.widgets.find((w) => w.name === "server_url");
    const spkW = node.widgets.find((w) => w.name === "sft_spk");
    if (!spkW || !spkW.options) return;
    const base = (urlW ? urlW.value : "http://127.0.0.1:8000") || "http://127.0.0.1:8000";
    let data;
    try {
        const resp = await fetch(base.replace(/\/+$/, "") + "/spks");
        if (!resp.ok) return;
        data = await resp.json();
    } catch (e) {
        return; // 服务未启动等, 不报错、不重试堆积
    }
    if (!data || !data.ok || !Array.isArray(data.available_spks)) return;
    const list = data.available_spks.filter((s) => typeof s === "string" && s);
    if (!list.length) return;

    // 原地更新 options.values, 保持数组引用(canvas/Vue 两种渲染都能拿到新数据)
    if (Array.isArray(spkW.options.values)) {
        spkW.options.values.length = 0;
        spkW.options.values.push(...list);
    } else {
        spkW.options.values = list.slice();
    }

    // 当前选中值已不在新列表(如刚删除当前音色) → 切到第一条
    if (!list.includes(spkW.value)) {
        if (typeof spkW.setValue === "function") {
            spkW.setValue(list[0], { node, canvas: app.canvas });
        } else {
            spkW.value = list[0];
        }
        if (typeof spkW.callback === "function") spkW.callback(spkW.value);
    }
    node.setDirtyCanvas?.(true, true);
};

const refreshAll = () => {
    for (const n of getTTSNodes()) refreshSpks(n);
};

// 防抖: 一次工作流里多个节点的 executed 事件会密集到达, 合并为一次刷新
let _lastSchedule = 0;
const scheduleRefresh = (delay = 300) => {
    const now = Date.now();
    if (now - _lastSchedule < delay) return;
    _lastSchedule = now;
    setTimeout(refreshAll, delay);
};

/* ================= 2. 模式联动: 无关参数灰色禁用 ================= */

// 各模式下需禁用的参数; 其余(含全部全局参数)始终可用
const MODE_DISABLED = {
    sft: ["prompt_wav", "prompt_text", "instruct_text"],
    instruct: ["prompt_text"],
    zero_shot: ["sft_spk", "instruct_text"],
};
const MANAGED_WIDGETS = ["prompt_wav", "prompt_text", "sft_spk", "instruct_text"];

const applyMode = (node) => {
    if (!node || !node.widgets || !TTS_CLASSES.includes(node.comfyClass)) return;
    const voiceW = node.widgets.find((w) => w.name === "voice");
    if (!voiceW) return;
    const disabledSet = new Set(MODE_DISABLED[voiceW.value] || []);
    for (const w of node.widgets) {
        if (MANAGED_WIDGETS.includes(w.name)) {
            const target = disabledSet.has(w.name);
            if (w.disabled !== target) w.disabled = target;
        }
    }
    node.setDirtyCanvas?.(true, true);
};

const applyModeAll = () => {
    for (const n of getTTSNodes()) applyMode(n);
};

/* ================= 扩展注册 ================= */

app.registerExtension({
    name: "CosyVoice.FrontendEnhance",

    setup() {
        // 全局事件兜底: 任何节点执行完成 / 整次执行成功 → 刷新下拉
        api.addEventListener("executed", () => scheduleRefresh(300));
        api.addEventListener("execution_success", () => scheduleRefresh(200));

        // 多时点初始化(覆盖页面加载早期 graph 未就绪)
        [800, 2000, 5000].forEach((ms) =>
            setTimeout(() => {
                refreshAll();
                applyModeAll();
            }, ms)
        );

        // 轻量轮询: 每 20s 同步一次, 覆盖外部增删/服务后启动等场景(fetch 失败静默)
        setInterval(() => {
            if (getTTSNodes().length) refreshAll();
        }, 20000);
    },

    beforeRegisterNodeDef(nodeType, nodeData) {
        // Register/Delete 执行后自动同步所有主节点下拉
        if (REGISTER_CLASSES.includes(nodeData.name) || DELETE_CLASSES.includes(nodeData.name)) {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (...args) {
                const r = onExecuted ? onExecuted.apply(this, args) : undefined;
                scheduleRefresh(400);
                return r;
            };
        }
        // 主节点: 执行后刷新 + 选中时刷新/重应用模式
        if (TTS_CLASSES.includes(nodeData.name)) {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (...args) {
                const r = onExecuted ? onExecuted.apply(this, args) : undefined;
                setTimeout(() => refreshSpks(this), 300);
                return r;
            };
            const onSelected = nodeType.prototype.onSelected;
            nodeType.prototype.onSelected = function (...args) {
                const r = onSelected ? onSelected.apply(this, args) : undefined;
                refreshSpks(this);
                applyMode(this);
                return r;
            };
        }
    },

    nodeCreated(node) {
        if (!TTS_CLASSES.includes(node.comfyClass)) return;

        // 手动刷新按钮
        try {
            if (typeof node.addWidget === "function") {
                node.addWidget("button", "刷新说话人列表", null, () => refreshSpks(node));
            }
        } catch (e) { /* 个别渲染模式按钮不渲染, 自动刷新仍生效 */ }

        // voice 变化 → 即时灰色禁用无关参数
        const voiceW = node.widgets && node.widgets.find((w) => w.name === "voice");
        if (voiceW) {
            const origCb = voiceW.callback;
            voiceW.callback = function (...args) {
                const r = origCb ? origCb.apply(this, args) : undefined;
                applyMode(node);
                return r;
            };
        }
        setTimeout(() => {
            refreshSpks(node);
            applyMode(node);
        }, 800);
    },
});
