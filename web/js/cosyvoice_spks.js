// CosyVoice-Timestamps 前端扩展:
// 1) 自动/手动从服务端 GET /spks 同步 sft 说话人下拉列表;
// 2) 注册/删除说话人节点执行后, 自动刷新所有 TTS 主节点的下拉;
// 3) 关键(浏览器实测): ComfyUI 0.34 ComboWidget —
//    a) options.values 才是下拉数据源(整体替换 options 对象无效);
//    b) 切换当前值必须用 widget.setValue(value, {node, canvas})(直接赋值 w.value 无效, 受控属性)。
import { app } from "../../../scripts/app.js";

const TTS_CLASS = "CosyVoiceTTSWithTimestamps";

const refreshSpks = async (node, attempt = 0) => {
    if (!node || !node.widgets) return;
    const urlW = node.widgets.find((w) => w.name === "server_url");
    const spkW = node.widgets.find((w) => w.name === "sft_spk");
    if (!urlW || !spkW) return;
    const base = (urlW.value || "http://127.0.0.1:8000").replace(/\/+$/, "");
    try {
        const resp = await fetch(base + "/spks");
        const data = await resp.json();
        if (!data.ok || !Array.isArray(data.available_spks)) return;
        // a) 更新下拉数据源
        spkW.options.values = data.available_spks;
        // b) 当前值不在新列表时, 用 setValue 切换到首项
        if (!data.available_spks.includes(spkW.value)) {
            const next = data.available_spks.length ? data.available_spks[0] : "";
            if (typeof spkW.setValue === "function") {
                spkW.setValue(next, { node, canvas: app.canvas });
            } else {
                spkW.value = next;
            }
            if (typeof spkW.callback === "function") spkW.callback(spkW.value);
        }
        if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
    } catch (e) {
        // 页面加载早期 graph/网络未就绪时重试几次, 不打扰用户
        if (attempt < 3) {
            setTimeout(() => refreshSpks(node, attempt + 1), 1500);
        }
    }
};

const refreshAll = () => {
    if (!app.graph || !app.graph._nodes) return;
    for (const n of app.graph._nodes) {
        if (n.comfyClass === TTS_CLASS) refreshSpks(n);
    }
};

// nodeCreated 在 0.34 有时不触发, 从 setup 兜底补"刷新按钮"
const ensureButtons = () => {
    if (!app.graph || !app.graph._nodes) return;
    for (const n of app.graph._nodes) {
        if (n.comfyClass !== TTS_CLASS) continue;
        const hasBtn = n.widgets && n.widgets.some((w) => w.type === "button" && String(w.name || "").includes("刷新"));
        if (!hasBtn && typeof n.addWidget === "function") {
            try {
                n.addWidget("button", "🔄 刷新说话人列表", null, () => refreshSpks(n));
            } catch (e) { /* 0.34 个别渲染模式下按钮不渲染, 自动刷新仍生效 */ }
        }
    }
};

app.registerExtension({
    name: "CosyVoice.SpkRefresh",
    setup() {
        // 多次触发兜底(页面加载早期 graph 未就绪/网络未稳时也能同步)
        [1200, 3500, 7000].forEach((ms) => {
            setTimeout(() => {
                if (!app.graph) return;
                app.graph.addEventListener("nodeAdded", () => setTimeout(refreshAll, 300));
                ensureButtons();
                refreshAll();
            }, ms);
        });
    },
    beforeRegisterNodeDef(nodeType, nodeData) {
        // Register/Delete 执行后自动同步所有主节点下拉
        if (nodeData.name === "CosyVoiceRegisterSpeaker" || nodeData.name === "CosyVoiceDeleteSpeaker") {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (...args) {
                const r = onExecuted ? onExecuted.apply(this, args) : undefined;
                setTimeout(refreshAll, 400);
                return r;
            };
        }
        // 主节点执行完成后再同步一次(防服务端注册状态变化)
        if (nodeData.name === TTS_CLASS) {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (...args) {
                const r = onExecuted ? onExecuted.apply(this, args) : undefined;
                setTimeout(() => refreshSpks(this), 300);
                return r;
            };
        }
    },
    async nodeCreated(node) {
        if (node.comfyClass !== TTS_CLASS) return;
        try {
            node.addWidget("button", "🔄 刷新说话人列表", null, () => refreshSpks(node));
        } catch (e) { /* 同上 */ }
        setTimeout(() => refreshSpks(node), 1200);
    },
});
