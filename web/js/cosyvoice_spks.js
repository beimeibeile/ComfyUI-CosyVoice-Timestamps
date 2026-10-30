// CosyVoice-CosyVoice-Timestamps 前端扩展:
// 1) 自动/手动从服务端 GET /spks 同步 sft 说话人下拉列表;
// 2) 注册/删除说话人节点执行后, 自动刷新所有 TTS 主节点的下拉(无需手动点按钮);
// 3) 更新 options 后强制 setDirtyCanvas 重绘(仅改 options 不触发 ComboWidget 重绘是此前的 bug)。
import { app } from "../../../scripts/app.js";

const TTS_CLASS = "CosyVoiceTTSWithTimestamps";

const refreshSpks = async (node) => {
    if (!node || !node.widgets) return;
    const urlW = node.widgets.find((w) => w.name === "server_url");
    const spkW = node.widgets.find((w) => w.name === "sft_spk");
    if (!urlW || !spkW) return;
    const base = (urlW.value || "http://127.0.0.1:8000").replace(/\/+$/, "");
    try {
        const resp = await fetch(base + "/spks");
        const data = await resp.json();
        if (!data.ok || !Array.isArray(data.available_spks)) return;
        const opts = data.available_spks.map((v) => ({ value: v, text: v }));
        const cur = spkW.value;
        spkW.options = opts;
        if (!data.available_spks.includes(cur)) {
            spkW.value = data.available_spks.length ? data.available_spks[0] : "";
            if (typeof spkW.callback === "function") spkW.callback(spkW.value);
        }
        // 关键: 强制画布重绘, 让 ComboWidget 按新 options 渲染(否则下拉仍显示旧列表)
        if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
    } catch (e) {
        // 服务未启动/未加载时保持现状, 不打扰用户
    }
};

const refreshAll = () => {
    if (!app.graph || !app.graph._nodes) return;
    for (const n of app.graph._nodes) {
        if (n.comfyClass === TTS_CLASS) refreshSpks(n);
    }
};

app.registerExtension({
    name: "CosyVoice.SpkRefresh",
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
        node.addWidget("button", "🔄 刷新说话人列表", null, () => refreshSpks(node));
        setTimeout(() => refreshSpks(node), 1200); // 进图自动同步一次
    },
});
