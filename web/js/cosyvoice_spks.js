// CosyVoice-CosyVoice-Timestamps 前端扩展:
// 自动从服务端 GET /spks 同步 sft 说话人下拉列表(节点上带「刷新说话人列表」按钮)。
// 注册/删除说话人后点刷新按钮即可同步, 无需重启 ComfyUI。
import { app } from "../../../scripts/app.js";

const refreshSpks = async (node) => {
    const urlW = node.widgets && node.widgets.find((w) => w.name === "server_url");
    const spkW = node.widgets && node.widgets.find((w) => w.name === "sft_spk");
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
    } catch (e) {
        // 服务未启动/未加载时保持现状, 不打扰用户
    }
};

app.registerExtension({
    name: "CosyVoice.SpkRefresh",
    async nodeCreated(node) {
        if (node.comfyClass !== "CosyVoiceTTSWithTimestamps") return;
        node.addWidget("button", "🔄 刷新说话人列表", null, () => refreshSpks(node));
        setTimeout(() => refreshSpks(node), 1200); // 首次进图自动同步一次
    },
});
