# -*- coding: utf-8 -*-
"""
ComfyUI-CosyVoice-Timestamps: CosyVoice 同源字级时间戳 TTS 客户端节点。

架构:
    ComfyUI (本节点, 轻量 HTTP 客户端)
        ⇄  CosyVoice 独立服务 (server/tts_server.py, Python 3.10 venv)
                ⇄  CosyVoice2-0.5B (流式合成 + 句级精确计时 + 字级分配)

节点输出:
    AUDIO           : ComfyUI 标准 AUDIO 字典 {"waveform": Tensor[1, 1, N], "sample_rate": int}
    WORD_TIMESTAMPS : JSON 字符串 [{word, start_ms, end_ms}, ...]
    SRT             : 标准 SRT 字幕文本
    DURATION_SECONDS: 合成音频总时长(秒)

依赖隔离: 本节点仅用标准库 urllib, 不向 ComfyUI 环境注入 CosyVoice 依赖。
"""
import base64
import json
import os
import re
import time
import urllib.request
import urllib.error
from datetime import datetime

import torch
import numpy as np


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# sft 说话人列表缓存(短 TTL, 避免每次构建 widget 都阻塞网络)
_SPKS_CACHE = {"ts": 0.0, "data": None}
_SPKS_TTL = 5.0  # 秒


def _fetch_available_spks(server_url="http://127.0.0.1:8000"):
    """同步拉取服务端可用说话人列表(带 5s 缓存); 服务未启动/异常返回 None."""
    now = time.time()
    if now - _SPKS_CACHE["ts"] < _SPKS_TTL:
        return _SPKS_CACHE["data"]
    try:
        url = server_url.rstrip("/") + "/spks"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        spks = None
        if data.get("ok") and isinstance(data.get("available_spks"), list):
            spks = [s for s in data["available_spks"] if isinstance(s, str) and s.strip()]
        _SPKS_CACHE.update({"ts": now, "data": spks})
        return spks
    except Exception:
        _SPKS_CACHE.update({"ts": now, "data": None})
        return None


def _post_json(server_url, payload, timeout=300, path="/tts_with_timestamps"):
    """向 CosyVoice 服务发送 JSON 请求并返回响应字典."""
    url = server_url.rstrip("/") + path
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"[CosyVoice] 服务返回 HTTP {e.code}: {body[:500]}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"[CosyVoice] 无法连接服务 {server_url}，请确认已启动 "
            f"server/tts_server.py（详见 README 部署章节）: {e.reason}"
        ) from e


def _srt_timestamp(ms):
    """毫秒 → SRT 时间戳 HH:MM:SS,mmm."""
    ms = max(0, int(ms))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, msec = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{msec:03d}"


def _build_srt(sentences):
    """由句级时间戳列表生成 SRT 文本. 每条字幕为一句."""
    if not sentences:
        return ""
    lines = []
    idx = 1
    for s in sentences:
        start, end = s.get("start_ms", 0), s.get("end_ms", 0)
        if start < 0 or end < start:
            continue
        txt = (s.get("sentence") or "").strip()
        if not txt:
            continue
        lines.append(f"{idx}\n{_srt_timestamp(start)} --> {_srt_timestamp(end)}\n{txt}\n")
        idx += 1
    return "\n".join(lines)


# ---- 文件名占位符 -------------------------------------------------------
_DATE_PLACEHOLDER_RE = re.compile(r"%date:([^%]+)%")
_NODE_REF_RE = re.compile(r"%([^%]+?)\.([^%]+?)%")
_ANY_PLACEHOLDER_RE = re.compile(r"%[^%]+%")
# ComfyUI 日期 token → strftime token
_DATE_TOKEN_MAP = {"yyyy": "%Y", "MM": "%m", "dd": "%d", "HH": "%H", "mm": "%M", "ss": "%S"}
_INVALID_FS_RE = re.compile(r'[<>:"|?*\x00-\x1f]')


def _build_title_map(extra_pnginfo) -> dict:
    """从 extra_pnginfo.workflow.nodes 建立 {node_id(str): 节点标题}。

    节点引用支持按用户在界面看到的标题匹配(标题可能被重命名),
    无标题时回落到节点类型名。
    """
    titles = {}
    try:
        nodes = extra_pnginfo.get("workflow", {}).get("nodes", [])
    except (AttributeError, TypeError):
        return titles
    for n in nodes:
        if isinstance(n, dict) and "id" in n:
            titles[str(n["id"])] = n.get("title") or n.get("type") or ""
    return titles


def _compact_name(s) -> str:
    """去除所有空白, 用于显示名/类名的宽松等价比较。"""
    return re.sub(r"\s+", "", str(s or ""))


def _lookup_node_value(node_name, field, prompt, title_map):
    """按节点类型名/显示名/标题匹配节点, 返回其字面值字段(标量)的字符串形式。

    节点名支持三种等价写法:
      - class_type(如 CosyVoiceTTSWithTimestamps, 无空格);
      - 界面显示名(如 CosyVoice TTS With Timestamps, 带空格);
      - 用户重命名后的标题(title)。
    只接受字符串/数值/布尔等字面输入; 若该字段连接自其他节点(list/dict),
    无法作为文件名, 返回 None。
    """
    target = (node_name or "").strip()
    target_compact = _compact_name(target)
    for nid, nd in (prompt or {}).items():
        if not isinstance(nd, dict):
            continue
        cls = nd.get("class_type", "")
        title = title_map.get(str(nid), "")
        display = NODE_DISPLAY_NAME_MAPPINGS.get(cls, "")
        names = (cls, title, display)
        if target in names or target_compact in tuple(_compact_name(n) for n in names):
            inputs = nd.get("inputs", {})
            if field in inputs and isinstance(inputs[field], (str, int, float, bool)):
                return str(inputs[field])
    return None


def _resolve_placeholders(prefix: str, prompt=None, extra_pnginfo=None) -> str:
    """解析文件名前缀中的占位符(全部在 Python 端完成, 不依赖前端 JS)。

    - %date:yyyyMMdd%: 日期时间(ComfyUI 风格, 支持 yyyy/MM/dd/HH/mm/ss);
    - %节点名.字段%: 跨节点引用, 按节点类型名或界面标题匹配, 读取其字面字段
      (如 %CosyVoice TTS With Timestamps.voice% → instruct);
    - 无法解析的残留占位符安全移除, 避免产生非法文件名。
    """
    def _date_sub(m):
        py_fmt = m.group(1)
        for k, v in _DATE_TOKEN_MAP.items():
            py_fmt = py_fmt.replace(k, v)
        return datetime.now().strftime(py_fmt)

    prefix = _DATE_PLACEHOLDER_RE.sub(_date_sub, prefix or "")
    title_map = _build_title_map(extra_pnginfo)

    def _node_sub(m):
        val = _lookup_node_value(m.group(1).strip(), m.group(2).strip(), prompt, title_map)
        return val if val is not None else ""

    prefix = _NODE_REF_RE.sub(_node_sub, prefix)
    return _ANY_PLACEHOLDER_RE.sub("", prefix)


def _sanitize_path_part(part: str) -> str:
    return _INVALID_FS_RE.sub("_", part).strip()


class CosyVoiceTTSWithTimestamps:
    """CosyVoice 同源字级时间戳 TTS 主节点."""

    @classmethod
    def INPUT_TYPES(cls):
        # sft_spk 选项与默认值直接从服务端 /spks 拉取(启动/注册节点时即拿到真实列表),
        # 服务未启动时兜底为内置默认; 运行中注册/删除后由前端 JS 再同步.
        spks = _fetch_available_spks()
        if not spks:
            spks = ["中文男"]
        sft_spk_default = spks[0]
        return {
            "required": {
                "tts_text": (
                    "STRING",
                    {"multiline": True, "default": "你好，欢迎使用 CosyVoice 同源时间戳合成。", "placeholder": "输入待合成文本"},
                ),
                "server_url": ("STRING", {"default": "http://127.0.0.1:8000"}),
                "voice": (
                    ["zero_shot", "instruct", "sft"],
                    {"default": "zero_shot", "tooltip": "zero_shot=参考音频克隆; instruct=指令控制; sft=预置音色"},
                ),
            },
            "optional": {
                "prompt_wav": (
                    "STRING",
                    {"default": "", "tooltip": "zero_shot/instruct 的参考音频路径(服务端可访问绝对路径)。"
                                               "instruct 模式可留空, 改用 sft_spk 指定已注册预置音色做参考"},
                ),
                "prompt_text": (
                    "STRING",
                    {"default": "", "tooltip": "参考音频对应的文字内容"},
                ),
                "sft_spk": (
                    spks,
                    {
                        "default": sft_spk_default,
                        "tooltip": "sft 预置说话人(选项/默认值自动取自服务端 /spks 真实列表)。"
                                   "用 CosyVoice Register Speaker 注册/Delete 删除后自动同步; 服务未启动时显示内置默认",
                    },
                ),
                "instruct_text": (
                    "STRING",
                    {"default": "", "tooltip": "voice=instruct 时的指令(如：请用四川话表达。)；参考音色来自 prompt_wav 或 sft_spk 预置音色"},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "tooltip": "随机种子，同种子可复现（服务端在合成前对 torch/np 置种）"}),
                "stream": ("BOOLEAN", {"default": True, "tooltip": "流式合成（推荐，时间戳更精确）。注意：speed≠1.0 时自动切换为非流式（CosyVoice 变速仅支持非流式）"}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1, "tooltip": "语速倍率（0.5~2.0），1.0 为原速。仅非流式支持变速，变速时自动关闭流式"}),
                "pause_ms": (
                    "INT",
                    {"default": 120, "min": 0, "max": 2000, "tooltip": "句间插入的停顿时长(ms)，避免句子粘连"},
                ),
                "device": (
                    ["", "auto", "cuda", "cpu"],
                    {
                        "default": "",
                        "tooltip": "模型加载设备(运行中动态切换, 无需重启服务): 空=不干预(用服务端启动参数); "
                                   "cpu=内存加载(不占显存, 与 H3 视频生成同机推荐); "
                                   "cuda=显存加载(合成更快); auto=有 CUDA 自动用 GPU。"
                                   "切换会卸载并重载模型(本次合成含加载耗时)",
                    },
                ),
                "idle_timeout": (
                    "INT",
                    {
                        "default": 0, "min": 0, "max": 86400,
                        "tooltip": "空闲自动卸载秒数: 0=不干预(用服务端 --idle_timeout, 默认 300)。"
                                   "服务常驻, 空闲超时自动卸载模型释放内存/显存, 下次请求自动重载",
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("AUDIO", "WORD_TIMESTAMPS", "SRT", "DURATION_SECONDS")
    FUNCTION = "generate"
    CATEGORY = "TTS/CosyVoice"

    def generate(
        self,
        tts_text,
        server_url,
        voice="zero_shot",
        prompt_wav="",
        prompt_text="",
        sft_spk="",
        instruct_text="",
        seed=0,
        stream=True,
        speed=1.0,
        pause_ms=120,
        device="",
        idle_timeout=0,
    ):
        if not tts_text or not tts_text.strip():
            raise ValueError("[CosyVoice] tts_text 为空，请输入待合成文本")

        payload = {
            "text": tts_text,
            "voice": voice,
            "prompt_wav": prompt_wav,
            "prompt_text": prompt_text,
            "sft_spk": sft_spk,
            "instruct_text": instruct_text,
            "seed": int(seed),
            "stream": bool(stream),
            "speed": float(speed),
            "pause_ms": int(pause_ms),
        }
        if device:  # 仅在显式指定时下发, 空=服务端保持当前配置
            payload["device"] = device
        if int(idle_timeout) > 0:  # 0=不干预服务端当前空闲卸载配置
            payload["idle_timeout"] = float(int(idle_timeout))
        result = _post_json(server_url, payload)
        if not result.get("ok"):
            raise RuntimeError(f"[CosyVoice] 服务端错误: {result.get('error', '未知错误')}")

        audio_b64 = result.get("audio_base64")
        if not audio_b64:
            raise RuntimeError(f"[CosyVoice] 服务未返回音频: {json.dumps(result)[:300]}")

        raw = np.frombuffer(base64.b64decode(audio_b64), dtype=np.int16).astype(np.float32) / 32768.0  # 服务端输出 16-bit PCM WAV，先按 int16 解码再归一化到 [-1,1]
        if raw.size == 0:
            raise RuntimeError("[CosyVoice] 合成音频为空")
        waveform = torch.from_numpy(raw).reshape(1, 1, -1).float()  # [batch=1, channels=1, samples] 兼容 ComfyUI 0.34+ AUDIO 格式
        sample_rate = int(result.get("sample_rate", 22050))

        words = result.get("word_timestamps") or []
        sentences = result.get("sentence_timestamps") or []
        srt = _build_srt(sentences)
        duration = float(result.get("duration_seconds", raw.size / sample_rate))

        audio_out = {"waveform": waveform, "sample_rate": sample_rate}
        return (audio_out, json.dumps(words, ensure_ascii=False, indent=2), srt, duration)


NODE_CLASS_MAPPINGS["CosyVoiceTTSWithTimestamps"] = CosyVoiceTTSWithTimestamps
NODE_DISPLAY_NAME_MAPPINGS["CosyVoiceTTSWithTimestamps"] = "CosyVoice TTS With Timestamps"


class CosyVoiceSRTFileWriter:
    """把 CosyVoiceTTSWithTimestamps 的 SRT 输出落盘为 .srt 字幕文件.

    用于 H3 短剧/口播成片闭环: 音频(SaveAudioAdvanced) + SRT 字幕文件一起输出,
    导入剪映等剪辑软件时字幕轨与口播严格同源对齐.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "srt": ("STRING", {"multiline": True, "tooltip": "来自 CosyVoiceTTSWithTimestamps 的 SRT 输出"}),
            },
            "optional": {
                "filename_prefix": (
                    "STRING",
                    {"default": "srt/cosyvoice",
                     "tooltip": "输出路径前缀, 支持子目录与占位符:\n"
                                "1) %date:yyyyMMdd% 日期(还可用 MM/dd/HH/mm/ss);\n"
                                "2) %节点名.字段% 引用其他节点参数(如 %CosyVoice TTS With Timestamps.voice%);\n"
                                "示例: audio/TTS/cosyvoice/%date:yyyyMMdd%/%CosyVoice TTS With Timestamps.voice%_"},
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID", "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("FILE_PATH", "SRT")
    FUNCTION = "write"
    CATEGORY = "TTS/CosyVoice"

    def write(self, srt, filename_prefix="srt/cosyvoice", unique_id=None, prompt=None, extra_pnginfo=None):
        srt = srt or ""
        if not srt.strip():
            return ("", srt)
        try:
            from folder_paths import get_output_directory
            base_dir = get_output_directory()
        except Exception:
            # 兜底: 仓库根 output 目录
            base_dir = os.path.abspath(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
            )
        # 解析占位符(日期+跨节点引用) → 拆目录/文件名 → 清洗非法字符 → 自动建目录
        resolved = _resolve_placeholders(
            filename_prefix, prompt=prompt, extra_pnginfo=extra_pnginfo
        ).replace("\\", "/").strip("/")
        parts = [_sanitize_path_part(p) for p in resolved.split("/") if p and p.strip()]
        if not parts:
            parts = ["cosyvoice"]
        name_part = parts[-1]
        out_dir = os.path.join(base_dir, *parts[:-1]) if parts[:-1] else base_dir
        os.makedirs(out_dir, exist_ok=True)

        # 同名自动递增序号, 不覆盖已有字幕
        if name_part.endswith(".srt"):
            stem, path = name_part[:-4], os.path.join(out_dir, name_part)
            i = 1
            while os.path.exists(path):
                path = os.path.join(out_dir, f"{stem}_{i:05}.srt")
                i += 1
        else:
            i, path = 1, os.path.join(out_dir, f"{name_part}00001.srt")
            while os.path.exists(path):
                path = os.path.join(out_dir, f"{name_part}{i:05}.srt")
                i += 1
        with open(path, "w", encoding="utf-8") as f:
            f.write(srt)
        return (path, srt)


NODE_CLASS_MAPPINGS["CosyVoiceSRTFileWriter"] = CosyVoiceSRTFileWriter
NODE_DISPLAY_NAME_MAPPINGS["CosyVoiceSRTFileWriter"] = "CosyVoice SRT File Writer"


class CosyVoiceRegisterSpeaker:
    """把一段参考音频注册为 sft 预置说话人(服务端 POST /register_spk, 持久化到模型 spk2info.pt).

    注册一次后, CosyVoiceTTSWithTimestamps 节点 voice=sft + sft_spk=<刚注册的 id> 即可直接使用.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "spk_id": ("STRING", {"default": "中文男", "tooltip": "要注册的说话人 ID(如 中文男/主播小张/旁白老王)"}),
                "prompt_wav": ("STRING", {"default": "", "tooltip": "参考音频绝对路径(服务端可访问, 如 D:/voices/ref.wav)"}),
            },
            "optional": {
                "prompt_text": ("STRING", {"default": "", "tooltip": "参考音频对应文字(可留空)"}),
                "overwrite": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "spk_id 已存在时: False=拒绝并提示(防误覆盖); True=用新参考音频覆盖该说话人"},
                ),
                "server_url": ("STRING", {"default": "http://127.0.0.1:8000"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("RESULT", "AVAILABLE_SPKS")
    FUNCTION = "register"
    CATEGORY = "TTS/CosyVoice"

    def register(self, spk_id, prompt_wav, prompt_text="", overwrite=False, server_url="http://127.0.0.1:8000"):
        if not spk_id.strip():
            raise ValueError("[CosyVoice] spk_id 不能为空")
        if not prompt_wav or not prompt_wav.strip():
            raise ValueError("[CosyVoice] 请提供参考音频绝对路径 prompt_wav")
        payload = {"spk_id": spk_id, "prompt_wav": prompt_wav,
                   "prompt_text": prompt_text or "", "overwrite": bool(overwrite)}
        result = _post_json(server_url, payload, path="/register_spk")
        if not result.get("ok"):
            raise RuntimeError(f"[CosyVoice] 注册失败: {result.get('error', '未知错误')}")
        spks = ", ".join(result.get("available_spks") or [])
        return (f"OK: 已注册说话人 '{spk_id}', 持久化到 spk2info.pt", spks)


class CosyVoiceDeleteSpeaker:
    """删除已注册的 sft 说话人(服务端 POST /delete_spk, 同步持久化到 spk2info.pt).

    删除后 CosyVoiceTTSWithTimestamps 的 sft_spk 下拉列表点『刷新说话人列表』按钮即可同步.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "spk_id": ("STRING", {"default": "", "tooltip": "要删除的说话人 ID(如 中文男)"}),
            },
            "optional": {
                "server_url": ("STRING", {"default": "http://127.0.0.1:8000"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("RESULT", "AVAILABLE_SPKS")
    FUNCTION = "delete"
    CATEGORY = "TTS/CosyVoice"

    def delete(self, spk_id, server_url="http://127.0.0.1:8000"):
        if not spk_id.strip():
            raise ValueError("[CosyVoice] spk_id 不能为空")
        result = _post_json(server_url, {"spk_id": spk_id}, path="/delete_spk")
        if not result.get("ok"):
            raise RuntimeError(f"[CosyVoice] 删除失败: {result.get('error', '未知错误')}")
        spks = ", ".join(result.get("available_spks") or ["(无)"])
        return (f"OK: 已删除说话人 '{spk_id}'", spks)


NODE_CLASS_MAPPINGS["CosyVoiceRegisterSpeaker"] = CosyVoiceRegisterSpeaker
NODE_DISPLAY_NAME_MAPPINGS["CosyVoiceRegisterSpeaker"] = "CosyVoice Register Speaker"

NODE_CLASS_MAPPINGS["CosyVoiceDeleteSpeaker"] = CosyVoiceDeleteSpeaker
NODE_DISPLAY_NAME_MAPPINGS["CosyVoiceDeleteSpeaker"] = "CosyVoice Delete Speaker"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
