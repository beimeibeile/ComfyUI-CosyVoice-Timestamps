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
import urllib.request
import urllib.error

import torch
import numpy as np


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _post_json(server_url, payload, timeout=300):
    """向 CosyVoice 服务发送 JSON 请求并返回响应字典."""
    url = server_url.rstrip("/") + "/tts_with_timestamps"
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


class CosyVoiceTTSWithTimestamps:
    """CosyVoice 同源字级时间戳 TTS 主节点."""

    @classmethod
    def INPUT_TYPES(cls):
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
                    {"default": "", "tooltip": "零样本克隆的参考音频路径（服务端机器可访问的绝对路径，如 D:/voices/ref.wav）"},
                ),
                "prompt_text": (
                    "STRING",
                    {"default": "", "tooltip": "参考音频对应的文字内容"},
                ),
                "sft_spk": (
                    "STRING",
                    {"default": "", "tooltip": "voice=sft 时的预置说话人 ID（如 中文男/中文女）"},
                ),
                "instruct_text": (
                    "STRING",
                    {"default": "", "tooltip": "voice=instruct 时的指令（如：请用广东话表达。）"},
                ),
                "language": ("STRING", {"default": "中文"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "stream": ("BOOLEAN", {"default": True, "tooltip": "流式合成（推荐，时间戳更精确）"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.5, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
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
        language="中文",
        seed=0,
        stream=True,
        temperature=0.7,
        top_p=0.8,
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
            "language": language,
            "seed": int(seed),
            "stream": bool(stream),
            "temperature": float(temperature),
            "top_p": float(top_p),
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
                    {"default": "srt/cosyvoice", "tooltip": "输出文件名前缀，可带子目录（如 srt/短剧01）"},
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("FILE_PATH", "SRT")
    FUNCTION = "write"
    CATEGORY = "TTS/CosyVoice"

    def write(self, srt, filename_prefix="srt/cosyvoice"):
        srt = srt or ""
        if not srt.strip():
            return ("", srt)
        try:
            from folder_paths import get_output_directory, get_save_image_path
            full_output_folder, filename, counter, _subfolder, _prefix = get_save_image_path(
                filename_prefix, get_output_directory()
            )
        except Exception:
            # 兜底: 相对仓库根 output 目录
            import os
            full_output_folder = os.path.abspath(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
            )
            filename = "cosyvoice_srt"
            counter = 1
            os.makedirs(full_output_folder, exist_ok=True)
        import os
        path = os.path.join(full_output_folder, f"{filename}_{counter:05}.srt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(srt)
        return (path, srt)


NODE_CLASS_MAPPINGS["CosyVoiceSRTFileWriter"] = CosyVoiceSRTFileWriter
NODE_DISPLAY_NAME_MAPPINGS["CosyVoiceSRTFileWriter"] = "CosyVoice SRT File Writer"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
