# -*- coding: utf-8 -*-
"""
CosyVoice 独立 TTS 服务（带同源字级时间戳）。

用法:
    # 1) 创建 Python 3.10 venv 并安装依赖(见 README)
    # 2) 下载模型: modelscope download --model iic/CosyVoice2-0.5B --local_dir ./models/CosyVoice2-0.5B
    # 3) 启动服务
    python server/tts_server.py --model_dir models/CosyVoice2-0.5B --port 8000

接口:
    GET  /health                    健康检查
    POST /tts_with_timestamps       合成文本 → 音频 + 字级/句级同源时间戳

依赖隔离: 本服务独立运行于 venv, 与 ComfyUI 环境完全隔离;
ComfyUI 侧仅通过 HTTP 调用(见根目录 nodes.py)。
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import time
import traceback

import numpy as np

# ---- FastAPI -------------------------------------------------------------
try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
except ImportError:  # pragma: no cover
    print("缺少 fastapi/uvicorn/pydantic, 请执行: pip install fastapi uvicorn pydantic")
    sys.exit(1)

# CosyVoice 源码目录（clone 后放入 models/_cosyvoice_src，或由 --cosyvoice_dir 指定）
_COSYVOICE_SRC_HINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "_cosyvoice_src")


class TTSRequest(BaseModel):
    text: str
    voice: str = "zero_shot"          # zero_shot | instruct | sft
    prompt_wav: str = ""
    prompt_text: str = ""
    sft_spk: str = ""
    instruct_text: str = ""
    language: str = "中文"
    seed: int = 0
    stream: bool = True
    temperature: float = 0.7
    top_p: float = 0.8
    speed: float = 1.0
    pause_ms: int = 120


app = FastAPI(title="CosyVoice TTS With Timestamps", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_cosyvoice = None
_SAMPLE_RATE = 22050


@app.on_event("startup")
def _load_model():
    global _cosyvoice
    model_dir = app.state.model_dir
    print(f"[CosyVoice] 加载模型: {model_dir}")
    from cosyvoice.cli.cosyvoice import AutoModel
    _cosyvoice = AutoModel(model_dir=model_dir)
    _cosyvoice.eval()
    global _SAMPLE_RATE
    _SAMPLE_RATE = int(getattr(_cosyvoice, "sample_rate", 22050))
    print(f"[CosyVoice] 模型就绪, 采样率 {_SAMPLE_RATE}Hz")


@app.get("/health")
def health():
    return {
        "status": "ok" if _cosyvoice is not None else "loading",
        "model_loaded": _cosyvoice is not None,
        "sample_rate": _SAMPLE_RATE,
    }


@app.post("/tts_with_timestamps")
def tts_with_timestamps(req: TTSRequest):
    if _cosyvoice is None:
        raise RuntimeError("模型尚未加载完成，请稍后重试")
    t0 = time.time()
    from timestamps import synthesize_with_timestamps  # 与 server 同目录

    try:
        # 同源: 同一种子下的随机采样, 保证可复现
        if req.seed is not None:
            import torch
            torch.manual_seed(int(req.seed))
            np.random.seed(int(req.seed))

        out = synthesize_with_timestamps(
            _cosyvoice,
            text=req.text,
            voice=req.voice,
            prompt_wav=req.prompt_wav,
            prompt_text=req.prompt_text,
            sft_spk=req.sft_spk,
            instruct_text=req.instruct_text,
            seed=req.seed,
            stream=req.stream,
            pause_ms=req.pause_ms,
            speed=req.speed,
        )

        audio: np.ndarray = out["audio"]
        wav_io = io.BytesIO()
        _save_wav(wav_io, audio, out["sample_rate"])
        audio_b64 = base64.b64encode(wav_io.getvalue()).decode("ascii")

        elapsed = round(time.time() - t0, 2)
        return {
            "ok": True,
            "audio_base64": audio_b64,
            "sample_rate": out["sample_rate"],
            "word_timestamps": out["word_timestamps"],
            "sentence_timestamps": out["sentence_timestamps"],
            "duration_seconds": out["duration_seconds"],
            "elapsed_seconds": elapsed,
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
        }


def _save_wav(buf: io.BytesIO, audio: np.ndarray, sample_rate: int) -> None:
    """用标准库写 WAV(16-bit PCM), 避免依赖 torchaudio/scipy."""
    import wave
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def main():
    parser = argparse.ArgumentParser(description="CosyVoice TTS 服务(同源时间戳)")
    parser.add_argument("--model_dir", required=True, help="CosyVoice 模型目录(如 models/CosyVoice2-0.5B)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--cosyvoice_dir", default="", help="CosyVoice 源码目录(自动加入 sys.path)")
    args = parser.parse_args()

    src = args.cosyvoice_dir or _COSYVOICE_SRC_HINT
    if os.path.isdir(src):
        sys.path.insert(0, src)
        print(f"[CosyVoice] 已加入源码路径: {src}")
    else:
        print("[CosyVoice] 提示: 未找到 CosyVoice 源码目录, 将依赖已 pip 安装的 cosyvoice 包")

    app.state.model_dir = os.path.abspath(args.model_dir)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
