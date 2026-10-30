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
import gc
import io
import os
import sys
import threading
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
    # 动态配置(空/0=不干预, 用服务端启动参数)
    device: str = ""                   # auto | cuda | cpu: 模型加载设备(内存/显存)
    idle_timeout: float = 0            # 空闲卸载秒数


app = FastAPI(title="CosyVoice TTS With Timestamps", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_cosyvoice = None
_SAMPLE_RATE = 22050
_model_lock = threading.Lock()          # 串行化加载/卸载
_last_activity = 0.0                    # 最近一次请求完成时间(epoch 秒)
_lazy_enabled = False                   # --lazy: 惰性加载(用时才加载)
_idle_timeout = 300.0                   # 空闲 N 秒后自动卸载模型(0=不卸载)
_LOADING = False                        # 加载中标记(供 health 展示)
_original_cuda_available = None         # torch.cuda.is_available 原函数(设备切换用)


def _set_cuda_visible(on: bool):
    """控制 torch 视角的 CUDA 可用性: on=False 时 patch is_available 为 False,
    CosyVoice 各组件(model/frontend/flow)据此全部落到 CPU 加载与推理(不占显存);
    on=True 恢复原函数(正常 GPU 路径). 无需重启进程, 运行中可动态切换."""
    global _original_cuda_available
    import torch
    if _original_cuda_available is None:
        _original_cuda_available = torch.cuda.is_available
    if on:
        torch.cuda.is_available = _original_cuda_available
    else:
        torch.cuda.is_available = lambda: False


def _apply_config(device: str = "", idle_timeout: float = 0):
    """按请求动态调整配置: 更新空闲卸载时间; 设备不一致时卸载模型并切换(下次请求自动重载)."""
    global _idle_timeout
    changed = False
    if idle_timeout and idle_timeout > 0 and idle_timeout != _idle_timeout:
        _idle_timeout = idle_timeout
        print(f"[CosyVoice] 空闲卸载时间更新为 {_idle_timeout:.0f}s")
        changed = True
    if device and device != app.state.device:
        print(f"[CosyVoice] 切换模型加载设备: {app.state.device} -> {device} (卸载后按新设备重载)")
        _unload_model()
        app.state.device = device
        _set_cuda_visible(device != "cpu")
        changed = True
    if changed:
        print(f"[CosyVoice] 当前配置: device={app.state.device}, idle_timeout={_idle_timeout:.0f}s")


def _load_model_impl():
    """真正加载模型(须持 _model_lock 调用). 设备由 app.state.device 决定."""
    global _cosyvoice, _SAMPLE_RATE
    model_dir = app.state.model_dir
    print(f"[CosyVoice] {'惰性' if _lazy_enabled else ''}加载模型({app.state.device}): {model_dir}")
    t0 = time.time()
    _set_cuda_visible(app.state.device != "cpu")  # cpu 时强制 torch 视角无 CUDA
    from cosyvoice.cli.cosyvoice import AutoModel
    _cosyvoice = AutoModel(model_dir=model_dir)
    if hasattr(_cosyvoice, "eval"):  # CosyVoice2 是业务封装类，无 eval；CosyVoice3 等 nn.Module 容器可 eval
        _cosyvoice.eval()
    _SAMPLE_RATE = int(getattr(_cosyvoice, "sample_rate", 22050))
    print(f"[CosyVoice] 模型就绪({app.state.device}, 耗时 {round(time.time() - t0, 1)}s), 采样率 {_SAMPLE_RATE}Hz")


def _ensure_loaded():
    """确保模型已加载; 惰性模式下首次请求时自动加载(阻塞至完成)."""
    global _cosyvoice, _LOADING
    if _cosyvoice is not None:
        return
    with _model_lock:
        if _cosyvoice is None:
            _LOADING = True
            try:
                _load_model_impl()
            finally:
                _LOADING = False


def _unload_model():
    """卸载模型并释放显存/内存."""
    global _cosyvoice
    with _model_lock:
        if _cosyvoice is None:
            return
        print("[CosyVoice] 空闲超时, 卸载模型释放资源")
        _cosyvoice = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        print("[CosyVoice] 模型已卸载, 显存/内存已释放")


def _idle_monitor():
    """后台守护线程: 周期性检查空闲时长, 超过阈值则卸载模型."""
    while True:
        time.sleep(10)
        try:
            if _cosyvoice is not None and _idle_timeout > 0 and _last_activity > 0:
                if time.time() - _last_activity >= _idle_timeout:
                    _unload_model()
        except Exception:
            traceback.print_exc()


@app.on_event("startup")
def _startup():
    # 非惰性模式: 启动即加载; 惰性模式: 首次请求时才加载
    if not _lazy_enabled:
        _ensure_loaded()
    # 空闲监控线程(守护)
    threading.Thread(target=_idle_monitor, daemon=True, name="cosyvoice-idle-monitor").start()


@app.get("/health")
def health():
    status = "ok" if _cosyvoice is not None else ("loading" if _LOADING else "idle")
    return {
        "status": status,
        "model_loaded": _cosyvoice is not None,
        "sample_rate": _SAMPLE_RATE,
        "device": getattr(app.state, "device", "auto"),
        "lazy": _lazy_enabled,
        "idle_timeout": _idle_timeout,
    }


@app.post("/tts_with_timestamps")
def tts_with_timestamps(req: TTSRequest):
    global _last_activity
    _apply_config(req.device, req.idle_timeout)  # 先应用动态配置(设备切换=卸载+下次加载)
    _ensure_loaded()  # 首次/超时/设备切换后自动加载(阻塞等待, 节点端 HTTP timeout=300s)
    _last_activity = time.time()
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


class RegisterSpkRequest(BaseModel):
    spk_id: str
    prompt_wav: str
    prompt_text: str = ""


@app.post("/register_spk")
def register_spk(req: RegisterSpkRequest):
    """把一段参考音频注册为 sft 预置说话人, 写入模型目录 spk2info.pt(持久化).

    注册后 sft 模式即可用 sft_spk=spk_id 调用; 说话人 embedding 由当前模型前端生成,
    与 CosyVoice2 完全同源(比直接引用其它模型的 spk2info.pt 兼容).
    """
    global _last_activity
    try:
        _ensure_loaded()
        _last_activity = time.time()
        spk_id = (req.spk_id or "").strip()
        if not spk_id:
            return {"ok": False, "error": "spk_id 不能为空"}
        if not req.prompt_wav or not os.path.isfile(req.prompt_wav):
            return {"ok": False, "error": f"参考音频不存在: {req.prompt_wav}"}
        if not hasattr(_cosyvoice, "frontend") or not hasattr(_cosyvoice.frontend, "spk2info"):
            return {"ok": False, "error": "当前模型前端不支持注册说话人(spk2info)"}

        # 生成与当前模型完全同源的说话人 embedding(与 zero_shot 同一条前端提取链路)。
        # 注意: 不能用官方 add_zero_shot_spk——它存入的 key 是 llm_embedding/flow_embedding,
        # 而 frontend_sft 读取的是 spk2info[spk_id]['embedding'], 结构不匹配(CosyVoice2 官方 sft 未启用)。
        model_input = _cosyvoice.frontend.frontend_zero_shot(
            '', req.prompt_text or '', req.prompt_wav, _cosyvoice.sample_rate, ''
        )
        embedding = model_input['llm_embedding']  # 说话人 embedding(CosyVoice2 的 llm/flow 共用)
        _cosyvoice.frontend.spk2info[spk_id] = {'embedding': embedding}
        _cosyvoice.save_spkinfo()  # 写入 {model_dir}/spk2info.pt, 重启后仍可用
        spks = list(_cosyvoice.frontend.spk2info.keys())
        return {"ok": True, "spk_id": spk_id, "available_spks": spks,
                "note": "已持久化到模型目录 spk2info.pt, 重启服务后 sft 仍可用"}
    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


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
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu"],
        help="推理设备: auto=有 CUDA 用 GPU, 默认; cuda=强制 GPU; cpu=强制 CPU(不占显存,"
             "与 ComfyUI/H3 视频生成同机跑时推荐, 避免显存冲突)",
    )
    parser.add_argument(
        "--lazy", action="store_true", default=False,
        help="惰性加载: 服务常驻但启动时不加载模型, 首次合成请求时才加载; "
             "空闲超时后自动卸载释放显存/内存",
    )
    parser.add_argument(
        "--idle_timeout", type=float, default=300.0,
        help="空闲自动卸载阈值(秒), 默认 300(5 分钟); 0=空闲不卸载",
    )
    args = parser.parse_args()

    src = args.cosyvoice_dir or _COSYVOICE_SRC_HINT
    if os.path.isdir(src):
        sys.path.insert(0, src)
        print(f"[CosyVoice] 已加入源码路径: {src}")
    else:
        print("[CosyVoice] 提示: 未找到 CosyVoice 源码目录, 将依赖已 pip 安装的 cosyvoice 包")

    # 设备控制统一走 torch.cuda.is_available 的 patch 机制(见 _set_cuda_visible),
    # 启动时不设 CUDA_VISIBLE_DEVICES 环境变量——运行时改 env 对已初始化的 torch 无效,
    # 动态切换内存/显存必须靠加载时 patch torch 视角的 CUDA 可用性。
    if args.device == "cpu":
        print("[CosyVoice] --device cpu: 模型将加载到内存推理(不占 GPU 显存, 可随时切回显存)")
    elif args.device == "cuda":
        print("[CosyVoice] --device cuda: 强制 GPU 推理")

    app.state.device = args.device
    app.state.model_dir = os.path.abspath(args.model_dir)
    global _lazy_enabled, _idle_timeout
    _lazy_enabled = bool(args.lazy)
    _idle_timeout = float(args.idle_timeout)
    if _lazy_enabled:
        print(f"[CosyVoice] --lazy: 惰性加载已启用, 空闲 {_idle_timeout:.0f}s 自动卸载模型")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
