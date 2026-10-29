# ComfyUI-CosyVoice-Timestamps

CosyVoice 本地 TTS + **同源字级时间戳**（source-aligned word-level timestamps）标准 ComfyUI 自定义节点。

> **同源**：时间戳直接取自 CosyVoice 本次合成的同一个音频流（句级为流式合成真实音频边界，字级为同源派生 + VAD 停顿校正），而非事后用 ASR/强制对齐测量。适合需要"逐词时间边界"驱动下游时序的工作流（如 MiniMax H3 短剧对话时序、字幕、口型同步）。

## 架构

```
┌────────────────────────────── ComfyUI ──────────────────────────────┐
│  custom_nodes/ComfyUI-CosyVoice-Timestamps/                         │
│    nodes.py  CosyVoiceTTSWithTimestamps 节点（轻量 HTTP 客户端）      │
│      输出 AUDIO / WORD_TIMESTAMPS / SRT / DURATION_SECONDS           │
└───────────────────────────────┬────────────────────────────────────┘
                                │ HTTP/JSON (127.0.0.1:8000)
┌───────────────────────────────▼────────────────────────────────────┐
│  独立服务（Python 3.10 venv，与 ComfyUI 环境完全隔离）               │
│  server/tts_server.py  FastAPI + WebSocket 就绪                      │
│  server/timestamps.py  流式逐句合成 + 句级精确计时 + 字级分配        │
│      └── CosyVoice2-0.5B（本地 GPU）                                │
└─────────────────────────────────────────────────────────────────────┘
```

依赖隔离是刻意设计：CosyVoice 依赖（matcha-tts / WeTextProcessing 等）不进入 ComfyUI 环境，避免破坏现有 Qwen3-TTS 的 transformers 4.57.x 环境。

## 目录结构

```
ComfyUI-CosyVoice-Timestamps/
├─ __init__.py              标准 custom node 入口
├─ nodes.py                 CosyVoiceTTSWithTimestamps 节点
├─ requirements.txt         ComfyUI 侧（零额外依赖，仅说明）
├─ pyproject.toml           打包配置
├─ README.md                本文件
├─ server/
│  ├─ tts_server.py         FastAPI 服务（模型加载 + 合成接口）
│  ├─ timestamps.py         同源时间戳推导核心（分句/计时/字级分配/VAD校正）
│  └─ requirements-server.txt
├─ examples/
│  └─ h3_workflow.json      H3 接入 API 示例
└─ docs/
   └─ 精度与调优.md
```

## 快速开始

### 第 1 步：部署独立服务（一次性）

```powershell
# 1.1 创建 Python 3.10 环境（建议 conda）
conda create -n cosyvoice -y python=3.10
conda activate cosyvoice

# 1.2 安装 CosyVoice 本体
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

# 1.3 安装本服务依赖
pip install -r requirements-server.txt

# 1.4 下载模型（二选一源）
pip install modelscope
modelscope download --model iic/CosyVoice2-0.5B --local_dir models\CosyVoice2-0.5B
```

> 若 HuggingFace 可访问：`pip install huggingface_hub && huggingface-cli download FunAudioLLM/CosyVoice2-0.5B --local-dir models/CosyVoice2-0.5B`

### 第 2 步：启动服务

```powershell
# 在仓库根目录执行
conda activate cosyvoice
python server/tts_server.py --model_dir models/CosyVoice2-0.5B --port 8000 --cosyvoice_dir <你的CosyVoice源码路径>
# 看到 "[CosyVoice] 模型就绪" 即成功；健康检查: http://127.0.0.1:8000/health
# 若 8000 端口被其他服务占用：换用 --port 8001（或任意空闲端口），并把节点参数 server_url 同步改为 http://127.0.0.1:8001
```

### 第 3 步：安装 ComfyUI 节点

```powershell
# 将仓库克隆/复制到 custom_nodes
cd <你的ComfyUI目录>\custom_nodes
git clone <本仓库地址> ComfyUI-CosyVoice-Timestamps
# 或直接把本项目目录拷贝为 custom_nodes/ComfyUI-CosyVoice-Timestamps
```

重启 ComfyUI，节点位于 **`TTS/CosyVoice` → `CosyVoice TTS With Timestamps`**。

## 节点参数

| 参数 | 说明 |
|---|---|
| `tts_text` | 待合成文本（自动分句，多行也可） |
| `server_url` | CosyVoice 服务地址，默认 `http://127.0.0.1:8000` |
| `voice` | `zero_shot`（参考音频克隆）/ `instruct`（指令控制）/ `sft`（预置音色） |
| `prompt_wav` | zero_shot 参考音频路径（服务端可访问的绝对路径） |
| `prompt_text` | 参考音频对应文字 |
| `sft_spk` | sft 预置说话人（如 `中文男`/`中文女`） |
| `instruct_text` | instruct 指令（如 `请用广东话表达。`） |
| `seed` | 随机种子，同种子可复现 |
| `stream` | 流式合成（推荐开启，时间戳基于流式真实音频） |
| `pause_ms` | 句间停顿时长，默认 120ms（防止句子粘连） |

## 输出

- **AUDIO**：ComfyUI 标准音频（22.05kHz 单声道 WAV 字节流）
- **WORD_TIMESTAMPS**：JSON 字符串，字级时间戳
  ```json
  [{"word": "你", "start_ms": 0.0, "end_ms": 230.5}, ...]
  ```
- **SRT**：按句聚合的标准字幕
- **DURATION_SECONDS**：合成总时长

## H3 短剧工作流接入

`examples/h3_workflow.json` 提供 API 调用格式的最小示例。典型流程：

```
剧本文本 → CosyVoiceTTSWithTimestamps（本节点）
              ├─ WORD_TIMESTAMPS → 驱动 H3 分段提示词的 <d> 对话时序
              └─ AUDIO → 与 H3 视频合成/二次配音
```

在 ComfyUI 中：添加本节点 → 将 `WORD_TIMESTAMPS`（STRING）接入你现有的提示词解析节点 → 时间戳即用于对齐每句台词的画面时序。

## 精度说明与调优

详见 [`docs/精度与调优.md`](docs/精度与调优.md)。要点：

- 句级时间戳：与合成音频**严格同源**（误差 < 1 帧 ≈ 0.045ms）
- 字级时间戳：同源派生（文本加权 + VAD 停顿吸附），启发式精度；如需更高字级精度，接入本机已有的 `Qwen3-ForcedAligner-0.6B`（ComfyUI-Qwen3-ASR）做交叉校验/修正
- 建议先用 2~3 句短文本验证时间戳分布是否符合预期，再调 `pause_ms` / 停顿阈值

## Roadmap

- [x] v0.1：CosyVoice2-0.5B + 流式同源时间戳 + 标准节点
- [ ] v0.2：WebSocket 流式接口（对接实时工作流）
- [ ] v0.3：切换 Fun-CosyVoice3-0.5B_RL（方言/指令更强，适配新模型推理签名）
- [ ] v0.4：Qwen3-ForcedAligner 交叉校验自动修正
