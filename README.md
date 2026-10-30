# ComfyUI-CosyVoice-Timestamps

CosyVoice 本地 TTS + **同源字级时间戳**（source-aligned word-level timestamps）标准 ComfyUI 自定义节点。

> **同源**
> ：时间戳直接取自 CosyVoice 本次合成的同一个音频流（句级为流式合成真实音频边界，字级为同源派生 + VAD 停顿校正），而非事后用 ASR / 强制对齐测量。适合需要 "逐词时间边界" 驱动下游时序的工作流（如 MiniMax H3 短剧对话时序、字幕、口型同步）。

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



```
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

> 若 HuggingFace 可访问：
> `pip install huggingface_hub && huggingface-cli download FunAudioLLM/CosyVoice2-0.5B --local-dir models/CosyVoice2-0.5B`

### 第 2 步：启动服务

```
# 在仓库根目录执行
conda activate cosyvoice
python server/tts_server.py --model_dir models/CosyVoice2-0.5B --port 8000 --cosyvoice_dir <你的CosyVoice源码路径>
# 看到 "[CosyVoice] 模型就绪" 即成功；健康检查: http://127.0.0.1:8000/health
# 若 8000 端口被其他服务占用：换用 --port 8001（或任意空闲端口），并把节点参数 server_url 同步改为 http://127.0.0.1:8001
```

**资源优化参数（与 ComfyUI/H3 视频生成同机时推荐）**：

| 参数 | 作用 |
| --- | --- |
| `--device cpu` | 强制 CPU 推理，不占 GPU 显存（显存全留给 ComfyUI/H3） |
| `--lazy` | **惰性加载**：服务常驻但启动时不加载模型，首次合成请求时才加载 |
| `--idle_timeout N` | 空闲 N 秒后自动卸载模型释放内存/显存（默认 300=5 分钟；0=不卸载） |

示例（推荐组合）：

```
python server/tts_server.py --model_dir models/CosyVoice2-0.5B --port 8000 --cosyvoice_dir <源码路径> --device cpu --lazy --idle_timeout 300
```

行为说明：服务进程常驻（端口不变，ComfyUI 节点无需改动）；`/health` 在模型未加载时返回
`"status":"idle"`；首次合成请求自动加载模型（CPU 约 50s / GPU 约 20-40s，节点 HTTP 超时 300s 足够），
空闲超过 `idle_timeout` 秒后自动卸载（GPU 模式释放显存，CPU 模式释放约 2GB 内存）。

### 第 3 步：安装 ComfyUI 节点



```
# 将仓库克隆/复制到 custom_nodes
cd <你的ComfyUI目录>\custom_nodes
git clone <本仓库地址> ComfyUI-CosyVoice-Timestamps
# 或直接把本项目目录拷贝为 custom_nodes/ComfyUI-CosyVoice-Timestamps
```

重启 ComfyUI，节点位于 `TTS/CosyVoice` → 包含三个节点：

* **`CosyVoice TTS With Timestamps`**：主合成节点（AUDIO + 字级/句级时间戳 + SRT + 时长）
* **`CosyVoice SRT File Writer`**：把 SRT 落盘为 `.srt` 字幕文件（H3 短剧/口播闭环）
* **`CosyVoice Register Speaker`**：把一段参考音频注册为 sft 预置说话人（填 `spk_id` + 参考音频路径，执行一次即写入模型目录 `spk2info.pt`，长期有效）

sft 使用流程：先用 `CosyVoice Register Speaker` 注册（如 `spk_id=中文男`，参考音频用周鸿祎.mp3），再在主节点 `voice=sft`、`sft_spk=中文男` 合成。

## 节点参数



| 参数              | 说明                                                                                                                                                           |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `tts_text`      | 待合成文本（自动分句，多行也可）                                                                                                                                             |
| `server_url`    | CosyVoice 服务地址，默认 `http://127.0.0.1:8000`                                                                                                                    |
| `voice`         | `zero_shot`（参考音频克隆，默认）/ `instruct`（参考音色+指令控制）/ `sft`（预置说话人）。**instruct** 需填 `prompt_wav`（参考音色）+ `instruct_text`（指令），`prompt_text` 不参与（CosyVoice2.inference_instruct2 签名）；**sft** 需先用 `CosyVoice Register Speaker` 节点（或 `POST /register_spk`）把一段参考音频注册为说话人，注册会生成与模型同源的 embedding 并持久化到模型目录 `spk2info.pt`，之后 `sft_spk` 直接填注册的 ID 即可 |
| `prompt_wav`    | zero\_shot 参考音频路径（服务端可访问的绝对路径；可先用 CosyVoice 源码自带的 `models/_cosyvoice_src/asset/zero_shot_prompt.wav` 验证） |
| `prompt_text`   | 参考音频对应文字                                                                                                                                                     |
| `sft_spk`       | sft 预置说话人（如 `中文男`/`中文女`）                                                                                                                                     |
| `instruct_text` | instruct 指令（如 `请用广东话表达。`）                                                                                                                                    |
| `seed`          | 随机种子，同种子可复现                                                                                                                                                  |
| `stream`        | 流式合成（推荐开启，时间戳基于流式真实音频）                                                                                                                                       |
| `pause_ms`      | 句间停顿时长，默认 120ms（防止句子粘连）                                                                                                                                      |
| `device`        | **模型加载设备（运行中动态切换，无需重启服务）**：空=不干预（用服务端启动参数）；`cpu`=内存加载（不占显存，与 H3 视频生成同机推荐）；`cuda`=显存加载（合成更快）；`auto`=有 CUDA 自动用 GPU。切换会卸载并重载模型（本次合成含加载耗时） |
| `idle_timeout`  | **空闲自动卸载秒数**：0=不干预（用服务端 `--idle_timeout`，默认 300）。服务常驻，空闲超时自动卸载模型释放内存/显存，下次请求自动重载                                                        |

> 提示：`device`/`idle_timeout` 每次合成请求都会下发到服务端并即时生效（可组合使用：
> 跑 H3 时 `device=cpu`，合成提速时 `device=cuda`，两种模式共用一份 `idle_timeout`）。

## 输出



* **AUDIO**：ComfyUI 标准音频（CosyVoice2 为 24kHz 单声道，sample_rate 由服务端返回）

* **WORD\_TIMESTAMPS**：JSON 字符串，字级时间戳



```
[{"word": "你", "start_ms": 0.0, "end_ms": 230.5}, ...]
```



* **SRT**：按句聚合的标准字幕

* **DURATION\_SECONDS**：合成总时长

## H3 短剧工作流接入

`examples/h3_workflow.json` 提供 API 调用格式的完整闭环示例。ComfyUI 内流程：

```
剧本文本 → CosyVoiceTTSWithTimestamps（本节点）
              ├─ AUDIO  ──────────────→ SaveAudioAdvanced（存 flac/wav）
              ├─ SRT    ──────────────→ CosyVoiceSRTFileWriter（落盘 .srt 字幕文件）
              ├─ WORD_TIMESTAMPS ─────→ 驱动 H3 分段提示词的 <d> 对话时序 / 逐字对齐
              └─ DURATION_SECONDS ────→ 对齐 H3 视频片段时长
```

接入步骤（ComfyUI 画布）：

1. 添加 `CosyVoiceTTSWithTimestamps`，填入台词文本与 `prompt_wav`/`prompt_text`（参考音色）。
2. `AUDIO` → `SaveAudioAdvanced`（格式选 flac），输出口播音频文件。
3. `SRT` → `CosyVoiceSRTFileWriter`，输出同源字幕文件 `output/srt/*.srt`。
4. 剪辑合成（如剪映）：H3 生成的视频片段 + 口播音频 + `.srt` 字幕直接导入，
   字幕轨与口播严格同源对齐；如需逐字字幕，用 `WORD_TIMESTAMPS` 生成逐字 SRT/ASS 再导入。

时间戳与音频来自同一次流式合成，句级边界即真实音频边界，因此字幕不会出现
"口播已结束字幕还没走完"或反向错位的不同源漂移。

## 精度说明与调优

详见 [docs / 精度与调优.md](docs/精度与调优.md)。要点：



* 句级时间戳：与合成音频**严格同源**（误差 < 1 帧 ≈ 0.045ms）

* 字级时间戳：同源派生（文本加权 + VAD 停顿吸附），启发式精度；如需更高字级精度，接入本机已有的 `Qwen3-ForcedAligner-0.6B`（ComfyUI-Qwen3-ASR）做交叉校验 / 修正

* 建议先用 2\~3 句短文本验证时间戳分布是否符合预期，再调 `pause_ms` / 停顿阈值

## Roadmap



* [x] v0.1：CosyVoice2-0.5B + 流式同源时间戳 + 标准节点

* [ ] v0.2：WebSocket 流式接口（对接实时工作流）

* [ ] v0.3：切换 Fun-CosyVoice3-0.5B\_RL（方言 / 指令更强，适配新模型推理签名）

* [ ] v0.4：Qwen3-ForcedAligner 交叉校验自动修正