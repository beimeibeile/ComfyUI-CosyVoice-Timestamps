# -*- coding: utf-8 -*-
"""
timestamps.py 核心逻辑单测（不依赖 CosyVoice 真实模型，用 mock 验证同源性质）。
运行: python tests/test_timestamps.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))

import numpy as np
import torch

from timestamps import (
    split_sentences,
    keep_chars,
    assign_word_timestamps,
    synthesize_with_timestamps,
)

SAMPLE_RATE = 22050
FAIL = []


def check(name, cond, detail=""):
    if cond:
        print(f"  [PASS] {name}")
    else:
        print(f"  [FAIL] {name} {detail}")
        FAIL.append(name)


class MockCosy:
    """模拟 CosyVoice: 每 0.1s 输出一个音频 chunk, 总时长 = 有效字数 * 0.2s."""
    sample_rate = SAMPLE_RATE

    def inference_zero_shot(self, tts_text, prompt_text, prompt_wav, stream=True):
        n = len(keep_chars(tts_text))
        for _ in range(max(1, n)):
            yield {"tts_speech": torch.zeros(1, SAMPLE_RATE // 10)}


def test_split_sentences():
    print("[1] split_sentences")
    check("基本分句", split_sentences("你好。世界！测试？") == ["你好。", "世界！", "测试？"], str(split_sentences("你好。世界！测试？")))
    check("换行分句", split_sentences("第一行\n第二行") == ["第一行", "第二行"])
    check("空文本", split_sentences("") == [])
    check("无标点单句", split_sentences("没有标点的一句话") == ["没有标点的一句话"])


def test_keep_chars():
    print("[2] keep_chars")
    chars = keep_chars("你好，world 123。")
    words = [c["ch"] for c in chars]
    check("提取有效字", words == ["你", "好", "w", "o", "r", "l", "d", "1", "2", "3"], str(words))
    check("保留原串索引", chars[0]["idx"] == 0 and chars[1]["idx"] == 1)


def test_assign_word_timestamps():
    print("[3] assign_word_timestamps (纯函数)")
    # 模拟 1s 音频, 4 个有效字, 句起始 100ms
    audio = np.ones(SAMPLE_RATE, dtype=np.float32) * 0.3
    words = assign_word_timestamps("你好世界", audio, SAMPLE_RATE, 100.0, pause_ms=0)
    check("字数量", len(words) == 4)
    check("首字起点", words[0]["start_ms"] == 100.0)
    check("末字终点≈句尾", abs(words[-1]["end_ms"] - (100.0 + 1000.0)) < 1.0,
          f"{words[-1]['end_ms']} vs {100.0+1000.0}")
    check("时间单调递增", all(words[i]["end_ms"] <= words[i + 1]["start_ms"] for i in range(len(words) - 1)))
    # 带停顿: 末字 end 应 +pause
    words2 = assign_word_timestamps("你好世界", audio, SAMPLE_RATE, 0.0, pause_ms=120)
    check("句尾停顿并入末字", abs(words2[-1]["end_ms"] - 1120.0) < 1.0, str(words2[-1]))


def test_synthesize_with_timestamps():
    print("[4] synthesize_with_timestamps (mock 同源全流程)")
    cosy = MockCosy()
    out = synthesize_with_timestamps(cosy, "你好。世界！", voice="zero_shot", prompt_wav="x.wav", pause_ms=100)

    audio = out["audio"]
    sr = out["sample_rate"]
    check("音频非空", len(audio) > 0)
    check("句数", len(out["sentence_timestamps"]) == 2)
    check("句级时间轴覆盖", abs(out["sentence_timestamps"][0]["start_ms"] - 0.0) < 1.0)
    # 句2起点 = 句1时长 + 句间停顿100ms
    s1 = out["sentence_timestamps"][0]
    expect_s2_start = s1["end_ms"] + 100.0
    check("句间停顿写入", abs(out["sentence_timestamps"][1]["start_ms"] - expect_s2_start) < 2.0,
          f"{out['sentence_timestamps'][1]['start_ms']} vs {expect_s2_start}")
    # 同源: 最后一个字终点 <= 音频总时长(含尾停顿)
    last_w = out["word_timestamps"][-1]
    total_ms = len(audio) / sr * 1000
    check("字级终点≤音频总长", last_w["end_ms"] <= total_ms + 1.0, f"{last_w['end_ms']} vs {total_ms}")
    check("字级覆盖全部有效字", len(out["word_timestamps"]) == 4)  # 你好+世界 = 4字
    check("duration_seconds 与音频一致", abs(out["duration_seconds"] - len(audio) / sr) < 0.01)


if __name__ == "__main__":
    test_split_sentences()
    test_keep_chars()
    test_assign_word_timestamps()
    test_synthesize_with_timestamps()
    print()
    if FAIL:
        print(f"FAILED: {len(FAIL)} -> {FAIL}")
        sys.exit(1)
    print("ALL PASS")
