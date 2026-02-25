import gradio as gr
import torch
import soundfile as sf
import os
import datetime
import re
import shutil
import numpy as np
from qwen_tts import Qwen3TTSModel

# ================= 配置区域 =================
# 注意：GitHub 上不传大模型，让用户自己把模型解压到这个文件夹
MODEL_PATH = "Qwen3-Base"
VOICE_DIR = "my_voices"
OUTPUT_DIR = "recordings"

# 自动创建目录
for d in [VOICE_DIR, OUTPUT_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

print("="*60)
print("🚀 Qwen3-Voice-Studio (8GB 显存优化版) 启动中...")
print("="*60)

# ================= 核心引擎 (Engine) =================
try:
    print(f"⏳ 正在加载模型 (路径：{os.path.abspath(MODEL_PATH)})...")

    # 显存清理
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 核心优化：使用 bfloat16 和 flash-attention/sdpa 节省显存
    model = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa"
    )
    print("✅ 模型加载成功！")
except Exception as e:
    print(f"❌ 模型加载失败：{e}")
    print("💡 请确保你已经下载了 Qwen3-TTS 模型并放在 'Qwen3-Base' 文件夹中。")
    model = None

# ================= 业务逻辑 (Logic) =================

def get_voice_list():
    if not os.path.exists(VOICE_DIR): return []
    return [f for f in os.listdir(VOICE_DIR) if f.endswith(('.wav', '.mp3'))]

def split_text(text):
    """长文本切分算法"""
    sentences = re.split(r'([，。！？；,.!?;])', text)
    parts = []
    for i in range(0, len(sentences)-1, 2):
        parts.append(sentences[i] + sentences[i+1])
    if len(sentences) % 2 == 1:
        parts.append(sentences[-1])
    return [p.strip() for p in parts if p.strip()]

def clone_voice(audio, text, lang):
    if model is None: return None, "❌ 错误：模型未加载"
    if not audio: return None, "⚠️ 请先上传声音"

    try:
        torch.cuda.empty_cache()
        wavs, sr = model.generate_voice_clone(
            text=text,
            ref_audio=audio,
            x_vector_only_mode=True,
            language=lang
        )
        out_path = os.path.join(OUTPUT_DIR, "temp_preview.wav")
        sf.write(out_path, wavs[0], sr)
        return out_path, "✅ 合成成功"
    except Exception as e:
        return None, f"❌ 报错：{str(e)}"

def save_voice(audio, name):
    if not audio or not name: return gr.update(), "❌ 缺少音频或名称"
    safe_name = "".join([c for c in name if c.isalnum() or c in (' ', '_', '-')]).strip()
    save_path = os.path.join(VOICE_DIR, f"{safe_name}.wav")
    shutil.copy(audio, save_path)
    return gr.update(choices=get_voice_list(), value=f"{safe_name}.wav"), f"已保存：{safe_name}"

def tts_long(text, voice, lang):
    if model is None: return None, "❌ 模型未加载"
    if not voice: return None, "⚠️ 请选择音色"

    try:
        voice_path = os.path.join(VOICE_DIR, voice)
        parts = split_text(text)
        all_audio = []
        sr = 24000

        for part in parts:
            torch.cuda.empty_cache() # 防止爆显存
            wavs, sr = model.generate_voice_clone(
                text=part, ref_audio=voice_path,
                x_vector_only_mode=True, language=lang
            )
            all_audio.append(wavs[0])

        final_wav = np.concatenate(all_audio)
        out_name = f"tts_{datetime.datetime.now().strftime('%H%M%S')}.wav"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        sf.write(out_path, final_wav, sr)
        return out_path, f"✅ 长文本合成完毕：{out_name}"
    except Exception as e:
        return None, f"❌ 合成出错：{str(e)}"

# ================= 界面 (UI) =================
with gr.Blocks(title="Qwen3 Voice Studio", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🎙️ Qwen3-Voice-Studio (本地版)")
    gr.Markdown("基于 Qwen3-TTS | 支持 8GB 显存 | 长文本合成 | 音色克隆")

    with gr.Tab("✨ 音色克隆"):
        with gr.Row():
            with gr.Column():
                inp_audio = gr.Audio(type="filepath", label="1. 上传干声 (3-10s)")
                inp_text = gr.Textbox(value="你好，这是我的声音克隆测试。", label="2. 测试文本")
                btn_test = gr.Button("🚀 测试合成", variant="primary")
            with gr.Column():
                out_audio = gr.Audio(label="3. 试听结果")
                inp_name = gr.Textbox(placeholder="例如：老王", label="4. 音色命名")
                btn_save = gr.Button("💾 保存到音色库")
                log_1 = gr.Textbox(label="日志")

    with gr.Tab("📖 长文本合成"):
        with gr.Row():
            with gr.Column():
                drop_voice = gr.Dropdown(choices=get_voice_list(), label="选择音色")
                drop_lang = gr.Dropdown(choices=["Chinese", "English"], value="Chinese", label="语言")
                txt_long = gr.Textbox(lines=6, label="输入长文本")
                btn_run = gr.Button("开始合成", variant="primary")
            with gr.Column():
                out_final = gr.Audio(label="合成结果")
                log_2 = gr.Textbox(label="日志")

    btn_test.click(clone_voice, [inp_audio, inp_text, drop_lang], [out_audio, log_1])
    btn_save.click(save_voice, [inp_audio, inp_name], [drop_voice, log_1])
    btn_run.click(tts_long, [txt_long, drop_voice, drop_lang], [out_final, log_2])

if __name__ == "__main__":
    demo.launch(inbrowser=True)
