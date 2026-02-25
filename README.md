# 🎙️ Qwen3-Voice-Studio

一个基于 [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 的本地化语音克隆工作台。
针对消费级显卡（8GB VRAM）进行了深度优化，实现了长文本切分与流式合成。

## ✨ 特性 (Features)
- **显存优化**：使用 `bfloat16` 精度和 `SDPA` 加速，RTX 3070/4060 (8GB) 可流畅运行。
- **一键克隆**：上传 5 秒录音即可提取声纹并保存。
- **长文本支持**：自动切分长句，防止显存溢出。
- **隐私安全**：模型与数据完全在本地运行，无需联网 API。

## 🛠️ 安装 (Installation)

1. **克隆项目**
   ```bash
   git clone https://github.com/你的用户名/Qwen3-Voice-Studio.git
   cd Qwen3-Voice-Studio
   ```

2. **安装核心驱动 (GPU)**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   pip install git+https://github.com/QwenLM/Qwen3-TTS.git
   ```

4. **下载模型**
   请下载 Qwen3-TTS 权重，解压并重命名为 `Qwen3-Base`，放在项目根目录下。

## 🚀 运行

```bash
python app.py
```
