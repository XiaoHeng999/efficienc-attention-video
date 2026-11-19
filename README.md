# daVinci-MagiHuman

daVinci-MagiHuman 是一个基于 PyTorch 的音视频生成推理项目。当前仓库保留了
推理主链路、可选注意力后端、示例配置和最小运行依赖；模型权重不随仓库发布。

## 运行逻辑

入口是 `inference/pipeline/entry.py`，执行流程如下：

1. 读取 `--config-load-path` 指定的 JSON 配置。
2. 加载 DiT、文本编码器、音频模型和 VAE 权重。
3. 仅传入 `--prompt` 时运行 T2V；同时传入 `--image_path` 时运行 TI2V。
4. 生成视频和音频，并合成为 `<output_path>_...mp4`。

## 环境要求

- Linux、Python 3.12+
- NVIDIA GPU、对应 CUDA、PyTorch 2.10+
- FlashAttention（Hopper 使用 FA3；Blackwell 使用 FA4）
- ffmpeg
- 能够加载完整模型权重的显存/CPU 内存

安装基础依赖：

```bash
conda create -n davinci-magihuman python=3.12 -y
conda activate davinci-magihuman
conda install -c conda-forge ffmpeg -y
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
pip install -r requirements.txt
pip install --no-deps -r requirements-nodeps.txt
```

根据 GPU 安装 FlashAttention：

```bash
# Hopper / CUDA 12.x
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper && python setup.py install && cd ../..

# Blackwell / CUDA 13.x：使用下面命令替代上面的 PyTorch 和 FA3 安装
# pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
# pip install --pre "flash-attn-4[cu13]"
```

MagiCompiler 是推理编译运行时，请单独安装：

```bash
git clone https://github.com/SandAI-org/MagiCompiler.git
pip install -r MagiCompiler/requirements.txt
pip install -e MagiCompiler
```

如果已有可用的 MagiCompiler 安装，可跳过这一步。

## 模型与配置

从 [HuggingFace 模型页](https://huggingface.co/GAIR/daVinci-MagiHuman) 下载
daVinci-MagiHuman 权重，并准备以下外部模型：

- [t5gemma-9b-9b-ul2](https://huggingface.co/google/t5gemma-9b-9b-ul2)
- [stable-audio-open-1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0)
- [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)

编辑 `example/base/config.json`，至少修改以下路径：

```json
{
  "engine_config": {"load": "/path/to/checkpoints/base", "cp_size": 1},
  "evaluation_config": {
    "audio_model_path": "/path/to/stable-audio-open-1.0",
    "txt_model_path": "/path/to/t5gemma-9b-9b-ul2",
    "vae_model_path": "/path/to/Wan2.2-TI2V-5B"
  }
}
```

`cp_size` 应与实际使用的并行 GPU 数一致。

## 运行示例

先设置项目路径和配置，再运行单机单卡 T2V：

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
bash example/base/run_T2V.sh
```

运行 TI2V：

```bash
bash example/base/run_TI2V.sh
```

也可以直接调用入口：

```bash
torchrun --standalone --nproc_per_node=1 inference/pipeline/entry.py \
  --config-load-path example/base/config.json \
  --prompt "A person speaks naturally to the camera" \
  --output_path output/result
```

直接调用时，加入 `--image_path example/assets/image.png` 即切换为 TI2V。

## 注意力后端

默认使用 FlashAttention。实验性后端通过 `MAGI_ATTN_BACKEND` 选择，例如：

```bash
MAGI_ATTN_BACKEND=svg2 bash example/base/run_T2V.sh
```

可用后端以 `inference/model/dit/attention_backends/` 中的实现为准；未安装对应
CUDA 扩展时请使用默认后端。
