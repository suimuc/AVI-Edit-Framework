# [CVPR2026] Audio-sync Video Instance Editing with Granularity-Aware Mask Refiner

## [Project Page](https://hjzheng.net/projects/AVI-Edit/) | [arXiv](https://arxiv.org/abs/2512.10571) | [Model](https://huggingface.co/suimu/AVI-Edit)

This repository contains the official implementation of **AVI-Edit**, a framework for audio-sync video instance editing with granularity-aware mask refiner.
<table align='center' border="0" style="width: 100%; text-align: center; margin-top: 80px;">
  <tr>
    <td>
      <video align='center' src="https://github.com/user-attachments/assets/6f9e45fa-c9cb-4d1f-b801-4a6b121a37cf" autoplay loop></video>
    </td>
  </tr>
    <tr align="center">
    <td>
      <em>For the best experience, please enable audio.</em>
    </td>
  </tr>
</table>

## To Do List

- [x] Inference scripts
- [ ] Dataset
- [ ] Training scripts
- [ ] Full audio agent
- [ ] Controllable conditions
  - [ ] Reference
  - [ ] Pose
  - [ ] Sketch

## Environment

The current `avi_edit` environment in this repository uses:

- PyTorch `2.6.0+cu124`
- CUDA `12.4`

Use the following commands to install the environment:

```bash
git clone https://github.com/suimuc/AVI-Edit-Framework.git
cd AVI-Edit-Framework
conda create -n avi_edit python=3.10
conda activate avi_edit
pip install -r requirements.txt
pip install -e .
```

## Download Model Weights

Place all checkpoints under `model_weights/`.

Required models:

- `Wan-AI/Wan2.2-TI2V-5B`
- `Wan-AI/Wan2.2-S2V-14B`
  - only the `wav2vec2-large-xlsr-53-english` directory is required
- `suimu/MVSEP`
- `suimu/AVI-Edit`

If `huggingface-cli` is not available, install it first:

```bash
pip install -U huggingface_hub
```

Then download the weights with Hugging Face:

```bash
mkdir model_weights
hf download Wan-AI/Wan2.2-TI2V-5B \
  --local-dir model_weights/Wan2.2-TI2V-5B

hf download Wan-AI/Wan2.2-S2V-14B \
  --include "wav2vec2-large-xlsr-53-english/*" \
  --local-dir model_weights/Wan2.2-S2V-14B

hf download suimu/MVSEP \
  --local-dir model_weights/MVSEP

hf download suimu/AVI-Edit \
  --local-dir model_weights/AVI-Edit
```

After downloading, the local structure should match the paths used by the scripts:

```text
model_weights/
|-- AVI-Edit/
|   `-- avi_edit.pt
|-- MVSEP/
|   |-- 97d170e1-a778de4a.th
|   |-- 97d170e1-dbb4db15.th
|   `-- 97d170e1-e41a5468.th
|-- Wan2.2-S2V-14B/
|   `-- wav2vec2-large-xlsr-53-english/
`-- Wan2.2-TI2V-5B/
    |-- Wan2.2_VAE.pth
    |-- models_t5_umt5-xxl-enc-bf16.pth
    |-- diffusion_pytorch_model-00001-of-00003.safetensors
    |-- diffusion_pytorch_model-00002-of-00003.safetensors
    |-- diffusion_pytorch_model-00003-of-00003.safetensors
    `-- google/umt5-xxl/
```

## Inference

### 1. Inference with an Edited Audio Track

Use this script when you already have the edited audio:

```bash
python scripts/inference_with_edited_audio.py \
  --video-path /path/to/input_video.mp4 \
  --audio-path /path/to/edited_audio.wav \
  --mask-path /path/to/mask.mp4 \
  --prompt "Describe the edited scene here." \
  --output-dir /path/to/output_dir
```

Arguments:

- `--video-path`: path to the source video.
- `--audio-path`: path to the edited audio.
- `--mask-path`: path to the instance mask.
- `--prompt`: text description of the edited scene.
- `--output-dir`: directory used to save the final results.

### 2. Inference with the Agent

Use this script when you want the system to generate replacement audio automatically from the video, mask, and edit prompt:

```bash
python scripts/inference.py \
  --video-path /path/to/input_video.mp4 \
  --mask-path /path/to/mask.mp4 \
  --prompt "Describe the edited scene here." \
  --output-dir /path/to/output_dir \
  --dashscope-api-key "<YOUR_QWEN_OR_OPENAI_COMPATIBLE_API_KEY>" \
  --eleven-api-key "<YOUR_ELEVENLABS_API_KEY>"
```

Notes:

- `--dashscope-api-key` is used for the default Qwen-based audio agent pipeline.
- You can replace Qwen with another OpenAI-compatible API by changing `--openai-base-url`.
- `--eleven-api-key` is required for speech or sound generation through [ElevenLabs](https://elevenlabs.io/).
- Use your own API keys. Do not hardcode private keys into scripts or README files.

## Gradio Demo

The demo includes a local `sam2` package under `demo/` and uses local SAM2 checkpoints from `demo/checkpoints/`.

First, download the SAM2 checkpoints:

```bash
cd demo/checkpoints
bash download_ckpts.sh
cd ../..
```

Then launch the Gradio demo:

```bash
python demo/gradio_with_sam2.py \
  --server-port 7861 \
  --share \
  --dashscope-api-key "<YOUR_QWEN_OR_OPENAI_COMPATIBLE_API_KEY>" \
  --eleven-api-key "<YOUR_ELEVENLABS_API_KEY>"
```

The demo supports:

- interactive SAM2 point-based object selection;
- full-video mask propagation;
- audio-agent-driven editing with a single interface.

## Citation

If you find this project useful, please cite:

```bibtex
@article{avi-edit,
  title={Audio-sync Video Instance Editing with Granularity-Aware Mask Refiner},
  author={Zheng, Haojie and Weng, Shuchen and Liu, Jingqi and Yang, Siqi and Shi, Boxin and Wang, Xinlong},
  journal={arXiv preprint arXiv:2512.10571},
  year={2025}
}
```

## Acknowledgements

This project builds on several excellent open-source efforts, including Wan, SAM2, MVSEP, and DiffSynth-based components.
