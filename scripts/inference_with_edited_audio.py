import argparse
import os
import random
from pathlib import Path

import cv2
import imageio
import librosa
import moviepy.editor as mp
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from einops import rearrange, repeat
from safetensors import safe_open
from transformers import Wav2Vec2Processor

from diffsynth.models.wav2vec import WanS2VAudioEncoder
from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline


DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)
IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


def load_state_dict_from_safetensors(file_path, torch_dtype=None, device="cpu"):
    state_dict = {}
    with safe_open(file_path, framework="pt", device=device) as handle:
        for key in handle.keys():
            value = handle.get_tensor(key)
            if torch_dtype is not None:
                value = value.to(torch_dtype)
            state_dict[key] = value
    return state_dict


def crop_and_resize(image, target_height, target_width):
    width, height = image.size
    scale = max(target_width / width, target_height / height)
    image = torchvision.transforms.functional.resize(
        image,
        (round(height * scale), round(width * scale)),
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
    )
    image = torchvision.transforms.functional.center_crop(
        image, (target_height, target_width)
    )
    return image


def crop_and_resize_mask(image, target_height, target_width):
    width, height = image.size
    scale = max(target_width / width, target_height / height)
    image = torchvision.transforms.functional.resize(
        image,
        (round(height * scale), round(width * scale)),
        interpolation=torchvision.transforms.InterpolationMode.NEAREST,
    )
    image = torchvision.transforms.functional.center_crop(
        image, (target_height, target_width)
    )
    return image


def preprocess_image(image, torch_dtype=None, device=None):
    image = torch.tensor(np.array(image, dtype=np.float32))
    image = image.to(dtype=torch_dtype, device=device)
    image = repeat(image, "h w c -> b c h w", b=1)
    return image


def save_video_as_grid_and_mp4_with_audio(
    video_batch: torch.Tensor, save_path: str, audio_path: str, fps: float, name: str
):
    os.makedirs(save_path, exist_ok=True)

    for video in video_batch:
        frames = []
        for frame in video:
            frame = rearrange(frame, "c h w -> h w c")
            frame = (255.0 * frame).cpu().numpy().astype(np.uint8)
            frames.append(frame)

        tmp_video_path = os.path.join(save_path, f"{name}.mp4")
        with imageio.get_writer(tmp_video_path, fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)

        video_clip = mp.VideoFileClip(tmp_video_path)
        audio_clip = mp.AudioFileClip(audio_path)
        if audio_clip.duration > video_clip.duration:
            audio_clip = audio_clip.subclip(0, video_clip.duration)

        final_path = os.path.join(save_path, f"{name}_with_audio.mp4")
        video_with_audio = video_clip.set_audio(audio_clip)
        video_with_audio.write_videofile(final_path, fps=fps)

        video_with_audio.close()
        video_clip.close()
        audio_clip.close()
        os.remove(tmp_video_path)


def read_video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0:
        raise ValueError(f"Invalid FPS from video: {video_path}")
    return fps


def load_video_frames(video_path: str, height: int, width: int) -> list[Image.Image]:
    reader = imageio.get_reader(video_path)
    frames = []
    try:
        for frame in reader:
            frames.append(crop_and_resize(Image.fromarray(frame), height, width))
    finally:
        reader.close()
    if not frames:
        raise ValueError(f"No frames found in video: {video_path}")
    return frames


def binarize_mask_frame(mask_image: Image.Image, height: int, width: int, threshold: float):
    mask_image = crop_and_resize_mask(mask_image.convert("L"), height, width)
    mask_array = np.array(mask_image, dtype=np.float32)
    if mask_array.max() > 1.0:
        mask_array = mask_array / 255.0
    mask_array = (mask_array > threshold).astype(np.uint8)
    mask_array = np.repeat(mask_array[:, :, None], 3, axis=2)
    return Image.fromarray(mask_array)


def load_mask_frames(
    mask_path: str,
    height: int,
    width: int,
    threshold: float,
    reference_num_frames: int,
) -> list[Image.Image]:
    path = Path(mask_path)

    if path.is_dir():
        files = sorted(
            file_path
            for file_path in path.iterdir()
            if file_path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not files:
            raise ValueError(f"No image masks found in directory: {mask_path}")
        return [
            binarize_mask_frame(Image.open(file_path), height, width, threshold)
            for file_path in files
        ]

    if path.suffix.lower() in IMAGE_EXTENSIONS:
        image = binarize_mask_frame(Image.open(path), height, width, threshold)
        return [image.copy() for _ in range(reference_num_frames)]

    reader = imageio.get_reader(mask_path)
    mask_frames = []
    try:
        for frame in reader:
            mask_frames.append(
                binarize_mask_frame(Image.fromarray(frame), height, width, threshold)
            )
    finally:
        reader.close()

    if not mask_frames:
        raise ValueError(f"No frames found in mask video: {mask_path}")
    return mask_frames


def get_target_num_frames(
    video_frames: int, mask_frames: int, max_num_frames: int | None
) -> int:
    target_frames = min(video_frames, mask_frames)
    if max_num_frames is not None:
        target_frames = min(target_frames, max_num_frames)
    if target_frames < 1:
        raise ValueError("Aligned frame count must be at least 1.")
    target_frames = ((target_frames - 1) // 4) * 4 + 1
    if target_frames < 1:
        raise ValueError("Aligned frame count is invalid after 4n+1 adjustment.")
    return target_frames


def build_audio_embeddings(
    audio_path: str,
    fps: float,
    num_frames: int,
    audio_encoder: WanS2VAudioEncoder,
    audio_processor: Wav2Vec2Processor,
    torch_dtype: torch.dtype,
    device: str,
) -> torch.Tensor:
    audio_waveform, sample_rate = librosa.load(audio_path, sr=16000)
    
    audio_embeds = audio_encoder.get_audio_feats_per_inference_1(
        audio_waveform, sample_rate, audio_processor,
        fps=fps, batch_frames=81-1, dtype=torch.bfloat16, device='cuda'
    )
    if audio_embeds.shape[0] < num_frames:
        padding = torch.zeros(
            (num_frames - audio_embeds.shape[0],) + tuple(audio_embeds.shape[1:]),
            device=audio_embeds.device,
            dtype=audio_embeds.dtype,
        )
        audio_embeds = torch.cat([audio_embeds, padding], dim=0)
    else:
        audio_embeds = audio_embeds[:num_frames]
    return audio_embeds.unsqueeze(0).to(device=device, dtype=torch_dtype)




def build_mask_tensor(
    mask_frames: list[Image.Image], torch_dtype: torch.dtype, device: str
) -> torch.Tensor:
    frames = [preprocess_image(frame, torch_dtype=torch_dtype, device=device) for frame in mask_frames]
    mask_tensor = torch.stack(frames, dim=2)[:, :1]
    mask_tensor = (mask_tensor > 0).to(device=device, dtype=torch_dtype)
    return mask_tensor


def downsample_mask_to_latent(mask_tensor: torch.Tensor, torch_dtype: torch.dtype, device: str):
    down_scale_spatial = 16
    down_scale_temporal = 4

    mask_first = F.interpolate(
        mask_tensor[:, :, 0],
        size=(
            mask_tensor.shape[3] // down_scale_spatial,
            mask_tensor.shape[4] // down_scale_spatial,
        ),
        mode="bilinear",
        align_corners=False,
    ).unsqueeze(2)

    mask_parts = [mask_first]
    if mask_tensor.shape[2] > 1:
        mask_rest = F.interpolate(
            mask_tensor[:, :, 1:],
            size=(
                (mask_tensor.shape[2] - 1) // down_scale_temporal,
                mask_tensor.shape[3] // down_scale_spatial,
                mask_tensor.shape[4] // down_scale_spatial,
            ),
            mode="trilinear",
            align_corners=False,
        )
        mask_parts.append(mask_rest)

    mask_latent = torch.cat(mask_parts, dim=2)
    return (mask_latent > 0).to(device=device, dtype=torch_dtype)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_audio_encoder(
    audio_processor_path: str,
    audio_encoder_weights: str,
    torch_dtype: torch.dtype,
    device: str,
):
    audio_processor = Wav2Vec2Processor.from_pretrained(audio_processor_path)
    audio_encoder = WanS2VAudioEncoder()
    pretrained_dict = load_state_dict_from_safetensors(
        audio_encoder_weights, torch_dtype=torch_dtype, device=device
    )
    state_dict = {"model." + key: value for key, value in pretrained_dict.items()}
    missing, unexpected = audio_encoder.load_state_dict(state_dict, strict=False)
    print(f"audio_encoder.load_state_dict -> miss: {missing}, unexpected: {unexpected}")
    audio_encoder = audio_encoder.to(device, dtype=torch_dtype).eval()
    return audio_processor, audio_encoder


def init_pipeline(
    model_paths: list[str],
    text_encoder_path: str,
    vae_path: str,
    tokenizer_path: str,
    torch_dtype: torch.dtype,
    device: str,
):
    return WanVideoPipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=device,
        model_configs=[
            ModelConfig(path=model_paths),
            ModelConfig(path=text_encoder_path),
            ModelConfig(path=vae_path),
        ],
        tokenizer_path=tokenizer_path,
    )


def run_single_inference(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")

    torch_dtype = torch.bfloat16
    device = "cuda"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed if args.seed is not None else random.randrange(2**32)
    set_seed(seed)
    print(f"Using seed: {seed}")

    video_frames = load_video_frames(args.video_path, args.height, args.width)
    fps = read_video_fps(args.video_path)
    mask_frames = load_mask_frames(
        args.mask_path,
        args.height,
        args.width,
        args.mask_threshold,
        len(video_frames),
    )

    num_frames = get_target_num_frames(
        len(video_frames), len(mask_frames), args.max_num_frames
    )
    print(
        f"Aligned frames -> video: {len(video_frames)}, mask: {len(mask_frames)}, used: {num_frames}"
    )

    video_frames = video_frames[:num_frames]
    mask_frames = mask_frames[:num_frames]

    audio_processor, audio_encoder = init_audio_encoder(
        args.audio_processor_path,
        args.audio_encoder_weights,
        torch_dtype,
        device,
    )
    pipe = init_pipeline(
        args.model_paths,
        args.text_encoder_path,
        args.vae_path,
        args.tokenizer_path,
        torch_dtype,
        device,
    )
    with torch.no_grad():
        
        audio_embed = build_audio_embeddings(
            args.audio_path,
            fps,
            num_frames,
            audio_encoder,
            audio_processor,
            torch_dtype,
            device,
        )
        mask_tensor = build_mask_tensor(mask_frames, torch_dtype, device)
        mask_latent = downsample_mask_to_latent(mask_tensor, torch_dtype, device)

        latent_frame_mask = torch.ones(
            (num_frames - 1) // 4 + 1, dtype=torch.bool, device=device
        )
        latent_frame_mask = latent_frame_mask.unsqueeze(0)

        video = pipe(
            input_video_ori=[video_frames],
            instance_mask_list=[mask_latent],
            mask_type_adain=args.mask_type_adain,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            seed=seed,
            tiled=True,
            height=args.height,
            width=args.width,
            audio_embed_s2v=[audio_embed],
            num_frames=num_frames,
            cfg_scale=args.cfg_scale,
            num_inference_steps=args.num_inference_steps,
            return_tensor=True,
            latent_frame_mask=latent_frame_mask,
        )

    base_name = args.output_name or Path(args.video_path).stem

    video_out = (video.permute(0, 2, 1, 3, 4).cpu() + 1.0) / 2.0
    save_video_as_grid_and_mp4_with_audio(
        video_out.float(),
        str(output_dir),
        args.audio_path,
        fps=fps,
        name=f"{base_name}_edited_{num_frames}",
    )

    print(f"Saved outputs to: {output_dir}")


def build_argparser():
    parser = argparse.ArgumentParser(description="Single video editing inference script.")
    parser.add_argument("--video-path", required=True, help="Input video path.")
    parser.add_argument("--mask-path", required=True, help="Mask path. Supports video, image, or image directory.")
    parser.add_argument("--audio-path", required=True, help="Input audio path.")
    parser.add_argument("--prompt", required=True, help="Editing prompt.")
    parser.add_argument("--output-dir", required=True, help="Directory to save outputs.")
    parser.add_argument("--output-name", default=None, help="Output file prefix. Defaults to video stem.")
    parser.add_argument("--height", type=int, default=704, help="Inference height.")
    parser.add_argument("--width", type=int, default=1280, help="Inference width.")
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.2,
        help="Mask binarization threshold after normalization to [0, 1].",
    )
    parser.add_argument(
        "--max-num-frames",
        type=int,
        default=None,
        help="Optional upper bound before applying the 4n+1 frame rule.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--cfg-scale", type=float, default=5.0, help="CFG scale.")
    parser.add_argument(
        "--num-inference-steps", type=int, default=50, help="Sampling steps."
    )
    parser.add_argument(
        "--mask-type-adain", type=int, default=1, help="Mask type passed into the model."
    )
    parser.add_argument(
        "--negative-prompt",
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt.",
    )
    parser.add_argument(
        "--audio-processor-path",
        default="./model_weights/Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english/",
        help="Wav2Vec2 processor path.",
    )
    parser.add_argument(
        "--audio-encoder-weights",
        default="./model_weights/Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english/model.safetensors",
        help="Audio encoder safetensors path.",
    )
    parser.add_argument(
        "--model-paths",
        nargs="+",
        default=[
            "./model_weights/AVI-Edit/avi_edit.pt",
        ],
        help="Primary Wan model paths.",
    )
    parser.add_argument(
        "--text-encoder-path",
        default="./model_weights/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth",
        help="Text encoder path.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default="./model_weights/Wan2.2-TI2V-5B/google/umt5-xxl",
        help="Tokenizer directory path.",
    )
    parser.add_argument(
        "--vae-path",
        default="./model_weights/Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
        help="VAE path.",
    )
    return parser


if __name__ == "__main__":
    run_single_inference(build_argparser().parse_args())
