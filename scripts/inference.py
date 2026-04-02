import argparse
import base64
import json
import os
import random
import re
import warnings
from pathlib import Path
from typing import Any, Optional

import cv2
import imageio
import librosa
import moviepy.editor as mp
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from demucs.apply import apply_model
from demucs.states import load_model
from einops import rearrange, repeat
from openai import OpenAI
from safetensors import safe_open
from transformers import Wav2Vec2Processor

from diffsynth.models.wav2vec import WanS2VAudioEncoder
from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline

try:
    from elevenlabs import ElevenLabs, VoiceSettings, save
except Exception:
    ElevenLabs = None
    VoiceSettings = None
    save = None


DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)
DEFAULT_OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_VL_MODEL = "qwen3-vl-plus-2025-12-19"
QWEN_TEXT_MODEL = "qwen3-max"
QWEN_VID_FPS = 1
IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}
VOICE_ID_MAP = {
    "girl": [
        "t4U671CQHG58R11znrVj",
        "BlgEcC0TfWpBak7FmvHW",
        "XJ2fW4ybq7HouelYYGcL",
        "BAdH0bMfq6VleQGLXj38",
        "JVVJ6VsnUPJAdfGmEBGP",
        "2bk7ULW9HfwvcIbMWod0",
        "LJk8s1oqYd18q18FQ9Eg",
        "ocZQ262SsZb9RIxcQBOj",
    ],
    "boy": [
        "tJHJUEHzOkMoPmJJ5jo2",
        "E95NigJoVU5BI8HjQeN3",
        "87n4zM8Wuy87vFILuKvE",
        "6BZyx2XekeeXOkTVn8un",
        "vBKc2FfBKJfcZNyEt1n6",
        "rU18Fk3uSDhmg5Xh41o4",
        "DTKMou8ccj1ZaWGBiotd",
        "UGTtbzgh3HObxRjWaSpr",
    ],
    "man": [
        "pNInz6obpgDQGcFmaJgB",
        "ErXwobaYiN019PkySvjV",
        "G17SuINrv2H9FC6nvetn",
        "7S3KNdLDL7aRgBVRQb1z",
        "8z82LG47qQ2qjeeQB8lk",
        "goT3UYdM9bhm0n2lmKQx",
        "UgBBYS2sOqTuMpoF3BR0",
        "MFZUKuGQUsGJPQjTS4wC",
        "EkK5I93UQWFDigLMpZcX",
        "uju3wxzG5OhpWcoi3SMy",
    ],
    "woman": [
        "Xb7hH8MSUJpSbSDYk0k2",
        "9BWtsMINqrJLrRacOk9x",
        "lxYfHSkYm1EzQzGhdbfc",
        "4RZ84U1b4WCqpu57LvIq",
        "Z3R5wn05IrDiVCyEkUrK",
        "K7W7zLWeGoxU9YqWoB7A",
        "si0svtk05vPEuvwAW93c",
        "v8DWAeuEGQSfwxqdH9t2",
        "XhNlP8uwiH6XZSFnH1yL",
        "ZF6FPAbjXT4488VcRRnw",
        "xctasy8XvGp2cVO9HL9k",
    ],
    "grandpa": [
        "NOpBlnGInO9m6vDvFkFC",
        "dPah2VEoifKnZT37774q",
        "abRFZIdN4pvo8ZPmGxHP",
        "ZthjuvLPty3kTMaNKVKb",
        "dPah2VEoifKnZT37774q",
        "7p1Ofvcwsv7UBPoFNcpI",
        "YXpFCvM1S3JbWEJhoskW",
        "L1aJrPa7pLJEyYlh3Ilq",
        "qAZH0aMXY8tw1QufPN0D",
        "6sFKzaJr574YWVu4UuJF",
        "zNsotODqUhvbJ5wMG7Ei",
    ],
    "grandma": [
        "q1Hhtkt94vkD6q7p50hW",
        "RILOU7YmBhvwJGDGjNmP",
        "NFFZBoF6tNodi008z7VH",
        "9Dbo4hEvXQ5l7MXGZFQA",
        "7NsaqHdLuKNFvEfjpUno",
        "0rEo3eAjssGDUCXHYENf",
        "kkPJzQOWz2Oz9cUaEaQd",
        "8WaMCGQzWsKvf7sGPqjE",
        "MHPwHxLx0nmGIb5Jnbly",
    ],
}
DEFAULT_DEMUCS_MODEL_PATHS = [
    "./model_weights/MVSEP/97d170e1-a778de4a.th",
    "./model_weights/MVSEP/97d170e1-dbb4db15.th",
    "./model_weights/MVSEP/97d170e1-e41a5468.th",
]
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
        try:
            if audio_clip.duration > video_clip.duration:
                audio_clip = audio_clip.subclip(0, video_clip.duration)

            final_path = os.path.join(save_path, f"{name}_with_audio.mp4")
            video_with_audio = video_clip.set_audio(audio_clip)
            video_with_audio.write_videofile(final_path, fps=fps, logger=None)
            video_with_audio.close()
        finally:
            video_clip.close()
            audio_clip.close()
            if os.path.exists(tmp_video_path):
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


def binarize_mask_frame(
    mask_image: Image.Image, height: int, width: int, threshold: float
) -> Image.Image:
    mask_image = crop_and_resize_mask(mask_image.convert("L"), height, width)
    mask_array = np.array(mask_image, dtype=np.float32)
    if mask_array.max() > 1.0:
        mask_array = mask_array / 255.0
    mask_array = (mask_array > threshold).astype(np.uint8)
    mask_array = np.repeat(mask_array[:, :, None], 3, axis=2) * 255
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
        audio_waveform,
        sample_rate,
        audio_processor,
        fps=fps,
        batch_frames=81 - 1,
        dtype=torch_dtype,
        device=device,
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
    frames = [
        preprocess_image(frame, torch_dtype=torch_dtype, device=device)
        for frame in mask_frames
    ]
    mask_tensor = torch.stack(frames, dim=2)[:, :1]
    mask_tensor = (mask_tensor > 0).to(device=device, dtype=torch_dtype)
    return mask_tensor


def downsample_mask_to_latent(
    mask_tensor: torch.Tensor, torch_dtype: torch.dtype, device: str
):
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


def ensure_file_exists(path: str, label: str):
    if not Path(path).exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def ensure_audio_generation_dependencies(api_key: str):
    if not api_key:
        raise ValueError(
            "ElevenLabs API key is required. Set --eleven-api-key or ELEVEN_API_KEY."
        )
    if ElevenLabs is None or VoiceSettings is None or save is None:
        raise ImportError(
            "elevenlabs is required to synthesize edited audio for this script."
        )


def read_as_base64(path: str) -> str:
    with open(path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("utf-8")


def encode_video(video_path: str) -> str:
    return read_as_base64(video_path)


def parse_audio_edit_output(label_text: str) -> tuple[str, str]:
    match_type = re.search(r"type\s*:\s*([^\n\r]+)", label_text, flags=re.IGNORECASE)
    raw_type = match_type.group(1).strip().lower() if match_type else ""
    if "human" in raw_type or "voice" in raw_type or "speech" in raw_type:
        sound_type = "human voice"
    elif "music" in raw_type:
        sound_type = "music"
    else:
        sound_type = "other sound effect"

    match_object = re.search(
        r"object\s*:\s*([^\n\r]+)", label_text, flags=re.IGNORECASE
    )
    object_name = match_object.group(1).strip() if match_object else ""
    return sound_type, object_name


def choose_voice_id(gender: str) -> Optional[str]:
    gender = (gender or "").strip().lower()
    if gender not in VOICE_ID_MAP:
        return None
    return random.choice(VOICE_ID_MAP[gender])


def build_openai_compatible_client(api_key: str, base_url: str) -> OpenAI:
    if not api_key:
        raise ValueError(
            "DashScope API key is required. Set --dashscope-api-key or DASHSCOPE_API_KEY."
        )
    return OpenAI(api_key=api_key, base_url=base_url)


def classify_video_mask_caption(
    client: OpenAI, video_path: str, mask_path: str, editing_caption: str
) -> tuple[str, str, str]:
    b64_video = encode_video(video_path)
    b64_video_mask = encode_video(mask_path)

    messages = [
        {
            "role": "system",
            "content": "You are a multimodal reasoning assistant.You will receive a video, an object mask, and a editing caption.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {
                        "url": f"data:video/mp4;base64,{b64_video}",
                        "fps": QWEN_VID_FPS,
                    },
                },
                {
                    "type": "video_url",
                    "video_url": {
                        "url": f"data:video/mp4;base64,{b64_video_mask}",
                        "fps": QWEN_VID_FPS,
                    },
                },
                {
                    "type": "text",
                    "text": (
                        "The first video is the original video. "
                        "The second video is the object mask in the video. "
                        "Determine the object category from the mask of video. "
                        f"According to the editing caption: '{editing_caption}', determine which object will be replaced in the mask area and "
                        "which type of sound this object will produce. "
                        "Specially, if the editing caption explicitly contains spoken words or quotes (e.g., a man says 'hello'), "
                        "the sound type must be 'human voice'. "
                        "Output one of the following options: 'human voice', 'music', or 'other sound effect' placed after 'type:' "
                        f"Then according to the editing caption: '{editing_caption}'. Output the object noun after editing will be replaced in the mask area without any other description placed after 'object:'."
                    ),
                },
            ],
        },
    ]

    response = client.chat.completions.create(model=QWEN_VL_MODEL, messages=messages)
    label = response.choices[0].message.content.strip()
    sound_type, object_name = parse_audio_edit_output(label)
    return label, sound_type, object_name


def generate_audio_prompt_effect(
    client: OpenAI,
    input_prompt: str,
    object_name: str,
    duration_seconds: float,
    save_path: str,
    eleven_api_key: str,
) -> dict[str, Any]:
    ensure_audio_generation_dependencies(eleven_api_key)
    sys_msg = [{"type": "text", "text": "You are a helpful assistant."}]
    user_text = f"""
You are provided with prompt designed for video generation model that describes a short video with sound. Your task is to generate a description of sound effects related only to the specified sound-producing object.
Here's an example for you:

input:
A close-up shot of a clear glass with a textured surface, placed on a wooden surface. The glass is partially filled with water, and the focus is on the water's movement as it is poured into the glass from above. The pouring action creates ripples and bubbles in the water, which are captured in detail. The background is blurred, emphasizing the glass and the water.
object_name: water
output:
Close-up water pouring sound, gentle bubbling, glass resonance, soft wood contact tone.

Now given an input as below, please describe only the sound effects produced by the specified object, without any other scene or visual descriptions.
Keep your output under 20 words.
input:
{input_prompt}
object_name: {object_name}
""".strip()

    response = client.chat.completions.create(
        model=QWEN_TEXT_MODEL,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
        ],
    )
    effect_prompt = response.choices[0].message.content.strip()

    elevenlabs = ElevenLabs(api_key=eleven_api_key)
    effects = elevenlabs.text_to_sound_effects.convert(
        text=effect_prompt,
        duration_seconds=max(1.0, duration_seconds),
    )
    save(effects, save_path)
    return {"audio_text": effect_prompt, "audio_file": save_path}


def generate_audio_prompt_music(
    client: OpenAI,
    input_prompt: str,
    object_name: str,
    duration_seconds: float,
    save_path: str,
    eleven_api_key: str,
) -> dict[str, Any]:
    ensure_audio_generation_dependencies(eleven_api_key)
    sys_msg = [{"type": "text", "text": "You are a helpful assistant."}]
    user_text = f"""
You are provided with prompt designed for video generation model that describes a short video with sound.
Your task is to generate a concise description of music that is produced by the specified object or by an instrument or item the object interacts with.
Here's an example for you:

input:
a man with short dark hair, wearing a black and white striped shirt, is playing the guitar...
object_name: man
output:
Guitar strumming, fingerpicking, string resonance, subtle fret noise.

Now given an input as below, please describe only the music produced by the specified object or by the item it interacts with, without any other scene or visual descriptions.
Keep your output under 20 words.
input:
{input_prompt}
object_name: {object_name}
""".strip()

    response = client.chat.completions.create(
        model=QWEN_TEXT_MODEL,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
        ],
    )
    music_prompt = response.choices[0].message.content.strip()

    elevenlabs = ElevenLabs(api_key=eleven_api_key)
    music = elevenlabs.text_to_sound_effects.convert(
        text=music_prompt,
        duration_seconds=max(1.0, duration_seconds),
    )
    save(music, save_path)
    return {"audio_text": music_prompt, "audio_file": save_path}


def generate_audio_prompt_one(
    client: OpenAI,
    input_prompt: str,
    duration_seconds: float,
    save_path: str,
    eleven_api_key: str,
) -> dict[str, Any]:
    ensure_audio_generation_dependencies(eleven_api_key)
    sys_msg = [{"type": "text", "text": "You are a helpful assistant."}]
    extract_text_prompt = r"""If the input contains direct speech or quoted dialogue (i.e., text enclosed in quotation marks), extract and output only the quoted content without the quotation marks. Do not include any other text, explanation, or formatting. If there is no quoted content, return None.
Here are some examples:
Input:
an angry and mad man speaks "dont prompt me anymore because I know I am not human"
Output:
dont prompt me anymore because I know I am not human

Input:
a man with headphones appears to be singing or speaking passionately into a microphone, saying 'Feel the rhythm in your soul, let it move you.'
Output:
Feel the rhythm in your soul, let it move you.

Input:
A woman and a man is talking
Output:
None

Now here's a new input text prompt, please extract the speech from the following input, and do not output any other words:
"""
    extract_response = client.chat.completions.create(
        model=QWEN_TEXT_MODEL,
        messages=[
            {"role": "system", "content": sys_msg},
            {
                "role": "user",
                "content": [{"type": "text", "text": extract_text_prompt + input_prompt}],
            },
        ],
    )
    user_words = (extract_response.choices[0].message.content or "").strip()

    json_prompt_prefix = (
        "You are provided with a prompt designed for a video generation model that describes a short video with sound. "
        f"The duration of the video is {duration_seconds:.2f} seconds. Please generate a JSON containing the spoken lines for the persons appearing in the video. "
        f"Make sure the generated lines can be naturally spoken within {duration_seconds:.2f} seconds at a normal speaking rate.\n"
        'If user gives specific lines, your output must include it exactly. The gender of the person must be specified clearly: '
        'If a male is younger than 15 years old, he should be specified as "boy"; if he is older than 50 years old, he should be specified as "grandpa"; otherwise he will be specified as "man". '
        'Likely, if a female is younger than 15 years old, she should be specified as "girl"; if she is older than 50 years old, she should be specified as "grandma"; otherwise she will be specified as "woman". '
        "Here's an example for you:"
    )
    json_example = """
input:
In a dimly lit room, a male stands facing to his right, while an older, white man in a suit and glasses is positioned to his left, engaging in a conversation. A vertical arrangement of lights, centrally placed between them, casts stark reflections that dominate the backdrop. The male speaks, his mouth moving demonstrably, while the older listens intently, his expression serious and focused. Their discussion appears to be intense and perhaps adversarial, given the contrasting tones and the somber environment.

output:
{
  "gender": "man",
  "lines": "I will never give up on my dreams."
}

Now given an input as below, please generate a raw JSON object, without any markdown formatting, without any other descriptions:
""".strip()
    json_response = client.chat.completions.create(
        model=QWEN_TEXT_MODEL,
        messages=[
            {"role": "system", "content": sys_msg},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": json_prompt_prefix + "\n\n" + json_example + "\n" + input_prompt,
                    }
                ],
            },
        ],
    )
    json_text = (json_response.choices[0].message.content or "").strip()
    gender, lines = "man", ""
    try:
        data = json.loads(json_text)
        gender = str(data.get("gender", "man")).strip().lower() or "man"
        lines = str(data.get("lines", "")).strip()
    except Exception:
        lines = ""
    if user_words and user_words.lower() != "none":
        lines = user_words
    if not lines:
        raise ValueError("Audio agent failed to generate spoken lines for human voice.")

    voice_id = choose_voice_id(gender) or choose_voice_id("man")
    elevenlabs = ElevenLabs(api_key=eleven_api_key)
    audio = elevenlabs.text_to_speech.convert(
        text=lines,
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
        voice_settings=VoiceSettings(speed=0.7),
    )
    save(audio, save_path)
    return {"audio_text": lines, "gender": gender, "audio_file": save_path}


def load_demucs_model_compat(model_path: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            package = torch.load(model_path, map_location="cpu", weights_only=False)
        except TypeError:
            package = torch.load(model_path, map_location="cpu")
    return load_model(package)


class Demucs4_SeparationModel:
    def __init__(self, model_paths: list[str], device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_list = model_paths
        self.models = []

        for model_path in self.model_list:
            if not os.path.isfile(model_path):
                filename = os.path.basename(model_path)
                remote_url = (
                    "https://github.com/ZFTurbo/MVSEP-CDX23-Cinematic-Sound-Demixing/"
                    f"releases/download/v.1.0.0/{filename}"
                )
                torch.hub.download_url_to_file(remote_url, model_path)
            model = load_demucs_model_compat(model_path)
            model.to(device)
            self.models.append(model)

    @property
    def instruments(self):
        return ["dialog", "effect", "music"]

    def separate_music_file(
        self,
        mixed_sound_array: np.ndarray,
        sample_rate: int,
    ) -> tuple[dict[str, np.ndarray], dict[str, int]]:
        separated_music_arrays = {}
        output_sample_rates = {}

        audio = np.expand_dims(mixed_sound_array.T, axis=0)
        audio = torch.from_numpy(audio).type(torch.FloatTensor).to(self.device)

        all_out = []
        with torch.no_grad():
            for model in self.models:
                out = apply_model(model, audio, shifts=1, overlap=0.8)[0].cpu().numpy()
                all_out.append(out)
            dnr_demucs = np.array(all_out).mean(axis=0)

        separated_music_arrays["dialog"] = dnr_demucs[2].T
        separated_music_arrays["music"] = dnr_demucs[0].T
        separated_music_arrays["effect"] = dnr_demucs[1].T
        for instrument in self.instruments:
            output_sample_rates[instrument] = sample_rate
        return separated_music_arrays, output_sample_rates


def save_frames_to_mp4(frames: list[Image.Image], output_path: str, fps: float):
    if not frames:
        raise ValueError("Cannot save an empty frame list to video.")
    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(np.array(frame.convert("RGB"), dtype=np.uint8))


def extract_audio_from_video(video_path: str, output_path: str, sample_rate: int):
    video_clip = mp.VideoFileClip(video_path)
    try:
        if video_clip.audio is None:
            raise ValueError(
                f"Input video must contain an audio track, but none was found: {video_path}"
            )
        video_clip.audio.write_audiofile(
            output_path,
            fps=sample_rate,
            codec="pcm_s16le",
            logger=None,
        )
    finally:
        if video_clip.audio is not None:
            video_clip.audio.close()
        video_clip.close()


def ensure_audio_samples_channels(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.shape[0] <= 2 and audio.shape[1] > 2:
        audio = audio.T
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    return audio.astype(np.float32)


def load_audio_samples(audio_path: str, sample_rate: int) -> tuple[np.ndarray, int]:
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=False)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)
    elif audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    return ensure_audio_samples_channels(audio.T), sr


def fit_audio_num_samples(audio: np.ndarray, target_num_samples: int) -> np.ndarray:
    audio = ensure_audio_samples_channels(audio)
    if audio.shape[0] >= target_num_samples:
        return audio[:target_num_samples]
    padding = np.zeros(
        (target_num_samples - audio.shape[0], audio.shape[1]), dtype=audio.dtype
    )
    return np.concatenate([audio, padding], axis=0)


def write_audio_samples(audio_path: str, audio: np.ndarray, sample_rate: int):
    audio = ensure_audio_samples_channels(audio)
    sf.write(audio_path, audio, sample_rate, subtype="FLOAT")


def separate_audio_stems(
    input_audio_path: str,
    demucs_model: Demucs4_SeparationModel,
    output_dir: Path,
    sample_rate: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, str]]:
    mixed_audio, sr = load_audio_samples(input_audio_path, sample_rate)
    separated, sample_rates = demucs_model.separate_music_file(mixed_audio, sr)

    stem_arrays: dict[str, np.ndarray] = {}
    stem_paths: dict[str, str] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for stem_name in demucs_model.instruments:
        stem_audio = ensure_audio_samples_channels(separated[stem_name])
        stem_arrays[stem_name] = stem_audio
        stem_path = output_dir / f"{stem_name}.wav"
        write_audio_samples(str(stem_path), stem_audio, sample_rates[stem_name])
        stem_paths[stem_name] = str(stem_path)
    return mixed_audio, stem_arrays, stem_paths


def clamp_audio(audio: np.ndarray, max_peak: float = 0.99) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= max_peak or peak < 1e-8:
        return audio.astype(np.float32)
    return (audio * (max_peak / peak)).astype(np.float32)


def generate_replacement_audio(
    client: OpenAI,
    sound_type: str,
    prompt: str,
    object_name: str,
    duration_seconds: float,
    eleven_api_key: str,
    generated_audio_path: str,
) -> dict[str, Any]:
    if sound_type == "human voice":
        return generate_audio_prompt_one(
            client=client,
            input_prompt=prompt,
            duration_seconds=duration_seconds,
            save_path=generated_audio_path,
            eleven_api_key=eleven_api_key,
        )
    if sound_type == "music":
        return generate_audio_prompt_music(
            client=client,
            input_prompt=prompt,
            object_name=object_name or "instrument",
            duration_seconds=duration_seconds,
            save_path=generated_audio_path,
            eleven_api_key=eleven_api_key,
        )
    return generate_audio_prompt_effect(
        client=client,
        input_prompt=prompt,
        object_name=object_name or "object",
        duration_seconds=duration_seconds,
        save_path=generated_audio_path,
        eleven_api_key=eleven_api_key,
    )


def build_background_tracks(
    original_audio: np.ndarray,
    original_stems: dict[str, np.ndarray],
    output_dir: Path,
    sample_rate: int,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    original_audio = ensure_audio_samples_channels(original_audio)
    backgrounds = {
        "human voice": original_audio - original_stems["dialog"],
        "music": original_stems["effect"],
        "other sound effect": original_stems["music"],
    }
    background_paths: dict[str, str] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for sound_type, audio in backgrounds.items():
        filename = {
            "human voice": "background_instrum.wav",
            "music": "background_effect.wav",
            "other sound effect": "background_music.wav",
        }[sound_type]
        audio_path = output_dir / filename
        write_audio_samples(str(audio_path), audio, sample_rate)
        background_paths[sound_type] = str(audio_path)
    return backgrounds, background_paths


def mix_generated_audio_into_background(
    background_audio: np.ndarray,
    replacement_audio_path: str,
    sample_rate: int,
    output_audio_path: str,
) -> str:
    background_audio = ensure_audio_samples_channels(background_audio)
    target_num_samples = background_audio.shape[0]
    replacement_audio, _ = load_audio_samples(replacement_audio_path, sample_rate)
    replacement_audio = fit_audio_num_samples(replacement_audio, target_num_samples)
    mixed = clamp_audio(background_audio + replacement_audio)
    write_audio_samples(output_audio_path, mixed, sample_rate)
    return output_audio_path


def run_single_inference(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")

    ensure_file_exists(args.video_path, "Input video")
    ensure_file_exists(args.mask_path, "Input mask")
    torch_dtype = torch.bfloat16
    device = "cuda"
    sample_rate = args.audio_sample_rate
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
    duration_seconds = num_frames / fps

    base_name = args.output_name or Path(args.video_path).stem
    work_dir = output_dir / f"{base_name}_audio_agent"
    work_dir.mkdir(parents=True, exist_ok=True)

    agent_video_path = work_dir / "agent_video.mp4"
    agent_mask_video_path = work_dir / "agent_mask.mp4"
    save_frames_to_mp4(video_frames, str(agent_video_path), fps=fps)
    save_frames_to_mp4(mask_frames, str(agent_mask_video_path), fps=fps)

    original_audio_path = work_dir / "original_audio.wav"
    extract_audio_from_video(args.video_path, str(original_audio_path), sample_rate)

    client = build_openai_compatible_client(
        api_key=args.dashscope_api_key,
        base_url=args.openai_base_url,
    )
    label_raw, sound_type, object_name = classify_video_mask_caption(
        client,
        str(agent_video_path),
        str(agent_mask_video_path),
        args.prompt,
    )
    print(f"Audio agent type: {sound_type}, object: {object_name}")

    demucs_model = Demucs4_SeparationModel(
        model_paths=args.demucs_model_paths,
        device=args.demucs_device,
    )
    original_audio_array, stem_arrays, stem_paths = separate_audio_stems(
        str(original_audio_path),
        demucs_model,
        work_dir / "stems",
        sample_rate,
    )
    background_arrays, background_paths = build_background_tracks(
        original_audio_array,
        stem_arrays,
        work_dir / "backgrounds",
        sample_rate,
    )

    generated_audio_path = work_dir / "generated_replacement.mp3"
    generated_audio_info = generate_replacement_audio(
        client=client,
        sound_type=sound_type,
        prompt=args.prompt,
        object_name=object_name,
        duration_seconds=duration_seconds,
        eleven_api_key=args.eleven_api_key,
        generated_audio_path=str(generated_audio_path),
    )

    edited_audio_path = work_dir / "edited_audio.wav"
    edited_audio_path_str = mix_generated_audio_into_background(
        background_audio=background_arrays[sound_type],
        replacement_audio_path=str(generated_audio_path),
        sample_rate=sample_rate,
        output_audio_path=str(edited_audio_path),
    )

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
            edited_audio_path_str,
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
        ).unsqueeze(0)

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

    video_out = (video.permute(0, 2, 1, 3, 4).cpu() + 1.0) / 2.0
    output_name = f"{base_name}_edited_{num_frames}"
    output_video_path = output_dir / f"{output_name}_with_audio.mp4"
    save_video_as_grid_and_mp4_with_audio(
        video_out.float(),
        str(output_dir),
        edited_audio_path_str,
        fps=fps,
        name=output_name,
    )
    metadata = {
        "seed": seed,
        "video_path": args.video_path,
        "mask_path": args.mask_path,
        "prompt": args.prompt,
        "fps": fps,
        "num_frames": num_frames,
        "duration_seconds": duration_seconds,
        "sound_type": sound_type,
        "object_name": object_name,
        "label_raw": label_raw,
        "background_audio_path": background_paths[sound_type],
        "generated_audio": generated_audio_info,
        "stem_paths": stem_paths,
        "background_paths": background_paths,
        "edited_audio_path": edited_audio_path_str,
        "output_video_path": str(output_video_path),
    }
    metadata_path = work_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(f"Saved outputs to: {output_dir}")
    return {
        "seed": seed,
        "output_dir": str(output_dir),
        "work_dir": str(work_dir),
        "metadata_path": str(metadata_path),
        "output_video_path": str(output_video_path),
        "edited_audio_path": edited_audio_path_str,
        "sound_type": sound_type,
        "object_name": object_name,
        "num_frames": num_frames,
        "fps": fps,
    }


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Single video editing inference script with audio-agent-guided audio replacement."
    )
    parser.add_argument("--video-path", required=True, help="Input video path.")
    parser.add_argument(
        "--mask-path",
        required=True,
        help="Mask path. Supports video, image, or image directory.",
    )
    parser.add_argument("--prompt", required=True, help="Editing prompt.")
    parser.add_argument("--output-dir", required=True, help="Directory to save outputs.")
    parser.add_argument(
        "--output-name",
        default=None,
        help="Output file prefix. Defaults to video stem.",
    )
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
        "--audio-sample-rate",
        type=int,
        default=44100,
        help="Sample rate used for extraction, Demucs separation, and final mixing.",
    )
    parser.add_argument(
        "--dashscope-api-key",
        default=os.getenv("DASHSCOPE_API_KEY", ""),
        help="DashScope API key. Defaults to DASHSCOPE_API_KEY.",
    )
    parser.add_argument(
        "--openai-base-url",
        default=os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL),
        help="OpenAI-compatible base URL for DashScope.",
    )
    parser.add_argument(
        "--eleven-api-key",
        default=os.getenv("ELEVEN_API_KEY", ""),
        help="ElevenLabs API key. Defaults to ELEVEN_API_KEY.",
    )
    parser.add_argument(
        "--demucs-model-paths",
        nargs="+",
        default=DEFAULT_DEMUCS_MODEL_PATHS,
        help="Demucs checkpoint paths for dialog/effect/music separation.",
    )
    parser.add_argument(
        "--demucs-device",
        default=None,
        help="Device for Demucs, e.g. cuda:0. Defaults to auto selection.",
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
