import argparse
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
DEMO_ROOT_DEFAULT = SCRIPT_DIR
for path in (SCRIPT_DIR, SCRIPTS_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import inference_single_audio_agent as base


def resolve_repo_path(path_str: str) -> str:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path.resolve())


def resolve_repo_paths(path_list: list[str]) -> list[str]:
    return [resolve_repo_path(path_str) for path_str in path_list]


def resolve_demo_path(path_str: str, demo_root: Path) -> str:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = demo_root / path
    return str(path.resolve())


def ensure_sam2_importable(demo_root: Path):
    if str(demo_root) not in sys.path:
        sys.path.insert(0, str(demo_root))
    from sam2.build_sam import build_sam2_video_predictor

    return build_sam2_video_predictor


def show_mask_on_image(
    image_path: str,
    mask: np.ndarray | None,
    points: list[list[float]] | None = None,
    labels: list[int] | None = None,
) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image, dtype=np.uint8)
    overlay = image_np.copy()

    if mask is not None:
        color = np.array([255, 0, 0], dtype=np.uint8)
        alpha = 0.5
        overlay = np.where(
            mask[:, :, None] > 0,
            image_np * (1.0 - alpha) + color * alpha,
            image_np,
        ).astype(np.uint8)

    if points:
        overlay = np.ascontiguousarray(overlay)
        labels = labels or []
        for point, label in zip(points, labels):
            color = (0, 255, 0) if label == 1 else (255, 0, 0)
            center = (int(point[0]), int(point[1]))
            cv2.circle(overlay, center, 5, color, -1)
            cv2.circle(overlay, center, 5, (255, 255, 255), 1)

    return overlay


def parse_optional_int(raw_value: str | float | int | None, label: str) -> int | None:
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError as exc:
        raise gr.Error(f"{label} must be an integer.") from exc


def normalize_video_path(video_value: Any) -> str | None:
    if video_value is None:
        return None
    if isinstance(video_value, Path):
        return str(video_value)
    if isinstance(video_value, str):
        return video_value
    if isinstance(video_value, (list, tuple)) and len(video_value) > 0:
        first_item = video_value[0]
        if isinstance(first_item, (str, Path)):
            return str(first_item)
    if isinstance(video_value, dict):
        for key in ("path", "video", "name"):
            value = video_value.get(key)
            if isinstance(value, (str, Path)):
                return str(value)
    raise gr.Error(f"Unsupported video input type: {type(video_value)!r}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch a Gradio app for SAM2-assisted AVI-Edit audio-agent inference."
    )
    parser.add_argument("--server-name", default="0.0.0.0", help="Gradio host.")
    parser.add_argument("--server-port", type=int, default=7860, help="Gradio port.")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link.")
    parser.add_argument(
        "--output-dir",
        default="./outputs/gradio_audio_agent",
        help="Base output directory for generated masks and edited videos.",
    )
    parser.add_argument(
        "--temp-root",
        default="./outputs/gradio_audio_agent_tmp",
        help="Temporary directory for SAM2 frame extraction.",
    )
    parser.add_argument(
        "--demo-root",
        default=str(DEMO_ROOT_DEFAULT),
        help="Root directory that contains the local sam2 package and checkpoints.",
    )
    parser.add_argument(
        "--sam2-config",
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="SAM2 Hydra config name.",
    )
    parser.add_argument(
        "--sam2-checkpoint",
        default="checkpoints/sam2.1_hiera_large.pt",
        help="SAM2 checkpoint path, relative to --demo-root if not absolute.",
    )
    parser.add_argument(
        "--sam2-max-frames",
        type=int,
        default=150,
        help="Maximum number of frames accepted by the SAM2 UI.",
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
    parser.add_argument("--cfg-scale", type=float, default=5.0, help="CFG scale.")
    parser.add_argument(
        "--num-inference-steps", type=int, default=50, help="Sampling steps."
    )
    parser.add_argument(
        "--mask-type-adain", type=int, default=1, help="Mask type passed into the model."
    )
    parser.add_argument(
        "--negative-prompt",
        default=base.DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt.",
    )
    parser.add_argument(
        "--audio-sample-rate",
        type=int,
        default=44100,
        help="Audio sample rate used for extraction and mixing.",
    )
    parser.add_argument(
        "--dashscope-api-key",
        default="",
        help="DashScope API key. Defaults to DASHSCOPE_API_KEY when empty.",
    )
    parser.add_argument(
        "--openai-base-url",
        default=base.DEFAULT_OPENAI_BASE_URL,
        help="OpenAI-compatible base URL for DashScope.",
    )
    parser.add_argument(
        "--eleven-api-key",
        default="",
        help="ElevenLabs API key. Defaults to ELEVEN_API_KEY when empty.",
    )
    parser.add_argument(
        "--demucs-model-paths",
        nargs="+",
        default=base.DEFAULT_DEMUCS_MODEL_PATHS,
        help="Demucs checkpoint paths.",
    )
    parser.add_argument(
        "--demucs-device",
        default=None,
        help="Device for Demucs. Defaults to auto selection.",
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
        default=["./model_weights/AVI-Edit/avi_edit.pt"],
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


def create_demo(runtime_args: argparse.Namespace) -> gr.Blocks:
    demo_root = Path(runtime_args.demo_root).expanduser()
    if not demo_root.is_absolute():
        demo_root = (SCRIPT_DIR / demo_root).resolve()
    else:
        demo_root = demo_root.resolve()
    if not demo_root.exists():
        raise FileNotFoundError(f"Demo root does not exist: {demo_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Gradio launcher.")

    build_sam2_video_predictor = ensure_sam2_importable(demo_root)
    sam2_checkpoint = Path(resolve_demo_path(runtime_args.sam2_checkpoint, demo_root))
    if not sam2_checkpoint.exists():
        raise FileNotFoundError(f"SAM2 checkpoint does not exist: {sam2_checkpoint}")

    output_root = Path(resolve_repo_path(runtime_args.output_dir))
    temp_root = Path(resolve_repo_path(runtime_args.temp_root))
    output_root.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)

    runtime_args.output_dir = str(output_root)
    runtime_args.temp_root = str(temp_root)
    runtime_args.sam2_checkpoint = str(sam2_checkpoint)
    runtime_args.demucs_model_paths = resolve_repo_paths(runtime_args.demucs_model_paths)
    runtime_args.audio_processor_path = resolve_repo_path(runtime_args.audio_processor_path)
    runtime_args.audio_encoder_weights = resolve_repo_path(
        runtime_args.audio_encoder_weights
    )
    runtime_args.model_paths = resolve_repo_paths(runtime_args.model_paths)
    runtime_args.text_encoder_path = resolve_repo_path(runtime_args.text_encoder_path)
    runtime_args.tokenizer_path = resolve_repo_path(runtime_args.tokenizer_path)
    runtime_args.vae_path = resolve_repo_path(runtime_args.vae_path)
    runtime_args.dashscope_api_key = (
        runtime_args.dashscope_api_key or os.getenv("DASHSCOPE_API_KEY", "")
    )
    runtime_args.eleven_api_key = (
        runtime_args.eleven_api_key or os.getenv("ELEVEN_API_KEY", "")
    )
    torch_dtype = torch.bfloat16
    device = "cuda"

    print("Loading SAM2 video predictor...")
    sam2_predictor = build_sam2_video_predictor(
        runtime_args.sam2_config,
        runtime_args.sam2_checkpoint,
        device=device,
    )
    print("Loading audio encoder...")
    audio_processor, audio_encoder = base.init_audio_encoder(
        runtime_args.audio_processor_path,
        runtime_args.audio_encoder_weights,
        torch_dtype,
        device,
    )
    print("Loading video diffusion pipeline...")
    video_pipe = base.init_pipeline(
        runtime_args.model_paths,
        runtime_args.text_encoder_path,
        runtime_args.vae_path,
        runtime_args.tokenizer_path,
        torch_dtype,
        device,
    )
    print("Loading Demucs separation model...")
    demucs_model = base.Demucs4_SeparationModel(
        model_paths=runtime_args.demucs_model_paths,
        device=runtime_args.demucs_device,
    )
    client = None

    def get_client():
        nonlocal client
        if client is None:
            client = base.build_openai_compatible_client(
                api_key=runtime_args.dashscope_api_key,
                base_url=runtime_args.openai_base_url,
            )
        return client

    def run_loaded_inference(
        video_path: str,
        mask_path: str,
        prompt: str,
        output_name: str,
        seed_value: str | float | int | None,
        max_num_frames_value: str | float | int | None,
    ) -> dict[str, Any]:
        base.ensure_file_exists(video_path, "Input video")
        base.ensure_file_exists(mask_path, "Input mask")

        max_num_frames = (
            parse_optional_int(max_num_frames_value, "Max Num Frames")
            if max_num_frames_value not in (None, "")
            else runtime_args.max_num_frames
        )
        seed = parse_optional_int(seed_value, "Seed")
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        base.set_seed(seed)
        print(f"Using seed: {seed}")

        torch.cuda.empty_cache()
        try:
            video_frames = base.load_video_frames(
                video_path, runtime_args.height, runtime_args.width
            )
            fps = base.read_video_fps(video_path)
            mask_frames = base.load_mask_frames(
                mask_path,
                runtime_args.height,
                runtime_args.width,
                runtime_args.mask_threshold,
                len(video_frames),
            )
            num_frames = base.get_target_num_frames(
                len(video_frames), len(mask_frames), max_num_frames
            )
            print(
                f"Aligned frames -> video: {len(video_frames)}, mask: {len(mask_frames)}, used: {num_frames}"
            )
            video_frames = video_frames[:num_frames]
            mask_frames = mask_frames[:num_frames]
            duration_seconds = num_frames / fps

            base_name = (output_name or "").strip() or Path(video_path).stem
            task_stamp = (
                datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                + "_"
                + uuid.uuid4().hex[:8]
            )
            request_dir = output_root / f"{base_name}_{task_stamp}"
            request_dir.mkdir(parents=True, exist_ok=True)
            work_dir = request_dir / f"{base_name}_audio_agent"
            work_dir.mkdir(parents=True, exist_ok=True)

            agent_video_path = work_dir / "agent_video.mp4"
            agent_mask_video_path = work_dir / "agent_mask.mp4"
            base.save_frames_to_mp4(video_frames, str(agent_video_path), fps=fps)
            base.save_frames_to_mp4(mask_frames, str(agent_mask_video_path), fps=fps)

            original_audio_path = work_dir / "original_audio.wav"
            base.extract_audio_from_video(
                video_path,
                str(original_audio_path),
                runtime_args.audio_sample_rate,
            )

            openai_client = get_client()
            label_raw, sound_type, object_name = base.classify_video_mask_caption(
                openai_client,
                str(agent_video_path),
                str(agent_mask_video_path),
                prompt,
            )
            print(f"Audio agent type: {sound_type}, object: {object_name}")

            original_audio_array, stem_arrays, stem_paths = base.separate_audio_stems(
                str(original_audio_path),
                demucs_model,
                work_dir / "stems",
                runtime_args.audio_sample_rate,
            )
            background_arrays, background_paths = base.build_background_tracks(
                original_audio_array,
                stem_arrays,
                work_dir / "backgrounds",
                runtime_args.audio_sample_rate,
            )

            generated_audio_path = work_dir / "generated_replacement.mp3"
            generated_audio_info = base.generate_replacement_audio(
                client=openai_client,
                sound_type=sound_type,
                prompt=prompt,
                object_name=object_name,
                duration_seconds=duration_seconds,
                eleven_api_key=runtime_args.eleven_api_key,
                generated_audio_path=str(generated_audio_path),
            )

            edited_audio_path = work_dir / "edited_audio.wav"
            edited_audio_path_str = base.mix_generated_audio_into_background(
                background_audio=background_arrays[sound_type],
                replacement_audio_path=str(generated_audio_path),
                sample_rate=runtime_args.audio_sample_rate,
                output_audio_path=str(edited_audio_path),
            )

            with torch.no_grad():
                audio_embed = base.build_audio_embeddings(
                    edited_audio_path_str,
                    fps,
                    num_frames,
                    audio_encoder,
                    audio_processor,
                    torch_dtype,
                    device,
                )
                mask_tensor = base.build_mask_tensor(mask_frames, torch_dtype, device)
                mask_latent = base.downsample_mask_to_latent(
                    mask_tensor, torch_dtype, device
                )
                latent_frame_mask = torch.ones(
                    (num_frames - 1) // 4 + 1, dtype=torch.bool, device=device
                ).unsqueeze(0)
                video = video_pipe(
                    input_video_ori=[video_frames],
                    instance_mask_list=[mask_latent],
                    mask_type_adain=runtime_args.mask_type_adain,
                    prompt=prompt,
                    negative_prompt=runtime_args.negative_prompt,
                    seed=seed,
                    tiled=True,
                    height=runtime_args.height,
                    width=runtime_args.width,
                    audio_embed_s2v=[audio_embed],
                    num_frames=num_frames,
                    cfg_scale=runtime_args.cfg_scale,
                    num_inference_steps=runtime_args.num_inference_steps,
                    return_tensor=True,
                    latent_frame_mask=latent_frame_mask,
                )

            video_out = (video.permute(0, 2, 1, 3, 4).cpu() + 1.0) / 2.0
            final_name = f"{base_name}_edited_{num_frames}"
            output_video_path = request_dir / f"{final_name}_with_audio.mp4"
            base.save_video_as_grid_and_mp4_with_audio(
                video_out.float(),
                str(request_dir),
                edited_audio_path_str,
                fps=fps,
                name=final_name,
            )

            metadata = {
                "seed": seed,
                "video_path": video_path,
                "mask_path": mask_path,
                "prompt": prompt,
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

            return {
                "seed": seed,
                "output_dir": str(request_dir),
                "work_dir": str(work_dir),
                "metadata_path": str(metadata_path),
                "output_video_path": str(output_video_path),
                "edited_audio_path": edited_audio_path_str,
                "sound_type": sound_type,
                "object_name": object_name,
                "num_frames": num_frames,
                "fps": fps,
            }
        finally:
            torch.cuda.empty_cache()

    def init_sam2_session(video_value: Any):
        video_path = normalize_video_path(video_value)
        if not video_path:
            raise gr.Error("Please upload a video first.")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise gr.Error("Failed to open the uploaded video.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > runtime_args.sam2_max_frames:
            cap.release()
            raise gr.Error(
                f"The video has {total_frames} frames, which exceeds the current limit of {runtime_args.sam2_max_frames}."
            )

        session_id = (
            datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        )
        frame_dir = temp_root / session_id / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)

        frame_count = 0
        first_frame_path = None
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame_path = frame_dir / f"{frame_count:05d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            if frame_count == 0:
                first_frame_path = frame_path
            frame_count += 1
        cap.release()

        if frame_count == 0 or first_frame_path is None:
            raise gr.Error("The uploaded video does not contain any readable frames.")

        sam2_state = sam2_predictor.init_state(video_path=str(frame_dir))
        sam2_predictor.reset_state(sam2_state)
        preview = Image.open(first_frame_path).convert("RGB")
        status = (
            f"SAM2 initialized successfully. Extracted {frame_count} frames. "
            "Click positive or negative points on the first frame, then generate the full mask video."
        )
        return (
            preview,
            str(first_frame_path),
            str(frame_dir),
            [],
            [],
            sam2_state,
            None,
            None,
            None,
            None,
            status,
        )

    def sam2_interactive_click(
        point_type: str,
        tracking_points: list[list[float]] | None,
        tracking_labels: list[int] | None,
        original_first_frame_path: str,
        sam2_state: Any,
        evt: gr.SelectData,
    ):
        if sam2_state is None or not original_first_frame_path:
            raise gr.Error("Please upload a video and initialize SAM2 first.")

        points = list(tracking_points or [])
        labels = list(tracking_labels or [])
        point = [float(evt.index[0]), float(evt.index[1])]
        label = 1 if point_type == "include" else 0
        points.append(point)
        labels.append(label)

        _, _, out_mask_logits = sam2_predictor.add_new_points(
            inference_state=sam2_state,
            frame_idx=0,
            obj_id=1,
            points=np.array(points, dtype=np.float32),
            labels=np.array(labels, dtype=np.int32),
        )

        mask = (out_mask_logits[0] > 0.0).detach().cpu().numpy()
        if mask.ndim == 3:
            mask = mask[0]
        preview_image = show_mask_on_image(
            original_first_frame_path,
            mask.astype(np.uint8),
            points,
            labels,
        )
        return preview_image, points, labels, None, None, f"Recorded {len(points)} points."

    def clear_sam2_points(original_first_frame_path: str, sam2_state: Any):
        if not original_first_frame_path:
            return None, [], [], None, None, "Please upload a video first."
        if sam2_state is not None:
            sam2_predictor.reset_state(sam2_state)
        clean_image = Image.open(original_first_frame_path).convert("RGB")
        return clean_image, [], [], None, None, "Cleared all interaction points."

    def generate_full_video_mask(
        video_frames_dir: str,
        video_value: Any,
        tracking_points: list[list[float]] | None,
        sam2_state: Any,
    ):
        video_input_path = normalize_video_path(video_value)
        if sam2_state is None or not video_frames_dir or not video_input_path:
            raise gr.Error("Please upload a video and initialize SAM2 first.")
        if not tracking_points:
            raise gr.Error("Add at least one interaction point before generating the mask.")

        cap = cv2.VideoCapture(video_input_path)
        if not cap.isOpened():
            raise gr.Error("Failed to read the original video, so the mask cannot be generated.")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        mask_output_path = Path(video_frames_dir).parent / "instance_mask.mp4"
        writer = cv2.VideoWriter(
            str(mask_output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise gr.Error("Failed to create the mask video output file.")

        try:
            for _, _, out_mask_logits in sam2_predictor.propagate_in_video(sam2_state):
                mask = (out_mask_logits[0] > 0.0).detach().cpu().numpy().astype(np.uint8)
                if mask.ndim == 3:
                    mask = mask[0]
                mask = mask * 255
                if mask.shape[:2] != (height, width):
                    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                writer.write(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
        finally:
            writer.release()

        status = f"Mask video generated: {mask_output_path}"
        return str(mask_output_path), str(mask_output_path), status

    def run_editing(
        video_value: Any,
        mask_path: str,
        prompt: str,
        output_name: str,
        seed_value: str | float | int | None,
        max_num_frames_value: str | float | int | None,
    ):
        video_path = normalize_video_path(video_value)
        if not video_path:
            raise gr.Error("Please upload a video first.")
        if not mask_path:
            raise gr.Error("Please generate the full mask first.")
        if not prompt or not prompt.strip():
            raise gr.Error("Please enter an editing prompt.")

        result = run_loaded_inference(
            video_path=video_path,
            mask_path=mask_path,
            prompt=prompt,
            output_name=output_name,
            seed_value=seed_value,
            max_num_frames_value=max_num_frames_value,
        )
        with open(result["metadata_path"], "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        status = f"Editing completed: {result['output_video_path']}"
        return result["output_video_path"], metadata, status

    with gr.Blocks(title="AVI-Edit SAM2 Audio Agent") as demo:
        gr.Markdown("## AVI-Edit: Audio-sync Video Instance Editing with Granularity-Aware Mask Refiner")

        tracking_points = gr.State([])
        tracking_labels = gr.State([])
        original_first_frame_path = gr.State()
        video_frames_dir = gr.State()
        sam2_state = gr.State()
        generated_mask_path = gr.State()

        with gr.Row():
            with gr.Column(scale=5):
                gr.Markdown("### Step 1: Upload a video and select the target on the first frame")
                video_input = gr.Video(label="Input Video", height=320)
                with gr.Row():
                    point_type = gr.Radio(
                        ["include", "exclude"],
                        value="include",
                        label="Point Type",
                    )
                    clear_btn = gr.Button("Clear Points and Reset")
                first_frame_display = gr.Image(
                    label="First-Frame Interaction Preview",
                    interactive=False,
                    height=480,
                )
                gen_mask_btn = gr.Button("Step 2: Generate Full Mask", variant="primary")

            with gr.Column(scale=5):
                gr.Markdown("### Step 3: Review the mask and run editing")
                mask_preview = gr.Video(label="Generated Mask Video", height=320, autoplay=True)
                prompt = gr.Textbox(
                    label="Editing Prompt",
                    placeholder="Example: A woman with blonde hair, wearing a blue coat over a pink top, is engaged in a conversation with a man in a brown coat and a man in a patterned scarf, saying 'No, please. When the latkes are frying, the kitchen should be closed.'.",
                    lines=3,
                )
                with gr.Accordion("Advanced", open=False):
                    output_name = gr.Textbox(
                        label="Output Filename Prefix (Optional)",
                        placeholder="Leave empty to use the source video filename.",
                    )
                    seed_value = gr.Textbox(
                        label="Random Seed (Optional)",
                        placeholder="Leave empty to use a random seed.",
                    )
                    max_num_frames_value = gr.Textbox(
                        label="Max Num Frames (Optional)",
                        placeholder=(
                            "Leave empty to use the launcher default."
                            if runtime_args.max_num_frames is None
                            else f"Leave empty to use the launcher default: {runtime_args.max_num_frames}."
                        ),
                    )
                run_btn = gr.Button("Step 4: Run Editing", variant="primary")
                final_output = gr.Video(label="Edited Result", height=320)
                metadata_output = gr.JSON(label="Inference Metadata")
                status_text = gr.Textbox(label="Status", interactive=False)

        video_input.upload(
            fn=init_sam2_session,
            inputs=[video_input],
            outputs=[
                first_frame_display,
                original_first_frame_path,
                video_frames_dir,
                tracking_points,
                tracking_labels,
                sam2_state,
                generated_mask_path,
                mask_preview,
                final_output,
                metadata_output,
                status_text,
            ],
        )
        first_frame_display.select(
            fn=sam2_interactive_click,
            inputs=[
                point_type,
                tracking_points,
                tracking_labels,
                original_first_frame_path,
                sam2_state,
            ],
            outputs=[
                first_frame_display,
                tracking_points,
                tracking_labels,
                generated_mask_path,
                mask_preview,
                status_text,
            ],
        )
        clear_btn.click(
            fn=clear_sam2_points,
            inputs=[original_first_frame_path, sam2_state],
            outputs=[
                first_frame_display,
                tracking_points,
                tracking_labels,
                generated_mask_path,
                mask_preview,
                status_text,
            ],
        )
        gen_mask_btn.click(
            fn=generate_full_video_mask,
            inputs=[video_frames_dir, video_input, tracking_points, sam2_state],
            outputs=[generated_mask_path, mask_preview, status_text],
        )
        run_btn.click(
            fn=run_editing,
            inputs=[
                video_input,
                generated_mask_path,
                prompt,
                output_name,
                seed_value,
                max_num_frames_value,
            ],
            outputs=[final_output, metadata_output, status_text],
        )

    return demo


def main():
    runtime_args = build_argparser().parse_args()
    demo = create_demo(runtime_args)
    demo.queue().launch(
        server_name=runtime_args.server_name,
        server_port=runtime_args.server_port,
        share=runtime_args.share,
    )


if __name__ == "__main__":
    main()
