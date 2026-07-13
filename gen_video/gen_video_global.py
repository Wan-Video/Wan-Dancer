import librosa
import os
import torch
import torchvision
import torch.distributed as dist
import numpy as np
import moviepy as mpy
from loguru import logger

import PIL
from PIL import Image, ImageFile

from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

FPS = 30


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="input parameters")
    parser.add_argument(
        "--image_path",
        type=str,
        default="",
        required=True,
        help="image_path",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="",
        required=True,
        help="prompt_path",
    )
    parser.add_argument(
        "--music_path",
        type=str,
        default="",
        required=True,
        help="music_path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        required=True,
        help="seed",
    )
    parser.add_argument(
        "--dit_model_path",
        type=str,
        default="models/wan_dancer_model/global_model.safetensors",
        required=False,
        help="dit model path",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1280,
        required=False,
        help="height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=720,
        required=False,
        help="width",
    )
    parser.add_argument(
        "--music_inject_layers",
        type=str,
        default="0, 4, 8, 12, 16, 20, 24, 27",
        help="music inject layers",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=149,
        help="num frames",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="outputs/global_video",
        help="output folder",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default="",
        help="timestamp",
    )
    parser.add_argument(
        "--sigma_shift",
        type=int,
        default=5,
        help="sigma shift",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=48,
        help="number inference steps",
    )
    parser.add_argument(
        "--cfg_scale",
        type=int,
        default=5,
        help="cfg scale",
    )
    args = parser.parse_args()
    return args


def init_dit_model(args):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    assert world_size == 8, "WORLD_SIZE must be 8"
    ## usp config
    ulysses_degree = world_size
    sequence_parallel_degree = world_size

    ring_degree = sequence_parallel_degree // ulysses_degree
    data_parallel_degree = 1
    usp_config = {
        "data_parallel_degree": data_parallel_degree,
        "sequence_parallel_degree": sequence_parallel_degree,
        "ring_degree": ring_degree,
        "ulysses_degree": ulysses_degree,
    }

    ## load models
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(
                model_id="Wan-AI/Wan-Dancer-14B",
                origin_file_pattern="global_model.safetensors",
                offload_device="cpu",
            ),
            ModelConfig(
                model_id="Wan-AI/Wan-Dancer-14B",
                origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                offload_device="cpu",
            ),
            ModelConfig(
                model_id="Wan-AI/Wan-Dancer-14B",
                origin_file_pattern="Wan2.1_VAE.pth",
                offload_device="cpu",
            ),
            ModelConfig(
                model_id="Wan-AI/Wan-Dancer-14B",
                origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                offload_device="cpu",
            ),
        ],
        tokenizer_config=ModelConfig(
            model_id="Wan-AI/Wan-Dancer-14B", origin_file_pattern="google/umt5-xxl/"
        ),
        skip_download=True,
        redirect_common_files=False,
        use_usp=True,
        usp_config=usp_config,
        dit_model_type=1,  # 1 for our trained model
        enable_music_inject=True,
        enable_refimage=True,
        enable_global=True,
        enable_dynamicfps=True,
        enable_unimodel=True,
    )

    pipe.enable_vram_management()
    return pipe


def get_music_base_feature(music_path, output_path, fps=30):
    hop_length = 512
    sr = fps * hop_length
    data, sr = librosa.load(music_path, sr=sr)
    sr = 22050
    envelope = librosa.onset.onset_strength(y=data, sr=sr)
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=20).T
    chroma = librosa.feature.chroma_cens(
        y=data, sr=sr, hop_length=hop_length, n_chroma=12
    ).T
    peak_idxs = librosa.onset.onset_detect(
        onset_envelope=envelope.flatten(), sr=sr, hop_length=hop_length
    )
    peak_onehot = np.zeros_like(envelope, dtype=np.float32)
    peak_onehot[peak_idxs] = 1.0
    start_bpm = librosa.beat.tempo(y=librosa.load(music_path)[0])[0]
    _, beat_idxs = librosa.beat.beat_track(
        onset_envelope=envelope,
        sr=sr,
        hop_length=hop_length,
        start_bpm=start_bpm,
        tightness=100,
    )
    beat_onehot = np.zeros_like(envelope, dtype=np.float32)
    beat_onehot[beat_idxs] = 1.0
    audio_feature = np.concatenate(
        [envelope[:, None], mfcc, chroma, peak_onehot[:, None], beat_onehot[:, None]],
        axis=-1,
    )
    logger.info(f"audio feature shape: {audio_feature.shape} for {music_path}")
    np.save(output_path, audio_feature)


def crop_and_resize(image: PIL.Image.Image, target_width=720, target_height=1280):
    width, height = image.size
    scale = min(target_width / width, target_height / height)
    resized_height = round(height * scale)
    resized_width = round(width * scale)
    image = torchvision.transforms.functional.resize(
        image,
        (resized_height, resized_width),
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
    )
    ## pad 127 to target size
    target_image = np.ones((target_height, target_width, 3), dtype=np.uint8) * 127
    tl_x = (target_width - resized_width) // 2
    tl_y = (target_height - resized_height) // 2
    br_x = tl_x + resized_width
    br_y = tl_y + resized_height
    target_image[tl_y:br_y, tl_x:br_x, :] = np.array(image, dtype=np.uint8)
    image = Image.fromarray(target_image)
    return image, (tl_x, tl_y, br_x, br_y)


def gen_video_single(pipe, prompt, input_config):
    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    img_path = input_config["img_path"]

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img = Image.open(img_path)
    target_height = input_config["target_height"]
    target_width = input_config["target_width"]
    ## resize input image
    img, (tl_x, tl_y, br_x, br_y) = crop_and_resize(
        img, target_width=target_width, target_height=target_height
    )
    ## refimage
    enable_refimage = input_config.get("enable_refimage", False)
    if enable_refimage:
        refimage_path = input_config["refimage_path"]
        refimage = Image.open(refimage_path)
        refimage, (tl_x, tl_y, br_x, br_y) = crop_and_resize(
            refimage, target_width=target_width, target_height=target_height
        )
    else:
        refimage = None

    ## music feature
    enable_music_inject = input_config["enable_music_inject"]
    music_feature = input_config["music_feature"]

    ## input parameters
    seed = input_config["seed"]
    num_frames = input_config["num_frames"]
    interp_mode = input_config["interp_mode"]
    enable_global = input_config.get("enable_global", False)
    enable_dynamicfps = input_config.get("enable_dynamicfps", False)
    input_fps = input_config.get("input_fps", 30.0)
    enable_vae_decode_framewise = input_config.get("enable_vae_decode_framewise", False)
    enable_skip_layer = input_config.get("enable_skip_layer", False)
    enable_unimodel = input_config.get("enable_unimodel", False)
    sigma_shift = input_config.get("sigma_shift", 5)
    num_inference_steps = input_config.get("num_inference_steps", 48)
    cfg_scale = input_config.get("cfg_scale", 5)

    ## mask
    mask = np.zeros(num_frames, dtype=np.int32)
    mask[0] = 1
    keyframes = np.zeros((num_frames, target_height, target_width, 3), dtype=np.uint8)
    keyframes[mask == 1] = np.array(img, dtype=np.uint8)
    keyframes = [
        Image.fromarray(img.astype("uint8")) if isinstance(img, np.ndarray) else img
        for img in keyframes
    ]
    mask = torch.tensor(mask).to(torch.int32)
    keyframes_mask = mask

    ## generated video
    video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        input_image=None,
        num_inference_steps=num_inference_steps,
        seed=seed,
        tiled=True,
        height=target_height,
        width=target_width,
        enable_music_inject=enable_music_inject,
        music_feature=music_feature,
        num_frames=num_frames,
        interp_mode=interp_mode,
        enable_refimage=enable_refimage,
        refimage=refimage,
        enable_global=enable_global,
        keyframes=keyframes,
        keyframes_mask=keyframes_mask,
        enable_dynamicfps=enable_dynamicfps,
        input_fps=input_fps,
        enable_vae_decode_framewise=enable_vae_decode_framewise,
        enable_skip_layer=enable_skip_layer,
        enable_unimodel=enable_unimodel,
        sigma_shift=sigma_shift,
        cfg_scale=cfg_scale,
    )

    if dist.get_rank() == 0:
        save_video_path = input_config["save_video_path"]
        tmp_video_path = save_video_path[:-4] + "_tmp.mp4"
        if enable_vae_decode_framewise:
            fps = 8  # 8fps for vae decode framewise
            save_video(video, tmp_video_path, fps=fps, quality=5)
        else:
            save_video(video, tmp_video_path, fps=FPS, quality=5)

        ## crop generated video
        video = mpy.VideoFileClip(tmp_video_path)
        croper = mpy.video.fx.Crop(x1=tl_x, y1=tl_y, x2=br_x, y2=br_y)
        video = croper.apply(video)

        video.write_videofile(save_video_path, codec="libx264", audio_codec="aac")
        os.remove(tmp_video_path)


def gen_video(
    pipe,
    music_feature_path,
    img_path,
    prompt,
    output_video_path,
    seed=0,
    max_pixels=1280 * 720,
    height=1280,
    width=720,
    num_frames=149,
    enable_refimage=False,
    refimage_path=None,
    enable_global=False,
    enable_dynamicfps=False,
    enable_vae_decode_framewise=False,
    enable_skip_layer=False,
    enable_unimodel=False,
    sigma_shift=5,
    num_inference_steps=48,
    cfg_scale=5,
):
    ## input parameters
    input_config = {}
    input_config["img_path"] = img_path
    input_config["enable_music_inject"] = True
    music_feature = np.load(music_feature_path)
    music_feature = torch.from_numpy(music_feature).to(
        dtype=torch.bfloat16, device="cuda"
    )
    input_config["music_feature"] = music_feature
    input_config["max_pixels"] = max_pixels
    input_config["save_video_path"] = output_video_path
    input_config["seed"] = seed
    input_config["target_height"] = height
    input_config["target_width"] = width
    input_config["num_frames"] = num_frames
    input_config["interp_mode"] = "bilinear"
    input_config["enable_refimage"] = enable_refimage
    input_config["refimage_path"] = refimage_path
    input_config["enable_global"] = enable_global
    input_fps = 30.0 / int(music_feature.shape[0] / 149.0 + 0.5)
    input_fps = "{:.4f}".format(input_fps)
    input_config["input_fps"] = float(input_fps)
    logger.info(f"input fps: {input_fps}")
    ## update prompt
    prompt += f"帧率是{input_fps}"
    logger.info(f"prompt: {prompt}")
    input_config["enable_dynamicfps"] = enable_dynamicfps
    input_config["enable_vae_decode_framewise"] = enable_vae_decode_framewise
    input_config["enable_skip_layer"] = enable_skip_layer
    input_config["enable_unimodel"] = enable_unimodel
    input_config["sigma_shift"] = sigma_shift
    input_config["num_inference_steps"] = num_inference_steps
    input_config["cfg_scale"] = cfg_scale

    ## gen video
    gen_video_single(pipe, prompt, input_config)


def main():
    args = parse_args()
    prompt_path = args.prompt_path
    ## prompt
    with open(prompt_path, "r") as f:
        prompt = f.read().strip()

    ## init dit model
    pipe = init_dit_model(args)

    final_name = (
        args.image_path.split("/")[-1].split(".")[0]
        + "_"
        + args.music_path.split("/")[-1].split(".")[0]
    )
    time_name = args.timestamp
    music_folder = "outputs/tmp_results/" + final_name + "_" + str(time_name)
    os.makedirs(music_folder, exist_ok=True)

    ## encode music
    original_music_path = args.music_path
    music_feature_path = os.path.join(music_folder, final_name + "_librosa_feature.npy")
    if dist.get_rank() == 0:
        get_music_base_feature(original_music_path, music_feature_path, fps=30)
    dist.barrier(device_ids=[dist.get_rank()])

    ## generate global video
    seed = args.seed
    img_path = args.image_path
    refimage_path = args.image_path
    output_video_folder = args.output_folder
    os.makedirs(output_video_folder, exist_ok=True)
    output_video_path = os.path.join(
        output_video_folder,
        final_name + "_seed" + str(seed) + "_" + str(time_name) + ".mp4",
    )

    gen_video(
        pipe,
        music_feature_path,
        img_path,
        prompt,
        output_video_path,
        seed=seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        enable_refimage=True,
        refimage_path=refimage_path,
        enable_global=True,
        enable_dynamicfps=True,
        enable_vae_decode_framewise=True,
        enable_skip_layer=True,
        enable_unimodel=True,
        sigma_shift=args.sigma_shift,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
    )


if __name__ == "__main__":
    main()
