import cv2
import librosa
import os
import time
import torch
import torchvision
import torch.distributed as dist
import numpy as np
import moviepy as mpy
from loguru import logger
from tqdm import tqdm
import soundfile as sf

import PIL
from PIL import Image, ImageFile
from moviepy import AudioFileClip

from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

FPS = 30
LAYER = 66

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="input parameters")
    parser.add_argument(
        "--image_path",
        type=str,
        default='',
        required=True,
        help="image_path",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default='',
        required=True,
        help="prompt_path",
    )
    parser.add_argument(
        "--music_path",
        type=str,
        default='',
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
        "--enable_music_inject", 
        default=False, 
        action="store_true", 
        help="Whether to inject music."
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
        help="music inject layers"
    )
    parser.add_argument(
        "--enable_refimage", 
        default=False, 
        action="store_true", 
        help="enable refimage."
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
        default='outputs/local_video',
        help="output folder",
    )
    parser.add_argument(
        "--global_video_path",
        type=str,
        default='',
        required=True,
        help="global video path",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default='',
        required=True,
        help="timestamp",
    )
    parser.add_argument(
        "--enable_dynamicfps", 
        default=False, 
        action="store_true", 
        help="enable dynamicfps."
    )
    parser.add_argument(
        "--enable_skip_layer", 
        default=False, 
        action="store_true", 
        help="enable skip dit layer."
    )
    parser.add_argument(
        "--sigma_shift",
        type=int,
        default=5,
        help="sigma_shift",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=48,
        help="num_inference_steps",
    )
    parser.add_argument(
        "--cfg_scale",
        type=int,
        default=5,
        help="cfg scale",
    )

    args = parser.parse_args()
    return args

def get_music_clip_149f(original_music_path, target_music_folder):
    audio = AudioFileClip(original_music_path)  
    total_duration = audio.duration 
    audio, sr = librosa.load(original_music_path, sr=None)
    duration = float(149) / FPS
    idx = 0
    t = 0
    while t + 0.2 < total_duration: 
        start_time = t
        end_time = t + duration
        if end_time >= total_duration:
            end_time = total_duration
        sliced_audio = audio[int(start_time * sr):int(end_time * sr)]
        timestamp = time.time()
        save_path = os.path.join(target_music_folder, str(idx).zfill(3) + '_' + str(timestamp).replace('.', '') + '.wav')
        sf.write(save_path, sliced_audio, sr)
        t += duration
        idx += 1

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
    logger.info(f'audio feature shape: {audio_feature.shape} for {music_path}')
    np.save(output_path, audio_feature)

def get_music_features(music_folder):
    dirs = [f for f in sorted(os.listdir(music_folder)) if f.endswith('.wav')]
    for idx, name in tqdm(enumerate(dirs)):
        music_path = os.path.join(music_folder, name)
        output_path = os.path.join(music_folder, name.replace('.wav', '_librosa_feature.npy'))
        if os.path.exists(output_path) is False:
            get_music_base_feature(music_path, output_path)

def crop_and_resize(image: PIL.Image.Image, target_width=720, target_height=1280):
    width, height = image.size
    scale = min(target_width / width, target_height / height)
    resized_height = round(height * scale)
    resized_width = round(width * scale)
    image = torchvision.transforms.functional.resize(
        image,
        (resized_height, resized_width),
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR
    )
    ## pad 127 to target size
    target_image = np.ones((target_height, target_width, 3), dtype=np.uint8) * 127
    tl_x = (target_width - resized_width) // 2
    tl_y = (target_height - resized_height) // 2
    br_x = tl_x + resized_width
    br_y = tl_y + resized_height
    target_image[tl_y: br_y, tl_x: br_x, :] = np.array(image, dtype=np.uint8)
    image = Image.fromarray(target_image)
    return image, (tl_x, tl_y, br_x, br_y)

def process_global_video_firstlastframe(video_path, height, width, total_frames):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    N = len(frames) 
    logger.info(f'global video frame number: {N}')

    seg_num = int(np.ceil(total_frames / 149)) # 149 cannot be changeg
    frame_interval_num = float(total_frames) / N

    keyframes_list = []
    keyframes_mask_list = []
    for i in range(seg_num): 
        mask = np.zeros(149, dtype=np.int32)
        if i != seg_num - 1:
            cnt = 0
            while (cnt * frame_interval_num < 149 - frame_interval_num):
                index = int(np.ceil(frame_interval_num * cnt))
                mask[index] = 1
                cnt += 1
        else:
            end_index = total_frames - 149 * i - 1
            mask[end_index] = 1
            cnt = 0
            while (cnt * frame_interval_num < end_index - frame_interval_num):
                index = int(np.ceil(frame_interval_num * cnt))
                mask[index] = 1
                cnt += 1

        keyframes_mask_list.append(mask)

    sum = 0
    for mask in keyframes_mask_list:
        sum += np.sum(mask)

    ## fill keyframes_list
    index = 0
    for mask in keyframes_mask_list:
        keyframes = np.zeros((149, height, width, 3), dtype=np.uint8)
        keyframes = [Image.fromarray(img.astype('uint8')) for img in keyframes]
        for j in range(len(mask)):
            if mask[j] == 1:
                frame = Image.fromarray(frames[index].astype('uint8'))
                frame, _ = crop_and_resize(frame, target_height=height, target_width=width)
                keyframes[j] = frame.copy()
                index += 1
        keyframes_list.append(keyframes)

    for i in range(len(keyframes_list) - 1):
        keyframes_list[i][-1] = keyframes_list[i + 1][0]
        keyframes_mask_list[i][-1] = 1
    
    return keyframes_list, keyframes_mask_list

def gen_video_single(pipe, prompt, input_config):
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    target_height = input_config['target_height']
    target_width = input_config['target_width']
    ## refimage
    enable_refimage = input_config.get('enable_refimage', False)
    if enable_refimage:
        refimage_path = input_config['refimage_path']
        refimage = Image.open(refimage_path)
        refimage, (tl_x, tl_y, br_x, br_y) = crop_and_resize(refimage, target_width=target_width, target_height=target_height) # resize
    else:
        refimage = None
    ## music feature
    enable_music_inject = input_config['enable_music_inject']
    music_feature = input_config['music_feature']

    ## generated video
    seed = input_config['seed']
    num_frames = input_config['num_frames']
    interp_mode = input_config['interp_mode']
    keyframes = input_config['keyframes']
    keyframes_mask = input_config['keyframes_mask']
    enable_dynamicfps = input_config.get('enable_dynamicfps', False)
    input_fps = input_config.get('input_fps', 30.0)
    enable_skip_layer = input_config.get('enable_skip_layer', False)
    sigma_shift = input_config.get('sigma_shift', 5)
    num_inference_steps = input_config.get('num_inference_steps', 48)
    cfg_scale = input_config.get('cfg_scale', 5)

    video = pipe(
        prompt=prompt, 
        negative_prompt=negative_prompt, 
        num_inference_steps=num_inference_steps, 
        seed=seed, tiled=True, height=target_height, width=target_width,
        enable_music_inject=enable_music_inject,
        music_feature=music_feature,
        num_frames=num_frames,
        interp_mode=interp_mode,
        enable_refimage=enable_refimage,
        refimage=refimage,
        keyframes=keyframes,
        keyframes_mask=keyframes_mask,
        enable_dynamicfps=enable_dynamicfps,
        input_fps=input_fps,
        enable_skip_layer=enable_skip_layer,
        sigma_shift=sigma_shift,
        cfg_scale=cfg_scale)
    
    if dist.get_rank() == 0:
        save_video_path = input_config['save_video_path']
        save_video(video, save_video_path, fps=FPS, quality=5)

        ## crop generated video
        video = mpy.VideoFileClip(save_video_path)
        croper = mpy.video.fx.Crop(x1=tl_x, y1=tl_y, x2=br_x, y2=br_y)
        video = croper.apply(video)

        ## add music
        music_path = input_config['music_path']
        video.audio = mpy.AudioFileClip(music_path)
        save_video_path = save_video_path[:-4] + "_music.mp4"
        video.write_videofile(save_video_path, codec='libx264', audio_codec='aac')

    dist.barrier(device_ids=[dist.get_rank()])

def init_dit_model(args):
    world_size = int(os.environ.get("WORLD_SIZE", 1))   

    ## usp config
    ulysses_degree = world_size
    sequence_parallel_degree = world_size
    ring_degree=sequence_parallel_degree // ulysses_degree
    data_parallel_degree = 1
    usp_config = {
        'data_parallel_degree': data_parallel_degree,
        'sequence_parallel_degree': sequence_parallel_degree,
        'ring_degree': ring_degree,
        'ulysses_degree': ulysses_degree
    }

    ## load models
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan-Dancer-14B", 
                        origin_file_pattern="local_model.safetensors", 
                        offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan-Dancer-14B", 
                        origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", 
                        offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan-Dancer-14B", 
                        origin_file_pattern="Wan2.1_VAE.pth", 
                        offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan-Dancer-14B", 
                        origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", 
                        offload_device="cpu"),
        ],
        tokenizer_config=ModelConfig(model_id="Wan-AI/Wan-Dancer-14B",origin_file_pattern="google/umt5-xxl/"),
        skip_download=True,
        redirect_common_files=False,
        use_usp=True,
        usp_config=usp_config,
        dit_model_type=1, # 1 for our trained model
        enable_music_inject=True,
        enable_refimage=True,
        enable_global=True,
        enable_dynamicfps=True,
        enable_unimodel=True
    )

    pipe.enable_vram_management()
    return pipe

def gen_video(pipe, music_path, music_feature_path, prompt, output_video_path, 
              seed=0, max_pixels=1280*720, height=1280, width=720,
              num_frames=81, enable_refimage=False, refimage_path=None,
              keyframes=None, keyframes_mask=None, enable_dynamicfps=False, 
              enable_skip_layer=False, sigma_shift=5, num_inference_steps=48):
    ## input parameters
    input_config = {}
    input_config['music_path'] = music_path
    input_config['enable_music_inject'] = True
    music_feature = np.load(music_feature_path)
    music_feature = torch.from_numpy(music_feature).to(dtype=torch.bfloat16, device='cuda')
    input_config['music_feature'] = music_feature
    input_config['max_pixels'] = max_pixels
    input_config['save_video_path'] = output_video_path
    input_config['seed'] = seed
    input_config['target_height'] = height
    input_config['target_width'] = width
    input_config['num_frames'] = num_frames
    input_config['interp_mode'] = 'bilinear'
    input_config['enable_refimage'] = enable_refimage
    input_config['refimage_path'] = refimage_path
    input_config['keyframes'] = keyframes
    input_config['keyframes_mask'] = keyframes_mask
    input_config['input_fps'] = 30
    input_config['enable_dynamicfps'] = enable_dynamicfps
    input_config['enable_skip_layer'] = enable_skip_layer
    input_config['sigma_shift'] = sigma_shift
    input_config['num_inference_steps'] = num_inference_steps

    # gen video
    gen_video_single(pipe, prompt, input_config)

def process():
    args = parse_args()
    final_name = args.image_path.split('/')[-1].split('.')[0] + '_' + \
        args.music_path.split('/')[-1].split('.')[0]
    time_name = args.timestamp
    music_folder = 'outputs/tmp_results/' + final_name + '_' + str(time_name)
    os.makedirs(music_folder, exist_ok=True)

    ## 0. process global video
    video_path = args.global_video_path
    height = args.height
    width = args.width
    audio = AudioFileClip(args.music_path)  # load the audio file
    total_duration = audio.duration  # total duration of the audio file
    total_frames = int(total_duration * FPS)
    logger.info(f'total frames: {total_frames}')
    keyframes_list, keyframes_mask_list = process_global_video_firstlastframe(video_path, height, width, total_frames)

    # replace the first frame of the first segment with the input image
    input_image = Image.open(args.image_path)
    video_h, video_w = keyframes_list[0][0].size[1], keyframes_list[0][0].size[0]
    if min(video_h, video_w) < 512: 
        input_image_resized, _ = crop_and_resize(input_image, target_height=height, target_width=width)
        keyframes_list[0][0] = input_image_resized

    # save keyframe videos
    for i in range(len(keyframes_list)):
        save_keyframes_path = os.path.join(music_folder, f'keyframes_{str(i).zfill(2)}.mp4')
        save_video(keyframes_list[i], save_keyframes_path, fps=FPS, quality=5)

    ## 1. init model
    prompt_path = args.prompt_path
    # prompt
    with open(prompt_path, 'r') as f:
        prompt = f.read().strip()
        prompt += ', 帧率是30fps。'
    logger.info(f'---- prompt: {prompt} ----')
    pipe = init_dit_model(args)

    ## 2. slice music to 5s segments；
    original_music_path = args.music_path
    target_music_root_folder = music_folder
    if dist.get_rank() == 0:
        get_music_clip_149f(original_music_path, target_music_root_folder)
    dist.barrier(device_ids=[dist.get_rank()])

    ## 3. encode music
    if dist.get_rank() == 0:
        get_music_features(music_folder)
    dist.barrier(device_ids=[dist.get_rank()])
        
    ## 4. generate video
    video_paths = []
    dirs = [f for f in sorted(os.listdir(music_folder)) if f.endswith('.wav')]
    for idx, name in tqdm(enumerate(dirs)):
        music_path = os.path.join(music_folder, name)
        music_feature_path = os.path.join(music_folder, name[:-4] + '_librosa_feature.npy')
        refimage_path = args.image_path 
        seed = idx * 10 + args.seed
        output_video_path = os.path.join(music_folder, name[:-4] + "_seed" + str(seed) + ".mp4")
        gen_video(pipe, music_path, music_feature_path, prompt, 
                    output_video_path, seed=seed,
                    height=args.height, width=args.width, num_frames=args.num_frames,
                    enable_refimage=True, refimage_path=refimage_path,
                    keyframes=keyframes_list[idx], keyframes_mask=keyframes_mask_list[idx],
                    enable_dynamicfps=True,
                    enable_skip_layer=True,
                    sigma_shift=args.sigma_shift,
                    num_inference_steps=args.num_inference_steps)
        video_paths.append(output_video_path[:-4] + "_music.mp4")

    ## 5. combine with music
    music_path = args.music_path
    music = mpy.AudioFileClip(music_path)
    total_duration = music.duration  
    output_video_folder = args.output_folder
    os.makedirs(output_video_folder, exist_ok=True)
    seed = args.seed
    if len(video_paths) > 0:
        output_video_path = os.path.join(output_video_folder, final_name + '_' + str(time_name) + "_seed" + str(seed) + '.mp4')
        clips = [mpy.VideoFileClip(vp) for vp in video_paths]
        final_clip = mpy.concatenate_videoclips(clips, method="compose")
        final_clip.audio = music
        final_clip = final_clip[:total_duration-0.2] 
        final_clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac', fps=FPS)
        logger.info(f"Final video saved to {output_video_path}")
    else:
        logger.warning("No video files found to concatenate.")

def main():
    process()

if __name__ == '__main__':
    main()