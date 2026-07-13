# Activate your venv, and run the script in the project root dir

seed=0
image_path='gen_video/ref_image/3001.jpg'
prompt_path='gen_video/prompt/kpop_global.txt'
music_path='gen_video/music/KPopDance.WAV'
output_folder="outputs/global_video/"
timestamp=$(date "+%Y%m%d_%H%M%S_%N")
num_inference_steps=48
cfg_scale=5

main_process_ip=${MASTER_ADDR:-localhost} 
main_process_port=${MASTER_PORT:-8899}
machine_rank=${RANK:-0}
num_machines=${WORLD_SIZE:-1}
gpu_count=$(nvidia-smi -L | wc -l)
num_processes=$((num_machines * gpu_count))

torchrun \
    --nproc_per_node=${gpu_count} \
    --master_addr=${main_process_ip} \
    --master_port=${main_process_port} \
    --nnodes=${num_machines} \
    --node_rank=${machine_rank} \
    gen_video/gen_video_global.py \
    --seed $seed \
    --image_path $image_path \
    --prompt_path $prompt_path \
    --music_path $music_path \
    --output_folder $output_folder \
    --timestamp ${timestamp} \
    --num_inference_steps $num_inference_steps \
    --cfg_scale $cfg_scale