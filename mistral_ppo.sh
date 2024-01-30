cd ~/trlx/examples/hh

export HF_HOME=/scratch/banghua

CUDA_VISIBLE_DEVICES=4,5,6 accelerate launch --num_processes 3 --main_process_port 29501 --config_file ../../configs/accelerate/zero3.yaml mistral_ppo.py