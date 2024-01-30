cd ~/trlx/examples/hh

export HF_HOME=/scratch/banghua

accelerate launch --num_processes 4 --main_process_port 29501 --config_file ../../configs/accelerate/zero3.yaml mixtral_ppo.py