#!/bin/bash

# Check if two arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 PROMPT_PATH MODEL_PATH"
    exit 1
fi

# Assign the arguments to variables
PROMPT_PATH=$1
MODEL_PATH=$2

CUDA_VISIBLE_DEVICES=7 python '/home/banghua/trlx/trlx/trainer/vllm_generate.py' \
--prompt_path "$PROMPT_PATH" \
--model_path "$MODEL_PATH"