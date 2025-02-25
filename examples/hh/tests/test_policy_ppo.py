import json
import math
import os
import sys
from itertools import islice

import numpy as np
import torch
import tritonclient.grpc as client_util
from datasets import load_dataset
from huggingface_hub import snapshot_download
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tritonclient.utils import np_to_triton_dtype
from utils import from_openchat_to_llama, from_list_to_openchat

import trlx
from trlx.data.default_configs import (
    ModelConfig,
    OptimizerConfig,
    PPOConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

# export HOME=/scratch/banghua
# cd trlx/examples/hh
# accelerate launch --num_processes 2 --config_file ../../configs/accelerate/zero2-bf16.yaml tests/test_policy.py
# python -m debugpy --listen 5679 -m accelerate.commands.launch  --num_processes 2 --config_file ../../configs/accelerate/zero2-bf16.yaml tests/test_policy.py

default_config = TRLConfig(
    train=TrainConfig(
        seq_length=1024,
        epochs=10000,
        total_steps=10000,
        batch_size=1,
        eval_batch_size=1,
        checkpoint_interval=10000,
        eval_interval=500,
        pipeline="PromptPipeline",
        trainer="AcceleratePPOTrainer",
        checkpoint_dir="checkpoints/ppo_hh",
    ),
    model=ModelConfig(model_path="openchat/openchat_3.5", num_layers_unfrozen=2),
    tokenizer=TokenizerConfig(tokenizer_path="openchat/openchat_3.5", truncation_side="left"),
    optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=5e-7, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)),
    scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=5e-7)),
    method=PPOConfig(
        name="PPOConfig",
        num_rollouts=4,
        chunk_size=2,
        ppo_epochs=4,
        init_kl_coef=0.05,
        target=6,
        horizon=10000,
        gamma=1,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=1,
        scale_reward="running",
        ref_mean=None,
        ref_std=None,
        cliprange_reward=10,
        gen_kwargs=dict(
            max_new_tokens=128,
            top_k=0,
            top_p=1.0,
            do_sample=True,
        ),
    ),
)


if __name__ == "__main__":
    config = default_config

    dataset = load_dataset("ThWu/rlhf_cleaned_prompt", split="train[:40]")
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    dataset = dataset.map(from_list_to_openchat)
    prompts = [{"prompt": x["prompt"]} for x in dataset["train"]]
    eval_prompts = [{"prompt": x["prompt"]} for x in islice(dataset["test"], 280)]

    def fake_reward_fn(samples, **kwargs):
        return torch.zeros(len(samples))

    trlx.train(
        prompts=prompts,
        eval_prompts=eval_prompts,
        reward_fn=fake_reward_fn,
        config=config,
        stop_sequences=["GPT4 Correct User:", "GPT4 Correct Assistant:"],
    )
