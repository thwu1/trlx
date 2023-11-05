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

default_config = TRLConfig(
    train=TrainConfig(
        seq_length=1024,
        epochs=10000,
        total_steps=10000,
        batch_size=4,
        eval_batch_size=32,
        checkpoint_interval=10000,
        eval_interval=500,
        pipeline="PromptPipeline",
        trainer="AcceleratePPOTrainer",
        checkpoint_dir="checkpoints/ppo_hh",
    ),
    model=ModelConfig(model_path="openchat/openchat_3.5", num_layers_unfrozen=2),
    tokenizer=TokenizerConfig(tokenizer_path="openchat/openchat_3.5", truncation_side="left"),
    optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=8e-6, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)),
    scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=8e-6)),
    method=PPOConfig(
        name="PPOConfig",
        num_rollouts=64,
        chunk_size=16,
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
            max_new_tokens=4096,
            top_k=0,
            top_p=1.0,
            do_sample=True,
        ),
    ),
)

# TODO: test the reward model
# implement reward template, need to substitute the special tokens with the reward template when evaluate
# the reward model should get sample
# dataset template
# implement the policy template, figure out padding


def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)

    dataset = load_dataset("ThWu/rlhf_cleaned_prompt", split="train[:200]")
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    def apply_openchat_format_from_list(sample):
        str = ""
        for i, content in enumerate(sample["conversations"]):
            if i % 2 == 0:
                str += "GPT4 Correct User: " + content + "<|end_of_turn|>"
            else:
                str += "GPT4 Correct Assistant: " + content + "<|end_of_turn|>"
        str += "GPT4 Correct Assistant:"
        return {"prompt": str}

    dataset = dataset.map(apply_openchat_format_from_list)
    prompts = [{"prompt": x["prompt"]} for x in dataset["train"]]
    eval_prompts = [{"prompt": x["prompt"]} for x in islice(dataset["test"], 280)]

    def fake_reward_fn(samples, prompts, **kwargs):
        return torch.zeros(len(samples))

    reward_fn = fake_reward_fn

    trlx.train(
        prompts=prompts,
        eval_prompts=eval_prompts,
        reward_fn=reward_fn,
        config=config,
        stop_sequences=["GPT4 Correct User:", "GPT4 Correct Assistant:"],
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)
