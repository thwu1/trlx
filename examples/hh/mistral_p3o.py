import json
import math
import os
import sys
from itertools import islice

# import numpy as np
import torch

# import tritonclient.grpc as client_util
from datasets import load_dataset
from huggingface_hub import snapshot_download
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType

# from tritonclient.utils import np_to_triton_dtype
from utils import from_openchat_to_llama, from_list_to_openchat, fake_reward, create_reward_fn

import trlx
from trlx.data.default_configs import (
    ModelConfig,
    OptimizerConfig,
    P3OConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

default_config = TRLConfig(
    train=TrainConfig(
        seq_length=2048,
        epochs=10000,
        total_steps=30000,
        batch_size=2,
        eval_batch_size=4,
        checkpoint_interval=500,
        eval_interval=500,
        pipeline="PromptPipeline",
        save_best=True,
        save_optimizer=False,
        trainer="AccelerateP3OTrainer",
        checkpoint_dir="checkpoints/p3o_hh",
    ),
    model=ModelConfig(
        model_path="openchat/openchat_3.5",
        num_layers_unfrozen=2,
        # peft_config=LoraConfig(
        #     r=8,
        #     task_type=TaskType.CAUSAL_LM,
        #     lora_alpha=32,
        #     lora_dropout=0.1,
        # ),
    ),
    tokenizer=TokenizerConfig(tokenizer_path="openchat/openchat_3.5", truncation_side="left"),
    optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=4e-8, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)),
    scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=4e-8)),
    method=P3OConfig(
        name="P3OConfig",
        num_responses_per_query=2,
        num_rollouts=32,
        chunk_size=4,
        p3o_epochs=2,
        kl_coef=0.01,
        cliprange=0.2,
        cliprange_ratio=10.0,
        scale_reward="running",
        ref_mean=None,
        ref_std=None,
        cliprange_reward=10,
        gen_kwargs=dict(
            max_new_tokens=1024,
            top_k=0,
            top_p=1.0,
            do_sample=True,
        ),
        clip_tokenwise=False,
        avg_tokenwise=False,
        scale_q=False,
    ),
)
"""
old_version: 
clip_tokenwise=False,
avg_tokenwise=False,

new_version:
clip_tokenwise=True,
avg_tokenwise=True/False, doesn't matter

old_version(normalize loss tokenwise):
clip_tokenwise=False,
avg_tokenwise=True,
"""

# accelerate launch --num_processes 1 --config_file ../../configs/accelerate/zero2-bf16.yaml mistral_p3o.py
# TODO: test the reward model (done)
# implement reward template, need to substitute the special tokens with the reward template when evaluate (done)
# dataset template (done)
# implement the policy template, figure out padding
# review the mistral model structure and figure out how to add value head
# implement the mistral conpatiable modeling
# check the generation
# for openchat-3.5, pad token should be eos token = <|end_of_turn|>


# def prepare_tensor(name: str, input):
#     t = client_util.InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
#     t.set_data_from_numpy(input)
#     return t
test_run = False

def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)
    dataset = load_dataset("ThWu/cleaned_prompt_r", split="train")
    dataset = dataset.train_test_split(test_size=0.001, seed=42)
    dataset = dataset.map(from_list_to_openchat)

    prompts = [{"prompt": x["prompt"]} for x in dataset["train"]]
    eval_prompts = [{"prompt": x["prompt"]} for x in islice(dataset["test"], 100)]
    reward_fn = fake_reward if test_run else create_reward_fn()

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
