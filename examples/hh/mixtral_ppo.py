import json
import math
import os
import sys
from itertools import islice

import torch
import tritonclient.grpc as client_util
from datasets import load_dataset
from huggingface_hub import snapshot_download
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tritonclient.utils import np_to_triton_dtype
from utils import fake_reward, create_reward_fn, from_list_to_mixtral
from peft import LoraConfig, TaskType

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
        seq_length=2048,
        epochs=10000,
        total_steps=20000,
        batch_size=1,
        eval_batch_size=4,
        checkpoint_interval=500,
        eval_interval=500,
        save_best=True,
        save_optimizer=False,
        pipeline="PromptPipeline",
        trainer="AcceleratePPOTrainer",
        checkpoint_dir="/scratch/banghua/trlx_mixtral_checkpoints",
    ),
    model=ModelConfig(
        model_path="mistralai/Mixtral-8x7B-Instruct-v0.1",
        num_layers_unfrozen=1,
        # peft_config=LoraConfig(
        #     r=8,
        #     task_type=TaskType.CAUSAL_LM,
        #     lora_alpha=32,
        #     lora_dropout=0.1,
        # ),
    ),
    tokenizer=TokenizerConfig(tokenizer_path="mistralai/Mixtral-8x7B-Instruct-v0.1", truncation_side="left"),
    optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=2e-7, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)),
    scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=2e-7)),
    method=PPOConfig(
        name="PPOConfig",
        num_rollouts=64,
        chunk_size=4,
        ppo_epochs=2,
        init_kl_coef=0.01,
        target=None,
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
            max_new_tokens=1024,
            top_k=0,
            top_p=1.0,
            do_sample=True,
        ),
    ),
)

# accelerate launch --num_processes 2 --config_file ../../configs/accelerate/zero2-bf16.yaml mistral_ppo.py
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
test_run = True

def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)
    dataset = load_dataset("ThWu/rlhf_cleaned_prompt", split="train")
    dataset = dataset.train_test_split(test_size=0.001, seed=42)
    dataset = dataset.map(from_list_to_mixtral, num_proc=4)

    prompts = [{"prompt": x["prompt"]} for x in dataset["train"]]
    eval_prompts = [{"prompt": x["prompt"]} for x in islice(dataset["test"], 100)]
    reward_fn = fake_reward if test_run else create_reward_fn()

    trlx.train(
        prompts=prompts,
        eval_prompts=eval_prompts,
        reward_fn=reward_fn,
        config=config,
        stop_sequences=["[INST]", "[/INST]"],
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)
