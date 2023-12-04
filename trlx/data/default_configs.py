from pathlib import Path

from trlx.models.modeling_p3o import P3OConfig
from trlx.models.modeling_ppo import PPOConfig
from trlx.trainer.accelerate_sft_trainer import SFTConfig

from .configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)


def default_p3o_config():
    return TRLConfig(
        train=TrainConfig(
            seq_length=1024,
            epochs=10000,
            total_steps=10000,
            batch_size=4,
            eval_batch_size=32,
            checkpoint_interval=10000,
            eval_interval=500,
            pipeline="PromptPipeline",
            trainer="AccelerateP3OTrainer",
            checkpoint_dir="checkpoints/p3o_hh",
        ),
        model=ModelConfig(model_path="EleutherAI/gpt-j-6B", num_layers_unfrozen=2),
        tokenizer=TokenizerConfig(tokenizer_path="EleutherAI/gpt-j-6B", truncation_side="left"),
        optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=8e-6, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)),
        scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=8e-6)),
        method=P3OConfig(
            name="P3OConfig",
            num_rollouts=64,
            chunk_size=16,
            p3o_epochs=4,
            kl_coef=0.05,
            cliprange=0.2,
            cliprange_ratio=10.0,
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


def default_ppo_config():
    return TRLConfig(
        train=TrainConfig(
            seq_length=1024,
            epochs=100,
            total_steps=10000,
            batch_size=32,
            eval_batch_size=32,
            checkpoint_interval=10000,
            eval_interval=100,
            pipeline="PromptPipeline",
            trainer="AcceleratePPOTrainer",
        ),
        model=ModelConfig(model_path="lvwerra/gpt2-imdb", num_layers_unfrozen=2),
        tokenizer=TokenizerConfig(tokenizer_path="gpt2", truncation_side="right"),
        optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=3e-5, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)),
        scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=1e12, eta_min=3e-5)),
        method=PPOConfig(
            name="PPOConfig",
            num_rollouts=128,
            chunk_size=128,
            ppo_epochs=4,
            init_kl_coef=0.001,
            target=None,
            horizon=10000,
            gamma=1,
            lam=0.95,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=1,
            scale_reward="ignored",
            ref_mean=None,
            ref_std=None,
            cliprange_reward=10,
            gen_kwargs=dict(
                max_new_tokens=40,
                top_k=0,
                top_p=1.0,
                do_sample=True,
            ),
        ),
    )


def default_sft_config():
    return TRLConfig(
        train=TrainConfig(
            seq_length=1024,
            epochs=100,
            total_steps=1000,
            batch_size=8,
            checkpoint_interval=10000,
            eval_interval=100,
            pipeline="PromptPipeline",
            trainer="AccelerateSFTTrainer",
        ),
        model=ModelConfig(model_path="gpt2", num_layers_unfrozen=-1),
        tokenizer=TokenizerConfig(tokenizer_path="gpt2", truncation_side="right"),
        optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=1.0e-4, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)),
        scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=1e12, eta_min=1.0e-4)),  # train.total_steps
        method=SFTConfig(
            name="sftconfig",
            gen_kwargs=dict(max_new_tokens=40, top_k=0, top_p=1.0, do_sample=True),
        ),
    )
