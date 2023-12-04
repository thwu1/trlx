from typing import Callable, List

# Register load pipelines via module import
from trlx.pipeline import _DATAPIPELINE
from trlx.pipeline.offline_pipeline import PromptPipeline

# Register load trainers via module import
from trlx.trainer import _TRAINERS, register_trainer
from trlx.trainer.accelerate_ppo_trainer import AcceleratePPOTrainer
from trlx.trainer.accelerate_p3o_trainer import AccelerateP3OTrainer
from trlx.trainer.accelerate_rft_trainer import AccelerateRFTTrainer
from trlx.trainer.accelerate_sft_trainer import AccelerateSFTTrainer


def get_trainer(name: str) -> Callable:
    """
    Return constructor for specified RL model trainer
    """
    name = name.lower()
    if name in _TRAINERS:
        return _TRAINERS[name]
    else:
        raise Exception("Error: Trying to access a trainer that has not been registered")


def get_pipeline(name: str) -> Callable:
    """
    Return constructor for specified pipeline
    """
    name = name.lower()
    if name in _DATAPIPELINE:
        return _DATAPIPELINE[name]
    else:
        raise Exception("Error: Trying to access a pipeline that has not been registered")
