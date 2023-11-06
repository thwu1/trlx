import json
import sys

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import from_openchat_to_llama, from_list_to_openchat


def main(hparams={}):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    print(tokenizer)
    dataset = load_dataset("ThWu/rlhf_cleaned_prompt", split="train[:200]")
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    dataset = dataset.map(from_list_to_openchat)
    prompts = [{"prompt": x["prompt"]} for x in dataset["train"]]

    if isinstance(prompts[0], dict):
        metadata = prompts
        prompts = [x.pop("prompt") for x in metadata]
    else:
        metadata = [{}] * len(prompts)
    model_inputs = tokenizer(prompts, truncation=True, padding=False, max_length=4096, add_special_tokens=False)


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)
