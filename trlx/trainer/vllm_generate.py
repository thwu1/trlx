import argparse
from vllm import LLM, SamplingParams
import json
import torch
import gc
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

parser = argparse.ArgumentParser()

parser.add_argument("--prompt_path", type=str, help="Path to a file containing prompts, one per line")
parser.add_argument("--model_path", type=str, help="Path to the model")

args = parser.parse_args()

prompts_dict = json.load(open(args.prompt_path))

def serialize(all_prompts_dict):
    """all_prompts_dict: [{local_process_idx: batch},...]
    batch: {input_ids: torch.LongTensor, attention_mask: torch.LongTensor}
    """
    def _serialize(batch):
        batch = {k: v.tolist() for k, v in batch.items()}
        return batch
    for i, batch in enumerate(all_prompts_dict):
        for k, v in batch.items():
            all_prompts_dict[i][k] = _serialize(v)
    return all_prompts_dict

# prompts_dict = [{0: {'input_ids': torch.tensor([[2,2,3],[2,5,6]]), 'attention_mask': torch.tensor([[1,1,1],[1,1,1]])}, 1: {'input_ids': torch.tensor([[1,2,3],[2,5,6]]), 'attention_mask': torch.tensor([[1,1,1],[1,1,1]])}}]
# prompts_dict = serialize(prompts_dict)

# llm = LLM(model=args.model_path, tensor_parallel_size=1)
llm = LLM(model=args.model_path, tensor_parallel_size=1)

def remove_left_padding(prompts, pad_token=2):
    lst = []
    for prompt in prompts:
        first_non_pad_index = next((i for i, x in enumerate(prompt) if x != 2), len(prompt))
        lst.append(prompt[first_non_pad_index:])
    return lst

def flatten_to_list(prompts_dict):
    prompts = []
    for prompt in prompts_dict:
        for k, ls in prompt.items():
            for v in ls:
                prompts.extend(remove_left_padding(v["input_ids"]))
    return prompts

def pad_sequences(sequences, max_length = None, padding_value=2):
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    padded_sequences = []
    for seq in sequences:
        padded_seq = seq + [padding_value] * (max_length - len(seq))
        padded_sequences.append(padded_seq[:max_length])
    
    return padded_sequences


def update_dict(output_tokens, prompts, prompts_dict):
    idx = 0
    for i, prompt in enumerate(prompts_dict):
        for k, ls in prompt.items():
            for j, v in enumerate(ls):
                assert remove_left_padding(v["input_ids"]) == prompts[idx: idx + len(v["input_ids"])], f"remove_left_padding(v['input_ids']): {remove_left_padding(v['input_ids'])}, prompts[idx: idx + len(v['input_ids'])]: {prompts[idx: idx + len(v['input_ids'])]}"
                prompts_dict[i][k][j]["output_tokens"] = pad_sequences(output_tokens[idx: idx + len(v["input_ids"])])
                idx += len(v["input_ids"])
    return prompts_dict

prompts = flatten_to_list(prompts_dict)
# print(prompts)

outputs = llm.generate(prompt_token_ids=prompts, sampling_params=SamplingParams(temperature=1.0, top_p=0.99, max_tokens=1024))

generated_text = []
output_tokens = []

assert len(prompts) == len(outputs)
for prompt, output in zip(prompts, outputs):
    assert prompt == output.prompt_token_ids, f"prompt: {prompt}, output.prompt_token_ids: {output.prompt_token_ids}"
    output_tokens.append(output.outputs[0].token_ids)

prompts_dict = update_dict(output_tokens, prompts, prompts_dict)

with open(args.model_path + '/output.json', 'w') as f:
    json.dump(prompts_dict, f)
