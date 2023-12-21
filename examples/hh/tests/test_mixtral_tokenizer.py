from transformers import AutoTokenizer
from utils import from_list_to_mixtral
from datasets import load_dataset


dataset = load_dataset("ThWu/rlhf_cleaned_prompt", split="train")

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")

def tokenize(texts):
    tensor = []
    tensor.append(tokenizer.bos_token_id)
    for i, t in enumerate(texts):
        if i % 2 == 0:
            tensor += tokenizer.encode("[INST]", add_special_tokens=False) + tokenizer.encode(t, add_special_tokens=False)
        else:
            tensor += tokenizer.encode("[/INST]", add_special_tokens=False) + tokenizer.encode(t, add_special_tokens=False) + [tokenizer.eos_token_id]
    tensor += tokenizer.encode("[/INST]", add_special_tokens=False)
    return tensor

for i in range(10000):
    texts = dataset[i]["conversations"]

    prompt = from_list_to_mixtral({"conversations": texts})["prompt"]
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

    assert tokenize(texts) == input_ids[0].tolist()
    # print(prompt)
    # print(input_ids[0].tolist())
    # print(tokenizer.decode(tokenize(texts)))
    # print(tokenize(texts))
    # print(tokenizer.eos_token)