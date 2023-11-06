def from_openchat_to_llama(str_sample):
    str_sample.strip()
    if str_sample.startswith("<s>"):
        str_sample = str_sample[len("<s>") :].strip()
    assert str_sample.startswith("GPT4 Correct User: ")
    str = str_sample.replace("<|end_of_turn|>GPT4 Correct User: ", "</s> [INST] ")
    str = str.replace("<|end_of_turn|>GPT4 Correct Assistant: ", " [/INST] ")
    str = "[INST]" + str[len("GPT4 Correct User:") :]
    str = str.replace("<|end_of_turn|>", "</s>")
    return str


def from_list_to_openchat(sample):
    str = ""
    for i, content in enumerate(sample["conversations"]):
        if i % 2 == 0:
            str += "GPT4 Correct User: " + content + "<|end_of_turn|>"
        else:
            str += "GPT4 Correct Assistant: " + content + "<|end_of_turn|>"
    str += "GPT4 Correct Assistant:"
    return {"prompt": str}
