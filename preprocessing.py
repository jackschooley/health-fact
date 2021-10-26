from datasets import load_dataset

dataset = load_dataset("health_fact")
train = dataset["train"]
validation = dataset["validation"]
test = dataset["test"]

def tokenize(tokenizer, data):
    output = tokenizer(data["claim"], 
                       data["explanation"], 
                       padding = "max_length",
                       truncation = True,
                       return_tensors = "pt",
                       return_attention_mask = True)
    input_ids = output["input_ids"]
    attention_mask = output["attention_mask"]
    return input_ids, attention_mask