from datasets import load_dataset
from transformers import DistilBertTokenizer

dataset = load_dataset("health_fact")
train = dataset["train"]
validation = dataset["validation"]
test = dataset["test"]

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(data):
    output = tokenizer(data["claim"], 
                       data["explanation"], 
                       padding = True,
                       return_tensor = "pt",
                       return_attention_mask = True)
    input_ids = output["input_ids"]
    attention_mask = output["attention_mask"]
    return input_ids, attention_mask