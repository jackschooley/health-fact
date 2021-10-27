import datasets
import numpy
import pandas
import torch
from transformers import DistilBertTokenizer

def fetch_data(data_split = "train"):
    dataset = datasets.load_dataset("health_fact")
    if data_split == "train":
        data = pandas.DataFrame(dataset["train"])
    elif data_split == "validation":
        data = pandas.DataFrame(dataset["validation"])
    else:
        data = pandas.DataFrame(dataset["test"])
    return data

def remove_bad_observations(data):
    good_indices = numpy.asarray(data["label"] != -1).nonzero()[0] #returns a tuple
    cleaned_data = data.iloc[good_indices]
    cleaned_data.reset_index(inplace = True) #needed to avoid torch compatibility issues
    
    claims = cleaned_data["claim"].tolist()
    explanations = cleaned_data["explanation"].tolist()
    labels = torch.tensor(cleaned_data["label"])
    return claims, explanations, labels

def tokenize(tokenizer, claims, explanations):
    output = tokenizer(claims, 
                       explanations, 
                       padding = "max_length",
                       truncation = "longest_first",
                       return_tensors = "pt",
                       return_attention_mask = True)
    input_ids = output["input_ids"]
    attention_mask = output["attention_mask"]
    return input_ids, attention_mask


def preprocess(tokenizer, data_split = "train"):
    data = fetch_data(data_split)
    claims, explanations, labels = remove_bad_observations(data)
    input_ids, attention_mask = tokenize(tokenizer, claims, explanations)
    return input_ids, attention_mask, labels

if __name__ == "__main__":
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    input_ids, attention_mask, labels = preprocess(tokenizer)