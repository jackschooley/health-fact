from datasets import load_dataset

dataset = load_dataset("health_fact")
train = dataset["train"]
validation = dataset["validation"]
test = dataset["test"]