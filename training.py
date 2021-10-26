import datasets
import torch
from transformers import DistilBertModel, DistilBertTokenizer

from model import Classifier
from preprocessing import tokenize

def train(model, optimizer, input_ids, attention_mask, labels, epochs, batch_size,
          use_gpu = True):
    
    if use_gpu:
        model = model.cuda()
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        labels = labels.cuda()
    
    model.train()
    for epoch in range(epochs):
        print("Epoch", epoch)
        batch_start = 0
        
        while batch_start <= input_ids.size(0):
            batch_end = batch_start + batch_size
            batch_inputs = input_ids[batch_start:batch_end, :]
            batch_attentions = attention_mask[batch_start:batch_end, :]
            batch_labels = labels[batch_start:batch_end]
            
            model_output = model(batch_inputs, batch_attentions, batch_labels)
            loss = model_output.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_start = batch_end
            if batch_start % 100 == 0:
                print("Batch starting with observation", batch_start, "loss is", loss.item())

if __name__ == "__main__":
    dataset = datasets.load_dataset("health_fact")
    train_data = dataset["train"]
    
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    db_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    
    input_ids, attention_mask = tokenize(tokenizer, train_data)
    labels = torch.tensor(train_data["label"])
    
    learning_rate = 0.0001
    batch_size = 2
    epochs = 1
    
    model = Classifier(db_model)
    optimizer = torch.optim.SGD(model.parameters(), learning_rate)

    train(model, optimizer, input_ids, attention_mask, labels, epochs, batch_size)
    torch.save(model, "model.pth")
    torch.save(model.state_dict(), "model_weights.pth")