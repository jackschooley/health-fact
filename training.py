import torch
from transformers import DistilBertModel, DistilBertTokenizer

from model import Classifier
from preprocessing import preprocess

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

    torch.save(model, "model.pth")
    torch.save(model.state_dict(), "model_weights.pth")
    
if __name__ == "__main__":
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    input_ids, attention_mask, labels = preprocess(tokenizer)
    
    learning_rate = 0.0001
    batch_size = 2
    epochs = 1
    
    db_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    model = Classifier(db_model)
    optimizer = torch.optim.SGD(model.parameters(), learning_rate)

    train(model, optimizer, input_ids, attention_mask, labels, epochs, batch_size)