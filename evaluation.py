import numpy
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from transformers import DistilBertTokenizer

from preprocessing import preprocess

def evaluate(model, input_ids, attention_mask, batch_size, use_gpu = True):
    
    if use_gpu:
        model = model.cuda()
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
    
    model.eval()
    batch_start = 0
    predicted_labels = None
    while batch_start < input_ids.size(0):
        batch_end = batch_start + batch_size
        batch_inputs = input_ids[batch_start:batch_end, :]
        batch_attentions = attention_mask[batch_start:batch_end, :]
        
        with torch.no_grad():
            model_output = model(batch_inputs, batch_attentions)
            
        logits = model_output.logits
        predictions = torch.argmax(logits, 1)
        if use_gpu:
            predictions = predictions.cpu()
        
        if predicted_labels is not None:
            predicted_labels = numpy.append(predicted_labels, predictions.numpy())
        else:
            predicted_labels = predictions
        
        if batch_start % 100 == 0:
            print("Evaluating batch starting with observation", batch_start)
        batch_start = batch_end
    return predicted_labels

def get_metrics(labels, predicted_labels):
    accuracy = accuracy_score(labels, predicted_labels)
    confuse = confusion_matrix(labels, predicted_labels)
    return accuracy, confuse

if __name__ == "__main__":
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    input_ids, attention_mask, labels = preprocess(tokenizer, "validation")
    
    batch_size = 2
    
    model = torch.load("model.pth")
    model.load_state_dict(torch.load("model_weights.pth"))
    predicted_labels = evaluate(model, input_ids, attention_mask, batch_size)
    
    accuracy, confuse = get_metrics(labels.numpy(), predicted_labels)
    print("Exact match accuracy is", accuracy)
    print(confuse)