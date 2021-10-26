from torch import nn
from transformers import DistilBertModel

class ModelOutputs:
    
    def __init__(self, logits, loss = None):
        self.logits = logits
        self.loss = loss

class Classifier(nn.Module):
     
    def __init__(self, db_config):
        super(Classifier, self).__init__()
        db_model = DistilBertModel.from_pretrained("distilbert-base-uncased",
                                                   config = db_config)
        self.distilbert = db_model
        self.linear = nn.Linear(db_config.dim, 4)
        
    def forward(self, input_ids, attention_mask, labels = None):
        db_outputs = self.distilbert(input_ids, attention_mask)
        hidden_states = db_outputs.last_hidden_state
        cls_hidden_states = hidden_states[:, 0, :]
        logits = self.linear(cls_hidden_states)
        
        loss = None
        if labels:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return ModelOutputs(logits, loss)