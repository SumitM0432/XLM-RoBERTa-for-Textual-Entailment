import torch
import transformers
import config
from torch import nn

class roberta_model(nn.Module):
    
    def __init__(self, n_classes):
        super(roberta_model, self).__init__()
        self.roberta = transformers.XLMRobertaForSequenceClassification.from_pretrained(config.ROBERTA_PATH, num_labels = 3)
        self.drop = nn.Dropout(p = 0.3)
        self.out = nn.Linear(self.roberta.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask, labels):
        _, pooled_output = self.roberta(
        input_ids = input_ids,
        attention_mask = attention_mask,
        labels = labels
        )
        output = self.drop(pooled_output)
        return self.out(output)
