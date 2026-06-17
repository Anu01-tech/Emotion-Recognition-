import torch
import torch.nn as nn
from transformers import DistilBertModel

class TextEmotionModel(nn.Module):
    def __init__(self, num_classes, model_name="distilbert-base-uncased"):
        super(TextEmotionModel, self).__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Extract the pooled output (CLS token representation)
        pooled_output = outputs.last_hidden_state[:, 0]
        output = self.drop(pooled_output)
        return self.out(output)
