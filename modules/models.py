import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from transformers import BertForSequenceClassification


class SiameseBERT(nn.Module):
    def __init__(self, bert_type: str, device: torch.device):
        super(SiameseBERT, self).__init__()
        
        self.name = 'siam_bert'
        self.encoder = BertModel.from_pretrained(bert_type)
        self.tokenizer = BertTokenizer.from_pretrained(bert_type)
        self.metric = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.device = device

    def forward(self, inputs):
        encoded1 = self.tokenizer(inputs[0], padding=True, truncation=True, return_tensors="pt")
        encoded2 = self.tokenizer(inputs[1], padding=True, truncation=True, return_tensors="pt")
        
        encoded1 = encoded1.to(self.device)
        encoded2 = encoded2.to(self.device)

        outputs1 = self.encoder(encoded1['input_ids'], encoded1['token_type_ids'], encoded1['attention_mask'])
        outputs2 = self.encoder(encoded2['input_ids'], encoded2['token_type_ids'], encoded2['attention_mask'])
        
        return self.metric(outputs1[0][:, 0, :], outputs2[0][:, 0, :])
  
  
from transformers import BertForSequenceClassification


class SiameseBERT2(BertForSequenceClassification):
    def __init__(self, *args, **kwargs):
        super(SiameseBERT2, self).__init__(*args, **kwargs)
        self.metric = nn.CosineSimilarity(dim=1, eps=1e-6)
        
    def forward(self, **kwargs):
        outputs = self.bert(**kwargs)
        return outputs[0][:, 0, :]  # (loss), logits, (hidden_states), (attentions)