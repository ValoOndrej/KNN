from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from transformers import BertForSequenceClassification
from .embeddings import EmbeddedVocab


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


class SiameseLSTM(nn.Module):
  '''
  architechture follows example proposed by javiersuweijie from fast.ai forum blogpost:
  https://forums.fast.ai/t/siamese-network-architecture-using-fast-ai-library/15114/3
  '''
  def __init__(self, hidden_size: int, pretrained_embeddings: EmbeddedVocab, embedding_dim: int, num_layers: int, n_token: int, train_embeddings: bool = True, use_pretrained:bool = False,  dropouth: float=0.5):
      super(SiameseLSTM, self).__init__()
      self.name = 'siam_lstm'
      self.init_range=0.1
      if use_pretrained:
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings.embeddings, freeze=not train_embeddings)
        self.embedding.weight = nn.Parameter(pretrained_embeddings.embeddings)
        self.embedding.weight.requires_grad = train_embeddings
      else:
        self.embedding = nn.Embedding(n_token, embedding_dim, padding_idx=0)
        self.embedding.weight.data.uniform_(-self.init_range, self.init_range)
      
      self.encoder1 = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropouth, bidirectional=True)
      self.encoder2 = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropouth, bidirectional=True)

      self.encoder1 = self.encoder1.float()
      self.encoder2 = self.encoder2.float()
      self.metric = nn.CosineSimilarity(dim=1, eps=1e-6)
      
  def forward(self, inputs):
    
    embedded1 = self.embedding(inputs[0])
    embedded2 = self.embedding(inputs[1])
    
    outputs1, hiddens1 = self.encoder1(embedded1)
    outputs1, hiddens1 = self.encoder2(outputs1, hiddens1)
    
    outputs2, hiddens2 = self.encoder1(embedded2)
    outputs2, hiddens2 = self.encoder2(outputs2, hiddens2)

    return self.metric(outputs1[:, -1, :], outputs2[:, -1, :])

  @staticmethod
  def similarity(h1, h2):
    return torch.exp(-torch.norm(h1-h2, dim=1))


class CNN(nn.Module):
    
    def __init__(self, n_classes,
                 pretrained_embeddings: EmbeddedVocab,
                 embedding_dim: int, n_token: int, 
                 train_embeddings: bool = True,
                 use_pretrained:bool = False,
                 dropouth: float=0.5):

        super(CNN, self).__init__()
        self.init_range=0.1
        if use_pretrained:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings.embeddings, freeze=not train_embeddings)
            self.embedding.weight = nn.Parameter(pretrained_embeddings.embeddings)
            self.embedding.weight.requires_grad = train_embeddings
        else:
            self.embedding = nn.Embedding(n_token, embedding_dim, padding_idx=0)
            self.embedding.weight.data.uniform_(-self.init_range, self.init_range)

        self.convs = nn.ModuleList([nn.Conv2d(1, 128, (5, embedding_dim)) for _ in range(1)])
        self.dropout = nn.Dropout(dropouth)
        self.act = nn.ReLU()


    def forward(self, x):        
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        return x


class SiameseCNN(nn.Module):
    def __init__(self, n_classes,
                 pretrained_embeddings: EmbeddedVocab,
                 embedding_dim: int, n_token: int, 
                 train_embeddings: bool = True,
                 use_pretrained:bool = False,
                 dropouth: float=0.5):
        
        super(SiameseCNN, self).__init__()
        self.cnn = CNN(n_classes=n_classes,
                       pretrained_embeddings=pretrained_embeddings,
                       n_token=n_token,
                       embedding_dim=embedding_dim,
                       train_embeddings=train_embeddings,
                       use_pretrained=use_pretrained,
                       dropouth=dropouth)
        self.metric = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, x):    
        out1 = self.cnn(x[0])
        out2 = self.cnn(x[1])
        return self.metric(out1, out2)


class SiameseBERT2(BertForSequenceClassification):
    def __init__(self, *args, **kwargs):
        super(SiameseBERT2, self).__init__(*args, **kwargs)
        self.metric = nn.CosineSimilarity(dim=1, eps=1e-6)
        
    def forward(self, **kwargs):
        outputs = self.bert(**kwargs)
        return outputs[0][:, 0, :]  # (loss), logits, (hidden_states), (attentions)



if __name__ == '__main__':
    data = torch.randn([1,10,40])
    net = CNN(10,2)
    xd = net(data)
    print(xd)