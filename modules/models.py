from turtle import forward
import torch
import torch.nn as nn
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
  def __init__(self, hidden_size: int, pretrained_embeddings: EmbeddedVocab, embedding_dim: int, num_layers: int, n_token: int, train_embeddings: bool = True, use_pretrained:bool = False,  dropouth: float=0.3):
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
      
      self.encoder = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropouth)
      self.encoder = self.encoder.float()
      self.metric = nn.CosineSimilarity(dim=1, eps=1e-6)
      
  def forward(self, inputs):
    
    embedded1 = self.embedding(inputs[0])
    embedded2 = self.embedding(inputs[1])
    
    outputs1, hiddens1 = self.encoder(embedded1)
    outputs2, hiddens2 = self.encoder(embedded2)

    return self.metric(outputs1[:, -1, :], outputs2[:, -1, :])

  @staticmethod
  def similarity(h1, h2):
    return torch.exp(-torch.norm(h1-h2, dim=1))


class CNN(nn.Module):
    
    def __init__(self, input_shape, n_classes, pretrained_embeddings,
                 n_token, embedding_dim, train_embeddings,
                 use_pretrained) -> None:
        super(CNN, self).__init__()
        self.init_range=0.1
        if use_pretrained:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings.embeddings, freeze=not train_embeddings)
            self.embedding.weight = nn.Parameter(pretrained_embeddings.embeddings)
            self.embedding.weight.requires_grad = train_embeddings
        else:
            self.embedding = nn.Embedding(n_token, embedding_dim, padding_idx=0)
            self.embedding.weight.data.uniform_(-self.init_range, self.init_range)
        self.conv1d = nn.Conv1d(input_shape, 128, kernel_size=5)
        self.act = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=128, out_features=20)
        self.linear2 = nn.Linear(in_features=20, out_features=n_classes)


    def forward(self, x):        
        x = self.embedding(x)
        out = self.pool(self.act(self.conv1d(x)))
        out = self.flatten(out)
        out = self.act(self.linear1(out))
        return self.linear2(out)


class SiameseCNN(nn.Module):
    def __init__(self, input_shape, n_classes, pretrained_embeddings,
                 n_token, embedding_dim, train_embeddings=True, 
                 use_pretrained=False) -> None:
        super(SiameseCNN, self).__init__()
        self.cnn = CNN(input_shape=input_shape,
                       n_classes=n_classes,
                       pretrained_embeddings=pretrained_embeddings,
                       n_token=n_token,
                       embedding_dim=embedding_dim,
                       train_embeddings=train_embeddings,
                       use_pretrained=use_pretrained)
        self.metric = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, x):    
        out1 = self.cnn(x[0])
        out2 = self.cnn(x[1])
        self.metric(out1, out2)


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