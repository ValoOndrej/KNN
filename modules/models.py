from turtle import forward

from cv2 import batchDistance
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
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
    def __init__(self, hidden_size: int, pretrained_embeddings: EmbeddedVocab,
                 embedding_dim: int, num_layers: int, n_token: int, device,
                 train_embeddings: bool = True, use_pretrained:bool = False,
                 dropouth: float=0.5):
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
        self.encoder1 = self.encoder1.float()
        self.metric = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.direction = 2
        self.device = device


    def initHiddenCell(self, batch_size):
        rand_hidden = Variable(torch.randn(self.direction * self.num_layers, batch_size, self.hidden_size))
        rand_cell = Variable(torch.randn(self.direction * self.num_layers, batch_size, self.hidden_size))
        return rand_hidden.to(self.device), rand_cell.to(self.device)


    def forward(self, inputs):
    
        h1, c1 = self.initHiddenCell(inputs[0].shape[0])
        h2, c2 = self.initHiddenCell(inputs[1].shape[0])

        embedded1 = self.embedding(inputs[0])
        embedded2 = self.embedding(inputs[1])

        outputs1, _ = self.encoder1(embedded1, (h1, c1))
        outputs2, _ = self.encoder1(embedded2, (h2, c2))

        return self.metric(outputs1[:, -1, :], outputs2[:, -1, :])

    @staticmethod
    def similarity(h1, h2):
      return torch.exp(-torch.norm(h1-h2, dim=1))


class SiameseLSTMCNN(nn.Module):
    def __init__(self, hidden_size: int, pretrained_embeddings: EmbeddedVocab,
                 embedding_dim: int, num_layers: int, n_token: int, device,
                 train_embeddings: bool = True, use_pretrained:bool = False,
                 dropouth: float=0.5):
        super(SiameseLSTMCNN, self).__init__()
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
        self.encoder1 = self.encoder1.float()
        self.metric = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.cnn = CNN(embedding_dim=embedding_dim,
                       dropouth=dropouth,
                       convs=[5])
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.direction = 2
        self.device = device


    def initHiddenCell(self, batch_size):
        rand_hidden = Variable(torch.randn(self.direction * self.num_layers, batch_size, self.hidden_size))
        rand_cell = Variable(torch.randn(self.direction * self.num_layers, batch_size, self.hidden_size))
        return rand_hidden.to(self.device), rand_cell.to(self.device)


    def forward(self, inputs):
    
        h1, c1 = self.initHiddenCell(inputs[0].shape[0])
        h2, c2 = self.initHiddenCell(inputs[1].shape[0])

        embedded1 = self.embedding(inputs[0])
        embedded2 = self.embedding(inputs[1])

        outputs1, _ = self.encoder1(embedded1, (h1, c1))
        outputs2, _ = self.encoder1(embedded2, (h2, c2))

        out1 = self.cnn(embedded1)
        out2 = self.cnn(embedded2)

        comb1 = torch.cat((outputs1[:, -1, :], out1), dim=1)
        comb2 = torch.cat((outputs2[:, -1, :], out2), dim=1)

        #return self.metric(outputs1[:, -1, :], outputs2[:, -1, :])
        return self.metric(comb1, comb2)

    @staticmethod
    def similarity(h1, h2):
      return torch.exp(-torch.norm(h1-h2, dim=1))


class CNN(nn.Module):
    
    def __init__(self,
                 embedding_dim: int, 
                 dropouth: float=0.5,
                 convs=[5,5,3,3]):

        super(CNN, self).__init__()
        self.init_range=0.1
        self.convs = nn.ModuleList([nn.Conv2d(1, 128, (K, embedding_dim)) for K in convs])
        self.dropout = nn.Dropout(dropouth)


    def forward(self, x):        
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
        
        self.init_range=0.1
        if use_pretrained:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings.embeddings,
                                                          freeze=not train_embeddings)
            self.embedding.weight = nn.Parameter(pretrained_embeddings.embeddings)
            self.embedding.weight.requires_grad = train_embeddings
        else:
            self.embedding = nn.Embedding(n_token, embedding_dim, padding_idx=0)
            self.embedding.weight.data.uniform_(-self.init_range, self.init_range)
        
        
        
        self.cnn = CNN(embedding_dim=embedding_dim,
                       dropouth=dropouth)
        self.metric = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, x):    

        x1 = self.embedding(x[0])
        x2 = self.embedding(x[1])
        out1 = self.cnn(x1)
        out2 = self.cnn(x2)
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