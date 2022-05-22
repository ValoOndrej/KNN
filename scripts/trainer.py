import torch.nn as nn
import torch
import sys
import numpy as numpy
import torch.nn.functional as F

from blocks import *

class SiameseTrainer:

    def __init__(self, start_epoch=0, end_epoch=1000,
                 criterion=None, logger=None, model_name="",
                 load=False, step_size=int(50* 0.8), metric=None,
                 arch=BertForSequenceClassification, in_channels=1, out_classes=1,
                 learning_rate=0.001, dropout_chance=0.0):
        
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.criterion = criterion
        self.metric = metric
        self.logger = logger        

        self.model = arch()
        self.optimizer = torch.optim.Adam(self.model.parameters(), learning_rate)

        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sys.stdout.flush()

        if load:
            checkpoint = torch.load(f"../pretrained_weights/{model_name}.pt")
            state_dict = checkpoint['state_dict']
            unParalled_state_dict = {}
            for key in state_dict.keys():
                unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
            self.model.load_state_dict(unParalled_state_dict, strict=False)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model = nn.DataParallel(self.model, device_ids = [i for i in range(torch.cuda.device_count())])
        self.model.to(self.device)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                         step_size=step_size,
                                                         gamma=0.5)


    def fit(self, train_data=None, valid_data=None):

        for epoch in range(self.start_epoch, self.end_epoch):
            self.train(train_data, epoch)
            self.validate(valid_data, epoch)
            self.scheduler.step()
            if self.logger.epoch(epoch, self.model.state_dict(), self.optimizer.state_dict()):
                print("Early Stopping")
                break
            self.logger.update_metrics(clear=True)
            sys.stdout.flush()


    def train(self, train_loader, epoch):
        self.model.train()
        self.logger.time()
        for index, patch in enumerate(train_loader):
            self.logger.time(iteration=True)
            x, y = patch['x'].to(self.device), patch['y'].to(self.device)
            loss, metric = 0, 0
            pred=self.model(x)
            for item in pred:
                loss += self.criterion(item, y)
                metric += self.metric(item, y)
            loss /= len(pred)
            metric /= len(pred)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.logger.update_metrics(index=index, train_loss=round(loss.item(), 2),
                              train_dice=metric)
            if (index + 1) % 5 == 0:
                self.logger.iteration(epoch, index, False)
                sys.stdout.flush()


    def validate(self, valid_loader, epoch):
        
        with torch.no_grad():
            self.model.eval()
            for index, patch in enumerate(valid_loader):
                x, y = patch['x'].to(self.device), patch['y'].to(self.device)
                loss, metric = 0, 0
                pred=self.model(x)
                for item in pred:
                    loss += self.criterion(item, y)
                    metric += self.metric(item, y)
                loss /= len(pred)
                metric /= len(pred)
                self.logger.update_metrics(index=index, valid_loss=loss.item(),
                                      valid_dice=metric)
                if (index + 1) % 5 == 0:
                    self.logger.iteration(epoch, index, False)
                    sys.stdout.flush()


    def infer(self, data):
        """Performs inference on given data, calculates dice coeff. if annotations 
        are provided.
        Args:
            data (tensor): data to be infered. Shape: (BxCxHxWxD)
            metric (DiceCoefficient): 
        Returns:
            np.array: network predictions
            list: list with dice results.
        """
        results = []
        seg_results = []

        with torch.no_grad():
            self.model.eval()
            for index, patch in enumerate(data):
                x, y = patch['x'].to(self.device), patch['y'].to(self.device)
                pred=self.model(x)
                pred = pred[-1]
                self.metric.add(torch.argmax(F.softmax(pred, dim=1), dim=1).long().detach().cpu(), y.long().detach().cpu())
                results.append(F.softmax(pred, dim=1).cpu().numpy())
        return results