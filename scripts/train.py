import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from random import randrange
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

from blocks import *
from utils import *

print("Loading datasets...")
train_path = "../data/quora-IR-dataset/classification/train_pairs.tsv"
test_path = "../data/quora-IR-dataset/classification/test_pairs.tsv"
val_path = "../data/quora-IR-dataset/classification/dev_pairs.tsv"


train_df = pd.read_csv(train_path,nrows=5000, sep="\t")
test_df = pd.read_csv(test_path,nrows=5000, sep="\t")
val_df = pd.read_csv(val_path,nrows=5000, sep="\t")

print("Loading model...")
num_labels = 2
model = BertForSequenceClassification(num_labels)


print("Assigning data to train,test,validate ...")
X_train = train_df[["question1","question2"]]
y_train = train_df[["is_duplicate"]]
X_test = test_df[["question1","question2"]]
y_test = test_df[["is_duplicate"]]
X_validation = val_df[["question1","question2"]]
y_validation = val_df[["is_duplicate"]]


print("Converting data to torch ...")
train = convert_to_dataset_torch(X_train, y_train)
test = convert_to_dataset_torch(X_test, y_test)
validation = convert_to_dataset_torch(X_validation, y_validation)


max_seq_length = 128


batch_size = 1

core_number = 1

# Prepare the training dictionaries
dataloaders_dict = {'train': torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=core_number),
                   'val':torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=core_number)
                   }
dataset_sizes = {'train':len(train[0]),
                'val':len(test[0])}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

train_acc = []
val_acc = []
train_loss = []
val_loss = []

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    print('starting')
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100
    best_acc = 0

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            
            sentiment_corrects = 0
            
            
            # Iterate over data.
            for inputs, sentiment in dataloaders_dict[phase]:
                
                inputs1 = inputs[0] # News statement input
                inputs2 = inputs[1] # Meta data input
                inputs3 = inputs[2] # Credit scores input
                
                inputs1 = inputs1.to(device) 
                inputs2 = inputs2.to(device) 
                inputs3 = inputs3.to(device) 

                sentiment = sentiment.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    #print(inputs)
                    outputs = model(inputs1, inputs2, inputs3,)

                    outputs = F.softmax(outputs,dim=1)
                    
                    loss = criterion(outputs, torch.max(sentiment.float(), 1)[1])
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs1.size(0)

                
                sentiment_corrects += torch.sum(torch.max(outputs, 1)[1] == torch.max(sentiment, 1)[1])

                
            epoch_loss = running_loss / dataset_sizes[phase]

            
            sentiment_acc = sentiment_corrects.double() / dataset_sizes[phase]

            print('{} total loss: {:.4f} '.format(phase,epoch_loss ))
            print('{} sentiment_acc: {:.4f}'.format(
                phase, sentiment_acc))

            # Saving training acc and loss for each epoch
            sentiment_acc1 = sentiment_acc.data
            sentiment_acc1 = sentiment_acc1.cpu()
            sentiment_acc1 = sentiment_acc1.numpy()
            train_acc.append(sentiment_acc1)
            
            #epoch_loss1 = epoch_loss.data
            #epoch_loss1 = epoch_loss1.cpu()
            #epoch_loss1 = epoch_loss1.numpy()
            train_loss.append(epoch_loss)
                
            if phase == 'val' and sentiment_acc > best_acc:
                print('Saving with accuracy of {}'.format(sentiment_acc),
                      'improved over previous {}'.format(best_acc))
                best_acc = sentiment_acc
                
                # Saving val acc and loss for each epoch
                sentiment_acc1 = sentiment_acc.data
                sentiment_acc1 = sentiment_acc1.cpu()
                sentiment_acc1 = sentiment_acc1.numpy()
                val_acc.append(sentiment_acc1)
            
                #epoch_loss1 = epoch_loss.data
                #epoch_loss1 = epoch_loss1.cpu()
                #epoch_loss1 = epoch_loss1.numpy()
                val_loss.append(epoch_loss)
                
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'bert_model_test_noFC1_triBERT.pth')

        print('Time taken for epoch'+ str(epoch+1)+ ' is ' + str((time.time() - epoch_start)/60) + ' minutes')
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(float(best_acc)))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_acc, val_acc, train_loss, val_loss

model.to(device)


lrlast = .0005
lrmain = .00001
optim1 = optim.Adam(
    [
        {"params":model.bert.parameters(),"lr": lrmain},
        {"params":model.classifier.parameters(), "lr": lrlast},
       
   ])

#optim1 = optim.Adam(model.parameters(), lr=0.001)#,momentum=.9)
# Observe that all parameters are being optimized
optimizer_ft = optim1
criterion = nn.CrossEntropyLoss()

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)


model_ft1, train_acc, val_acc, train_loss, val_loss = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=20)

# Accuracy plots

'''print(val_acc)
print(val_loss)
#plt.plot(train_acc)
plt.plot(val_acc)
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['val'], loc='upper left')
#plt.show()
plt.savefig('accuracy.png')
plt.close()
print('Saved Accuracy plot')
# Loss plots
#plt.plot(train_loss)
plt.plot(val_loss)
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['val'], loc='upper right')
#plt.show()
plt.savefig('loss.png')
plt.close()
print('Saved Loss plot')'''
