import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
import sys
import logging
import logging.handlers
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split


def setup_logger(path) -> logging.Logger:
    logger = logging.getLogger(__name__)
    setattr(logger, 'out_path', '/'.join(path.split('/')[:-1]))
    logger.setLevel(logging.INFO)
    if len(logger.handlers) == 0:
        formatter = logging.Formatter('%(asctime)s | %(message)s')
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        fh = logging.handlers.WatchedFileHandler(path)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def collate_fn_bert(batch, tokenizer, bert_type):
    batch = np.array(batch)
    sent1 = batch[:, 0].tolist()
    sent2 = batch[:, 1].tolist()
    target = torch.tensor(batch[:, 2].tolist())
    if bert_type == 'classifier':
        out = tokenizer(sent1, sent2, padding=True, truncation=True, return_tensors="pt")
        out['labels'] = target
    elif bert_type == 'siamese':
        out1 = tokenizer(sent1, padding=True, truncation=True, return_tensors="pt")
        out2 = tokenizer(sent2, padding=True, truncation=True, return_tensors="pt")
        out = {'sent1':out1, 'sent2':out2, 'labels':target}
    else:
        raise ValueError("Incorrect bert type: should be 'siamese' or 'classifier'")
    return out


def collate_fn_lstm(batch):
	padding_list = [torch.LongTensor(item[0]) for item in batch]
	batch_size = len(padding_list)
	[padding_list.append(torch.LongTensor(item[1])) for item in batch]
	data = pad_sequence(padding_list, padding_value=0).T
	sentences1 = data[:batch_size]
	sentences2 = data[batch_size:]
	labels = [item[2] for item in batch]
	return [torch.stack([sentences1, sentences2]), torch.tensor(labels)]


def get_quora_huggingface(export_path: Path) -> None:
    if not export_path.exists():
      os.makedirs(export_path)
    dataset = load_dataset('quora')
    dataset['train'].set_format('pandas')
    dataset_pd = dataset['train'][:]
    dataset_pd['question1'] = dataset_pd.questions.apply(lambda x: x['text'][0])
    dataset_pd['question2'] = dataset_pd.questions.apply(lambda x: x['text'][1])
    dataset_pd.is_duplicate = dataset_pd.is_duplicate.astype('int8')
    dataset_pd[['question1', 'question2', 'is_duplicate']].to_csv(str(export_path/'dataset.csv'))


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def compute_metrics_siamBERT(pred):
    labels = pred.label_ids
    preds = pred.predictions>0.5
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def split_to_train_test(path):
    
    get_quora_huggingface(Path(path))

    df = pd.read_csv(f'{path}/dataset.csv')
    train, test = train_test_split(df, test_size=0.25, train_size=0.75, shuffle=True)
    
    train.to_csv(f'{path}/dataset.csv')
    test.to_csv(f'{path}/test_dataset.csv')


def train(model, optimizer, criterion, train_dataloader, device, epoch_loss, preds_train, labels_train, gradient_clipping_norm, epoch, logger):
    model.train()   
    for epoch_iteration, (batch) in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()
        
        X_batch = batch[0].to(device)
        y_batch = batch[1].float().to(device)
        y_hat = model(X_batch)
        preds_train.append(y_hat.detach().cpu().numpy())#((y_hat>=0.5).float()==y_batch).sum().item()
        labels_train.append(y_batch.detach().cpu().numpy())
        loss = criterion(y_hat, y_batch)
        loss.backward()

        epoch_loss.append(loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping_norm)

        optimizer.step()

        if epoch_iteration % 1000 == 0.:
          logger.info('Mean loss till {}th iteration of epoch {}: {}'.format(epoch_iteration, epoch, np.mean(epoch_loss)))

def eval(model, criterion, test_dataloader, device,  eval_loss, preds_test, labels_test):
    model.eval()
    for batch in test_dataloader:
        with torch.no_grad():
            X_batch = batch[0].to(device)
            y_batch = batch[1].float().to(device)
            y_hat = model(X_batch)
            loss = criterion(y_hat, y_batch)
            eval_loss.append(loss.item())
            preds_test.append(y_hat.detach().cpu().numpy())#((y_hat>=0.5).float()==y_batch).sum().item()
            labels_test.append(y_batch.detach().cpu().numpy())


def compute_metrics_siamLSTM(pred, y_true):
    labels = y_true
    preds = pred>0.5
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


if __name__ == '__main__' :
    split_to_train_test('../logs/data')
