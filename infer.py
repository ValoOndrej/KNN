import os
from pathlib import Path
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from time import time
from datetime import date
import argparse
import matplotlib.pyplot as plt
from transformers import BertForSequenceClassification, Trainer, TrainingArguments, BertTokenizer, BertConfig

from modules.data import ImportData
from modules.models import SiameseBERT2
from modules.utils import collate_fn_bert, setup_logger, compute_metrics, compute_metrics_siamBERT, get_quora_huggingface
from modules.train import CustomTrainer

import transformers
transformers.logging.set_verbosity_info()

path = Path('./logs/data/')
if not (path/'dataset.csv').exists():
    get_quora_huggingface(path)

today = str(date.today())
path = Path(f'./logs/train_job_{today}/')
data_path = Path('./logs/data')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="Model path")  
    parser.add_argument("-log", "--logdir", type=str, help="Directory to save all downloaded files, and model checkpoints.", default=path)  
    parser.add_argument("-df", "--data_file", type=str, help="Path to dataset.", default=data_path/"dataset.csv")
    parser.add_argument("-s", "--split_seed", type=int, help="Seed for splitting the dataset.", default=44)
    parser.add_argument("-b", "--batch_size", type=int, help="Batch Size.", default=4)
    parser.add_argument("-epo", "--n_epoch", type=int, help="Number of epochs.", default=4)
    parser.add_argument("-sot", "--size_of_train", type=int, help="Number of train data.", default=5000)
    parser.add_argument("-ai", "--aug_intensity", type=int, help="Number of train data.", default=9)
    parser.add_argument("-a", "--augmentation", action='store_true', help="Augment data")
    parser.add_argument("-bert_cls", "--bert_cls", type=str, help="Type of BERT trained (classificator, siamese).", default='siamese')
    parser.add_argument("-bert_backbone", "--bert_backbone", type=str, help="Either path to the model, or name of the BERT model that should be used, compatible with HuggingFace Transformers.", default='bert-base-uncased')

    args = parser.parse_args()
    args.logdir = args.logdir/(args.bert_cls+'_'+'bert')

    model_path = args.logdir/'best_model/'
    if not args.logdir.exists():
        os.makedirs(args.logdir)

    logger = setup_logger(str(args.logdir/'logs.log'))
    logger.info("Begining job. All files and logs will be saved at: {}".format(args.logdir))

    config = BertConfig.from_json_file(f"{args.model}/config.json")


    device=torch.device('cuda' if torch.cuda.is_available() else  'cpu')
    model = SiameseBERT2.from_pretrained(args.model) if args.bert_cls=='siamese' else BertForSequenceClassification.from_pretrained(args.model)
    #model = torch.nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    

    training_args = TrainingArguments(
        output_dir=str(args.logdir/'results'),          # output directory
        do_train=False,
        do_eval=True,
        dataloader_num_workers = 4,
        evaluation_strategy="steps",
        logging_first_step = True,
        num_train_epochs=4,              # total # of training epochs
        per_device_train_batch_size=args.batch_size,  # batch size per device during training
        per_device_eval_batch_size=256,   # batch size for evaluation
        logging_dir=str(args.logdir/'logs'),            # directory for storing logs
    )

    test_data = ImportData.get_test_data(f'logs/data/test_dataset.csv')


    trainer_class = CustomTrainer if args.bert_cls == 'siamese' else Trainer
    trainer_args = {'model':model, 'args':training_args,
                    'eval_dataset': test_data.values,
                    'data_collator':lambda x: collate_fn_bert(x, tokenizer, args.bert_cls),
                    'compute_metrics':compute_metrics_siamBERT if args.bert_cls == 'siamese' else compute_metrics}
    if args.bert_cls == 'siamese':
        trainer_args['logger'] = logger
    trainer = trainer_class(**trainer_args)

    pred = trainer.predict(test_data.values)
    print(pred)