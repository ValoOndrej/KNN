import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from time import time
from datetime import date
import argparse

from modules.data import ImportData, QuoraQuestionDataset
from modules.embeddings import EmbeddedVocab
from modules.models import SiameseLSTMCNN
from modules.utils import collate_fn_lstm, train, eval, setup_logger, count_parameters, compute_metrics_siamLSTM


today = str(date.today())
path = Path(f'./logs/train_job_{today}/')
emb_path = Path('./logs/')
data_path = Path('./logs/data')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", "--model_name", type=str, help="Name of trained model. Needed only for correct logs output", default='siam_lstmcnn')  
    parser.add_argument("-log", "--logdir", type=str, help="Directory to save all downloaded files, and model checkpoints.", default=path)  
    parser.add_argument("-df", "--data_file", type=str, help="Path to dataset.", default=data_path/"dataset.csv")
    parser.add_argument("-pr", "--use_pretrained", action='store_true', help="Boolean, whether use pretrained embeddings.", default=True)
    parser.add_argument("-dim", "--emb_dim", type=int, help="Dimensions of pretrained embeddings", default=100)
    parser.add_argument("-empth", "--emb_path", type=str, help="path to file with pretrained embeddings", default=emb_path)
    parser.add_argument("-s", "--split_seed", type=int, help="Seed for splitting the dataset.", default=44)
    parser.add_argument('-noprep', "--preprocessing", action='store_false', help="Preprocess dataset before training the model", default=True)
    parser.add_argument("-hid", "--n_hidden", type=int, help="Number of hidden units in LSTM layer.", default=50)
    parser.add_argument("-b", "--batch_size", type=int, help="Batch Size.", default=64)
    parser.add_argument("-epo", "--n_epoch", type=int, help="Number of epochs.", default=25)
    parser.add_argument("-nl", "--n_layer", type=int, help="Number of LSTM layers.", default=2)
    parser.add_argument("-gc", "--gradient_clipping_norm", type=float, help="Gradient clipping norm", default=1.25)
    parser.add_argument("-note", "--train_embeddings", action='store_false', help="Whether to fine-tune embedding weights during training", default=True)
    parser.add_argument("-sot", "--size_of_train", type=int, help="Number of train data.", default=0)
    parser.add_argument("-ai", "--aug_intensity", type=int, help="Number of augmentations done on train data.", default=9)
    parser.add_argument("-a", "--augmentation", action='store_true', help="Augment data")


    args = parser.parse_args()
    args.logdir = args.logdir/f'{args.model_name}_{args.emb_dim}dglove'
    model_path = args.logdir/'best_model/'
    if not args.logdir.exists():
        os.makedirs(args.logdir)

    logger = setup_logger(str(args.logdir/'logs.log'))
    logger.info("Begining job. All files and logs will be saved at: {}".format(args.logdir))

    if args.use_pretrained:
        logger.info('Building Embedding Matrix...')
        embedded_vocab_class = EmbeddedVocab(args.emb_path/'glove.6B.100d.txt', args.emb_dim, args.emb_path, logger)
    else:
        embedded_vocab_class = None

    logger.info('Reading Dataset and splitting into train and test datasets with seed: {}'.format(args.split_seed))
    data = ImportData(str(args.data_file))
    data.train_test_split(seed=args.split_seed,
                         augment=args.augmentation,
                         size_of_train=args.size_of_train,
                         num_arg=args.aug_intensity)

    logger.info('Preprocessing Train Dataset...')
    train_dataset = QuoraQuestionDataset(data.train, use_pretrained_emb=args.use_pretrained, reverse_vocab=embedded_vocab_class.reverse_vocab, preprocess = args.preprocessing, train=True, logger=logger)
    train_dataset.words_to_ids()
    logger.info('Preprocessing Test Dataset...')
    test_dataset = QuoraQuestionDataset(data.test, use_pretrained_emb=True, reverse_vocab=train_dataset.reverse_vocab, preprocess = args.preprocessing, train = False, logger=logger)
    test_dataset.words_to_ids()


    logger.info('')
    logger.info('Number of training samples        :{}'.format(len(train_dataset)))
    logger.info('Number of validation samples      :{}'.format(len(test_dataset)))
    logger.info('Number of unique words          :{}'.format(train_dataset.unique_words))
    logger.info('')

    n_hidden = args.n_hidden
    gradient_clipping_norm = args.gradient_clipping_norm
    batch_size = args.batch_size
    embeddings_dim = args.emb_dim
    n_epoch = args.n_epoch
    n_layer = args.n_layer
    n_token = train_dataset.unique_words
    use_pretrained_embeddings = args.use_pretrained
    train_emb = args.train_embeddings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, collate_fn = collate_fn_lstm)
    test_dataloader = DataLoader(test_dataset, batch_size=1000, shuffle=False, collate_fn = collate_fn_lstm)

    model = SiameseLSTMCNN(n_hidden, embedded_vocab_class, embeddings_dim, n_layer, n_token,
                           train_embeddings = train_emb, use_pretrained = use_pretrained_embeddings,
                           dropouth=0.3, device=device)

    model = model.float()
    model = nn.DataParallel(model)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    logger.info('Building model.')
    logger.info('--------------------------------------')
    logger.info('Model Parameters:')
    logger.info('Hidden Size                  :{}'.format(args.n_hidden))
    logger.info('Number of layers             :{}'.format(args.n_layer))
    logger.info('Use pretrained Embeddings    :{}'.format(args.use_pretrained))
    logger.info('Dimensions of Embeddings     :{}'.format(args.emb_dim))
    logger.info('Train/fine tune Embeddings   :{}'.format(args.train_embeddings))
    logger.info('Gradient clipping            :{}'.format(args.gradient_clipping_norm))
    logger.info('--------------------------------------')
    logger.info('Training Parameters:')
    logger.info('Device                       :{}'.format(' GPU' if torch.cuda.is_available() else ' CPU'))
    logger.info('Optimizer                    :{}'.format(' Adam'))
    logger.info('Loss function                :{}'.format(' MSE'))
    logger.info('Batch Size                   :{}'.format(args.batch_size))
    logger.info('Number of Epochs             :{}'.format(args.n_epoch))
    logger.info("Parameters count             :{}".format(count_parameters(model)))
    logger.info('--------------------------------------')

    start = time()
    all_train_losses = []
    all_test_losses = []
    all_train_metrics = []
    all_test_metrics = []
    best_acc = 0.5
    logger.info("Training the model...")
    for epoch in range(n_epoch):
        epoch_time = time()
        epoch_iteration = 0
        epoch_loss=[]
        preds_train = []
        labels_train = []
        labels_test = []

        train(model, optimizer, criterion, train_dataloader, device, epoch_loss, preds_train, labels_train, args.gradient_clipping_norm, epoch, logger)

        eval_loss = []
        preds_test = []
        eval(model, criterion, test_dataloader, device, eval_loss, preds_test, labels_test)

        train_loss = np.mean(epoch_loss)
        train_metrics = compute_metrics_siamLSTM(np.concatenate(preds_train), np.concatenate(labels_train))#np.sum(preds_train)/data.train.shape[0]
        train_metrics = pd.DataFrame(train_metrics.values(), index = train_metrics.keys(), columns=['train']).T
        test_loss = np.mean(eval_loss)
        test_metrics = compute_metrics_siamLSTM(np.concatenate(preds_test), np.concatenate(labels_test))#np.sum(preds_test)/data.test.shape[0]
        test_metrics = pd.DataFrame(test_metrics.values(), index = test_metrics.keys(), columns=['test']).T
        stats_df = train_metrics.append(test_metrics)

        if test_metrics['accuracy'][0]>best_acc:
            if not model_path.exists():
                os.mkdir(model_path)
            logger.info('Saving best model at: {}'.format(str(model_path/'checkpoint.pth')))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_loss': test_loss,
                'test_accuracy':test_metrics['accuracy'][0],
                'test_fscore':test_metrics['f1'][0]
                }, str(model_path/'checkpoint.pth'))
        
        all_train_losses.append(train_loss)
        all_test_losses.append(test_loss)
        all_train_metrics.append(train_metrics)
        all_test_metrics.append(test_metrics)

        logger.info('Mean loss of epoch {} - train: {}, test: {}. Calculation time: {} hours'.format(epoch, train_loss, test_loss, (time() - epoch_time)/3600))
        logger.info('Detailed stats of epoch {}:\n{}\n'.format(epoch, stats_df.to_string()))
        logger.info('')

    all_train_metrics = pd.concat(all_train_metrics).reset_index(drop=True)
    all_test_metrics = pd.concat(all_test_metrics).reset_index(drop=True)
    all_train_metrics['epoch'] = [i for i in range(1, len(all_train_metrics)+1)]
    all_test_metrics['epoch'] = [i for i in range(1, len(all_test_metrics)+1)]

    all_losses = pd.DataFrame(zip(all_train_losses, all_test_losses), columns=['train_loss', 'test_loss'])
    all_losses['epoch'] = [i for i in range(1, len(all_losses)+1)]

    logger.info('Preprocessing Test Dataset...')
    test_dataset = QuoraQuestionDataset(ImportData.get_test_data(f'logs/data/test_dataset.csv'), use_pretrained_emb=True, reverse_vocab=train_dataset.reverse_vocab, preprocess = args.preprocessing, train = False, logger=logger)
    test_dataset.words_to_ids()
    test_dataloader = DataLoader(test_dataset, batch_size=1000, shuffle=False, collate_fn = collate_fn_lstm)

    test_loss = []
    preds_test = []
    labels_test = []

    eval(model, criterion, test_dataloader, device, test_loss, preds_test, labels_test)
    test1_metrics = compute_metrics_siamLSTM(np.concatenate(preds_test), np.concatenate(labels_test))#np.sum(preds_test)/data.test.shape[0]
    test1_metrics = pd.DataFrame(test1_metrics.values(), index = test1_metrics.keys(), columns=['test'])
    logger.info(test1_metrics)
    logger.info('')


    all_train_metrics.reset_index(drop=True).to_csv(args.logdir/f'{args.model_name}_{args.emb_dim}dglove_train_metrics.csv', sep=',', index=False)
    all_test_metrics.reset_index(drop=True).to_csv(args.logdir/f'{args.model_name}_{args.emb_dim}dglove_test_metrics.csv', sep=',', index=False)
    all_losses.to_csv(args.logdir/f'{args.model_name}_{args.emb_dim}dglove_train_losses.csv', sep=',', index=False)