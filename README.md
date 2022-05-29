
# KNN

## Trainning Bert
```sh 
python3 train_bert.py [--help] [--model_name STRING] [--logdir PATH] [--data_file PATH] 
[--split_seed INT] [--batch_size INT] [--n_epoch INT] [--size_of_train INT] 
[--augmentation] [--bert_cls STRING] [--bert_backbone STRING]
    
    '--help', '-h', Prints the short help message.
    '--model_name STRING', '-model_name STRING', Name of trained model. Needed only for correct logs output.
    '--logdir PATH', '-log PATH', Directory to save all downloaded files, and model checkpoints.
    '--data_file PATH', '-df PATH', Path to dataset.
    '--split_seed INT', '-s INT', Seed for splitting the dataset.
    '--batch_size INT', '-b INT', Batch size.
    '--n_epoch INT', '-epo INT', Number of epochs.
    '--size_of_train INT', '-sot INT', Number of train data.
    '--n_epoch INT', '-epo INT', Number of epochs.
    '--aug_intensity INT', '-ai INT', Number of train data.
    '--bert_cls STRING', '-bert_cls STRING', Type of BERT trained (classificator, siamese).
    '--bert_backbone STRING', '-bert_backbone STRING', Either path to the model, or name of the BERT model 
    that should be used, compatible with HuggingFace Transformers.output.
```

## Inference Bert
```sh 
python3 infer.py [--help] [--model_name STRING] [--logdir PATH] [--data_file PATH] 
[--split_seed INT] [--batch_size INT] [--n_epoch INT] [--size_of_train INT] 
[--augmentation] [--bert_cls STRING] [--bert_backbone STRING]
    
    '--help', '-h', Prints the short help message.
    '--model_name STRING', '-model_name STRING', Name of trained model. Needed only for correct logs output.
    '--logdir PATH', '-log PATH', Directory to save all downloaded files, and model checkpoints.
    '--data_file PATH', '-df PATH', Path to dataset.
    '--split_seed INT', '-s INT', Seed for splitting the dataset.
    '--batch_size INT', '-b INT', Batch size.
    '--n_epoch INT', '-epo INT', Number of epochs.
    '--size_of_train INT', '-sot INT', Number of train data.
    '--n_epoch INT', '-epo INT', Number of epochs.
    '--aug_intensity INT', '-ai INT', Number of train data.
    '--bert_cls STRING', '-bert_cls STRING', Type of BERT trained (classificator, siamese).
    '--bert_backbone STRING', '-bert_backbone STRING', Either path to the model, or name of the BERT model 
    that should be used, compatible with HuggingFace Transformers.output.
```

## Trainning Siamese CNN
```sh 
python3 train_cnn.py [--help] [--model_name STRING] [--logdir PATH] [--data_file PATH] 
[--use_pretrained] [--emb_dim INT] [--emb_path STRING] [--split_seed INT] 
[--preprocessing] [--batch_size INT] [--n_epoch INT] [--gradient_clipping_norm FLOAT] 
[--train_embeddings]  [--size_of_train INT] [--aug_intensity INT] [--augmentation] 
    
    '--help', '-h', Prints the short help message.
    '--model_name STRING','-model STRING' Name of trained model. Needed only for correct logs output.
    '--logdir STRING','-log STRING', Directory to save all downloaded files, and model checkpoints.
    '--data_file STRING', '-df STRING', Path to dataset.
    '--use_pretrained', '-pr', Boolean, whether use pretrained embeddings.
    '--emb_dim INT', '-dim INT', Dimensions of pretrained embeddings.
    '--emb_path STRING', '-empth STRING', Path to file with pretrained embeddings.
    '--split_seed INT', '-s INT', Seed for splitting the dataset.
    '--preprocessing', '-noprep', Preprocess dataset before training the model.
    '--batch_size INT', '-b INT', Batch Size.
    '--n_epoch INT', '-epo INT', Number of epochs.
    '--gradient_clipping_norm FLOAT', '-gc FLOAT', Gradient clipping norm.
    '--train_embeddings', '-note', Whether to fine-tune embedding weights during training.
    '--size_of_train INT', '-sot INT', Number of train data.
    '--aug_intensity INT', '-ai INT', Number of train data.
    '--augmentation', '-a', Augment data
```

## Trainning Siamese LSTM
```sh 
python3 train_lstm.py [--help] [--model_name STRING] [--logdir PATH] [--data_file PATH] 
[--use_pretrained] [--emb_dim INT] [--emb_path STRING] [--split_seed INT] 
[--preprocessing] [--n_hidden INT] [--batch_size INT] [--n_epoch INT] 
[--n_epoch INT] [--gradient_clipping_norm FLOAT] [--train_embeddings] 
[--n_layer INT] [--aug_intensity INT] [--augmentation] 
    
    '--help', '-h', Prints the short help message.
    '--model_name STRING','-model STRING' Name of trained model. Needed only for correct logs output.
    '--logdir STRING','-log STRING', Directory to save all downloaded files, and model checkpoints.
    '--data_file STRING', '-df STRING', Path to dataset.
    '--use_pretrained', '-pr', Boolean, whether use pretrained embeddings.
    '--emb_dim INT', '-dim INT', Dimensions of pretrained embeddings.
    '--emb_path STRING', '-empth STRING', Path to file with pretrained embeddings.
    '--split_seed INT', '-s INT', Seed for splitting the dataset.
    '--preprocessing', '-noprep', Preprocess dataset before training the model.
    '--n_hidden INT', '-hid INT', Batch Size.
    '--batch_size INT', '-b INT', Batch Size.
    '--n_epoch INT', '-epo INT', Number of epochs.
    '--n_layer', '-nl INT', Number of LSTM layers.
    '--gradient_clipping_norm FLOAT', '-gc FLOAT', Gradient clipping norm.
    '--train_embeddings', '-note', Whether to fine-tune embedding weights during training.
    '--size_of_train INT', '-sot INT', Number of train data.
    '--aug_intensity INT', '-ai INT', Number of train data.
    '--augmentation', '-a', Augment data
```

## Trainning Siamese LSTMCNN
```sh 
python3 train_lstmcnn.py [--help] [--model_name STRING] [--logdir PATH] [--data_file PATH] 
[--use_pretrained] [--emb_dim INT] [--emb_path STRING] [--split_seed INT] 
[--preprocessing] [--n_hidden INT] [--batch_size INT] [--n_epoch INT] 
[--n_epoch INT] [--gradient_clipping_norm FLOAT] [--train_embeddings] 
[--n_layer INT] [--aug_intensity INT] [--augmentation] 
    
    '--help', '-h', Prints the short help message.
    '--model_name STRING','-model STRING' Name of trained model. Needed only for correct logs output.
    '--logdir STRING','-log STRING', Directory to save all downloaded files, and model checkpoints.
    '--data_file STRING', '-df STRING', Path to dataset.
    '--use_pretrained', '-pr', Boolean, whether use pretrained embeddings.
    '--emb_dim INT', '-dim INT', Dimensions of pretrained embeddings.
    '--emb_path STRING', '-empth STRING', Path to file with pretrained embeddings.
    '--split_seed INT', '-s INT', Seed for splitting the dataset.
    '--preprocessing', '-noprep', Preprocess dataset before training the model.
    '--n_hidden INT', '-hid INT', Batch Size.
    '--batch_size INT', '-b INT', Batch Size.
    '--n_epoch INT', '-epo INT', Number of epochs.
    '--n_layer', '-nl INT', Number of LSTM layers.
    '--gradient_clipping_norm FLOAT', '-gc FLOAT', Gradient clipping norm.
    '--train_embeddings', '-note', Whether to fine-tune embedding weights during training.
    '--size_of_train INT', '-sot INT', Number of train data.
    '--aug_intensity INT', '-ai INT', Number of train data.
    '--augmentation', '-a', Augment data
```



## doc
https://www.overleaf.com/read/ctjbjgctsrvg


