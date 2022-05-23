
# KNN

## Trainning Bert and SiameseBert
```sh 
python3 train.py [-h] [--model_name STRING] [--logdir PATH] [--data_file PATH] [--split_seed INT] 
				 [--batch_size INT] [--n_epoch INT] [--bert_cls STRING] [--bert_backbone STRING]
    
    '--help', '-h' Prints the short help message.
    '--model_name STRING' '-model_name STRING' Name of trained model. Needed only for correct logs output.
    '--logdir PATH' '-log PATH' Directory to save all downloaded files, and model checkpoints.
    '--data_file PATH' '-df PATH' Path to dataset.
    '--split_seed INT' '-s INT' Seed for splitting the dataset.
    '--batch_size INT' '-b INT' Batch size.
    '--n_epoch INT' '-epo INT' Number of epochs.
    '--bert_cls STRING' '-bert_cls STRING' Type of BERT trained (classificator, siamese).
    '--bert_backbone STRING' '-bert_backbone STRING' Either path to the model, or name of the BERT model 	
												     that should be used, compatible with HuggingFace 
													 Transformers.output
```

## modules/


