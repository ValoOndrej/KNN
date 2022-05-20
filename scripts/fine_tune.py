
from transformers import get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset
from transformers import BertTokenizerFast
from transformers import AdamW
import multiprocessing
from tqdm import tqdm
import pandas as pd
import datetime
import random
import numpy
import torch
import time
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
numpy.random.seed(seed_val)
torch.manual_seed(seed_val)

device=torch.device('cuda' if torch.cuda.is_available() else  'cpu')

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def fit_batch(dataloader, model, optimizer, epoch):
    total_train_loss = 0
    
    for batch in tqdm(dataloader, desc=f"Training epoch:{epoch}", unit="batch"):
        # Unpack batch from dataloader.
        input_ids, attention_masks, token_type_ids, labels = batch
        
        # clear any previously calculated gradients before performing a backward pass.
        model.zero_grad()
        
        # Perform a forward pass (evaluate the model on this training batch).
        outputs = model(input_ids, 
                             token_type_ids=token_type_ids, 
                             attention_mask=attention_masks, 
                             labels=labels)
        loss = outputs['loss']
        
        total_train_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()
        
    return total_train_loss
    

def eval_batch(dataloader, model, metric=accuracy_score):
    total_eval_accuracy = 0
    total_eval_loss = 0
    predictions , predicted_labels = [], []
    
    for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
        # Unpack batch from dataloader.
        input_ids, attention_masks, token_type_ids, labels = batch
        
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            outputs = model(input_ids, 
                                   token_type_ids=token_type_ids, 
                                   attention_mask=attention_masks,
                                   labels=labels)
        loss, logits = outputs['loss'], outputs['logits']
        
        # Calculate the accuracy for this batch of validation sentences, and
        # accumulate it over all batches.
        y_pred = numpy.argmax(logits.detach().cpu().numpy(), axis=1).flatten()
        total_eval_accuracy += metric(labels, y_pred)
        
        predictions.extend(logits.detach().cpu().numpy().tolist())
        predicted_labels.extend(y_pred.tolist())
    
    return total_eval_accuracy, total_eval_loss, predictions ,predicted_labels


def train_BERT(train_dataloader, validation_dataloader, model, optimizer, epochs):
    # We'll store a number of quantities such as training and validation loss, 
    # validation accuracy, and timings.
    training_stats = []
    
    # Measure the total training time for the whole run.
    total_t0 = time.time()
    
    for epoch in range(0, epochs):
        
        # Measure how long the training epoch takes.
        t0 = time.time()
        
        # Reset the total loss for this epoch.
        total_train_loss = 0
        
        # Put the model into training mode. 
        model.train()
        
        total_train_loss = fit_batch(train_dataloader, model, optimizer, epoch)
        
        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)
        
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)
        
        t0 = time.time()
        
        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()
        
        total_eval_accuracy, total_eval_loss, _, _ = eval_batch(validation_dataloader, model)
        
        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        
        print(f"  Accuracy: {avg_val_accuracy}")
    
        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)
    
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
    
        print(f"  Validation Loss: {avg_val_loss}")
    
        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )
        

    print("")
    print("Training complete!")

    print("saving model to ../model/")
    os.makedirs('../model', exist_ok=True)
    torch.save(model, f"../model/BERT-{time.time()}")

    print(f"Total training took {format_time(time.time()-total_t0)}")
    return training_stats


def convert_to_dataset_torch(data: pd.DataFrame, labels: pd.Series) -> TensorDataset:
    input_ids = []
    attention_masks = []
    token_type_ids = []
    for _, row in tqdm(data.iterrows(), total=data.shape[0]):
        encoded_dict = tokenizer.encode_plus(row["question1"], row["question2"], padding='max_length', pad_to_max_length=True, 
                      return_attention_mask=True, return_tensors='pt', truncation=True)
        # Add the encoded sentences to the list.
        input_ids.append(encoded_dict['input_ids'])
        token_type_ids.append(encoded_dict["token_type_ids"])
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])
    
    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels.values)
    
    return TensorDataset(input_ids, attention_masks, token_type_ids, labels)


print("Loading Data ...")
questions_dataset = pd.read_csv("../data/quora-IR-dataset/quora_duplicate_questions.tsv", index_col="id", nrows=5000, sep='\t')
augmented_train_pairs = pd.read_csv("../data/quora-IR-dataset/classification/augmented_train_pairs.tsv", nrows=5000, sep='\t')
dev_pairs = pd.read_csv("../data/quora-IR-dataset/classification/dev_pairs.tsv", nrows=5000, sep='\t')
test_pairs = pd.read_csv("../data/quora-IR-dataset/classification/test_pairs.tsv", nrows=5000, sep='\t')
train_pairs = pd.read_csv("../data/quora-IR-dataset/classification/train_pairs.tsv", nrows=5000, sep='\t')

print("Loading Tokenizer ...")
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)

print("Assigning data to train,test,validate ...")
X_train = train_pairs[["question1","question2"]]
y_train = train_pairs[["is_duplicate"]]
X_test = test_pairs[["question1","question2"]]
y_test = test_pairs[["is_duplicate"]]
X_validation = dev_pairs[["question1","question2"]]
y_validation = dev_pairs[["is_duplicate"]]

#X_train, X_test, y_train, y_test = train_test_split(questions_dataset[["question1", "question2"]], 
#                                                    questions_dataset["is_duplicate"], test_size=0.2, random_state=42)
#X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
#max_length = 512
#print(tokenizer.encode_plus(X_train.iloc[0]["question1"], X_train.iloc[0]["question2"],  max_length=max_length, 
#                      pad_to_max_length=True, return_attention_mask=True, return_tensors='pt', truncation=True))

print("Converting data to torch ...")
train = convert_to_dataset_torch(X_train, y_train)
validation = convert_to_dataset_torch(X_validation, y_validation)

# The DataLoader needs to know our batch size for training, so we specify it 
# here.
batch_size = 1

core_number = 1

# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order. 
train_dataloader = torch.utils.data.DataLoader(
            train,  # The training samples.
            sampler = torch.utils.data.RandomSampler(train), # Select batches randomly
            batch_size = batch_size, # Trains with this batch size.
            num_workers = core_number
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = torch.utils.data.DataLoader(
            validation, # The validation samples.
            sampler = torch.utils.data.SequentialSampler(validation), # Pull out batches sequentially.
            batch_size = batch_size, # Evaluate with this batch size.
            num_workers = core_number
        )


# Load BertForSequenceClassification, the pretrained BERT model with a single 
# linear classification layer on top. 
bert_model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels=2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
    output_attentions=False, # Whether the model returns attentions weights.
    output_hidden_states=False, # Whether the model returns all hidden-states.
)

# Note: AdamW is a class from the huggingface library (as opposed to pytorch)
adamw_optimizer = torch.optim.AdamW(bert_model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )


print(f"Using device: {device}")
bert_model = torch.nn.DataParallel(bert_model, device_ids = [i for i in range(torch.cuda.device_count())])


# Number of training epochs. The BERT authors recommend between 2 and 4. 
epochs = 1

# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(adamw_optimizer, 
                                            num_warmup_steps = 0, 
                                            num_training_steps = total_steps)


training_stats = train_BERT(train_dataloader, validation_dataloader, bert_model, adamw_optimizer, epochs)

df_stats = pd.DataFrame(training_stats).set_index('epoch')
print(df_stats)