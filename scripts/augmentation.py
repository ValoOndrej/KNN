
from collections import defaultdict
from traceback import print_tb
from tracemalloc import stop
from sentence_transformers import util
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import STSBenchmarkDataReader, InputExample
from datetime import datetime
import nlpaug.augmenter.word as naw
import logging
import torch
import random
import sys
import os
import csv
import tqdm
import nltk
import gzip

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)
#### /print debug information to stdout

random.seed(69)

probability = 0.1


model_name = sys.argv[1] if len(sys.argv) > 1 else 'bert-base-uncased'
models_dir = "../models/Word2Vec/"
device = "cuda" if torch.cuda.is_available() else "cpu"

source_file = "../data/quora-IR-dataset/quora_duplicate_questions.tsv"
train_file = "../data/quora-IR-dataset/classification/train_pairs.tsv"
stop_words = set(['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'i', 'if', 'in', 'into', 'is', 'isn', "isn't", "it's", 'its', 'itself', 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 'she', "she's", 'should', "should've", 'shouldn', "shouldn't", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', 'were', 'weren', "weren't", 'which', 'while', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'])

if not os.path.exists(source_file):
    print("Missing", source_file)

#### Synonym replacement using Word2Vec ####
# Download the word2vec pre-trained Google News corpus (GoogleNews-vectors-negative300.bin)
# link: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
# aug = naw.WordEmbsAug(
#     model_type='word2vec', model_path=models_dir+'GoogleNews-vectors-negative300.bin',
#     action="substitute")

#### Synonym replacement using WordNet ####
aug = naw.SynonymAug(aug_src='wordnet')

#### Synonym replacement using BERT ####
#aug = naw.ContextualWordEmbsAug(
#    model_path=model_name, action="insert", device=device)


#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
#nltk.download('omw-1.4')

def synonym_replacement(question):
    new_question = question.copy()
    n = int(len(new_question) * probability)
    for i in range(n):
        m = random.randint(0,len(new_question)-1)
        while new_question[m] in stop_words:
            m = random.randint(0,len(new_question)-1)
        augmented_texts = aug.augment(new_question[m])
        new_question[m] = augmented_texts
    return new_question


def random_insertion(question):
    new_question = question.copy()
    n = int(len(new_question) * probability)
    
    for i in range(n):
        m = random.randint(0,len(new_question)-1)
        while new_question[m] in stop_words:
            m = random.randint(0,len(new_question)-1)
        augmented_texts = aug.augment(new_question[m])
        new_question.insert(random.randint(0,len(new_question)-1), augmented_texts)
    return new_question


def random_swap(question):
    new_question = question.copy()
    n = int(len(new_question) * probability)
    for i in range(n):
        k = random.randint(0,len(new_question)-1)
        m = random.randint(0,len(new_question)-1)
        word_k = new_question[k]
        word_m = new_question[m]
        new_question[n] = word_m
        new_question[m] = word_k
    return new_question

def random_deletion(question):
    new_question = []
    for word in question:
        if random.randint(0,1) > probability:
            new_question.append(word)
    return new_question


sentences = {}
duplicates = defaultdict(lambda: defaultdict(bool))
rows = []
with open(source_file, encoding='utf8') as fIn:
    final_line = fIn.readlines()[-1].split()
    final_id = final_line[0]
    final_quid = final_line[1] if(final_line[1] > final_line[2]) else final_line[2]

##########  load original train to augment train withou augmantation ##########
with open(train_file, encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    progress = tqdm.tqdm(unit="docs", total=len(final_id))
    for row in reader:
        id1 = row['qid1']
        id2 = row['qid2']
        question1 = row['question1'].replace("\r", "").replace("\n", " ").replace("\t", " ").replace("\r", "").replace("\n", " ").replace("\t", " ")\
       .replace("/", " ").replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")\
       .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
       .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
       .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
       .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
       .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
       .replace("€", " euro ").replace("'ll", " will").split()
        question2 = row['question1'].replace("\r", "").replace("\n", " ").replace("\t", " ").replace("\r", "").replace("\n", " ").replace("\t", " ")\
       .replace("/", " ").replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")\
       .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
       .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
       .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
       .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
       .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
       .replace("€", " euro ").replace("'ll", " will").split()
        is_duplicate = row['is_duplicate']
        if question1 == [] or question2 == [] or is_duplicate == None:
            continue
            
        new_sr_id1 = str(int(final_quid) + (1 * int(id1)))
        new_ri_id1 = str(int(final_quid) + (2 * int(id1)))
        new_rs_id1 = str(int(final_quid) + (3 * int(id1)))
        new_rd_id1 = str(int(final_quid) + (4 * int(id1)))

        new_sr_q1 = synonym_replacement(question1)
        new_ri_q1 = random_insertion(question1)
        new_rs_q1 = random_swap(question1)
        new_rd_q1 = random_deletion(question1)

        new_sr_id2 = str(int(final_quid) + (1 * int(id2)))
        new_ri_id2 = str(int(final_quid) + (2 * int(id2)))
        new_rs_id2 = str(int(final_quid) + (3 * int(id2)))
        new_rd_id2 = str(int(final_quid) + (4 * int(id2)))

        new_sr_q2 = synonym_replacement(question2)
        new_ri_q2 = random_insertion(question2)
        new_rs_q2 = random_swap(question2)
        new_rd_q2 = random_deletion(question2)

        question1 = " ".join(question1)
        question2 = " ".join(question2)

        new_sr_q1 = " ".join(new_sr_q1)
        new_ri_q1 = " ".join(new_ri_q1)
        new_rs_q1 = " ".join(new_rs_q1)
        new_rd_q1 = " ".join(new_rd_q1)

        new_sr_q2 = " ".join(new_sr_q2)
        new_ri_q2 = " ".join(new_ri_q2)
        new_rs_q2 = " ".join(new_rs_q2)
        new_rd_q2 = " ".join(new_rd_q2)

        sentences[id1] = question1
        sentences[id2] = question2

        sentences[new_sr_id1] = new_sr_q1
        sentences[new_ri_id1] = new_ri_q1
        sentences[new_rs_id1] = new_rs_q1
        sentences[new_rd_id1] = new_rd_q1

        sentences[new_sr_id2] = new_sr_q2
        sentences[new_ri_id2] = new_ri_q2
        sentences[new_rs_id2] = new_rs_q2
        sentences[new_rd_id2] = new_rd_q2
        
        rows.append({'qid1': id1, 'qid2': id2, 'question1': question1, 'question2': question2, 'is_duplicate': is_duplicate})
        
        rows.append({'qid1': new_sr_id1, 'qid2': id2, 'question1': new_sr_q1, 'question2': question2, 'is_duplicate': is_duplicate})
        rows.append({'qid1': new_ri_id1, 'qid2': id2, 'question1': new_ri_q1, 'question2': question2, 'is_duplicate': is_duplicate})
        rows.append({'qid1': new_rs_id1, 'qid2': id2, 'question1': new_rs_q1, 'question2': question2, 'is_duplicate': is_duplicate})
        rows.append({'qid1': new_rd_id1, 'qid2': id2, 'question1': new_rd_q1, 'question2': question2, 'is_duplicate': is_duplicate})

        rows.append({'qid1': id1, 'qid2': new_sr_id2, 'question1': question1, 'question2': new_sr_q2, 'is_duplicate': is_duplicate})
        rows.append({'qid1': id1, 'qid2': new_ri_id2, 'question1': question1, 'question2': new_ri_q2, 'is_duplicate': is_duplicate})
        rows.append({'qid1': id1, 'qid2': new_rs_id2, 'question1': question1, 'question2': new_rs_q2, 'is_duplicate': is_duplicate})
        rows.append({'qid1': id1, 'qid2': new_rd_id2, 'question1': question1, 'question2': new_rd_q2, 'is_duplicate': is_duplicate})

        rows.append({'qid1': new_sr_id1, 'qid2': new_sr_id2, 'question1': new_sr_q1, 'question2': new_sr_q2, 'is_duplicate': is_duplicate})
        rows.append({'qid1': new_ri_id1, 'qid2': new_ri_id2, 'question1': new_ri_q1, 'question2': new_ri_q2, 'is_duplicate': is_duplicate})
        rows.append({'qid1': new_rs_id1, 'qid2': new_rs_id2, 'question1': new_rs_q1, 'question2': new_rs_q2, 'is_duplicate': is_duplicate})
        rows.append({'qid1': new_rd_id1, 'qid2': new_rd_id2, 'question1': new_rd_q1, 'question2': new_rd_q2, 'is_duplicate': is_duplicate})

        progress.update(1)

progress.reset()
progress.close()
logging.info("Textual augmentation completed....")

##### save dataset #####
with open('../data/quora-IR-dataset/classification/augmented_train_pairs.tsv', 'w', encoding='utf8') as fOutTrain:
    fOutTrain.write("\t".join(['qid1', 'qid2', 'question1', 'question2', 'is_duplicate'])+"\n")

    for row in rows:
        id1 = row['qid1']
        id2 = row['qid2']

        fOutTrain.write("\t".join([row['qid1'], row['qid2'], sentences[id1], sentences[id2], row['is_duplicate']]))
        fOutTrain.write("\n")
