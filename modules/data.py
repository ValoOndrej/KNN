import torch
import pandas as pd
import numpy as np
from re import sub
from pathlib import Path
import re
import random
from random import shuffle
from sklearn.model_selection import train_test_split
import nlpaug.augmenter.word as naw
from torch.utils.data import Dataset
from logging import Logger
from typing import Union
from tqdm import tqdm

random.seed(1)
#for the first time you use wordnet
#import nltk
#nltk.download('wordnet')
from nltk.corpus import wordnet
path = Path('./logs/data/')
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 
			'ours', 'ourselves', 'you', 'your', 'yours', 
			'yourself', 'yourselves', 'he', 'him', 'his', 
			'himself', 'she', 'her', 'hers', 'herself', 
			'it', 'its', 'itself', 'they', 'them', 'their', 
			'theirs', 'themselves', 'what', 'which', 'who', 
			'whom', 'this', 'that', 'these', 'those', 'am', 
			'is', 'are', 'was', 'were', 'be', 'been', 'being', 
			'have', 'has', 'had', 'having', 'do', 'does', 'did',
			'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
			'because', 'as', 'until', 'while', 'of', 'at', 
			'by', 'for', 'with', 'about', 'against', 'between',
			'into', 'through', 'during', 'before', 'after', 
			'above', 'below', 'to', 'from', 'up', 'down', 'in',
			'out', 'on', 'off', 'over', 'under', 'again', 
			'further', 'then', 'once', 'here', 'there', 'when', 
			'where', 'why', 'how', 'all', 'any', 'both', 'each', 
			'few', 'more', 'most', 'other', 'some', 'such', 'no', 
			'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
			'very', 's', 't', 'can', 'will', 'just', 'don', 
			'should', 'now', '']

class ImportData:
    def __init__(self, dataset: Union[str, pd.DataFrame]):
        if isinstance(dataset, str):
            self.data = pd.read_csv(dataset).dropna().copy()[['question1', 'question2', 'is_duplicate']]
        elif isinstance(dataset, pd.DataFrame):
            self.data = dataset
        else:
            raise ValueError('Wrong value of dataset parameter, should be either string or pd.DataFrame!')
    
    @classmethod
    def get_test_data(cls, path):
        return pd.read_csv(path).dropna().copy()[['question1', 'question2', 'is_duplicate']]


    def train_test_split(self, seed: int=44, augment: bool=False, size_of_train: int=5000, num_arg: int=9):
        if (path/f'augmented_train_{size_of_train}_{num_arg}_dataset.csv').exists():
            self.test  = pd.read_csv(str(path/f'val_{size_of_train}_dataset.csv')).dropna().copy()[['question1', 'question2', 'is_duplicate']]
            if augment:
                self.train = pd.read_csv(str(path/f'augmented_train_{size_of_train}_{num_arg}_dataset.csv')).dropna().copy()[['question1', 'question2', 'is_duplicate']]
            else:
                self.train = pd.read_csv(str(path/f'train_{size_of_train}_dataset.csv')).dropna().copy()[['question1', 'question2', 'is_duplicate']]
        else:
            if (path/f'train_{size_of_train}_dataset.csv').exists():
                self.train = pd.read_csv(path/f'train_{size_of_train}_dataset.csv').dropna().copy()[['question1', 'question2', 'is_duplicate']]
                self.test = pd.read_csv(path/f'val_{size_of_train}_dataset.csv').dropna().copy()[['question1', 'question2', 'is_duplicate']]
            else:
                if not size_of_train:
                    size_of_train = 0.75
                    size_of_val = 0.25
                else:
                    size_of_val = size_of_train//4
                self.train, self.test = train_test_split(self.data, test_size=size_of_val, random_state=seed, train_size=size_of_train)
                self.train.to_csv(str(path/f'train_{size_of_train}_dataset.csv'))
                self.test.to_csv(str(path/f'val_{size_of_train}_dataset.csv'))
            if augment:
                data = self.train.copy()
                data = data.reset_index()
                data_list = [self.train]
                for i in tqdm(range(data.shape[0]), desc="Augmenting"):
                    qs1 = self.eda(data.iloc[i].question1, num_aug=num_arg)
                    qs2 = self.eda(data.iloc[i].question2, num_aug=num_arg)
                    is_ds = np.full(len(qs1),data.iloc[i].is_duplicate)
                    pairs = pd.DataFrame({'question1': qs1, 
                                        'question2': qs2, 
                                        'is_duplicate': is_ds})
                    data_list.append(pairs)
                self.train = pd.concat(data_list)
                self.train.to_csv(str(path/f'augmented_train_{size_of_train}_{num_arg}_dataset.csv'))


    def __getitem__(self, idx: int):
        ex = self.data.loc[idx]
        return ex.question1, ex.question2, ex.is_duplicate


    def __len__(self):
        return self.data.shape[0]


    def synonym_replacement(self, question, n):
        if len(question) <= 1:
            return question
        new_question = question.copy()
        for _ in range(n):
            
            indexes = []        
            for m in range(len(new_question)):
                if not new_question[m] in stop_words:
                    indexes.append(m)
            if not indexes:
                return question

            index = random.choice(indexes)
            synonyms = self.get_synonyms(new_question[index])
            if len(synonyms) == 0:
                continue
            j = random.randint(0,len(synonyms)-1)
            new_question[m] = synonyms[j]
        return new_question


    def random_insertion(self, question, n):
        if len(question) <= 1:
            return question
        new_question = question.copy()
        for _ in range(n):
            
            indexes = []        
            for m in range(len(new_question)):
                if not new_question[m] in stop_words:
                    indexes.append(m)
            if not indexes:
                return question

            index = random.choice(indexes)

            synonyms = self.get_synonyms(new_question[index])
            if len(synonyms) == 0:
                continue
            j = random.randint(0,len(synonyms)-1)
            new_question.insert(random.randint(0,len(new_question)-1), synonyms[j])
        return new_question


    def random_swap(self, question, n):
        if len(question) <= 1:
            return question
        new_question = question.copy()
        for i in range(n):
            while True:
                k = random.randint(0,len(new_question)-1)
                m = random.randint(0,len(new_question)-1)
                if k != m:
                    break
            new_question[m] = question[k]
            new_question[k] = question[m]
        return new_question

    def random_deletion(self, question, p):
        if len(question) <= 1:
            return question
        new_question = []
        for word in question:
            if random.randint(0,1) > p:
                new_question.append(word)
        return new_question


    def get_synonyms(self, word):

        synonyms = set()
        for syn in wordnet.synsets(word): 
            for l in syn.lemmas(): 
                synonym = l.name().replace("_", " ").replace("-", " ").lower()
                synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
                synonyms.add(synonym) 
        if word in synonyms:
            synonyms.remove(word)
        return list(synonyms)


    def eda(self, sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):
        
        sentence = QuoraQuestionDataset.text_to_word_list(sentence)
        words = sentence.split(' ')
        words = [word for word in words if word != '']
        num_words = len(words)
        
        augmented_sentences = []
        num_new_per_technique = int(num_aug/4)+1
    
        #sr
        if (alpha_sr > 0):
            n_sr = max(1, int(alpha_sr*num_words))
            for _ in range(num_new_per_technique):
                a_words = self.synonym_replacement(words, n_sr)
                augmented_sentences.append(' '.join(a_words))
        #ri
        if (alpha_ri > 0):
            n_ri = max(1, int(alpha_ri*num_words))
            for _ in range(num_new_per_technique):
                a_words = self.random_insertion(words, n_ri)
                augmented_sentences.append(' '.join(a_words))
        #rs
        if (alpha_rs > 0):
            n_rs = max(1, int(alpha_rs*num_words))
            for _ in range(num_new_per_technique):
                a_words = self.random_swap(words, n_rs)
                augmented_sentences.append(' '.join(a_words))
        #rd
        if (p_rd > 0):
            for _ in range(num_new_per_technique):
                a_words = self.random_deletion(words, p_rd)
                augmented_sentences.append(' '.join(a_words))
        augmented_sentences = [QuoraQuestionDataset.text_to_word_list(sentence) for sentence in augmented_sentences]
        shuffle(augmented_sentences)

        if num_aug >= 1:
            augmented_sentences = augmented_sentences[:num_aug]
        else:
            keep_prob = num_aug / len(augmented_sentences)
            augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

        augmented_sentences.append(sentence)

        return augmented_sentences


class QuoraQuestionDataset(Dataset):
    def __init__(self, datasetvar: ImportData, use_pretrained_emb: bool=False, reverse_vocab: dict = None, preprocess: bool = True, train: bool = True, logger: Logger = None):
        self.data = datasetvar.copy()
        self.type = 'train' if train==True else 'test'
        self.logger = logger

        if preprocess == True:
            self.preprocessing()  

        if not use_pretrained_emb and train:
            self.logger.info('Initializing vocab with ids of all unique words present in training dataset...')
            unique_words = self.data.question1.str.split(' ').append(self.data.question2.str.split(' '))
            unique_words = pd.Series([i for j in unique_words.values for i in j]).unique().tolist()
            unique_words.insert(0, 'pad')
            self.unique_words = len(unique_words)
            self.reverse_vocab = dict(zip(unique_words, range(0,self.unique_words)))

        elif isinstance(reverse_vocab, dict):
            self.reverse_vocab = reverse_vocab
            self.unique_words = len(reverse_vocab.keys())

        else:
            raise Exception("Invalid reverse_vocab arg (cannot create dictionary with mapping of words to their indices).")
  
    def preprocessing(self):
        self.logger.info('Cleaning {} dataset...'.format(self.type))
        self.data.question1 = self.data.question1.apply(lambda x: QuoraQuestionDataset.text_to_word_list(x))
        self.data.question2 = self.data.question2.apply(lambda x: QuoraQuestionDataset.text_to_word_list(x))
    
    def words_to_ids(self):
        self.logger.info('Replacing all words in {} dataset with their ids...'.format(self.type))
        self.data.question1 = self.data.question1.apply(lambda x: list(map(lambda y: self.replace_words(y, self.reverse_vocab), x.split())))
        self.data.question2 = self.data.question2.apply(lambda x: list(map(lambda y: self.replace_words(y, self.reverse_vocab), x.split())))

    @classmethod    
    def text_to_word_list(cls, text: str):
        text = str(text)
        text = text.lower()

        # Clean the text
        text = sub(r"[^A-Za-z0-9^,!.\/'+-=)(]", " ", text)
        text = sub(r"\′", "'", text)
        text = sub(r"\’", "'", text)
        text = sub(r"\`", "'", text)
        text = sub(r"\`", "'", text)
        text = sub(r"she\'s", "she is ", text)
        text = sub(r"he\'s", "he is ", text)
        text = sub(r"what\'s", "what is ", text)
        text = sub(r"\'ve", " have ", text)
        text = sub(r"can\'t", "can not ", text)
        text = sub(r"cannot", "can not ", text)
        text = sub(r"won\'t", "will not ", text)
        text = sub(r"it\'s", "it is ", text)
        text = sub(r"i\'m", "i am ", text)
        text = sub(r"\'re", " are ", text)
        text = sub(r"\'d", " would ", text)
        text = sub(r"n\'t", " not ", text)
        text = sub(r"\'ll", " will ", text)
        text = sub(r"\'s", " own", text)
        text = sub(r",", " ", text)
        text = sub(r"\.", " ", text)
        text = sub(r"!", " ! ", text)
        text = sub(r"\/", " ", text)
        text = sub(r"\^", " ^ ", text)
        text = sub(r"\+", " + ", text)
        text = sub(r"\-", " - ", text)
        text = sub(r"\=", " = ", text)
        text = sub(r"\'", " ", text)
        text = sub(r"(\d+)(k)", r"\g<1>000", text)
        text = sub(r":", " : ", text)
        text = sub(r" e g ", " eg ", text)
        text = sub(r" b g ", " bg ", text)
        text = sub(r" u s ", " american ", text)
        text = sub(r"\0s", "0", text)
        text = sub(r" 9 11 ", "911", text)
        text = sub(r"e - mail", "email", text)
        text = sub(r"j k", "jk", text)
        text = sub(r"\s{2,}", " ", text)
        text = sub(r"%", " percent ", text)
        text = sub(r"₹", " rupee ", text)
        text = sub(r"\$", " dollar ", text)
        text = sub(r"€", " euro ", text)
        text = sub(r",000,000", "m ", text)
        text = sub(r",000", "k ", text)

        return text  

  
    def replace_words(self, word: str, reverse_vocab: dict):
        if word in reverse_vocab.keys():
            return reverse_vocab[f'{word}']
        else:
            return 0
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
      
        ex = self.data.iloc[idx] 

        if type(idx)==list:
            return(ex.question1.values, ex.question2.values, ex.is_duplicate.values)
        else:
            return (ex.question1, ex.question2, ex.is_duplicate)
    
    def __len__(self):
        return self.data.shape[0]