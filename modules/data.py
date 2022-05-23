import torch
import pandas as pd
import numpy as np
from re import sub
import random
from sklearn.model_selection import train_test_split
import nlpaug.augmenter.word as naw
from torch.utils.data import Dataset
from logging import Logger
from typing import Union
import nltk

class ImportData:
    def __init__(self, dataset: Union[str, pd.DataFrame]):
        if isinstance(dataset, str):
            self.data = pd.read_csv(dataset).dropna().copy()[['question1', 'question2', 'is_duplicate']]
        elif isinstance(dataset, pd.DataFrame):
            self.data = dataset
        else:
            raise ValueError('Wrong value of dataset parameter, should be either string or pd.DataFrame!')
    
    def train_test_split(self, seed: int=44, test_size: int=40000):
        self.train, self.test = train_test_split(self.data, test_size=test_size, random_state=seed)
    
    def __getitem__(self, idx: int):
        ex = self.data.loc[idx]
        return ex.question1, ex.question2, ex.is_duplicate
  
    def __len__(self):
        return self.data.shape[0]


class QuoraQuestionDataset(Dataset):
    def __init__(self, datasetvar: ImportData, use_pretrained_emb: bool=False, reverse_vocab: dict = None, preprocess: bool = True, train: bool = True, logger: Logger = None, probability = 0.2):
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        self.stop_words = set(['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'i', 'if', 'in', 'into', 'is', 'isn', "isn't", "it's", 'its', 'itself', 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 'she', "she's", 'should', "should've", 'shouldn', "shouldn't", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', 'were', 'weren', "weren't", 'which', 'while', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'])
        self.aug = naw.SynonymAug(aug_src='wordnet')
        self.probability = probability
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
        self.data.question1 = self.data.question1.apply(lambda x: self.text_to_word_list(x))
        self.data.question2 = self.data.question2.apply(lambda x: self.text_to_word_list(x))
    
    def words_to_ids(self):
        self.logger.info('Replacing all words in {} dataset with their ids...'.format(self.type))
        self.data.question1 = self.data.question1.apply(lambda x: list(map(lambda y: self.replace_words(y, self.reverse_vocab), x.split())))
        self.data.question2 = self.data.question2.apply(lambda x: list(map(lambda y: self.replace_words(y, self.reverse_vocab), x.split())))

    
    def text_to_word_list(self, text: str):
        ''' 
        Preprocess method from: https://github.com/eliorc/Medium/blob/master/MaLSTM.ipynb
        '''
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
        text = sub(r"'", " ", text)
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
        text = sub(r"$", " dollar ", text)
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

    def synonym_replacement(self, question):
        new_question = question.copy()
        n = int(len(new_question) * self.probability)
        for i in range(n):
            m = random.randint(0,len(new_question)-1)
            while new_question[m] in self.stop_words:
                m = random.randint(0,len(new_question)-1)
            augmented_texts = aug.augment(new_question[m])
            new_question[m] = augmented_texts
        return new_question


    def random_insertion(self, question):
        new_question = question.copy()
        n = int(len(new_question) * self.probability)
        
        for i in range(n):
            m = random.randint(0,len(new_question)-1)
            while new_question[m] in self.stop_words:
                m = random.randint(0,len(new_question)-1)
            augmented_texts = self.aug.augment(new_question[m])
            new_question.insert(random.randint(0,len(new_question)-1), augmented_texts)
        return new_question


    def random_swap(self, question):
        new_question = question.copy()
        n = int(len(new_question) * self.probability)
        for i in range(n):
            k = random.randint(0,len(new_question)-1)
            m = random.randint(0,len(new_question)-1)
            word_k = new_question[k]
            word_m = new_question[m]
            new_question[n] = word_m
            new_question[m] = word_k
        return new_question

    def random_deletion(self, question):
        new_question = []
        for word in question:
            if random.randint(0,1) > self.probability:
                new_question.append(word)
        return new_question