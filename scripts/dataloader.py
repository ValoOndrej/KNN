import torch
import os
import copy
import json
import numpy as np
import pandas as pd
import torchvision.transforms.functional as TF
from torch.nn.functional import normalize
import threading

from PIL import Image
from torch.utils.data import Dataset

from utils import *

class Questions(Dataset):
    def __init__(self, data_path=None):
        self.data = pd.read_csv(data_path, nrows=5000, sep='\t')
        self.torchs = convert_to_dataset_torch(self.data[["question1","question2"]], self.data[["is_duplicate"]])

    def __getitem__(self, index):
        data = self.torchs[index]
        label = self.torchs[index]
        return {'x': data.float(), 'y': label.long()}

    def __len__(self):
        return self.torchs.size()[0]