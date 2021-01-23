import os
import numpy as np

import torch 
import librosa
import timm
import pandas as pd

from tqdm import tqdm

data_dir = 'test/'
files = os.listdir(data_dir)
# df = pd.read_csv('train_tp.csv')

for file in tqdm(files):
    w, r = librosa.load(data_dir + file)
#     print(file)
    w = torch.from_numpy(w)
    torch.save(w, 'test_torch/' + file[:-5] + '.pt')