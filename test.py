import os
import numpy as np


import torch 
import librosa
import timm
import pandas as pd
import torch.nn as nn

from tqdm import tqdm

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from torch.nn import functional as F

from model import Model

from torch.utils import data
from torch.utils.data import DataLoader

class TestDataset(data.Dataset):

    def __init__(self, data_dir):

        self.data_dir = data_dir
        self.files = os.listdir(data_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        
        file = self.files[index]
            
        w = torch.load(self.data_dir + file)
        
        return w, file[:-3]
    
test_dataset = TestDataset('test_torch/')

batch_size = 16
num_workers = 16

test_loader = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    drop_last=False, 
    num_workers=num_workers, 
    pin_memory=True)


sub_df = pd.read_csv('sample_submission.csv')
sub_df = sub_df.set_index('recording_id')

device = 'cuda:3'

model = Model.load_from_checkpoint(checkpoint_path="lightning_logs/version_20/checkpoints/val_loss=0.0801.ckpt")
model.eval()
model.to(device)

for W,F in tqdm(test_loader):
    
    with torch.no_grad():
        
#         l = torch.chunk(W.to(device),6, dim=1)
#         Y_hat = []
#         for x in l:
#             y_hat = model(x)
#             Y_hat.append(y_hat)
            
#         (y_hat,_) = torch.max(torch.stack(Y_hat, dim=1), dim=1)
        
        y_hat = model(W.to(device))
        
        for index, image_id in enumerate(F): 
            
            l = y_hat[index,:].tolist()
            l = list(map(lambda x: round(x,4), l))
            sub_df.loc[image_id]= l
            
# data_dir = 'test_torch/'
# files = os.listdir(data_dir)

# for file in tqdm(files):
#     w = torch.load(data_dir + file)
    
#     file_id = file[:-3]
    
#     y_hat = model(w.unsqueeze(0).to(device))
#     l = y_hat[0,:].tolist()
#     l = list(map(lambda x: round(x,4), l))
    
#     sub_df.loc[file_id] = l

# sub_df
sub_df.to_csv('submission.csv')


