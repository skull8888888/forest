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
from config import CONFIG
import glob



class TestDataset(data.Dataset):

    def __init__(self, data_dir):

        self.data_dir = data_dir
        self.files = os.listdir(data_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        
        file = self.files[index]
            
        sr = 32000
#         w = torch.load(self.data_dir + file)
        
        w, sr = librosa.load(self.data_dir + file, sr)
        w = torch.from_numpy(w)
        
        
        return w, file[:-5]
    
test_dataset = TestDataset('train_32000/test/')

batch_size = CONFIG.batch_size
num_workers = 32

test_loader = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    drop_last=False, 
    num_workers=num_workers, 
    pin_memory=True)


sub_df = pd.read_csv('sample_submission.csv')
sub_df = sub_df.set_index('recording_id')

device = 'cuda:0'

# dfs = []

# from multiprocessing import Process

# id_dict = {}

# def predict(fold_id, id_dict):
    
#     ckpt = glob.glob(f'checkpoints/fold_{fold_id}/epoch*.ckpt')[0]
    
#     model = Model.load_from_checkpoint(checkpoint_path=ckpt)
#     model = model.eval()
#     model.to(device)
    
#     for X,F in tqdm(test_loader):

#         with torch.no_grad():

#             X = X.unfold(1, CONFIG.sr * CONFIG.period, CONFIG.sr * CONFIG.skip)

#             Y_hat = []

#             for i in range(X.size(1)):

#                 y = model(X[:,i,:].to(device))
#                 Y_hat.append(y)

#             (y_hat,_) = torch.max(torch.stack(Y_hat, dim=1), dim=1)

#             for index, image_id in enumerate(F): 

#                 l = y_hat[index,:].tolist()
                
#                 if not image_id in id_dict:
#                     id_dict[image_id] = []
                    
#                 id_dict[image_id].append(l)

# processes = []
                
# for fold_id in range(5):
#     print('predicting for fold ', fold_id)
#     predict(fold_id, id_dict)
    
# for k, v in id_dict.items():
#     l = np.array(v).max(0)
#     l = list(map(lambda x: round(x,4), l))
#     l = l.tolist()
    
#     sub_df.loc[k]= l

# sub_df.to_csv('submission.csv')
    
models = nn.ModuleList()

for fold_id in range(5):
    
    ckpt = glob.glob(f'checkpoints_903/fold_{fold_id}/epoch*.ckpt')[0]
    
    model = Model.load_from_checkpoint(checkpoint_path=ckpt)
    model = model.eval()
    model.to(device)
    
    models.append(model)
    

for X,F in tqdm(test_loader):
    
    with torch.no_grad():
        
        X = X.unfold(1, CONFIG.sr * CONFIG.period, CONFIG.sr * CONFIG.skip)
        
        Y_hat = []
        
        for i in range(X.size(1)):
            
            y_step = []

            for model in models:
                y = model(X[:,i,:].to(device))
                y_step.append(y)
            
            y_step = torch.mean(torch.stack(y_step, dim=1), dim=1)
            Y_hat.append(y_step)
            
        (y_hat,_) = torch.max(torch.stack(Y_hat, dim=1), dim=1)
        
        for index, image_id in enumerate(F): 
            
            l = y_hat[index,:].tolist()
            l = list(map(lambda x: round(x,4), l))
            sub_df.loc[image_id]= l

sub_df.to_csv('submission.csv')


