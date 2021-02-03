#!/usr/bin/env python
# coding: utf-8

import os
import random

import torch
import torchvision
from torch.utils import data
from PIL import Image

import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold


from model import Model
import pandas as pd
import numpy as np

import math

from pytorch_lightning.callbacks import ModelCheckpoint

import librosa

from audiomentations import AddGaussianSNR, AddBackgroundNoise

from config import CONFIG

class Dataset(data.Dataset):

    def __init__(self, data_dir, df, train=True):

        self.data_dir = data_dir
        self.df = df.groupby('recording_id').agg(lambda x: list(x)).reset_index()
        self.train = train
        
        self.augment = AddBackgroundNoise( sounds_path='train_32000/fp/', min_snr_in_db=0.1, max_snr_in_db=1, p=0.5)
        
        self.files = os.listdir('train_32000/fp/')
        
    def __len__(self):
        return len(self.df)
    
    def crop(self, y, sr, period, record, noise):
        
        no_bird = y
        
        len_y = len(y)
        effective_length = sr * period

        rint = np.random.randint(len(record['t_min']))
        # random sound slice 
        time_start = record['t_min'][rint] * sr
        time_end = record['t_max'][rint] * sr

        ################
        # Positioning sound slice
        center = np.round((time_start + time_end) / 2)
        beginning = center - effective_length / 2

        if beginning < 0:
            beginning = 0

        beginning = np.random.randint(beginning, center)
        ending = beginning + effective_length

        if ending > len_y:
            ending = len_y

        beginning = ending - effective_length
        y = y[beginning:ending]
        ################


        beginning_time = beginning / sr
        ending_time = ending / sr

        label = torch.zeros(24)

        for i in range(len(record['t_min'])):
            if (record['t_min'][i] <= ending_time) and (record['t_max'][i] >= beginning_time):
                label[record['species_id'][i]] = 1
        
        return y, label
        
#         ######################
#         noise_start = np.random.randint(0, 50*sr)
#         noise = noise[noise_start:noise_start + effective_length]
#         ######################
        
# #         print(len(no_bird))
        
#         bird = np.random.choice([0,1], 1, p=[0.75, 0.25])

#         if bird == 0:
#             return y, label

#         return noise, torch.zeros(24)
        
        
    
    def __getitem__(self, index):
        
        r = self.df.iloc[index]
        recording_id = r['recording_id']
        
        noise = self.files[index]
        w_noise, sr = librosa.load('train_32000/fp/' + noise, CONFIG.sr)
        w_noise = torch.from_numpy(w_noise)
###############  audio
        
        w, sr = librosa.load(self.data_dir + recording_id + '.flac', CONFIG.sr)
        
#         if self.train:
#             w = self.augment(w, CONFIG.sr)    
        
        x = torch.from_numpy(w)
        

        if self.train:
            x_cut, label = self.crop(x, sr, CONFIG.period, r, w_noise)

            return x_cut, label
        
        label = torch.zeros(24)
                
        for i in r['species_id']:
            label[int(i)] = 1
                                                          
        return x, label
    
df = pd.read_csv('train_tp_folds.csv')

# folds = pd.read_csv('train_folds.csv')

# train_df, val_df = train_test_split(df, test_size=0.2)
    
batch_size = CONFIG.batch_size
num_workers = 32
# accumulate = 0
epochs = CONFIG.epochs

for valid_fold_id in range(5):
    
    checkpoint_dir = 'checkpoints/fold_' + str(valid_fold_id)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    val_acc_callback = ModelCheckpoint(
            monitor='val_acc', 
            dirpath=checkpoint_dir,
            filename='{epoch:02d}-{val_acc:.4f}',
            save_last=True, 
            mode='max')

    train_df = df[df['kfold'] != valid_fold_id]
    val_df = df[df['kfold'] == valid_fold_id]

    train_dataset = Dataset('train_32000/train/', train_df)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True, 
        num_workers=num_workers, 
        pin_memory=True)

    val_dataset = Dataset('train_32000/train/', val_df, False)

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=True, 
        num_workers=num_workers, 
        pin_memory=True)


    # checkpoint_path = 'lightning_logs/version_16/checkpoints/val_loss=0.0303.ckpt'
    model = Model(epochs, len(train_loader))
    # model = Model.load_from_checkpoint(checkpoint_path=checkpoint_path)

    trainer = Trainer(
        gpus=[0], 
        accelerator='ddp',
        callbacks=[val_acc_callback], 
        max_epochs=epochs)

    # trainer.tune(model, train_loader)
    trainer.fit(model, train_loader, val_loader)


