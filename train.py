#!/usr/bin/env python
# coding: utf-8

import os
import random

import torch
import torchvision
from torch.utils import data
import cv2
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

class Dataset(data.Dataset):

    def __init__(self, data_dir, df, train=True):

        self.data_dir = data_dir
        self.df = df.groupby('recording_id').agg(lambda x: list(x)).reset_index()
        self.train = train

    def __len__(self):
        return len(self.df)
    
    def crop(self, y, sr, period, record):

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

#         rand_choice = np.random.choice([0,1], 1, p=[0.7, 0.3])

#         if rand_choice == 0:
#             return y, label

#         return no_sounds, torch.zeros(24)

#     def crop(self, x, sr, period, record):
        
#         l = torch.chunk(x, period)
#         chunk_id = np.random.randint(0,6)

#         x_cut = l[chunk_id]

#         beginning_time = chunk_id*10
#         ending_time = (chunk_id + 1)*10

#         label = torch.zeros(24)

#         for i in range(len(record['t_min'])):
#             if (record['t_min'][i] <= ending_time) and (record['t_max'][i] >= beginning_time):
#                 label[record['species_id'][i]] = 1

#         return x_cut, label
    
    def __getitem__(self, index):
        
        r = self.df.iloc[index]
        recording_id = r['recording_id']
        
###############  audio
        
        sr = 32000
#         x = torch.load(self.data_dir + recording_id + '.pt')
        w, sr = librosa.load(self.data_dir + recording_id + '.flac', sr)
        x = torch.from_numpy(w)
        
        period = 10

        if self.train:
            x_cut, label = self.crop(x, sr, period, r)

            return x_cut, label
        
        label = torch.zeros(24)
                
        for i in r['species_id']:
            label[int(i)] = 1
                                                          
        return x, label
    
df = pd.read_csv('train_tp_folds.csv')

# folds = pd.read_csv('train_folds.csv')

train_df = df[df['kfold'] != 0]
val_df = df[df['kfold'] == 0]

# train_df, val_df = train_test_split(df, test_size=0.2)
    
batch_size = 16
num_workers = 16
# accumulate = 0
epochs = 50

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


# checkpoint_dir = 'video/lightning_logs/version_0/checkpoints/epoch=6.ckpt'
model = Model(epochs, len(train_loader))

seed_everything(42)

val_acc_callback = ModelCheckpoint(
        monitor='val_acc', 
        filename='{val_acc:.4f}', 
        save_last=True, 
        mode='max')
    
val_loss_callback = ModelCheckpoint(
    monitor='val_loss', 
    filename='{val_loss:.4f}', 
    mode='min')

trainer = Trainer(
#         logger=None,
#     auto_lr_find=True, 
    gpus=[0], 
    accelerator='ddp',
#     accumulate_grad_batches=accumulate, 
    callbacks=[val_acc_callback, val_loss_callback], 
    max_epochs=epochs,
#     precision=16
)

# trainer.tune(model, train_loader)
trainer.fit(model, train_loader, val_loader)


