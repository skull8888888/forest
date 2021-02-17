import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pytorch_lightning.metrics.functional as plm
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import numpy as np
from torch.distributions.beta import Beta
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
import timm
import math
from config import CONFIG
from transformer import ResidualAttentionBlock
from lwlrap import lwlrap
from sklearn.metrics import f1_score, precision_score, recall_score

from models import Encoder_Default


class Model(pl.LightningModule):
    
    def __init__(self, epochs=10, train_loader_len=100):
        super(Model, self).__init__()
         
        self.epochs = epochs
        self.learning_rate = CONFIG.lr
        self.train_loader_len = train_loader_len
        
        self.encoder = Encoder_Default(sample_rate=CONFIG.sr, 
            window_size=CONFIG.window_size,
            hop_size=CONFIG.hop_size,
            mel_bins=CONFIG.mel_bins,
            fmin=CONFIG.fmin,
            fmax=CONFIG.fmax,
            classes_num=24)
         
             
    def forward(self, x, y=None):
        x = self.encoder(x, y)
        return x
    
    def training_step(self, batch, batch_nb):
        
        X, y = batch
        
        y_hat, y = self.forward(X, y)
        
        loss = F.binary_cross_entropy(y_hat, y)
        
        lr = self.scheduler.get_last_lr()[0]
        self.log('lr', lr, prog_bar=True)

        return loss
    
    
#     def validation_step(self, batch, batch_nb):
        
#         X, y = batch

#         X = X.unfold(1, CONFIG.sr * CONFIG.period, int(CONFIG.sr * CONFIG.skip))
        
#         Y_hat = []
        
#         for i in range(X.size(1)):
            
#             x = X[:,i,:]
            
#             y_hat = self.forward(x) # (batch_size, classes, time)
            
#             Y_hat.append(y_hat)
            
#         Y_hat = torch.stack(Y_hat, dim=1)
       
#         (y_hat,_) = torch.max(Y_hat, dim=1)
       
#         return {'y_hat': y_hat, 'y': y}

    def validation_step(self, batch, batch_nb):
        
        X, y = batch
        
        Y_hat = []
        
        for i in range(X.size(1)):
            
            x = X[:,i,:]
            
            y_hat = self.forward(x) # (batch_size, classes)
            
            Y_hat.append(y_hat)
            
        y_hat = torch.mean(torch.stack(Y_hat, dim=1), dim=1)
       
        return {'y_hat': y_hat, 'y': y}
    
    
    def validation_epoch_end(self, outputs):

        y_hat = torch.cat([x['y_hat'] for x in outputs], dim=0).detach().cpu()
        
        threshold = 0.5
        
        y_hat[y_hat >= threshold] = 1
        y_hat[y_hat < threshold] = 0
        
        y = torch.cat([x['y'] for x in outputs], dim=0).detach().cpu()
                        
        f1 = f1_score(y, y_hat, average='weighted')
#         pre = precision_score(y, y_hat, average='weighted')
#         rec = recall_score(y, y_hat, average='weighted')
        
        self.log('f1', f1, prog_bar=True)
#         self.log('pre', pre, prog_bar=True)
#         self.log('rec', rec, prog_bar=True)
                
    
    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None, on_tpu=False, using_native_amp=True, using_lbfgs=False):
                
        optimizer.step(closure=second_order_closure)

        self.scheduler.step(current_epoch + batch_nb / self.train_loader_len)
        
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=CONFIG.l2)
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CONFIG.epochs, T_mult=1, eta_min=0, last_epoch=-1, verbose=True)
                
        return optimizer
        
#     def LWLRAP(self, preds, labels):
#         # Ranks of the predictions
#         ranked_classes = torch.argsort(preds, dim=-1, descending=True)
#         # i, j corresponds to rank of prediction in row i
#         class_ranks = torch.zeros_like(ranked_classes).type_as(labels)
        
#         for i in range(ranked_classes.size(0)):
#             for j in range(ranked_classes.size(1)):
#                 class_ranks[i, ranked_classes[i][j]] = float(j) + 1.0
                
#         # Mask out to only use the ranks of relevant GT labels
#         ground_truth_ranks = class_ranks * labels + (1e6) * (1 - labels)
#         # All the GT ranks are in front now
        
#         sorted_ground_truth_ranks, _ = torch.sort(ground_truth_ranks, dim=-1, descending=False)
#         # Number of GT labels per instance
#         num_labels = labels.sum(-1)
        
#         pos_matrix = torch.tensor(np.array([i+1 for i in range(labels.size(-1))])).unsqueeze(0).type_as(labels)
        
#         score_matrix = pos_matrix / sorted_ground_truth_ranks
        
#         score_mask_matrix, _ = torch.sort(labels, dim=-1, descending=True)
        
#         scores = score_matrix * score_mask_matrix
#         score = scores.sum() / labels.sum()
#         return score
