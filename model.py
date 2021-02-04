
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
# from panns_inference.models import Cnn14
# from audioset_tagging_cnn.pytorch.models import Wavegram_Logmel_Cnn14

import numpy as np

from torch.distributions.beta import Beta

from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

import timm
import math

from config import CONFIG
from transformer import MultiHead
from lwlrap import lwlrap

def do_mixup(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''

    batch_size = x.size()[0]
    
    lam = torch.FloatTensor([np.random.beta(alpha, alpha) for i in range(batch_size)]).type_as(x)
    lam.requires_grad = False
    lam = lam[:,None,None,None]

    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]

#     y_a, y_b = y, y[index]
    
    lam = lam.view(batch_size,1)
    
    y = lam * y + (1 - lam) * y[index,:]
#     y = y + y[index,:]
    
    return mixed_x, y

class AttBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super(AttBlock, self).__init__()
        
        self.att = nn.Conv1d(in_channels=n_in, out_channels=n_out, kernel_size=1, stride=1, padding=0, bias=True)
        self.cla = nn.Conv1d(in_channels=n_in, out_channels=n_out, kernel_size=1, stride=1, padding=0, bias=True)
         
    def forward(self, x):
        # x: (n_samples, n_in, n_time)
#         norm_att = torch.softmax(self.att(x) / math.sqrt(x.size(1)), dim=-1)

        norm_att = torch.softmax(torch.clamp(self.att(x), -10, 10), dim=-1)
#         norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = torch.sigmoid(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

# class AttBlock(nn.Module):
#     def __init__(self, in_features, classes_num):
#         super(AttBlock, self).__init__()

#         self.alpha = nn.Linear(in_features, 1)
        
#     def forward(self, x):
        
#         '''
#         x: (batch, time, features)
#         '''
        
#         dot_product = torch.sigmoid(self.alpha(x))
        
#         att_weights = dot_product / torch.sum(dot_product, dim=1).unsqueeze(2)
        
#         x = torch.sum(x*att_weights, dim=1)

#         return x 

# class AttBlock(nn.Module):
#     def __init__(self, in_features, classes_num):
#         super(AttBlock, self).__init__()

#         self.alpha = nn.Linear(in_features, 1)
#         self.beta = nn.Linear(in_features * 2, 1)
        
# #         self.decision_attention = nn.Linear(in_features, classes_num)
        
#         self.fc = nn.Linear(in_features * 2, classes_num)
        
#     def forward(self, x):
        
#         '''
#         x: (batch, time, features)
#         '''

#         a = torch.sigmoid(self.alpha(x))
#         a = a / torch.sum(a, dim=1).unsqueeze(2)
        
#         x_a = torch.sum(x*a, dim=1) # (batch, features)

#         x_a = x_a.unsqueeze(1).repeat(1,x.size(1),1) # prepare for stacking
        
#         x_b = torch.cat([x, x_a], dim=-1)
#         b = torch.sigmoid(self.beta(x_b))
        
#         ab = a * b
        
#         ab = ab / torch.sum(ab,dim=1).unsqueeze(2)
        
        
#         out = torch.sum(x_b*b, dim=1)
        
#         out = F.dropout(out, p=CONFIG.p, training=self.training)
        
#         out = self.fc(out)
# #         att_weights = torch.softmax(self.decision_attention(x) / math.sqrt(x.size(-1)), dim=1)        
# # #         x = F.dropout(x, p=0.5, training=self.training)
# #         x = self.fc(x)
        
# #         x = att_weights * x
        
#         return out


class Encoder_Default(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(Encoder_Default, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=4, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(mel_bins)
        
        
        fe = 1280
        fe_features = 2048
        
        self.fc1 = nn.Linear(fe, fe_features, bias=True)
            
        self.att_block = AttBlock(fe_features, classes_num)
    
#         self.fe = timm.models.resnest50d_4s2x40d(pretrained=True)
        self.fe = timm.models.tf_efficientnet_b0_ns(pretrained=True)
        self.fe = nn.Sequential(*list(self.fe.children())[:-2])  
        
#         self.fc = nn.Linear(fe_features, classes_num)
        
    def forward(self, x, y=None):
        """
        Input: (batch_size, data_length)"""
        

        x = self.spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        frames_num = x.shape[2]
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        alpha=1.0
        if self.training:
            x, y = do_mixup(x,y, alpha)

        x = torch.cat([x,x,x], dim=1)
                    
        x = self.fe(x)

        x = torch.mean(x, dim=3) # averaging across frequency dimension 
        
        stride = 1
        
        x1 = F.max_pool1d(x, kernel_size=3, stride=stride, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=stride, padding=1)
        x = x1 + x2
        
        x = F.dropout(x, p=CONFIG.p, training=self.training)
        x = x.transpose(1, 2)        
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)      
        x = F.dropout(x, p=CONFIG.p, training=self.training)
        
#         x = self.att_block(x)
#         x = self.fc(x)
        
#         if self.training:
#             return x, y
        
#         return x
        
        clipwise, weights, framewise = self.att_block(x)
        
        if self.training:
            
            return clipwise, y
        
        return torch.max(framewise, dim=-1)[0]


class Encoder_Transformer(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(Encoder_Transformer, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=4, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(mel_bins)
        
        
        fe = 1280
        fe_features = 1280
        
        self.fc1 = nn.Linear(fe, fe_features, bias=True)
        
        n_head = 4
        d_k = 64
        d_v = 64
        transformer_dropout = 0.2
        
        self.multihead = MultiHead(n_head, fe, d_k, d_v, transformer_dropout)
        
        self.att_block = AttBlock(fe_features, classes_num)
        
#         self.fe = timm.models.resnest50d_4s2x40d(pretrained=True)
        self.fe = timm.models.tf_efficientnet_b0_ns(pretrained=True)
        self.fe = nn.Sequential(*list(self.fe.children())[:-2])  
        
        self.fc = nn.Linear(fe_features, classes_num)
        
    def forward(self, x, y=None):
        """
        Input: (batch_size, data_length)"""
        

        x = self.spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        frames_num = x.shape[2]
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        alpha=1.0
        if self.training:
            x, y = do_mixup(x,y, alpha)

        x = torch.cat([x,x,x], dim=1)
                    
        x = self.fe(x)

        x = torch.mean(x, dim=3) # averaging across frequency dimension 
        
        stride = 1
        
        x1 = F.max_pool1d(x, kernel_size=3, stride=stride, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=stride, padding=1)
        x = x1 + x2
        
        
        x = F.dropout(x, p=CONFIG.p, training=self.training)
        x = x.transpose(1, 2)     # (batch_size, time, features)   
        x = self.multihead(x,x,x)
        x = x.transpose(1, 2)
        x = F.dropout(x, p=CONFIG.p, training=self.training)

        
        clipwise, weights, framewise = self.att_block(x)
        
        if self.training:
            
            return clipwise, y
        
        return torch.max(framewise, dim=-1)[0]

    
class Model(pl.LightningModule):
    
    def __init__(self, epochs=10, train_loader_len=100):
        super(Model, self).__init__()
         
        self.epochs = epochs
        self.learning_rate = 0.001
        self.train_loader_len = train_loader_len
        
        self.encoder = Encoder_Transformer(sample_rate=CONFIG.sr, 
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
    
    
    def validation_step(self, batch, batch_nb):
        
        X, y = batch

        X = X.unfold(1, CONFIG.sr * CONFIG.period, int(CONFIG.sr * CONFIG.skip))
        
        Y_hat = []
        
        for i in range(X.size(1)):
            
            x = X[:,i,:]
            
            y_hat = self.forward(x) # (batch_size, classes, time)
            
            Y_hat.append(y_hat)
            
        Y_hat = torch.stack(Y_hat, dim=1)
               
        (y_hat,_) = torch.max(Y_hat, dim=1)
       
        return {'y_hat': y_hat, 'y': y}
    
    def validation_epoch_end(self, outputs):

        y_hat = torch.cat([x['y_hat'] for x in outputs], dim=0)
        y = torch.cat([x['y'] for x in outputs], dim=0)
        
#         lwlrap_score = self.LWLRAP(y_hat, y)
                
        lwlrap_score = lwlrap(y_hat.detach().cpu().numpy(), y.detach().cpu().numpy())
        
        self.log('val_acc', lwlrap_score, prog_bar=True)
                    
#         self.log('lwlrap', lwlrap2, prog_bar=True)

    
    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None, on_tpu=False, using_native_amp=True, using_lbfgs=False):
                
        optimizer.step(closure=second_order_closure)

        self.scheduler.step(current_epoch + batch_nb / self.train_loader_len)
        
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=CONFIG.l2)
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CONFIG.epochs, T_mult=1, eta_min=0, last_epoch=-1, verbose=True)
                
        return optimizer
    
    def focal_loss(self, preds, targets):
        
        loss_fct = nn.BCEWithLogitsLoss(reduction='none')
        bce_loss = loss_fct(preds, targets)
        probas = torch.sigmoid(preds)
        loss = torch.where(targets >= 0.5, (1. - probas)**self.gamma * bce_loss, probas**self.gamma * bce_loss)
        loss = loss.mean()
        
        return loss
    
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
