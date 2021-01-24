
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

def do_mixup(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]

    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]

    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)
    
def interpolate(x: torch.Tensor, ratio: int):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    pad = framewise_output[:, -1:, :].repeat(
        1, frames_num - framewise_output.shape[1], 1)
    """tensor for padding"""

    output = torch.cat((framewise_output, pad), dim=1)
    """(batch_size, frames_num, classes_num)"""

    return output

# class AttBlock(nn.Module):
#     def __init__(self,
#                  in_features: int,
#                  out_features: int,
#                  activation="linear",
#                  temperature=1.0):
#         super().__init__()

#         self.activation = activation
#         self.temperature = temperature
        
#         self.att = nn.Conv1d(
#             in_channels=in_features,
#             out_channels=out_features,
#             kernel_size=1,
#             stride=1,
#             padding=0,
#             bias=False)
        
#         self.cla = nn.Conv1d(
#             in_channels=in_features,
#             out_channels=out_features,
#             kernel_size=1,
#             stride=1,
#             padding=0,
#             bias=True)

        
#         self.bn_att = nn.BatchNorm1d(out_features)
#         self.init_weights()
        
#     def init_weights(self):
#         init_layer(self.att)
#         init_layer(self.cla)
#         init_bn(self.bn_att)

#     def forward(self, x):
        
#         norm_att = torch.softmax(torch.clamp(self.att(x), -10, 10), dim=-1)    

#         cla = torch.sigmoid(self.cla(x))
       
#         x = torch.sum(norm_att * cla, dim=2)
# #         x = torch.clamp(x,0.0,1.0)
        
#         return x, norm_att, cla

    
# class Encoder(nn.Module):
#     def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
#         fmax, classes_num):
        
#         super(Encoder, self).__init__()

#         window = 'hann'
#         center = True
#         pad_mode = 'reflect'
#         ref = 1.0
#         amin = 1e-10
#         top_db = None

#         # Spectrogram extractor
#         self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
#             win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
#             freeze_parameters=True)

#         # Logmel feature extractor
#         self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
#             n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
#             freeze_parameters=True)

#         # Spec augmenter
#         self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
#             freq_drop_width=8, freq_stripes_num=2)

#         self.bn0 = nn.BatchNorm2d(mel_bins)
        
#         self.fc1 = nn.Linear(1792, 1792, bias=True)
# #         self.fc_final = nn.Linear(1280, classes_num, bias=True)
#         self.att_block = AttBlock(1792, classes_num, activation='sigmoid')
        
# #         self.fe = timm.models.resnest50d_4s2x40d(pretrained=True)
#         self.fe = timm.models.tf_efficientnet_b4_ns(pretrained=True)
#         self.fe = nn.Sequential(*list(self.fe.children())[:-2])

 
#     def forward(self, input, y=None):
#         """
#         Input: (batch_size, data_length)"""

#         x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
#         x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

#         frames_num = x.shape[2]
        
#         x = x.transpose(1, 3)
#         x = self.bn0(x)
#         x = x.transpose(1, 3)
        
#         if self.training:
#             x = self.spec_augmenter(x)

#         # Mixup on spectrogram
#         alpha = 0.5
#         if self.training:
#             x, y_a, y_b, lam = do_mixup(x,y, alpha)

#         x = torch.cat([x,x,x], dim=1)
#         x = self.fe(x)
# #         print(x.shape)
#         x = torch.mean(x, dim=3)
        
#         x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
#         x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
#         x = x1 + x2
        
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = x.transpose(1, 2)
#         x = F.relu_(self.fc1(x))
#         x = x.transpose(1, 2)
#         x = F.dropout(x, p=0.5, training=self.training)
        
#         (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        
#         segmentwise_output = segmentwise_output.transpose(1, 2)

# #         # Get framewise output
# #         framewise_output = interpolate(segmentwise_output, 32)
# #         framewise_output = pad_framewise_output(framewise_output, frames_num)

#         if self.training:
#             return clipwise_output, y_a, y_b, lam
        
#         return segmentwise_output


class AttBlock(nn.Module):
    def __init__(self, in_features: int):
        super(AttBlock, self).__init__()

        self.linear = nn.Linear(in_features, 1, bias=False)
        
    def forward(self, x):
        
        '''
        x: (batch, time, features)
        '''
        
        dot_product = torch.sigmoid(self.linear(x))
        
        att_weights = dot_product / torch.sum(dot_product, dim=1).unsqueeze(2)
        
        x = torch.sum(x*att_weights, dim=1)

        return x 

class Encoder(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(Encoder, self).__init__()

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
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(mel_bins)
        
        fe = 1280
        fe_features = 2048
#         fe_features = 1280
        
        self.fc1 = nn.Linear(fe, fe_features, bias=True)
        
        self.att_block = AttBlock(fe_features)
        
#         self.fe = timm.models.resnest50d_4s2x40d(pretrained=True)
        self.fe = timm.models.tf_efficientnet_b0_ns(pretrained=True)
        self.fe = nn.Sequential(*list(self.fe.children())[:-2])  
        
        self.fc = nn.Linear(fe_features, classes_num)
 
    def forward(self, input, y=None):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        frames_num = x.shape[2]
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        alpha = 1.0
        if self.training:
            x, y_a, y_b, lam = do_mixup(x,y, alpha)

        x = torch.cat([x,x,x], dim=1)
        x = self.fe(x)
#         print(x.shape)
        x = torch.mean(x, dim=3)

        
        stride = 1
        
        x1 = F.max_pool1d(x, kernel_size=3, stride=stride, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=stride, padding=1)
        x = x1 + x2
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))

        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.att_block(x)
        
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.fc(x)

        x = torch.sigmoid(x)
        
#         x = torch.clamp(x,0,1)
        
        if self.training:
            return x, y_a, y_b, lam
        
        return x
    

class Model(pl.LightningModule):
    
    def __init__(self, epochs=10, train_loader_len=100):
        super(Model, self).__init__()
        
        self.epochs = epochs
        self.learning_rate = 0.001
        self.train_loader_len = train_loader_len
        
        self.sr = 32000
        
        self.encoder = Encoder(sample_rate=self.sr, 
            window_size=2048,
            hop_size=512,
            mel_bins=328,
            fmin=50,
            fmax=self.sr // 2,
            classes_num=24)
             
    def forward(self, x, y=None):
        x = self.encoder(x, y)
        return x
    
    def training_step(self, batch, batch_nb):
        X, y = batch
        
#         print(X.shape)
#         y_hat = self.forward(X)
        
        y_hat, y_a, y_b, lam = self.forward(X, y)
        
        loss = lam * F.binary_cross_entropy(y_hat, y_a) + (1 - lam) * F.binary_cross_entropy(y_hat, y_b)

#         loss = F.binary_cross_entropy(y_hat, y)
    
        self.log('train_loss', loss)
        return loss
    
    
    def validation_step(self, batch, batch_nb):
        X, y = batch

#         y_hat = self.forward(X)
       

        X = X.unfold(1, self.sr * 10, self.sr * 5)
        
#         print(X.shape)
        Y_hat = []
        for i in range(X.size(1)):
            y_hat = self.forward(X[:,i,:])
            Y_hat.append(y_hat)
            
        (y_hat,_) = torch.max(torch.stack(Y_hat, dim=1), dim=1)
        
        loss = F.binary_cross_entropy(y_hat, y)
#         print(y_hat, y)
        return {'val_loss': loss, 'y_hat': y_hat.detach(), 'y': y.detach()}
    
    def validation_epoch_end(self, outputs):
        
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        
        y_hat = torch.cat([x['y_hat'] for x in outputs], dim=0)
        y = torch.cat([x['y'] for x in outputs], dim=0)
        
        lwlrap = self.LWLRAP(y_hat, y)
        
        self.log('val_loss', avg_loss,prog_bar=True)
        self.log('val_acc', lwlrap, prog_bar=True)
    
    
    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None, on_tpu=False, using_native_amp=True, using_lbfgs=False):
        
        optimizer.step(closure=second_order_closure)
#         optimizer.zero_grad()
        
        self.scheduler.step(current_epoch + batch_nb / self.train_loader_len // 4)

        
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.train_loader_len // 4, T_mult=1, eta_min=1e-6, last_epoch=-1)
        
#         scheduler = {
#             'scheduler': scheduler,
#             'interval': 'step'
#         }
        
        return optimizer
                                                                 
    def LWLRAP(self, preds, labels):
        # Ranks of the predictions
        ranked_classes = torch.argsort(preds, dim=-1, descending=True)
        # i, j corresponds to rank of prediction in row i
        class_ranks = torch.zeros_like(ranked_classes).type_as(labels)
        
        for i in range(ranked_classes.size(0)):
            for j in range(ranked_classes.size(1)):
                class_ranks[i, ranked_classes[i][j]] = float(j) + 1.0
                
        # Mask out to only use the ranks of relevant GT labels
        ground_truth_ranks = class_ranks * labels + (1e6) * (1 - labels)
        # All the GT ranks are in front now
        
        sorted_ground_truth_ranks, _ = torch.sort(ground_truth_ranks, dim=-1, descending=False)
        # Number of GT labels per instance
        num_labels = labels.sum(-1)
        
        pos_matrix = torch.tensor(np.array([i+1 for i in range(labels.size(-1))])).unsqueeze(0).type_as(labels)
        
        score_matrix = pos_matrix / sorted_ground_truth_ranks
        
        score_mask_matrix, _ = torch.sort(labels, dim=-1, descending=True)
        
        scores = score_matrix * score_mask_matrix
        score = scores.sum() / labels.sum()
        return score


    
