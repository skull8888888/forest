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


def do_mixup(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''

    batch_size = x.size()[0]
    
    lam = torch.FloatTensor([np.random.beta(alpha, alpha) for i in range(batch_size)]).type_as(x)
    lam.requires_grad = False
    lam = lam[:,None,None,None]

    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]

    lam = lam.view(batch_size,1)
    
    y = lam * y + (1 - lam) * y[index,:]
    
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


class CustomAttBlock(nn.Module):
    def __init__(self, in_features, classes_num):
        super(CustomAttBlock, self).__init__()

        self.alpha = nn.Linear(in_features, 1)
        
    def forward(self, x):
        
        '''
        x: (batch, time, features)
        '''
        
        dot_product = torch.sigmoid(self.alpha(x))
        
        att_weights = dot_product / torch.sum(dot_product, dim=1).unsqueeze(2)
        
        x = torch.sum(x*att_weights, dim=1)

        return x 

    
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
        x = F.relu_(self.fc1(x)) #(bs, time, features)
        x = x.transpose(1, 2) #(bs, features, time) 
        x = F.dropout(x, p=CONFIG.p, training=self.training)
        
        clipwise, weights, framewise = self.att_block(x)
        
        if self.training:
            
            return clipwise, y
        
        return torch.max(framewise, dim=-1)[0]

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
        
        clipwise, weights, framewise = self.att_block(x)
        
        if self.training:
            
            return clipwise, y
        
        return torch.max(framewise, dim=-1)[0]
    
class Encoder_B0_1024(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(Encoder_B0_1024, self).__init__()

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
        fe_features = 1024
        
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
        
        clipwise, weights, framewise = self.att_block(x)
        
        if self.training:
            
            return clipwise, y
        
        return torch.max(framewise, dim=-1)[0]
    
    
class Encoder_B0_Pretrained(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(Encoder_B0_Pretrained, self).__init__()

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
        self.bn0.load_state_dict(torch.load('pretrained_bn0_b0'))
        for p in self.bn0.parameters():
            p.requires_grad = False

        fe = 1280
        fe_features = 2048
        
        self.fc1 = nn.Linear(fe, fe_features, bias=True)
        self.fc1.load_state_dict(torch.load('pretrained_fc1_b0_fold_0'))
        for p in self.fc1.parameters():
            p.requires_grad = False
        
        self.att_block = AttBlock(fe_features, classes_num)
    
#         self.fe = timm.models.resnest50d_4s2x40d(pretrained=True)
        self.fe = timm.models.tf_efficientnet_b0_ns(pretrained=False)
        self.fe = nn.Sequential(*list(self.fe.children())[:-2])  
        
        self.fe.load_state_dict(torch.load('pretrained_fe_b0_fold_0'))
        for p in self.fe.parameters():
            p.requires_grad = False

        
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

#         Mixup on spectrogram
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
#         x = F.normalize(x, dim=1)
        x = F.dropout(x, p=CONFIG.p, training=self.training)
        
        clipwise, weights, framewise = self.att_block(x)
        
        if self.training:
            
            return clipwise, y
        
        return torch.max(framewise, dim=-1)[0]


    
class Encoder_B4(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(Encoder_B4, self).__init__()

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
        
        
        fe = 1792
        fe_features = 2048
        
        self.fc1 = nn.Linear(fe, fe_features, bias=True)
            
        self.att_block = AttBlock(fe_features, classes_num)
    
#         self.fe = timm.models.resnest50d_4s2x40d(pretrained=True)
        self.fe = timm.models.tf_efficientnet_b4_ns(pretrained=True)
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
        d_model = 512
        
        self.fc1 = nn.Linear(fe, d_model, bias=True)
        
        n_head = 4
        
        self.multihead = ResidualAttentionBlock(d_model, n_head)
        
        self.att_block = AttBlock(d_model, classes_num)
        
#         self.fe = timm.models.resnest50d_4s2x40d(pretrained=True)
        self.fe = timm.models.tf_efficientnet_b0_ns(pretrained=True)
        self.fe = nn.Sequential(*list(self.fe.children())[:-2])  
        
        self.pos_emb = nn.Embedding(32, d_model)
        
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
        
        x = F.dropout(x, p=CONFIG.p, training=self.training) # (batch_size, features, time)
        x = x.transpose(1, 2) # (batch_size, time, features)
        x = F.relu(self.fc1(x))
        
        pos = self.pos_emb(torch.arange(32).type_as(x).long()).unsqueeze(0)
        x = x + pos
        
        x = F.dropout(x, p=CONFIG.p, training=self.training)
        x = x.permute(1,0,2)     # (time, batch_size, features)   
        x = self.multihead(x)
        x = x.permute(1,2,0) # (batch_size, time, features)
        x = F.dropout(x, p=CONFIG.p, training=self.training)
        
#         print(x.shape)
        
        clipwise, weights, framewise = self.att_block(x)
        
        if self.training:
            
            return clipwise, y
        
        return torch.max(framewise, dim=-1)[0]

    