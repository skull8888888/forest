import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from collections import OrderedDict

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    
    
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 2)),
            ("gelu", QuickGELU()),
#             ("relu", nn.ReLU()),
            ("c_proj", nn.Linear(d_model * 2, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
#         self.ln_3 = nn.LayerNorm(d_model)
        
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        
#         x = self.ln_3(x)
        return x

# class ScaledDotProductAttention(nn.Module):
#     """Scaled Dot-Product Attention"""

#     def __init__(self, temperature, attn_dropout=0.1):
#         super().__init__()
#         self.temperature = temperature
#         self.dropout = nn.Dropout(attn_dropout)
#         self.softmax = nn.Softmax(dim=2)

#     def forward(self, q, k, v, mask=None):

#         attn = torch.bmm(q, k.transpose(1, 2))
#         attn = attn / self.temperature

#         if mask is not None:
#             attn = attn.masked_fill(mask, -np.inf)

#         attn = self.softmax(attn)
#         attn = self.dropout(attn)
#         output = torch.bmm(attn, v)

#         return output, attn


# class MultiHead(nn.Module):
#     """Multi-Head Attention module."""

#     def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
#         super().__init__()

#         self.n_head = n_head
#         self.d_k = d_k
#         self.d_v = d_v

#         self.w_qs = nn.Linear(d_model, n_head * d_k)
#         self.w_ks = nn.Linear(d_model, n_head * d_k)
#         self.w_vs = nn.Linear(d_model, n_head * d_v)
#         nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
#         nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
#         nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
#         self.w_qs.bias.data.fill_(0)
#         self.w_ks.bias.data.fill_(0)
#         self.w_vs.bias.data.fill_(0)

#         self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
#         self.layer_norm = nn.LayerNorm(d_model)

#         self.fc = nn.Linear(n_head * d_v, d_model)
#         nn.init.xavier_normal_(self.fc.weight)
#         self.fc.bias.data.fill_(0)

#         self.dropout = nn.Dropout(dropout)


#     def forward(self, q, k, v, mask=None):

#         d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

#         sz_b, len_q, _ = q.size()   # (batch_size, 80, 512)
#         sz_b, len_k, _ = k.size()
#         sz_b, len_v, _ = v.size()

#         residual = q

#         q = self.w_qs(q).view(sz_b, len_q, n_head, d_k) # (batch_size, T, 8, 64)
#         k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
#         v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        
#         # transpose so that each head is treated as a batch
#         q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk, (batch_size*8, T, 64)
#         k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
#         v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

#         # mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
#         output, attn = self.attention(q, k, v, mask=mask)   # (n_head * batch_size, T, 64), (n_head * batch_size, T, T)
        
#         output = output.view(n_head, sz_b, len_q, d_v)  # (n_head, batch_size, T, 64)

#         # reshaping equaivalent to concat in feature dimension (-1)
#         output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv), (batch_size, T, 512)
        
#         # final fc to project back to the original dimension space
#         output = F.relu_(self.dropout(self.fc(output)))
#         return output
    
    
    