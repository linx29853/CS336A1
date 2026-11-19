import torch
import math
from cs336_basics.softmax import softmax

def cross_attention(query, key, value, mask=None):
    '''Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V'''
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask==False, float("-inf"))
    attention = torch.matmul(softmax(scores, dim=-1), value)
    return attention
