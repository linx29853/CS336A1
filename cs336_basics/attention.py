import torch
import math
from torch import nn
from cs336_basics.softmax import softmax

def cross_attention(query, key, value, mask=None):
    '''Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V'''
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask==False, float("-inf"))
    attention = torch.matmul(softmax(scores, dim=-1), value)
    return attention

class MultiHeadSelfAttention(nn.Module):     
    def _init_weights(self):
        sigma = math.sqrt(1 / self.d_model)

        nn.init.trunc_normal_(
            self.W_Q, mean=0, std=sigma, a=-3 * sigma, b=3 * sigma
        )
        nn.init.trunc_normal_(
            self.W_K, mean=0, std=sigma, a=-3 * sigma, b=3 * sigma
        )
        nn.init.trunc_normal_(
            self.W_V, mean=0, std=sigma, a=-3 * sigma, b=3 * sigma
        )
        nn.init.trunc_normal_(
            self.W_O, mean=0, std=sigma, a=-3 * sigma, b=3 * sigma
        )

    def __init__(self, d_model: int, num_heads: int):         
        super().__init__()
        assert num_heads > 0, "注意力头数必须大于0"
        assert d_model % num_heads == 0, "特征维度必须能被注意力头数整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        W_Q_tensor = torch.empty(d_model, d_model)
        W_K_tensor = torch.empty(d_model, d_model)
        W_V_tensor = torch.empty(d_model, d_model)
        W_O_tensor = torch.empty(d_model, d_model)

        self.W_Q = nn.Parameter(W_Q_tensor)
        self.W_K = nn.Parameter(W_K_tensor)
        self.W_V = nn.Parameter(W_V_tensor)
        self.W_O = nn.Parameter(W_O_tensor)

        self._init_weights()

    def forward(self, x, mask=None):
        batch_shape = x.shape[:-2]
        seq_len = x.shape[-2]
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool().to(x.device)  # 下三角掩码
        Q = torch.matmul(x, self.W_Q.T)
        K = torch.matmul(x, self.W_K.T)
        V = torch.matmul(x, self.W_V.T)

        q = Q.reshape(
            *batch_shape, seq_len, self.num_heads, self.head_dim
        ).transpose(-3, -2)

        k = K.reshape(
            *batch_shape, seq_len, self.num_heads, self.head_dim
        ).transpose(-3, -2)

        v = V.reshape(
            *batch_shape, seq_len, self.num_heads, self.head_dim
        ).transpose(-3, -2)

        if mask is None:
            mask = causal_mask
        else:
            mask = mask & causal_mask

        attention_output = cross_attention(q, k, v, mask)
        attention_output = attention_output.transpose(-3, -2).contiguous().view(*batch_shape, seq_len, self.d_model)
        output = torch.matmul(attention_output, self.W_O.T)
        return output




