import logging

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
     # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=None):
        super(ScaledDotProductAttention, self).__init__()
        self.scale_factor = np.sqrt(d_k)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout_sign = dropout
        if self.dropout_sign:
            self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, q, k, v, attn_mask=None):
        # q: [b_size x n_heads x len_q x d_k]
        # k: [b_size x n_heads x len_k x d_k]
        # v: [b_size x n_heads x len_v x d_v] note: (len_k == len_v)

        # attn: [b_size x n_heads x len_q x len_k]
        scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale_factor
        if attn_mask is not None:
            assert attn_mask.size() == scores.size()
            scores.masked_fill_(attn_mask, -1e9)
        if self.dropout_sign:
            attn = self.dropout(self.softmax(scores))
        else:
            attn = self.softmax(scores)

        # outputs: [b_size x n_heads x len_q x d_v]
        context = torch.matmul(attn, v)

        return context, attn


class _MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, dropout):
        # d_model = 128,d_k=d_v = 32,n_heads = 4
        super(_MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads

        # self.w_q = Linear(d_model, d_k * n_heads)
        # self.w_k = Linear(d_model, d_k * n_heads)
        # self.w_v = Linear(d_model, d_v * n_heads)
        self.w_q = nn.Linear(d_model, d_k * n_heads)
        self.w_k = nn.Linear(d_model, d_k * n_heads)
        self.w_v = nn.Linear(d_model, d_v * n_heads)

        self.attention = ScaledDotProductAttention(d_k, dropout)

    def forward(self, q, k, v, attn_mask):
        # q: [b_size x len_q x d_model]
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_k x d_model]
        b_size = q.size(0)

        # q_s: [b_size x n_heads x len_q x d_k]
        # k_s: [b_size x n_heads x len_k x d_k]
        # v_s: [b_size x n_heads x len_k x d_v]
        q_s = self.w_q(q).view(b_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.w_k(k).view(b_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.w_v(v).view(b_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        if attn_mask:  # attn_mask: [b_size x len_q x len_k]
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # context: [b_size x n_heads x len_q x d_v], attn: [b_size x n_heads x len_q x len_k]
        context, attn = self.attention(q_s, k_s, v_s, attn_mask=attn_mask)
        # context: [b_size x len_q x n_heads * d_v]
        context = context.transpose(1, 2).contiguous().view(b_size, -1, self.n_heads * self.d_v)

        # return the context and attention weights
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, dropout=None):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.multihead_attn = _MultiHeadAttention(d_model, d_k, d_v, n_heads, dropout)
        self.proj = nn.Linear(n_heads * d_v, d_model)
        # self.layer_norm = LayerNormalization(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout_sign = dropout
        if self.dropout_sign:
            self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, q, k, v, attn_mask=None):
        # q: [b_size x len_q x d_model]
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_v x d_model] note (len_k == len_v)
        residual = q
        # context: a tensor of shape [b_size x len_q x n_heads * d_v]
        context, attn = self.multihead_attn(q, k, v, attn_mask=attn_mask)

        context = torch.where(torch.isnan(context), torch.full_like(context, 0), context)
        attn = torch.where(torch.isnan(attn), torch.full_like(attn, 0), attn)

        # project back to the residual size, outputs: [b_size x len_q x d_model]
        if self.dropout_sign:
            output = self.dropout(self.proj(context))
        else:
            output = self.proj(context)

        ro = residual + output
        no = self.layer_norm(ro)
        if torch.isnan(no).any() and not torch.isnan(ro).any():
            return ro, attn
        return no, attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout=None):
        super(PoswiseFeedForwardNet, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout_sign = dropout
        if self.dropout_sign:
            self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, inputs):
        # inputs: [b_size x len_q x d_model]
        residual = inputs
        output = self.relu(self.conv1(inputs.transpose(1, 2)))

        # outputs: [b_size x len_q x d_model]
        output = self.conv2(output).transpose(1, 2)
        if self.dropout_sign:
            output = self.dropout(output)

        return self.layer_norm(residual + output)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, d_ff, n_heads, dropout=None):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads, dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout)

    def forward(self, enc_inputs, self_attn_mask=None):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs,
                                               enc_inputs, attn_mask=self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)

        return enc_outputs


class SelectOperations(nn.Module):
    def __init__(self, d_model, operations):
        super(SelectOperations, self).__init__()
        self.selector = nn.Sequential(
            nn.Linear(d_model, operations),
        )

    def forward(self, enc_outputs):
        x = enc_outputs.squeeze()[0:-1]
        output = self.selector(x)
        # out = torch.softmax(output, dim=-1)
        return output


class StatisticLearning(nn.Module):
    def __init__(self, statistic_nums, d_model):
        super(StatisticLearning, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(statistic_nums, statistic_nums * 2),
            nn.ReLU(),
            nn.Linear(statistic_nums * 2, d_model),
        )

    def forward(self, input):
        return self.layer(input)


class ReductionDimension(nn.Module):
    def __init__(self, statistic_nums, d_model):
        super(ReductionDimension, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm1d(statistic_nums),
            nn.Linear(statistic_nums, d_model),
            nn.BatchNorm1d(d_model),
        )

    def forward(self, input):
        out = self.layer(input).unsqueeze(dim=0)
        return out


