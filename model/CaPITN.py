import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from util import get_clones

# CaPITN

class WaveBiasAct(nn.Module):
    pass # The full code will be released immediately upon the paper's acceptance and publication.


class CharacteristicAwareAttention(nn.Module):
    pass # The full code will be released immediately upon the paper's acceptance and publication.


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256):
        super(EncoderLayer, self).__init__()
        self.attn = CharacteristicAwareAttention(d_model=d_model, num_heads=n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            WaveBiasAct(),
            nn.Linear(d_ff, d_model))

    def forward(self, src, coords=None):
        attn_out, _ = self.attn(src, src, src, coords=coords)
        src = src + attn_out
        src2 = self.ffn(src)
        src = src + src2
        return src


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256):
        super(DecoderLayer, self).__init__()
        self.attn = CharacteristicAwareAttention(d_model=d_model, num_heads=n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            WaveBiasAct(),
            nn.Linear(d_ff, d_model))

    def forward(self, src, e_outputs, coords=None):
        attn_out, _ = self.attn(src, e_outputs, e_outputs, coords=coords)
        src = src + attn_out
        src2 = self.ffn(src)
        src = src + src2
        return src


class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.layers = get_clones(EncoderLayer(d_model, n_heads), n_layers)

    def forward(self, src, coords=None):
        for i in range(self.n_layers):
            src = self.layers[i](src, coords=coords)
        return src


class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.layers = get_clones(DecoderLayer(d_model, n_heads), n_layers)

    def forward(self, src, e_outputs, coords=None):
        for i in range(self.n_layers):
            src = self.layers[i](src, e_outputs, coords=coords)
        return src


class CaPITN(nn.Module):
    def __init__(self, in_dim, d_model, n_heads, d_ff, n_layers, out_dim):
        super(CaPITN, self).__init__()
        self.embedding = nn.Linear(in_dim, d_model)
        self.encoder = Encoder(d_model, n_heads, n_layers)
        self.decoder = Decoder(d_model, n_heads, n_layers)
        self.linear_out = nn.Sequential(
            nn.Linear(d_model, d_ff),
            WaveBiasAct(),
            nn.Linear(d_ff, d_ff),
            WaveBiasAct(),
            nn.Linear(d_ff, out_dim))

    def forward(self, x, t):
        coords = torch.cat([x, t], dim=-1)
        src = self.embedding(coords)
        e_out = self.encoder(src, coords=coords)
        d_out = self.decoder(src, e_out, coords=coords)
        output = self.linear_out(d_out)
        return output


