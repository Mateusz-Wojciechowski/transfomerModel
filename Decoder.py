import torch.nn as nn
import torch.nn.functional as fun
from PositionalEncoding import PositionalEncoding
from MultiHeadAttention import MultiHeadAttention
from FeedForwardNet import FeedForwardNet
from MultiHeadAttention import create_padding_mask, create_look_ahead_mask


class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, target_vocab_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(d_model, max_seq_len)
        self.pos_embedding = PositionalEncoding(d_model, max_seq_len)
        self.multi_head_att = MultiHeadAttention(d_model, num_heads)
        self.multi_head_att2 = MultiHeadAttention(d_model, num_heads)
        self.ff_net = FeedForwardNet(d_model, d_ff)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, target_vocab_size)

    def forward(self, x, encoder_output, single_mask, double_mask):
        attention_output = self.multi_head_att(x, x, x, double_mask)
        norm1 = self.layer_norm1(attention_output)

        attention_output2 = self.multi_head_att2(norm1, encoder_output, encoder_output, single_mask)
        norm2 = self.layer_norm2(attention_output2)
        ff_output = self.ff_net(norm2)
        norm3 = self.layer_norm3(ff_output)

        linear_output = self.output_layer(norm3)
        return fun.softmax(linear_output, dim=-1)
