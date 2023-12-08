import torch.nn as nn
import torch.nn.functional as fun
from EncoderBlock import EncoderBlock
from Decoder import Decoder
from FeedForwardNet import FeedForwardNet
from PositionalEncoding import PositionalEncoding
from MultiHeadAttention import create_masks


class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, src_vocab_size, target_vocab_size):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(target_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.decoder = Decoder(d_model, num_heads, d_ff, max_seq_len, target_vocab_size)
        self.encoder = EncoderBlock(d_model, d_ff, num_heads, max_seq_len)

    def forward(self, src, target):
        single_mask, double_mask = create_masks(src, target)
        embedded_src = self.pos_encoding(self.encoder_embedding(src))
        embedded_target = self.pos_encoding(self.decoder_embedding(target))

        encoder_output = self.encoder(embedded_src, single_mask)
        decoder_output = self.decoder(embedded_target, encoder_output, single_mask, double_mask)
        return decoder_output

