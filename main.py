import torch
from TransformerTraining import TransformerTraining
import torch.nn as nn
from Transformer import Transformer
import torch.optim as optim
from MultiHeadAttention import create_padding_mask, create_look_ahead_mask

d_model = 512
num_heads = 8
max_seq_len = 100
d_ff = 2048
scr_vocab_size = 5000
target_vocab_size = 5000
learning_rate = 0.001

model = Transformer(d_model, num_heads, d_ff, max_seq_len, scr_vocab_size, target_vocab_size)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

src_data = torch.randint(1, max_seq_len, (64, max_seq_len))
tgt_data = torch.randint(1, max_seq_len, (64, max_seq_len))

model.train()

for epoch in range(100):
    optimizer.zero_grad()

    output = model(src_data, tgt_data[:, :-1])

    output_dim = output.shape[-1]
    output = output.contiguous().view(-1, output_dim)

    tgt = tgt_data[:, 1:].contiguous().view(-1)

    loss = loss_fn(output, tgt)
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

