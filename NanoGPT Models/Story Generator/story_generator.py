# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import sentencepiece as spm
import os
import pandas as pd
import math

# Data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    max_index = len(data) - block_size
    if max_index <= 0:
        raise ValueError("The block_size is too large for the given dataset. Consider reducing it.")
    ix = torch.randint(max_index, (batch_size,), device=device)
    x_batches = []
    y_batches = []
    for i in ix:
        end = min(i + block_size, len(data))
        x = data[i:end]
        y = data[i+1:end+1]
        if len(x) < block_size:
            x = torch.cat([x, torch.zeros(block_size - len(x), dtype=torch.long, device=device)])
            y = torch.cat([y, torch.zeros(block_size - len(y), dtype=torch.long, device=device)])
        x_batches.append(x)
        y_batches.append(y)
    x = torch.stack(x_batches).to(device)
    y = torch.stack(y_batches).to(device)
    return x, y

@torch.no_grad()
def estimate_loss_and_perplexity():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        avg_loss = losses.mean()
        perplexity = math.exp(avg_loss)
        out[split] = (avg_loss, perplexity)
    model.train()
    return out

#Checkpointing function
def save_checkpoint(model, optimizer, epoch, path, loss):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss' : loss
    }, path)

class Head(nn.Module):
    """ One head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False).to(device)
        self.query = nn.Linear(n_embd, head_size, bias=False).to(device)
        self.value = nn.Linear(n_embd, head_size, bias=False).to(device)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size, device=device)))
        self.dropout = nn.Dropout(dropout).to(device)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd).to(device)
        self.dropout = nn.Dropout(dropout).to(device)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd).to(device),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd).to(device),
            nn.Dropout(dropout).to(device),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size).to(device)
        self.ffwd = FeedForward(n_embd).to(device)
        self.ln1 = nn.LayerNorm(n_embd).to(device)
        self.ln2 = nn.LayerNorm(n_embd).to(device)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(sp.get_piece_size(), n_embd).to(device)
        self.position_embedding_table = nn.Embedding(block_size, n_embd).to(device)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head).to(device) for _ in range(n_layer)]).to(device)
        self.ln_f = nn.LayerNorm(n_embd).to(device)
        self.lm_head = nn.Linear(n_embd, sp.get_piece_size()).to(device)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens=100):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Hyperparameters
batch_size = 512
block_size = 32
max_iters = 5000
eval_interval = 500
learning_rate = 1e-4
weight_decay = 1e-5  # Weight decay for regularization
patience = 3  # Early stopping patience
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 512
n_head = 4
n_layer = 4
dropout = 0.2

# Load the Story Cloze dataset
story_2016 = pd.read_csv('cloze_2016.csv')
story_2016['story'] = story_2016['storytitle'] + ' ' + story_2016[['sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5']].agg(' '.join, axis=1)
story_2018 = pd.read_csv('cloze_2018.csv')
story_2018['story'] = story_2018['storytitle'] + ' ' + story_2018[['sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5']].agg(' '.join, axis=1)
story_cloze = pd.concat([story_2016, story_2018], axis=0)

with open('text_data.txt', 'w') as f:
    f.write('\n'.join(story_cloze['story'].tolist()))
# Train SentencePiece model
spm.SentencePieceTrainer.train(input='text_data.txt', model_prefix='spm_model', vocab_size=10000, model_type='bpe')

# Initialize SentencePiece tokenizer
sp = spm.SentencePieceProcessor(model_file='spm_model.model')

# Encode and decode functions using SentencePiece
def encode(text):
    return sp.encode(text, out_type=int)

def decode(ids):
    return sp.decode(ids)

# Prepare data
text = ' '.join(story_cloze['story'])
data = torch.tensor(encode(text), dtype=torch.long).to(device)  # Move tensor to device
n = int(0.8 * len(data))
train_data = data[:n]
val_data = data[n:]


model = GPTLanguageModel().to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Training loop with checkpointing
best_val_loss = float('inf')
checkpoint_path = 'model_checkpoint.pth'
no_improvement_count = 0

# Hyperparameters
num_epochs = 1  # Number of epochs

# Training loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses_and_perplexities = estimate_loss_and_perplexity()
            train_loss, train_perplexity = losses_and_perplexities['train']
            val_loss, val_perplexity = losses_and_perplexities['val']

            print(f"Step {iter}: train loss {train_loss:.4f}, train perplexity {train_perplexity:.4f}, val loss {val_loss:.4f}, val perplexity {val_perplexity:.4f}")

            # Save the best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, iter, checkpoint_path, best_val_loss)
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Early stopping
            if no_improvement_count >= patience:
                print("Early stopping triggered.")
                break

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Print the epoch summary
    print(f"Epoch {epoch + 1} completed. Best validation loss: {best_val_loss:.4f}")

    # Early stopping after each epoch
    if no_improvement_count >= patience:
        print("Early stopping triggered after epoch.")
        break

def generate_story(prompt, num_sentences=5):
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    generated = model.generate(context, max_new_tokens=500)
    generated_text = decode(generated[0].tolist())
    sentences = generated_text.split('. ')
    return '.\n'.join(sentences[:num_sentences]) + '.'