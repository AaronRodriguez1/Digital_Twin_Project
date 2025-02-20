import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
import argparse


# Argument parser
parser = argparse.ArgumentParser(description='Training Script')
parser.add_argument('-batch_size', type=int, required=True, help='Batch size for training')
args = parser.parse_args()

# Configurations
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = args.batch_size
block_size = 128
max_iters = 200
learning_rate = 3e-4
eval_iters = 100
n_embd = 384
n_head = 1
n_layer = 1
dropout = 0.2
pad_token_id = 3  # Assuming [PAD] is assigned ID 3 in vocab.txt

print(f"Device: {device}")

# Load tokenized data
def load_tokenized_data(file_path, vocab_size):
    with open(file_path, "r", encoding="utf-8") as f:
        data = [[int(token) for token in line.strip().split()] for line in f]
    for seq in data:
        if len(seq) > block_size:
            raise ValueError(f"Sequence length {len(seq)} exceeds block_size {block_size}")
        for token in seq:
            if token >= vocab_size or token < 0:
                raise ValueError(f"Token {token} out of range [0, {vocab_size - 1}]")
    return torch.tensor(data, dtype=torch.long)

# Precompute vocab size
with open("vocab.txt", "r", encoding="utf-8") as f:
    vocab_size = len(f.readlines())

train_data = load_tokenized_data("preprocessed_train.txt", vocab_size)
val_data = load_tokenized_data("preprocessed_val.txt", vocab_size)

print(f"Vocab size: {vocab_size}")
print(f"Train data shape: {train_data.shape}, Val data shape: {val_data.shape}")

# Batch generation
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, data.size(0), (batch_size,))
    x = data[ix, :]  # Shape: (batch_size, block_size)
    y = torch.roll(x, shifts=-1, dims=1)  # Shift tokens to get targets
    y[:, -1] = pad_token_id  # Padding for last target position
    attention_mask = (x != pad_token_id).float()  # 1 for real tokens, 0 for padding
    return x.to(device), y.to(device), attention_mask.to(device)

# Loss estimation
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = []
        for _ in range(eval_iters):
            X, Y, attention_mask = get_batch(split)
            logits, loss = model(X, Y, attention_mask)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out

# Model components
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = (q @ k.transpose(-2, -1)) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, attention_mask=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
        loss = F.cross_entropy(logits, targets, ignore_index=pad_token_id)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Initialize model
model = GPTLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"Step {iter}: Train Loss: {losses['train']:.4f}, Val Loss: {losses['val']:.4f}")

    X, Y, attention_mask = get_batch('train')
    logits, loss = model(X, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"Final Loss: {loss.item():.4f}")

# Save the model's state_dict
torch.save(model.state_dict(), "model-01.pth")
print("Model saved as model-01.pth.")
