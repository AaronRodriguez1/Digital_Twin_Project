import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse

# Parser to accept batch size for flexibility
parser = argparse.ArgumentParser(description='Chatbot Inference Script')
parser.add_argument('-batch_size', type=int, default=1, help='Batch size for chatbot inference')
args = parser.parse_args()

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

n_embd = 384
n_head = 1
n_layer = 1
block_size = 128
dropout = 0.2


# Load vocabulary
print("Loading vocabulary...")
with open("vocab.txt", 'r', encoding='utf-8') as f:
    chars = f.read().strip().split()
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size}")

# Character encoding and decoding functions
string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [string_to_int.get(c, string_to_int['[UNK]']) for c in s]
decode = lambda l: ''.join([int_to_string.get(i, '[UNK]') for i in l])

# Define the GPT model (same as training)
class Head(nn.Module):
    """One head of self-attention."""
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
    """Multiple heads of self-attention in parallel."""
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
    """A simple feed-forward layer."""
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
    """Transformer block."""
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
    """The GPT Language Model."""
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
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

# Load the model
print("Loading model parameters...")
model = GPTLanguageModel(vocab_size).to(device)
model.load_state_dict(torch.load("model-01.pth", map_location=device))
model.eval()
print("Model loaded successfully!")

# Chatbot loop
print("Chatbot is ready! Type your prompt below.")
while True:
    prompt = input("Prompt: ")
    if not prompt.strip():
        print("Exiting chatbot. Goodbye!")
        break
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    generated_idx = model.generate(context, max_new_tokens=150)[0].tolist()
    generated_text = decode(generated_idx)
    print(f"Response: {generated_text}")
