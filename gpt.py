import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)

#read file
with open("./data/input.txt") as f:
    text = f.read()

#create vocab
chars = sorted(list(set(text)))
vocab_size = len(chars)

#create vocab-->id, id-->vocab mappings
ctoi = {c: id for id, c in enumerate(chars)}
itoc = {id: c for id, c in enumerate(chars)}

#create encoder decoder
encode = lambda x: [ctoi[ch] for ch in x] #input string, output list of ints
decode = lambda ids: "".join([itoc[id] for id in ids])

#train-test split
data = torch.tensor(encode(text), dtype=torch.long)
train_thresh = int(0.9*len(data))
train_data = data[:train_thresh]
val_data = data[train_thresh:]

#data batching
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, 1))
    x = torch.stack([data[i: i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """single headed attention"""
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        #attention scores
        wei = q @ k.transpose(-2, -1) * C**-0.5 #(B, T, C) @ (B, C, T) ----> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] ==0, float("-inf"))
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        out = wei @ v #(B, T, T) @ (B, T, C) -----> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""
    def __init__(self, num_heads, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
       out = torch.cat([h(x) for h in self.heads], dim=-1)
       out = self.proj(out)
       out = self.dropout(out)
       return  out

class FeedForward(nn.Module):
    """simple linear layer followed by a non-linear activation"""
    def __init__(self, n_embed) -> None:
        super().__init__()
        self.net  = nn.Sequential(
            nn.Linear(n_embed,4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embed,n_heads) -> None:
        super().__init__()
        head_size = n_embed//n_heads
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
#simple bigram language model
class BigramLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_heads=n_head) for _ in range(n_layer)])
        self.ln_f =   nn.LayerNorm(n_embed)
        self.lmhead = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        #idx, targets shape (B, T)
        B, T = idx.shape

        token_emb = self.embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) #(T, C)
        x = token_emb + pos_emb #(B, T, C)
        x = self.blocks(x) #(B, T, C)
        x = self.ln_f(x) #(B, T, C)
        logits = self.lmhead(x) # (B, T, vocab_size)


        if targets is None:
            loss = None
        else:
            #shape change necessary to canculate loss, Channels should be the 2nd dimnesion
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        """method to generate new sample T+1, T+2.... given initial sequence idx"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            #get predictions and losses
            logits, loss = model(idx_cond) # (B, T, C)
            #isolate the last time stamp as it contains the predicted token
            logits = logits[:,-1,:] # (B, C)
            #get the probaility distribution
            probs = F.softmax(logits, dim=-1)
            #sample from the probability distribution
            next_idx = torch.multinomial(probs, num_samples=1) #(B, 1)
            idx = torch.cat((idx, next_idx), dim=1) #(D, T+1)
        return idx
    
if __name__ == "__main__":

    model = BigramLanguageModel()
    m = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    print(f"Training starts now on device :{device}")

    for iter in range(max_iters):
        if iter% eval_iters ==0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:0.4f}")

        Xb, yb = get_batch("train")
        logits, loss = model(Xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
    #generate after training is complete
    print("######################################################################")
    print(f"Generating context characters:")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=800)[0].tolist()))
