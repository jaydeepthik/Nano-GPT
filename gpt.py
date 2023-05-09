import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 32
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

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        #attention scores
        wei = q @ k.transpose(-2, -1) * C**-0.5 #(B, T, C) @ (B, C, T) ----> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] ==0, float("-inf"))
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        out = wei @ v #(B, T, T) @ (B, T, C) -----> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""
    def __init__(self, num_heads, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    
    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)


#simple bigram language model
class BigramLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        #self.sa_head = Head(n_embed)
        self.sa_heads = MultiHeadAttention(4, n_embed//4) #4 heads of 8 self attention
        self.lmhead = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        #idx, targets shape (B, T)
        B, T = idx.shape

        token_emb = self.embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = token_emb + pos_emb
        x = self.sa_heads(x)
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
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
