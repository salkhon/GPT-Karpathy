import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparams
batch_size = 32  # how many independent sequences will we process in parallel?
context_len = 8  # what is the maximum context length for prediction?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200

torch.manual_seed(1337)

print(device)

with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers and vice versa
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


# encode: take a string, output a list of ints
def encode(s): return [stoi[c] for c in s]
# decoder: tale a list of ints, output a string
def decode(l): return "".join([itos[i] for i in l])


# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading


def get_batch(split):
    """Generate a small batch of inputs x and targets y

    Args:
        split (str): "train" or "val" split
    """
    data = train_data if split == "train" else val_data
    x_idxs = torch.randint(len(data) - context_len, (batch_size,))
    x = torch.stack([data[i:i+context_len] for i in x_idxs])
    y = torch.stack([data[i+1:i+1+context_len] for i in x_idxs])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
    
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()

    return out


# super simple Bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, xb, yb=None):
        # xb, yb are (B, T) tensor of ints
        logits = self.token_embedding_table(xb)

        if yb is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            yb = yb.view(B*T)
            loss = F.cross_entropy(logits, yb)
        
        return logits, loss
    
    def generate(self, xb, max_new_tokens):
        # xb is (B, T) array of token indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, _ = self(xb)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            yb_pred = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled token index 
            xb = torch.cat((xb, yb_pred), dim=1) # (B, T+1)
        
        return xb
    
model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

def train():
    for iter in range(max_iters):
        # every once in a while eval the loss on train and val sets
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data (dataloader)
        xb, yb = get_batch("trian")

        # eval the loss
        logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

train()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
