
import os
import torch
import time
import math
from data_loader import load_text_data, Tokenizer, DataLoader
from model import GPT

# Hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 10000
eval_interval = 500
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.backends.mps.is_available():
    device = 'mps'
eval_iters = 50
n_embd = 384
n_head = 8
n_layer = 8
dropout = 0.1
activation_types = ['relu'] * 8 # or 'relu' or 'arnold'
activation_types[4] = 'arnold'
lyapunov_gov_beta = 10
# ------------

torch.manual_seed(1337)

# Load data
dataset_dir = os.path.join(os.path.dirname(__file__), 'dataset')
text = load_text_data(dataset_dir)
if not text:
    print("Error: No text found in dataset directory!")
    exit(1)

tokenizer = Tokenizer(text)
vocab_size = tokenizer.vocab_size

# Create data loader
train_loader = DataLoader(text, tokenizer, block_size, batch_size, device, train_split=0.9)

# Model
model = GPT(vocab_size, n_embd, block_size, n_head, n_layer, dropout, device, activation_types=activation_types)
m = model.to(device)
# print the number of parameters in the model
print(str(sum(p.numel() for p in m.parameters())/1e6) + ' M parameters')

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = train_loader.get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Training loop
print("Starting training...")
start_time = time.time()

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # Save checkpoint
        checkpoint_path = f"checkpoint_step_{iter}.pt"
        torch.save(model.state_dict(), checkpoint_path)

        # Generate sample
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        print(f"Generating sample at step {iter}...")
        print(tokenizer.decode(m.generate(context, max_new_tokens=100)[0].tolist()))
        print("-" * 50)

    # sample a batch of data
    xb, yb = train_loader.get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # Lyapunov Governor
    if activation_type == 'arnold':
        max_lyap = model.get_max_lyapunov()
        # decay LR based on chaos (clamp at 0 for gov, but log true value)
        # Formula: exp(-max(0, lyap) * beta)
        new_lr = learning_rate * math.exp(-max(0, max_lyap) * lyapunov_gov_beta)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

        if iter % 10 == 0:
            # Collect K values
            k_values = []
            for block in model.blocks:
                if hasattr(block.ffwd, 'activation'):
                    k_values.append(block.ffwd.activation.K.item())
            
            if k_values:
                k_min, k_max, k_mean = min(k_values), max(k_values), sum(k_values)/len(k_values)
                print(f"Iter {iter}: Max Lyap = {max_lyap:.4f}, LR = {new_lr:.6f}, K(min/max/avg) = {k_min:.3f}/{k_max:.3f}/{k_mean:.3f}")
            else:
                 print(f"Iter {iter}: Max Lyap = {max_lyap:.4f}, LR = {new_lr:.6f}")

print(f"Training finished in {time.time() - start_time:.2f} seconds")
