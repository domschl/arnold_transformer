import os
import torch
import time
import math
import random
from data_loader import load_text_data, Tokenizer, DataLoader
from model import GPT

# Hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 128 # what is the maximum context length for predictions?
max_iters = 100000
eval_interval = 500
learning_rate = 1e-5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.backends.mps.is_available():
    device = 'mps'
eval_iters = 100
n_embd = 384
n_head = 16
n_layer = 24
dropout = 0.1
activation_types = ['relu'] * n_layer # 'relu' or 'arnold'
attention_types = ['standard'] * n_layer # 'standard' or 'arnold'
positional_encoding = 'standard' # 'standard' or 'arnold'
residual_attention_mix = False
K_phase = False
middle = n_layer // 2
attention_types[middle - 2] = 'arnold'
attention_types[middle - 1] = 'arnold'
attention_types[middle] = 'arnold'
attention_types[middle + 1] = 'arnold'
attention_types[middle + 2] = 'arnold'

# activation_types[middle] = 'arnold'
# activation_types[middle+1] = 'arnold'
# activation_types[middle-1] = 'arnold'
arnold_used = False
for act in activation_types:
    if act == 'arnold':
        arnold_used = True
    elif act == 'relu':
        pass
    else:
        raise ValueError("Must be 'relu' or 'arnold'")
arnold_att_used = False
for att in attention_types:
    if att == 'arnold':
        arnold_att_used = True
    elif att == 'standard':
        pass
    else:
        raise ValueError("Must be 'standard' or 'arnold'")
if positional_encoding not in ['standard', 'arnold']:
    ValueError("Must be 'standard' or 'arnold'")
lyapunov_gov_beta = 5
lyapunov_dampening_offset = -5
Omega = 0.618033988749895
Omega_rnd_std = 0.00001
init_K = 0.8
# ------------

torch.manual_seed(1337)
random.seed(1337)

# Load data
dataset_dir = os.path.join(os.path.dirname(__file__), 'dataset')
text = load_text_data(dataset_dir)
if not text:
    print("Error: No text found in dataset directory!")
    exit(1)

tokenizer = Tokenizer()
vocab_size = tokenizer.vocab_size

# Create data loader
train_loader = DataLoader(text, tokenizer, block_size, batch_size, device, cache_dir=dataset_dir, train_split=0.9)

# Model
model = GPT(vocab_size, n_embd, block_size, n_head, n_layer, dropout, device, 
            activation_types=activation_types, attention_types=attention_types,
            positional_encoding=positional_encoding, Omega=Omega, Omega_rnd_std=Omega_rnd_std, init_K=init_K,
            residual_attention_mix=residual_attention_mix, K_phase=K_phase)
m = model.to(device)

# print(model)

# print the number of parameters in the model
print(str(sum(p.numel() for p in m.parameters())/1e6) + ' M parameters')

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # , weight_decay=1e-2)

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
        print()
        print("=" * 50)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # Save checkpoint
        checkpoint_path = f"checkpoint_step_{iter}.pt"
        torch.save(model.state_dict(), checkpoint_path)

        # Generate sample
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        for temperature in [1.0]:
            print(f"Generating sample at step {iter}, temperature={temperature}...")
            print(tokenizer.decode(m.generate(context, max_new_tokens=128, temperature=temperature)[0].tolist()))
            print("-" * 50)

    # sample a batch of data
    xb, yb = train_loader.get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # Lyapunov Governor
    if arnold_used is True or arnold_att_used is True: # or positional_encoding=='arnold':
        max_lyap = model.get_max_lyapunov()
        # decay LR based on chaos (clamp at 0 for gov, but log true value)
        # Formula: exp(-max(0, lyap) * beta)
        new_lr = learning_rate * math.exp(-max(0, max_lyap + lyapunov_dampening_offset) * lyapunov_gov_beta)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

        if iter % 50 == 0:
            # Collect K values
            k_act, k_att = model.get_min_max_K()
            k_min = 1000.0
            k_max = -1000.0
            kn = 0
            ks = 0.0
            for i in range(len(k_att)):
                    ki=k_att[i]
                    ks += ki
                    kn += 1
                    if ki<k_min:
                        k_min = ki
                    if ki>k_max:
                        k_max = ki
            if kn>0:
                k_mean = ks / kn
            else:
                k_mean = 0.0
            print(f"Iter {iter}: Max Lyap = {max_lyap:.4f}, LR = {new_lr:.6f}, K_min,max,avg = {k_min:.3f},{k_max:.3f},{k_mean:.3f}")

print(f"Training finished in {time.time() - start_time:.2f} seconds")
