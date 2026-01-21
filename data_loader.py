
import os
import torch
import random

def load_text_data(directory):
    """
    Reads all .txt files in the given directory and concatenates them.
    """
    text = ""
    files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    print(f"Found {len(files)} text files in {directory}")
    for filename in sorted(files): # Sort for deterministic order
        path = os.path.join(directory, filename)
        with open(path, 'r', encoding='utf-8') as f:
            try:
                content = f.read()
                text += content + "\n" # Add newline between files
            except Exception as e:
                print(f"Error reading {path}: {e}")
    return text

class Tokenizer:
    def __init__(self, text):
        # Create a mapping from characters to integers
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }
        print(f"Tokenizer initialized with vocab size: {self.vocab_size}")

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])

class DataLoader:
    def __init__(self, text, tokenizer, block_size, batch_size, device='cpu', train_split=0.9):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        
        # Encode the entire text
        data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        
        # Split into train and validation
        n = int(train_split * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]
        print(f"Data split: {len(self.train_data)} train, {len(self.val_data)} val tokens")

    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y

if __name__ == "__main__":
    # fast test
    dataset_dir = os.path.join(os.path.dirname(__file__), 'dataset')
    raw_text = load_text_data(dataset_dir)
    if not raw_text:
        print("No text found. Make sure 'dataset' directory exists and has .txt files.")
    else:
        print(f"Loaded {len(raw_text)} characters")
        tokenizer = Tokenizer(raw_text)
        loader = DataLoader(raw_text, tokenizer, block_size=8, batch_size=4)
        xb, yb = loader.get_batch('train')
        print("Batch shape:", xb.shape)
        print("Input:", xb[0].tolist())
        print("Target:", yb[0].tolist())
        decoded = tokenizer.decode(xb[0].tolist())
        print("Decoded input sample:", decoded)
