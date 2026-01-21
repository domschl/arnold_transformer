
import os
import torch
import tiktoken
import numpy as np

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
    def __init__(self, text=None):
        # We use GPT-2 encoding
        self.enc = tiktoken.get_encoding("gpt2")
        # Pad vocab size to 50304 (multiple of 64) for efficiency
        self.vocab_size = 50304 
        self.eot_token = self.enc.eot_token
        print(f"Tiktoken tokenizer initialized. Native vocab: {self.enc.n_vocab}, Padded to: {self.vocab_size}")

    def encode(self, s):
        # Allow special tokens if needed, but for now just standard encode
        return self.enc.encode(s, allowed_special={'<|endoftext|>'})

    def decode(self, l):
        # Filter out potential padding tokens if any were generated (though we won't train on them)
        l = [x for x in l if x < self.enc.n_vocab]
        return self.enc.decode(l)

class DataLoader:
    def __init__(self, text, tokenizer, block_size, batch_size, device='cpu', train_split=0.9):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        
        # Encode the entire text
        print("Encoding text data (this might take a moment)...")
        # Use numpy for efficient storage before tensor conversion
        encoded = self.tokenizer.encode(text)
        data = torch.tensor(encoded, dtype=torch.long)
        print(f"Encoded {len(data)} tokens.")
        
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
        # Just use a subset for quick test
        raw_text = raw_text[:10000]
        tokenizer = Tokenizer() # No need to pass text for vocabulary building anymore
        loader = DataLoader(raw_text, tokenizer, block_size=8, batch_size=4)
        xb, yb = loader.get_batch('train')
        print("Batch shape:", xb.shape)
        print("Input:", xb[0].tolist())
        print("Target:", yb[0].tolist())
        decoded = tokenizer.decode(xb[0].tolist())
        print("Decoded input sample:", decoded)
