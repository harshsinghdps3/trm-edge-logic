import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class ChessMoveDataset(Dataset):
    def __init__(self, csv_file, seq_len=128, vocab=None):
        self.seq_len = seq_len
        df = pd.read_csv(csv_file)
        
        # Simple whitespace tokenizer for chess moves
        all_moves = [m.split() for m in df['moves']]
        
        if vocab is None:
            # Build vocab from scratch
            unique_moves = set(token for game in all_moves for token in game)
            self.vocab = {m: i + 1 for i, m in enumerate(sorted(unique_moves))}
            self.vocab['<PAD>'] = 0
            self.vocab_size = len(self.vocab)
        else:
            self.vocab = vocab
            self.vocab_size = len(vocab)
            
        self.data = []
        for moves in all_moves:
            # Convert to IDs and truncate/pad
            ids = [self.vocab[m] for m in moves if m in self.vocab]
            if len(ids) > seq_len:
                ids = ids[:seq_len]
            else:
                ids += [0] * (seq_len - len(ids))
            self.data.append(np.array(ids, dtype=np.int64))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Input: moves 0..N-1, Target: moves 1..N
        x = self.data[idx]
        return torch.from_numpy(x[:-1]), torch.from_numpy(x[1:])

if __name__ == "__main__":
    # Test with the actual games.csv
    csv_path = "data/games.csv"
    
    try:
        print(f"Loading dataset from {csv_path}...")
        dataset = ChessMoveDataset(csv_path, seq_len=128)
        print(f"Dataset loaded successfully.")
        print(f"Total games: {len(dataset)}")
        print(f"Vocabulary size: {dataset.vocab_size}")
        
        # Check a sample
        if len(dataset) > 0:
            x, y = dataset[0]
            print("\nSample 0:")
            print(f"Input shape: {x.shape}")
            print(f"Target shape: {y.shape}")
            print(f"First 10 input tokens: {x[:10]}")
            print(f"First 10 target tokens: {y[:10]}")
            
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}. Make sure you are running from the project root.")
    except Exception as e:
        print(f"An error occurred: {e}")


