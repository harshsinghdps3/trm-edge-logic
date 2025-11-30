import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ChessMoveDataset
from model import TinyRecursiveModel

# Hyperparameters
BATCH_SIZE = 32
DIM = 256  # Small dimension for "Tiny" constraint
RECURSION_DEPTH = 6  # T=6 recursions
LR = 3e-4
EPOCHS = 3

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Dataset
    print("Loading dataset...")
    ds = ChessMoveDataset("data/games.csv", seq_len=128)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Vocab size: {ds.vocab_size}")

    # 2. Model Initialization (< 7M Params target)
    model = TinyRecursiveModel(
        vocab_size=ds.vocab_size, 
        dim=DIM, 
        recursion_depth=RECURSION_DEPTH
    ).to(device)
    
    # Check param count
    params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 3. Training
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x) # (B, L, Vocab)
            
            # Flatten for loss
            loss = criterion(logits.reshape(-1, ds.vocab_size), y.reshape(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 50 == 0:
                print(f"Epoch {epoch+1} | Step {i} | Loss: {loss.item():.4f}")
        
        print(f"--> Epoch {epoch+1} Avg Loss: {total_loss/len(loader):.4f}")
        
    torch.save(model.state_dict(), "trm_chess_model.pth")
    print("Model saved.")

if __name__ == "__main__":
    train()