import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import ChessMoveDataset
from model import TinyRecursiveModel
import os

# Hyperparameters Matches training
BATCH_SIZE = 32
DIM = 256
RECURSION_DEPTH = 6

def evaluate_model(model_path, csv_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Training Vocab (CRITICAL: Must match model's training vocab)
    print("Loading training vocab from train.csv...")
    # We load train dataset just to get the vocab. 
    # In a production app, you'd save/load 'vocab.json' instead.
    train_ds = ChessMoveDataset("data/train.csv", seq_len=128)
    vocab = train_ds.vocab
    vocab_size = train_ds.vocab_size
    print(f"Training Vocab size: {vocab_size}")

    # 2. Load Test Dataset using Training Vocab
    print("Loading test dataset...")
    test_ds = ChessMoveDataset("data/test.csv", seq_len=128, vocab=vocab)
    
    print(f"Total Test samples: {len(test_ds)}")
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Load Model
    model = TinyRecursiveModel(
        vocab_size=vocab_size,
        dim=DIM,
        recursion_depth=RECURSION_DEPTH
    ).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model weights loaded successfully.")
    else:
        print(f"Error: Model file {model_path} not found.")
        return

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    model.eval()

    total_loss = 0
    correct_top1 = 0
    correct_top5 = 0
    total_tokens = 0

    print("Starting evaluation...")
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits = model(x)  # (B, L, Vocab)
            
            # Flatten for calculations
            flat_logits = logits.reshape(-1, test_ds.vocab_size)
            flat_y = y.reshape(-1)
            
            # Filter out padding (0) tokens from stats
            mask = flat_y != 0
            valid_logits = flat_logits[mask]
            valid_y = flat_y[mask]
            
            if valid_y.numel() == 0:
                continue

            # Calculate Loss
            loss = criterion(valid_logits, valid_y)
            total_loss += loss.item()

            # --- Calculate Accuracy ---
            # Top-1
            preds = torch.argmax(valid_logits, dim=1)
            correct_top1 += (preds == valid_y).sum().item()
            
            # Top-5
            # topk returns (values, indices)
            _, top5_preds = torch.topk(valid_logits, 5, dim=1)
            # valid_y unsqueezed: (N, 1), top5: (N, 5) -> broadcasting check
            correct_top5 += (top5_preds == valid_y.unsqueeze(1)).any(dim=1).sum().item()

            total_tokens += valid_y.numel()

    # Final Metrics
    avg_loss = total_loss / len(loader)
    acc_top1 = (correct_top1 / total_tokens) * 100
    acc_top5 = (correct_top5 / total_tokens) * 100

    print("-" * 30)
    print(f"Validation Loss:  {avg_loss:.4f}")
    print(f"Top-1 Accuracy:   {acc_top1:.2f}% (Exact match)")
    print(f"Top-5 Accuracy:   {acc_top5:.2f}% (Soft match)")
    print("-" * 30)

if __name__ == "__main__":
    evaluate_model("trm_chess_model.pth", "data/test.csv")
