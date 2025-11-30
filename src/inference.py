import torch
from model import TinyRecursiveModel
from dataset import ChessMoveDataset

def predict_next_move(model_path, csv_path, move_sequence_str):
    # Load reference vocab
    ds = ChessMoveDataset(csv_path) # Just to get vocab
    vocab = ds.vocab
    inv_vocab = {v: k for k, v in vocab.items()}
    
    # Load Model
    model = TinyRecursiveModel(ds.vocab_size, dim=256, recursion_depth=6)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Tokenize input
    moves = move_sequence_str.split()
    ids = [vocab.get(m, 0) for m in moves]
    input_tensor = torch.tensor([ids], dtype=torch.long)

    with torch.no_grad():
        logits = model(input_tensor)
        # Get prediction for the last token
        last_token_logits = logits[0, -1, :]
        predicted_id = torch.argmax(last_token_logits).item()
        
    return inv_vocab.get(predicted_id, "<UNK>")

if __name__ == "__main__":
    # Example: Sicilian Defense opening
    seq = "e4 e6 d4 d5"
    pred = predict_next_move("trm_chess_model.pth", "data/games.csv", seq)
    print(f"Sequence: {seq}")
    print(f"TRM Predicted Next Move: {pred}")