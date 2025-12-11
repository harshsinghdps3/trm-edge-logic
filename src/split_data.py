import pandas as pd
import os
from sklearn.model_selection import train_test_split

def split_dataset(input_csv, train_ratio=0.9):
    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Shuffle and Split
    train_df, test_df = train_test_split(df, test_size=(1 - train_ratio), random_state=42)
    
    print(f"Total samples: {len(df)}")
    print(f"Training samples: {len(train_df)}")
    print(f"Testing samples: {len(test_df)}")
    
    # Save splits
    train_path = "data/train.csv"
    test_path = "data/test.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Saved splits to:\n- {train_path}\n- {test_path}")

if __name__ == "__main__":
    split_dataset("data/games.csv")
