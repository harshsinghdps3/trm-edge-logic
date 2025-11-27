import torch
import torch.nn as nn
import torch.nn.functional as F

class RecursiveBlock(nn.Module):
    """
    A single block that will be recycled T times.
    Takes (x, z) and outputs updated z.
    """
    def __init__(self, dim, heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, z, mask=None):
        # Standard Transformer Block logic
        res = z
        z = self.norm1(z)
        attn_out, _ = self.attn(z, z, z, attn_mask=mask, need_weights=False)
        z = res + attn_out
        
        res = z
        z = self.norm2(z)
        z = res + self.mlp(z)
        return z

class TinyRecursiveModel(nn.Module):
    def __init__(self, vocab_size, dim=256, heads=4, recursion_depth=4):
        super().__init__()
        self.dim = dim
        self.recursion_depth = recursion_depth # 'T' parameter
        
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, dim))
        
        # ONE block, reused multiple times
        self.shared_block = RecursiveBlock(dim, heads)
        
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        B, L = x.shape
        # Initial Embedding (x -> z_0)
        z = self.embedding(x) + self.pos_embedding[:, :L, :]
        
        # Causal Mask
        mask = torch.triu(torch.ones(L, L) * float('-inf'), diagonal=1).to(x.device)

        # RECURSIVE REASONING LOOP
        # "Latent Recursion": passing z back into the same block T times
        for step in range(self.recursion_depth):
            z = self.shared_block(z, mask=mask)
            
        logits = self.head(z)
        return logits

if __name__ == "__main__":
    # Example usage
    vocab_size = 100
    dim = 64
    heads = 2
    depth = 2
    
    model = TinyRecursiveModel(vocab_size, dim, heads, depth)
    
    # Create dummy input (Batch size 1, Sequence length 10)
    x = torch.randint(0, vocab_size, (1, 10))
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output = model(x)
    
    print(f"Output shape: {output.shape}")
    print("Model forward pass successful!")