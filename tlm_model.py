"""
TLM (TCP Language Model) - A small transformer for TCP protocol sequences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from tcp_tokenizer import TCPTokenizer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(0.1)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and split into heads
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear transformation
        output = self.w_o(attention_output)
        
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class TLMModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=8, n_layers=4, d_ff=512, max_len=512):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_len = max_len
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        
        # Output layer
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def create_causal_mask(self, seq_len):
        """Create causal mask for autoregressive generation"""
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.positional_encoding(x.transpose(0, 1)).transpose(0, 1)
        
        # Create causal mask for autoregressive modeling
        causal_mask = self.create_causal_mask(seq_len).to(input_ids.device)
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, causal_mask)
        
        # Final layer norm and output projection
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits
    
    def generate(self, tokenizer, prompt_tokens=None, max_length=20, temperature=1.0, top_k=None):
        """Generate TCP sequence given a prompt"""
        self.eval()
        
        if prompt_tokens is None:
            # Start with START token
            input_ids = torch.tensor([[tokenizer.start_id]], dtype=torch.long)
        else:
            input_ids = torch.tensor([prompt_tokens], dtype=torch.long)
        
        generated_tokens = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                # Forward pass
                logits = self.forward(input_ids)
                
                # Get logits for the last token
                next_token_logits = logits[0, -1, :] / temperature
                
                # Apply top-k filtering if specified
                if top_k is not None:
                    top_k = min(top_k, next_token_logits.size(-1))
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(0, top_k_indices, top_k_logits)
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Stop if we generate END token
                if next_token.item() == tokenizer.end_id:
                    break
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=1)
        
        return generated_tokens[0].tolist()

class TLMTrainer:
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        
    def train_step(self, input_ids, target_ids, optimizer, criterion):
        """Single training step"""
        self.model.train()
        
        input_ids = torch.tensor(input_ids, dtype=torch.long).to(self.device)
        target_ids = torch.tensor(target_ids, dtype=torch.long).to(self.device)
        
        # Forward pass
        logits = self.model(input_ids)
        
        # Reshape for loss calculation
        logits = logits.view(-1, logits.size(-1))
        target_ids = target_ids.view(-1)
        
        # Calculate loss (ignore padding tokens)
        loss = criterion(logits, target_ids)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def train(self, train_data, num_epochs=10, batch_size=32, learning_rate=1e-4):
        """Train the model"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id)
        
        input_sequences, target_sequences = train_data
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            # Create batches
            for i in range(0, len(input_sequences), batch_size):
                batch_inputs = input_sequences[i:i+batch_size]
                batch_targets = target_sequences[i:i+batch_size]
                
                loss = self.train_step(batch_inputs, batch_targets, optimizer, criterion)
                total_loss += loss
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    def save_model(self, filepath):
        """Save model state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab_size': self.model.vocab_size,
            'd_model': self.model.d_model,
            'n_heads': self.model.n_heads,
            'n_layers': self.model.n_layers,
            'd_ff': self.model.d_ff,
            'max_len': self.model.max_len
        }, filepath)
    
    def load_model(self, filepath):
        """Load model state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

if __name__ == "__main__":
    # Test the model
    tokenizer = TCPTokenizer()
    
    # Create model
    model = TLMModel(
        vocab_size=tokenizer.vocab_size,
        d_model=64,  # Small model for testing
        n_heads=4,
        n_layers=2,
        d_ff=256,
        max_len=50
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    test_input = torch.randint(0, tokenizer.vocab_size, (2, 10))
    output = model(test_input)
    print(f"Output shape: {output.shape}")
    
    # Test generation
    generated = model.generate(tokenizer, max_length=10)
    decoded = tokenizer.decode_sequence(generated)
    print(f"Generated sequence: {' -> '.join(decoded)}")