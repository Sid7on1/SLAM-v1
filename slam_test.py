# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import time
import math
from typing import Optional, Tuple

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Linear projections and reshape
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch_size, num_heads, seq_len, seq_len]
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # [batch_size, seq_len, num_heads, head_dim]
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        # Final linear projection
        output = self.out_proj(attn_output)
        
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class StandardTransformer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class SLAMv1(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, ef_cycles, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.ef_cycles = ef_cycles
        
        # Initial full MHSA layer
        self.initial_block = EncoderBlock(d_model, num_heads, d_ff, dropout)
        
        # EF Cycle blocks
        self.ef_blocks_A = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(ef_cycles)
        ])
        
        self.ef_blocks_B = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(ef_cycles)
        ])
        
        # Final refinement block
        self.final_block = EncoderBlock(d_model, num_heads, d_ff, dropout)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Initial full MHSA layer
        x = self.initial_block(x, mask)
        
        # EF Cycles
        for i in range(self.ef_cycles):
            # Calculate segment indices
            first_60_percent = int(seq_len * 0.6)
            last_60_percent = int(seq_len * 0.6)
            start_40_percent = int(seq_len * 0.4)
            
            # Block A processes [1-60%] + [40-100%]
            segment_A1 = x[:, :first_60_percent, :]
            segment_A2 = x[:, start_40_percent:, :]
            segment_A = torch.cat([segment_A1, segment_A2], dim=1)
            
            # Block B processes [40-100%] + [1-60%]
            segment_B1 = x[:, start_40_percent:, :]
            segment_B2 = x[:, :first_60_percent, :]
            segment_B = torch.cat([segment_B1, segment_B2], dim=1)
            
            # Process segments through respective blocks
            processed_A = self.ef_blocks_A[i](segment_A)
            processed_B = self.ef_blocks_B[i](segment_B)
            
            # Split processed segments back to original positions
            processed_A1 = processed_A[:, :first_60_percent, :]
            processed_A2 = processed_A[:, first_60_percent:, :]
            
            processed_B1 = processed_B[:, :last_60_percent, :]
            processed_B2 = processed_B[:, last_60_percent:, :]
            
            # Fuse the outputs (average in overlapping regions)
            x_new = torch.zeros_like(x)
            
            # Non-overlapping regions from A
            x_new[:, :start_40_percent, :] = processed_A1[:, :start_40_percent, :]
            
            # Overlapping region (average of A and B)
            x_new[:, start_40_percent:first_60_percent, :] = (
                processed_A1[:, start_40_percent:first_60_percent, :] + 
                processed_B2[:, start_40_percent:first_60_percent, :]
            ) / 2
            
            # Non-overlapping regions from B
            x_new[:, first_60_percent:, :] = processed_B1[:, first_60_percent-start_40_percent:, :]
            
            # Update x for next cycle
            x = x_new
        
        # Final refinement block
        x = self.final_block(x, mask)
        
        return x

# DeepSeek Transformer implementation
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Create the cos and sin embeddings
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Create position indices
        self.register_buffer(
            "pos_ids", torch.arange(max_seq_len, dtype=torch.float)
        )
        
        self._update_cos_sin_tables()
    
    def _update_cos_sin_tables(self):
        # Calculate cos and sin values for the positions
        freqs = torch.outer(self.pos_ids, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype=torch.float))
        self.register_buffer("sin_cached", emb.sin().to(dtype=torch.float))
    
    def forward(self, x, seq_len=None):
        # Return the cached values for the positions we need
        if seq_len is None:
            seq_len = x.shape[1]
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype)
        )

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    # Apply rotary position embeddings to q and k
    # q, k: [batch_size, seq_len, num_heads, head_dim] or [batch_size, num_heads, seq_len, head_dim]
    
    # Check if tensors are in [batch, seq_len, heads, dim] format and transpose if needed
    q_orig_shape = q.shape
    k_orig_shape = k.shape
    
    # Ensure q and k are in [batch, seq, head, dim] format for applying rotary embeddings
    if q.shape[1] == q_orig_shape[2] and q.shape[2] == q_orig_shape[1]:
        # If q is in [batch, head, seq, dim] format, transpose to [batch, seq, head, dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        transpose_back = True
    else:
        transpose_back = False
    
    batch_size, seq_len, num_heads, head_dim = q.shape
    
    # If position_ids is None, use default positions
    if position_ids is None:
        position_ids = torch.arange(seq_len, device=q.device)
    
    # Select the cos and sin values for the positions we need
    cos = cos[position_ids].unsqueeze(1).unsqueeze(0)  # [1, seq_len, 1, head_dim]
    sin = sin[position_ids].unsqueeze(1).unsqueeze(0)  # [1, seq_len, 1, head_dim]
    
    # Apply rotary embeddings
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    # Transpose back if needed
    if transpose_back:
        q_embed = q_embed.transpose(1, 2)
        k_embed = k_embed.transpose(1, 2)
    
    return q_embed, k_embed

def rotate_half(x):
    # Rotate half the hidden dims of the input
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # Calculate RMS
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # Normalize and scale
        return self.weight * x * rms

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_model, d_ff)
        self.w3 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # SwiGLU activation
        swish = self.w1(x) * torch.sigmoid(self.w2(x) * 1.0)
        x = self.w3(self.dropout(swish))
        return x

class TrueGroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_query_groups=1, dropout=0.1, max_seq_len=2048):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_query_groups = num_query_groups
        self.head_dim = d_model // num_heads
        
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        assert num_heads % num_query_groups == 0, "num_heads must be divisible by num_query_groups"
        
        self.heads_per_group = num_heads // num_query_groups
        
        # True GQA: Only project queries for each group (fewer projections)
        # Each query group is shared across multiple attention heads
        self.q_proj = nn.Linear(d_model, self.head_dim * num_query_groups)
        
        # Keys and values are still full-sized (one per head)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
        # Add rotary embeddings
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)
    
    def forward(self, x, mask=None, is_causal=False):
        batch_size, seq_len, _ = x.shape
        
        # Validate mask shape if provided
        if mask is not None:
            # Check if mask is a 2D or 4D tensor
            if mask.dim() == 2:
                # Expand mask to 4D: [batch_size, 1, seq_len, seq_len]
                mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)
            elif mask.dim() == 3:
                # Expand mask to 4D: [batch_size, num_heads, seq_len, seq_len]
                mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            elif mask.dim() == 4:
                # Ensure mask has correct shape: [batch_size, num_heads, seq_len, seq_len]
                if mask.shape[0] != batch_size or mask.shape[1] != self.num_heads:
                    raise ValueError(f"Mask shape {mask.shape} doesn't match expected shape for batch_size={batch_size}, num_heads={self.num_heads}")
            else:
                raise ValueError(f"Mask must be 2D, 3D or 4D, got {mask.dim()}D")
        
        # Create causal mask if needed
        if is_causal:
            # Create a causal mask (lower triangular)
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            if mask is None:
                mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)
            else:
                # Combine with existing mask
                mask = mask * causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Linear projections
        # For queries, we project to a smaller dimension (num_query_groups * head_dim)
        q = self.q_proj(x)  # [batch_size, seq_len, num_query_groups * head_dim]
        
        # For keys and values, we project to the full dimension
        k = self.k_proj(x)  # [batch_size, seq_len, d_model]
        v = self.v_proj(x)  # [batch_size, seq_len, d_model]
        
        # Reshape queries for grouped query attention
        # [batch_size, seq_len, num_query_groups, head_dim]
        q = q.view(batch_size, seq_len, self.num_query_groups, self.head_dim)
        
        # Reshape keys and values
        # [batch_size, seq_len, num_heads, head_dim]
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Get rotary embeddings
        cos, sin = self.rotary_emb(x, seq_len)
        
        # Apply rotary embeddings to queries and keys
        # First, we need to expand queries to match the number of heads
        # Each query in a group is shared across multiple heads
        q_expanded = torch.repeat_interleave(q, self.heads_per_group, dim=2)  # [batch_size, seq_len, num_heads, head_dim]
        
        # Apply rotary position embeddings
        cos = cos.unsqueeze(1).unsqueeze(0)  # [1, seq_len, 1, head_dim]
        sin = sin.unsqueeze(1).unsqueeze(0)  # [1, seq_len, 1, head_dim]
        
        q_embed = (q_expanded * cos) + (rotate_half(q_expanded) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        
        # Transpose for attention matrix multiplication
        q_embed = q_embed.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
        k_embed = k_embed.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
        v = v.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(q_embed, k_embed.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # [batch_size, seq_len, num_heads, head_dim]
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        # Final linear projection
        output = self.out_proj(attn_output)
        
        return output

class DeepSeekBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_query_groups=1, dropout=0.1, max_seq_len=2048):
        super().__init__()
        self.pre_norm1 = RMSNorm(d_model)
        self.self_attn = TrueGroupedQueryAttention(d_model, num_heads, num_query_groups, dropout, max_seq_len)
        self.pre_norm2 = RMSNorm(d_model)
        self.ff = SwiGLU(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None, is_causal=False):
        # Pre-norm architecture
        residual = x
        x = self.pre_norm1(x)
        x = self.self_attn(x, mask, is_causal)
        x = residual + self.dropout(x)
        
        residual = x
        x = self.pre_norm2(x)
        x = self.ff(x)
        x = residual + self.dropout(x)
        
        return x

class DeepSeekTransformer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, num_query_groups=1, dropout=0.1, max_seq_len=2048):
        super().__init__()
        self.layers = nn.ModuleList([
            DeepSeekBlock(d_model, num_heads, d_ff, num_query_groups, dropout, max_seq_len)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)
    
    def forward(self, x, mask=None, is_causal=False):
        for layer in self.layers:
            x = layer(x, mask, is_causal)
        
        # Final normalization
        x = self.final_norm(x)
        return x

# Model wrapper for classification tasks
class TransformerClassifier(nn.Module):
    def __init__(self, encoder, d_model, num_classes, seq_len):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(d_model, num_classes)
        self.seq_len = seq_len
    
    def forward(self, x, mask=None):
        # Encode the sequence
        encoded = self.encoder(x, mask)
        
        # Use the [CLS] token (first token) for classification
        cls_representation = encoded[:, 0, :]
        
        # Classify
        logits = self.classifier(cls_representation)
        
        return logits

# Generate synthetic sequence classification data
def generate_synthetic_data(num_samples, seq_len, d_model, num_classes):
    # Generate random sequences
    X = torch.randn(num_samples, seq_len, d_model)
    
    # Generate random labels
    y = torch.randint(0, num_classes, (num_samples,))
    
    # Make the data task meaningful by embedding patterns based on class
    for i in range(num_samples):
        class_idx = y[i].item()
        # Add a class-specific pattern to the sequence
        pattern = torch.sin(torch.arange(0, seq_len) * (class_idx + 1) / seq_len)
        X[i, :, 0] += pattern * 2  # Add pattern to first dimension
    
    return X, y

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    return train_losses, val_losses, train_accs, val_accs

# Main execution
def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Hyperparameters
    d_model = 64
    num_heads = 4
    d_ff = 256
    num_classes = 5
    seq_len = 50
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    
    # Standard Transformer parameters
    std_num_layers = 4
    
    # SLAM parameters
    ef_cycles = 2
    
    # DeepSeek parameters
    deepseek_num_layers = 4
    deepseek_num_query_groups = 2  # Group query attention parameter
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate synthetic data
    print("Generating synthetic data...")
    num_train_samples = 1000
    num_val_samples = 200
    
    X_train, y_train = generate_synthetic_data(num_train_samples, seq_len, d_model, num_classes)
    X_val, y_val = generate_synthetic_data(num_val_samples, seq_len, d_model, num_classes)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize models
    print("Initializing models...")
    
    # Standard Transformer
    std_encoder = StandardTransformer(d_model, num_heads, d_ff, std_num_layers)
    std_model = TransformerClassifier(std_encoder, d_model, num_classes, seq_len)
    
    # SLAM v1
    slam_encoder = SLAMv1(d_model, num_heads, d_ff, ef_cycles)
    slam_model = TransformerClassifier(slam_encoder, d_model, num_classes, seq_len)
    
    # DeepSeek Transformer
    deepseek_encoder = DeepSeekTransformer(d_model, num_heads, d_ff, deepseek_num_layers, deepseek_num_query_groups)
    deepseek_model = TransformerClassifier(deepseek_encoder, d_model, num_classes, seq_len)
    
    # Loss function and optimizers
    criterion = nn.CrossEntropyLoss()
    std_optimizer = optim.Adam(std_model.parameters(), lr=learning_rate)
    slam_optimizer = optim.Adam(slam_model.parameters(), lr=learning_rate)
    deepseek_optimizer = optim.Adam(deepseek_model.parameters(), lr=learning_rate)
    
    # Train Standard Transformer
    print("\nTraining Standard Transformer...")
    start_time = time.time()
    std_train_losses, std_val_losses, std_train_accs, std_val_accs = train_model(
        std_model, train_loader, val_loader, criterion, std_optimizer, num_epochs, device
    )
    std_training_time = time.time() - start_time
    print(f"Standard Transformer training time: {std_training_time:.2f} seconds")
    
    # Train SLAM v1
    print("\nTraining SLAM v1...")
    start_time = time.time()
    slam_train_losses, slam_val_losses, slam_train_accs, slam_val_accs = train_model(
        slam_model, train_loader, val_loader, criterion, slam_optimizer, num_epochs, device
    )
    slam_training_time = time.time() - start_time
    print(f"SLAM v1 training time: {slam_training_time:.2f} seconds")
    
    # Train DeepSeek Transformer
    print("\nTraining DeepSeek Transformer...")
    start_time = time.time()
    deepseek_train_losses, deepseek_val_losses, deepseek_train_accs, deepseek_val_accs = train_model(
        deepseek_model, train_loader, val_loader, criterion, deepseek_optimizer, num_epochs, device
    )
    deepseek_training_time = time.time() - start_time
    print(f"DeepSeek Transformer training time: {deepseek_training_time:.2f} seconds")
    
    # Plot results
    epochs = range(1, num_epochs + 1)
    
    # Plot training and validation loss
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(epochs, std_train_losses, 'b-', label='Standard Train')
    plt.plot(epochs, std_val_losses, 'b--', label='Standard Val')
    plt.plot(epochs, slam_train_losses, 'r-', label='SLAM Train')
    plt.plot(epochs, slam_val_losses, 'r--', label='SLAM Val')
    plt.plot(epochs, deepseek_train_losses, 'g-', label='DeepSeek Train')
    plt.plot(epochs, deepseek_val_losses, 'g--', label='DeepSeek Val')
    plt.title('Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(epochs, std_train_accs, 'b-', label='Standard Train')
    plt.plot(epochs, std_val_accs, 'b--', label='Standard Val')
    plt.plot(epochs, slam_train_accs, 'r-', label='SLAM Train')
    plt.plot(epochs, slam_val_accs, 'r--', label='SLAM Val')
    plt.plot(epochs, deepseek_train_accs, 'g-', label='DeepSeek Train')
    plt.plot(epochs, deepseek_val_accs, 'g--', label='DeepSeek Val')
    plt.title('Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # Plot validation accuracy comparison
    plt.subplot(2, 2, 3)
    plt.plot(epochs, std_val_accs, 'b-', label='Standard')
    plt.plot(epochs, slam_val_accs, 'r-', label='SLAM')
    plt.plot(epochs, deepseek_val_accs, 'g-', label='DeepSeek')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # Plot training time comparison
    plt.subplot(2, 2, 4)
    models = ['Standard', 'SLAM', 'DeepSeek']
    times = [std_training_time, slam_training_time, deepseek_training_time]
    plt.bar(models, times, color=['blue', 'red', 'green'])
    plt.title('Training Time Comparison')
    plt.ylabel('Time (seconds)')
    
    plt.tight_layout()
    plt.savefig('transformer_comparison.png')
    plt.show()
    
    # Print final results
    print("\nFinal Results:")
    print(f"Standard Transformer - Val Accuracy: {std_val_accs[-1]:.2f}%, Training Time: {std_training_time:.2f}s")
    print(f"SLAM v1 - Val Accuracy: {slam_val_accs[-1]:.2f}%, Training Time: {slam_training_time:.2f}s")
    print(f"DeepSeek Transformer - Val Accuracy: {deepseek_val_accs[-1]:.2f}%, Training Time: {deepseek_training_time:.2f}s")
    
    # Count parameters
    std_params = sum(p.numel() for p in std_model.parameters())
    slam_params = sum(p.numel() for p in slam_model.parameters())
    deepseek_params = sum(p.numel() for p in deepseek_model.parameters())
    
    print(f"\nModel Parameters:")
    print(f"Standard Transformer: {std_params:,}")
    print(f"SLAM v1: {slam_params:,}")
    print(f"DeepSeek Transformer: {deepseek_params:,}")

if __name__ == "__main__":
    main()