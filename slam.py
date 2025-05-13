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

# Example usage
if __name__ == "__main__":
    # Example parameters
    d_model = 64
    num_heads = 4
    d_ff = 256
    ef_cycles = 2
    seq_len = 50
    batch_size = 8
    
    # Create a sample input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Initialize SLAM model
    slam_model = SLAMv1(d_model, num_heads, d_ff, ef_cycles)
    
    # Forward pass
    output = slam_model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("SLAM model successfully tested!")