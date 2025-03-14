import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, time_steps, d_model)
        Returns: x with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]

class SentenceDecoder(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=256, num_heads=4, num_layers=3, output_dim=768, dropout=0.1):
        super(SentenceDecoder, self).__init__()

        # Linear layer to project input into hidden dimension
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)  # Ensure this class exists

        # Transformer encoder layers
        self.transformer_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=512, dropout=dropout),
            num_layers=num_layers
        )

        # Final output layer to project hidden state to the desired output_dim (768)
        self.output_fc = nn.Linear(hidden_dim, output_dim)

        # Attention weight computation
        self.attention_fc = nn.Linear(output_dim, 1)  # Learnable attention scores

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, time_steps, 64)
        Returns: Tensor of shape (batch_size, 768) after attention-based aggregation
        """
        x = self.input_fc(x)  # (batch_size, time_steps, hidden_dim)
        x = self.relu(x)  # Apply ReLU activation
        x = self.positional_encoding(x)  # Add positional encoding

        # Transpose for transformer encoder
        x = x.transpose(0, 1)  # Convert to (time_steps, batch_size, hidden_dim)
        x = self.transformer_layers(x)
        x = x.transpose(0, 1)  # Convert back to (batch_size, time_steps, hidden_dim)

        x = self.output_fc(x)  # (batch_size, time_steps, 768)

        # Compute attention scores
        attn_scores = self.attention_fc(x).squeeze(-1) / (768 ** 0.5)  # Scaling for stability
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch_size, time_steps), normalized

        # Weighted sum of x using attention weights
        x = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # (batch_size, 768)

        return x
