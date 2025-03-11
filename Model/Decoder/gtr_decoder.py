import torch
import torch.nn as nn
import math

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

class SenteceDecoder(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=256, num_heads=4, num_layers=3, output_dim=768, dropout=0.1):
        super(SenteceDecoder, self).__init__()
        
        # Linear layer to project input into hidden dimension
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        
        # Transformer encoder layers
        self.transformer_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=512, dropout=dropout),
            num_layers=num_layers
        )
        
        # Final output layer to project hidden state to the desired output_dim (768)
        self.output_fc = nn.Linear(hidden_dim, output_dim)
        
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, time_steps, 64)
        Returns: Tensor of shape (batch_size, time_steps, 768)
        """
        x = self.input_fc(x)  # (batch_size, time_steps, hidden_dim)
        x = self.relu(x)  # Apply ReLU activation
        x = self.positional_encoding(x)  # Add positional encoding
        x = self.transformer_layers(x)  # (batch_size, time_steps, hidden_dim)
        x = self.output_fc(x)  # (batch_size, time_steps, 768)
        return x
