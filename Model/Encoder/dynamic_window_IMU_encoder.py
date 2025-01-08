import torch
import torch.nn as nn

class IMUSlidingWindowEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, window_size, stride):
        super(IMUSlidingWindowEncoder, self).__init__()
        self.window_size = window_size
        self.stride = stride
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True,bidirectional=True)
        self.fc = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        # x is of shape (batch_size, seq_length, input_dim)
        batch_size, seq_length, _ = x.shape
        embeddings = []
        
        # Slide the window over the input sequence with stride
        for i in range(0, seq_length - self.window_size + 1, self.stride):
            window = x[:, i:i+self.window_size, :]  # (batch_size, window_size, input_dim)
            lstm_out, _ = self.lstm(window)  # lstm_out: (batch_size, window_size, hidden_dim)
            last_out = lstm_out[:, -1, :]  # Use the last output of LSTM for embedding
            embedding = self.fc(last_out)  # (batch_size, embedding_dim)
            embeddings.append(embedding)

        return torch.stack(embeddings, dim=1)  # (batch_size, num_windows, embedding_dim)

# Example usage:
encoder = IMUSlidingWindowEncoder(input_dim=6, hidden_dim=128, embedding_dim=64, window_size=30, stride=5)

# Sample input: (batch_size=16, seq_length=100, input_dim=6)
sample_input = torch.randn(16, 200, 6)

# Forward pass
output = encoder(sample_input)

# Print shapes
print("Input shape:", sample_input.shape)  # (16, 100, 6)
print("Output shape:", output.shape)  # (16, num_windows, 64)

