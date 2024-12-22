import torch
import torch.nn as nn

class DeepConvLSTMTextEncoder(nn.Module):
    def __init__(self, num_filters=64, filter_size=5, num_units_lstm=128, output_dim=512):
        super(DeepConvLSTMTextEncoder, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(2, num_filters, kernel_size=(filter_size, 1))  # 2 channels (acc, gyro)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=(filter_size, 1))
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=(filter_size, 1))
        self.conv4 = nn.Conv2d(num_filters, num_filters, kernel_size=(filter_size, 1))

        # LSTM layers
        self.lstm1 = nn.LSTM(input_size=num_filters, hidden_size=num_units_lstm, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=num_units_lstm, hidden_size=num_units_lstm, batch_first=True)

        # Fully connected layer to get the embedding (512-dimensional output)
        self.fc = nn.Linear(num_units_lstm, output_dim)

    def forward(self, x):
        # x is of shape (batch_size, 2, 1200, 3)

        # Apply the convolutional layers
        x = self.conv1(x)  # Output shape: (batch_size, num_filters, 1196, 1)
        
        x = self.conv2(x)  # Output shape: (batch_size, num_filters, 1192, 1)
        
        x = self.conv3(x)  # Output shape: (batch_size, num_filters, 1188, 1)
        
        x = self.conv4(x)  # Output shape: (batch_size, num_filters, 1184, 1)
        
        # Now, the tensor has shape [batch_size, num_filters, 1184, 3]
        # Reshape the tensor to remove the last dimension (size 3) and combine it with the sequence dimension
        x = x.view(x.size(0), x.size(1), -1)  # Flatten the last dimension: shape [batch_size, num_filters, 1184 * 3]
        
        # Permute the dimensions: (batch_size, 1184 * 3, num_filters)
        x = x.permute(0, 2, 1)  # Output shape: (batch_size, 1184 * 3, num_filters)
        
        # Apply the LSTM layers
        x, _ = self.lstm1(x)  # Output shape: (batch_size, sequence_length, num_units_lstm)
        x, _ = self.lstm2(x)  # Output shape: (batch_size, sequence_length, num_units_lstm)

        x = x[:, -1, :]  # Output shape: (batch_size, num_units_lstm)
        # Apply the fully connected layer to get the 512-dimensional embedding
        x = self.fc(x)  # Output shape: (batch_size, output_dim)

        return x

def main():
    # Example usage:
    batch_size = 32
    sequence_length = 1200
    num_sensor_channels = 2
    num_dimensions = 3

    # Create a dummy input tensor with shape (batch_size, 2 (channels), 1200 (frames), 3 (x, y, z))
    input_data = torch.randn(batch_size, num_sensor_channels, sequence_length, num_dimensions)

    # Instantiate the model
    model = DeepConvLSTMTextEncoder()

    # Get the output embedding
    embedding = model(input_data)
    print(f"Output embedding shape: {embedding.shape}")  # Expected output shape: (batch_size, 512)

if __name__ == "__main__":
    main()
