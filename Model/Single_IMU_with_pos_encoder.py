import torch
import torch.nn as nn
import torch.nn.functional as F

class IMUSingleNodeEncoderWithClass(nn.Module):
    def __init__(self, feature_dim=3, embedding_size=512, temporal_hidden_size=256, num_classes=20):
        super(IMUSingleNodeEncoderWithClass, self).__init__()

        self.embedding_size = embedding_size
        self.temporal_hidden_size = temporal_hidden_size
        self.num_classes = num_classes
        
        # Fully connected layers for node feature processing (IMU data)
        self.fc1 = nn.Linear(feature_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 512)
        
        # Temporal modeling using LSTM
        self.lstm = nn.LSTM(input_size=512, hidden_size=temporal_hidden_size, batch_first=True, bidirectional=True)
        
        # Fully connected layer for final embedding (after LSTM)
        self.fc = nn.Linear(2 * temporal_hidden_size, self.embedding_size)
        
        # MLP for processing the one-hot class input
        self.class_fc1 = nn.Linear(num_classes, 64)
        self.class_fc2 = nn.Linear(64, 128)
        self.class_fc3 = nn.Linear(128, 256)
        self.class_fc4 = nn.Linear(256, 512)

        # Final layer for combining IMU data output and class output
        self.final_fc = nn.Linear(self.embedding_size + 512, self.embedding_size)
    
    def forward(self, imu_data, class_data):
        """
        Args:
            imu_data: A tensor of shape (batch_size, time_steps, 1, feature_dim)
            class_data: A tensor of shape (batch_size, num_classes) representing one-hot encoded classes
        Returns:
            A tensor of shape (batch_size, embedding_size)
        """
        batch_size, time_steps, num_nodes, feature_dim = imu_data.shape
        
        # Reshape and prepare imu_data for processing
        imu_data = imu_data.view(batch_size * time_steps, feature_dim)
        
        # Process IMU data through fully connected layers
        imu_x = F.relu(self.fc1(imu_data))
        imu_x = F.relu(self.fc2(imu_x))
        imu_x = F.relu(self.fc3(imu_x))
        imu_x = F.relu(self.fc4(imu_x))
        
        # Reshape to (batch_size, time_steps, 512)
        imu_x = imu_x.view(batch_size, time_steps, 512)
        
        # Temporal modeling with LSTM
        imu_x, _ = self.lstm(imu_x)  # Output shape: (batch_size, time_steps, 2 * temporal_hidden_size)
        imu_x = imu_x.mean(dim=1)  # Aggregate across time steps (mean pooling)
        
        # Process class_data through MLP
        class_x = F.relu(self.class_fc1(class_data))
        class_x = F.relu(self.class_fc2(class_x))
        class_x = F.relu(self.class_fc3(class_x))
        class_x = F.relu(self.class_fc4(class_x))
        
        # Concatenate IMU data output and class data output
        combined_x = torch.cat((imu_x, class_x), dim=1)
        
        # Final fully connected layer for the final embedding
        final_output = self.final_fc(combined_x)
        
        return final_output


def main():
    import torch

    # Define a test input tensor
    batch_size = 16
    time_steps = 1200
    num_nodes = 1  # Single node
    feature_dim = 6
    num_classes = 20
    embedding_size = 512

    # Generate random input data to simulate IMU data (batch_size, time_steps, 1, feature_dim)
    imu_data = torch.rand(batch_size, time_steps, num_nodes, feature_dim)

    # Generate random one-hot class data (batch_size, num_classes)
    class_data = torch.eye(num_classes)[torch.randint(0, num_classes, (batch_size,))]

    # Instantiate the model
    model = IMUSingleNodeEncoderWithClass(
        feature_dim=feature_dim,
        embedding_size=embedding_size,
        temporal_hidden_size=256,
        num_classes=num_classes
    )

    # Perform a forward pass
    output = model(imu_data, class_data)

    # Print input and output shapes
    print("IMU Data Shape: ", imu_data.shape)  # Expected: (batch_size, time_steps, 1, feature_dim)
    print("Class Data Shape: ", class_data.shape)  # Expected: (batch_size, num_classes)
    print("Output Shape: ", output.shape)    # Expected: (batch_size, embedding_size)


if __name__ == "__main__":
    main()
