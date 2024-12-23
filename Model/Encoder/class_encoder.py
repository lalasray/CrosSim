import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassEncoder(nn.Module):
    def __init__(self, num_classes, embedding_dim=512):
        """
        Initialize the encoder to project one-hot class vectors to a continuous embedding space.
        
        Args:
            num_classes (int): The number of classes (size of the one-hot encoding).
            embedding_dim (int): The size of the embedding space.
        """
        super(ClassEncoder, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        
        # MLP Encoder for the one-hot class vector
        self.fc1 = nn.Linear(num_classes, 1024)  # First layer
        self.fc2 = nn.Linear(1024, 512)  # Second layer
        self.fc3 = nn.Linear(512, embedding_dim)  # Output layer

    def forward(self, x):
        """
        Forward pass through the encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_classes).
        
        Returns:
            torch.Tensor: The embedding of shape (batch_size, embedding_dim).
        """
        x = F.relu(self.fc1(x))  # Apply ReLU activation
        x = F.relu(self.fc2(x))  # Apply ReLU activation
        x = self.fc3(x)  # Output embedding
        return x

# Example usage
if __name__ == "__main__":
    # Define number of classes and batch size
    num_classes = 300  # Example: 10 classes
    batch_size = 32
    embedding_dim = 512  # Size of the embedding space

    # Create a one-hot encoded batch (batch_size, num_classes)
    labels = torch.zeros(batch_size, num_classes)
    labels[torch.arange(batch_size), torch.randint(0, num_classes, (batch_size,))] = 1

    # Initialize the encoder model
    encoder = ClassEncoder(num_classes, embedding_dim)

    # Pass the one-hot encoded labels through the encoder
    embeddings = encoder(labels)
    
    print(f"Embeddings shape: {embeddings.shape}")  # Should be (batch_size, embedding_dim)