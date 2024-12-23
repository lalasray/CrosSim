import torch
import torch.nn as nn

class EmbeddingEncoder:
    def __init__(self, input_size=768, output_size=512):
        """
        Initialize the MLP encoder for dimensionality reduction.
        
        Args:
            input_size (int): The input size of the embedding (768).
            output_size (int): The output size of the embedding (512).
        """
        # Define the MLP encoder with two layers
        self.mlp_encoder = nn.Sequential(
            nn.Linear(input_size, 512),   # Map from input size (768) to 512
            nn.ReLU(),                    # ReLU activation function
            nn.Linear(512, output_size)   # Second layer to keep the final output size (512)
        )
        
        # Move the MLP encoder to the appropriate device (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mlp_encoder.to(self.device)

    def encode_embedding(self, embedding):
        """
        Encode the input random embedding.
        
        Args:
            embedding (torch.Tensor): The random embedding of shape (batch_size, 768).
        
        Returns:
            torch.Tensor: The reduced embedding vector of shape (batch_size, 512).
        """
        # Ensure the embedding is a torch tensor and move it to the correct device
        embedding = embedding.to(self.device)
        
        # Pass the embedding through the MLP encoder to reduce its dimensionality
        reduced_embedding = self.mlp_encoder(embedding)
        
        return reduced_embedding

def generate_random_embedding(batch_size=32, size=768):
    """
    Generate a random batch of embedding tensors.
    
    Args:
        batch_size (int): The number of embeddings in the batch (default 32).
        size (int): The size of each random embedding tensor (default 768).
        
    Returns:
        torch.Tensor: A batch of random embeddings with shape (batch_size, size).
    """
    return torch.randn(batch_size, size)  # Generate random tensor with normal distribution

def main():
    # Initialize the embedding encoder with the MLP encoder
    embedding_encoder = EmbeddingEncoder()

    # Generate a random batch of embeddings (batch_size, 768)
    batch_size = 32
    random_embeddings = generate_random_embedding(batch_size=batch_size, size=768)
    print(f"Random Embeddings shape: {random_embeddings.shape}")

    # Encode the random embeddings and print the resulting reduced embeddings
    reduced_embeddings = embedding_encoder.encode_embedding(random_embeddings)
    
    print(f"Reduced Embeddings shape: {reduced_embeddings.shape}")

if __name__ == "__main__":
    main()
