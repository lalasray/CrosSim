import vec2text
import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn

import torch
import torch.nn as nn

class EmbeddingEncoder(nn.Module):  # Inherit from nn.Module
    def __init__(self, input_size=768, output_size=512):
        """
        Initialize the MLP encoder for dimensionality reduction.
        
        Args:
            input_size (int): The input size of the embedding (default: 768).
            output_size (int): The output size of the embedding (default: 512).
        """
        super(EmbeddingEncoder, self).__init__()  # Initialize the parent class

        # Define the MLP encoder with two layers
        self.mlp_encoder = nn.Sequential(
            nn.Linear(input_size, 768),   # Map from input size (768) to 512
            nn.ReLU(),                    # ReLU activation function
            nn.Linear(768, output_size)   # Second layer to output the final size (512)
        )

        # Move the MLP encoder to the appropriate device (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  # Move the whole model to the device

    def forward(self, embedding):
        """
        Encode the input embedding.
        
        Args:
            embedding (torch.Tensor): The input embedding of shape (batch_size, input_size).
        
        Returns:
            torch.Tensor: The reduced embedding vector of shape (batch_size, output_size).
        """
        # Ensure the embedding is a torch tensor and move it to the correct device
        embedding = embedding.to(self.device)
        
        # Pass the embedding through the MLP encoder to reduce its dimensionality
        reduced_embedding = self.mlp_encoder(embedding)
        
        return reduced_embedding



def main():
	# Initialize the EmbeddingEncoder
	embedding_encoder = EmbeddingEncoder()

	# Encode the embeddings to reduce dimensionality
	reduced_embeddings = embedding_encoder.encode_embedding(embeddings)

	# Output the reduced embeddings
	print(reduced_embeddings.shape)
if __name__ == "__main__":
    main()

