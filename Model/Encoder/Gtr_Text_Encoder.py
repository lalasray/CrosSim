import vec2text
import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn

# EmbeddingEncoder class definition
class EmbeddingEncoder:
    def __init__(self, input_size=768, output_size=512):
        """
        Initialize the MLP encoder for dimensionality reduction.
        
        Args:
            input_size (int): The input size of the embedding (default: 768).
            output_size (int): The output size of the embedding (default: 512).
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

# Function to get embeddings from text
def get_gtr_embeddings(text_list,
                       encoder: nn.Module,
                       tokenizer: AutoTokenizer) -> torch.Tensor:
    inputs = tokenizer(text_list,
                       return_tensors="pt",
                       max_length=128,
                       truncation=True,
                       padding="max_length").to("cuda")

    with torch.no_grad():
        model_output = encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        hidden_state = model_output.last_hidden_state
        embeddings = vec2text.models.model_utils.mean_pool(hidden_state, inputs['attention_mask'])

    return embeddings

# Load models and tokenizer
encoder = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder.to("cuda")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
corrector = vec2text.load_pretrained_corrector("gtr-base")

def main():
	# Define a sample text string
	text_string = "The quick brown fox jumps over the lazy dog."

	# Convert the text string into a list (as the function expects a list)
	text_list = [text_string]

	# Get embeddings for the text
	embeddings = get_gtr_embeddings(text_list, encoder, tokenizer)

	# Initialize the EmbeddingEncoder
	embedding_encoder = EmbeddingEncoder()

	# Encode the embeddings to reduce dimensionality
	reduced_embeddings = embedding_encoder.encode_embedding(embeddings)

	# Output the reduced embeddings
	print(reduced_embeddings.shape)
if __name__ == "__main__":
    main()

