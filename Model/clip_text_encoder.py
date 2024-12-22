import torch
from transformers import CLIPProcessor, CLIPModel
import random
import string

class TextEncoder:
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32"):
        """
        Initialize the CLIP model and processor.
        
        Args:
            clip_model_name (str): The name of the pretrained CLIP model.
        """
        # Load the CLIP model and processor from Hugging Face
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        # Freeze the CLIP model parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Move model to the appropriate device (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model.to(self.device)

    def encode_text(self, text_data):
        """
        Encode the input text using the CLIP model.
        
        Args:
            text_data (str): The input text description to be encoded.
        
        Returns:
            torch.Tensor: The embedding vector for the input text.
        """
        # Process the text data
        inputs = self.clip_processor(text=text_data, return_tensors="pt", padding=True, truncation=True)
        
        # Move inputs to the same device as the model
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        # Extract the text features (embeddings) from CLIP
        with torch.no_grad():
            clip_embeddings = self.clip_model.get_text_features(**inputs)
        
        return clip_embeddings.squeeze(0)  # Remove batch dimension

def generate_random_text(length=10):
    """
    Generate a random string of specified length.
    
    Args:
        length (int): The length of the random string.
        
    Returns:
        str: A random string.
    """
    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for i in range(length))

def main():
    # Initialize the text encoder with the CLIP model
    text_encoder = TextEncoder()

    # Generate a random text description
    random_text = generate_random_text(length=20)  # Random string of 20 characters
    print(f"Random Text: {random_text}")

    # Encode the text and print the resulting embedding
    text_embedding = text_encoder.encode_text(random_text)
    
    print(f"Text embedding shape: {text_embedding.shape}")

if __name__ == "__main__":
    main()
