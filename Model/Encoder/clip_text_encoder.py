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
        Encode the input batch of text using the CLIP model.
        
        Args:
            text_data (list of str): The input batch of text descriptions to be encoded.
        
        Returns:
            torch.Tensor: The batch of embedding vectors for the input text.
        """
        # Process the batch of text data
        inputs = self.clip_processor(text=text_data, return_tensors="pt", padding=True, truncation=True)
        
        # Move inputs to the same device as the model
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        # Extract the text features (embeddings) from CLIP
        with torch.no_grad():
            clip_embeddings = self.clip_model.get_text_features(**inputs)
        
        return clip_embeddings  # No need to squeeze, as it's already batch_size, 512

def generate_random_text(batch_size=32, length=20):
    """
    Generate a batch of random strings of specified length.
    
    Args:
        batch_size (int): The number of random strings to generate.
        length (int): The length of each random string.
        
    Returns:
        list of str: A list of random strings.
    """
    letters = string.ascii_letters + string.digits
    return [''.join(random.choice(letters) for i in range(length)) for _ in range(batch_size)]

def main():
    # Initialize the text encoder with the CLIP model
    text_encoder = TextEncoder()

    # Generate a random batch of text descriptions
    batch_size = 32
    random_text_batch = generate_random_text(batch_size=batch_size, length=20)  # 32 random strings of 20 characters
    print(f"Random Text Batch: {random_text_batch[:3]}...")  # Print a sample of the batch

    # Encode the batch of text and print the resulting embeddings
    text_embeddings = text_encoder.encode_text(random_text_batch)
    
    print(f"Text embeddings shape: {text_embeddings.shape}")

if __name__ == "__main__":
    main()
