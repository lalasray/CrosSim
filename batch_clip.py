import os
import torch
from transformers import CLIPProcessor, CLIPModel

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
        
        return clip_embeddings

def process_text_file(file_path):
    """
    Read text from a file, generate CLIP embeddings, and save the embeddings to a .pt file.
    
    Args:
        file_path (str): The path to the input text file.
    """
    # Check if the file exists
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return

    # Read text data from the file
    with open(file_path, 'r') as f:
        text_data = [line.strip() for line in f if line.strip()]  # Remove empty lines

    if not text_data:
        print(f"No valid text found in the file: {file_path}")
        return

    # Initialize the text encoder
    text_encoder = TextEncoder()

    # Encode the text
    text_embeddings = text_encoder.encode_text(text_data)

    # Save the embeddings to a .pt file in the same directory
    base_name = os.path.splitext(file_path)[0]  # Remove the .txt extension
    output_path = f"{base_name}_clip.pt"
    torch.save(text_embeddings, output_path)
    print(f"Embeddings saved to {output_path}")

def process_all_text_files_in_directory(directory_path):
    """
    Process all text files in a directory and its subdirectories.
    
    Args:
        directory_path (str): The path to the directory containing text files.
    """
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                process_text_file(file_path)

if __name__ == "__main__":
    
    directory_path = "test_label"  # Replace with the desired directory path
    print(f"Processing text files from the directory: {directory_path}")

    # Process all text files in the directory and its subdirectories
    process_all_text_files_in_directory(directory_path)