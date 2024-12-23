import torch
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer
import numpy as np
from sklearn.decomposition import PCA
import os
from transformers import CLIPProcessor, CLIPModel

# Function to get GTR embeddings
def get_gtr_embeddings(text_list, encoder, tokenizer):
    inputs = tokenizer(text_list,
                       return_tensors="pt",
                       max_length=128,
                       truncation=True,
                       padding="max_length").to("cuda")

    with torch.no_grad():
        model_output = encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        hidden_state = model_output.last_hidden_state
        embeddings = hidden_state.mean(dim=1)  # Mean pooling
    return embeddings

# Function to get CLIP embeddings using the provided TextEncoder class
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

# Generate activity sentences
activities = [
    "Running in the park",
    "Jogging in the park",
    "Walking in the park",
    "Cycling around the city",
    "Riding a bicycle in the city",
    "Lifting weights at the gym",
    "Doing strength training at the gym",
    "Yoga session in the morning",
    "Morning yoga practice",
    "Sitting at the desk working on a computer"
]

# Load GTR model
device = "cuda" if torch.cuda.is_available() else "cpu"
gtr_encoder = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder.to(device)
gtr_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")

# Generate GTR embeddings
gtr_embeddings = get_gtr_embeddings(activities, gtr_encoder, gtr_tokenizer)

# Initialize the TextEncoder for CLIP
text_encoder = TextEncoder()

# Generate CLIP embeddings
clip_embeddings = text_encoder.encode_text(activities)

# Reduce dimensions using PCA for visualization
pca = PCA(n_components=2)

gtr_pca = pca.fit_transform(gtr_embeddings.cpu().numpy())
clip_pca = pca.fit_transform(clip_embeddings.cpu().numpy())

# Plot the embeddings
fig, axs = plt.subplots(1, 2, figsize=(14, 7))

# Plot GTR embeddings
axs[0].scatter(gtr_pca[:, 0], gtr_pca[:, 1], color='blue')
for i, activity in enumerate(activities):
    axs[0].text(gtr_pca[i, 0], gtr_pca[i, 1], activity, fontsize=9)

axs[0].set_title("GTR Embeddings")
axs[0].set_xlabel("Principal Component 1")
axs[0].set_ylabel("Principal Component 2")

# Plot CLIP embeddings
axs[1].scatter(clip_pca[:, 0], clip_pca[:, 1], color='green')
for i, activity in enumerate(activities):
    axs[1].text(clip_pca[i, 0], clip_pca[i, 1], activity, fontsize=9)

axs[1].set_title("CLIP Embeddings")
axs[1].set_xlabel("Principal Component 1")
axs[1].set_ylabel("Principal Component 2")

# Display plots
plt.tight_layout()
plt.show()
