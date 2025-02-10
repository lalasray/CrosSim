import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from transformers import AutoModel, AutoTokenizer
import numpy as np
from info_nce import InfoNCE, info_nce

#use predefined model
def predefined_infonce(query, positive_key):
    loss = InfoNCE()
    output = loss(query, positive_key)  # Computes the InfoNCE loss
    return output  # Return the actual loss value

# Cosine similarity function
def cosine_similarity(x, y):
    """
    Computes cosine similarity between two tensors.
    Args:
        x: Tensor of shape (batch_size, embedding_dim)
        y: Tensor of shape (batch_size, embedding_dim)
    Returns:
        Similarity score matrix (batch_size, batch_size)
    """
    return F.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0), dim=-1)

# InfoNCE Loss with Negative Sampling and Batch Handling
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07, hard_negative_mining=True):
        """
        InfoNCE Loss with optional hard negative mining.
        Args:
            temperature: Scaling factor for similarity.
            hard_negative_mining: Flag to indicate whether to perform hard negative mining.
        """
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.hard_negative_mining = hard_negative_mining

    def forward(self, anchor, positive):
        """
        Forward pass to compute the InfoNCE loss.
        Args:
            anchor: Embedding tensor for the anchor.
            positive: Embedding tensor for the positive pair.
        Returns:
            Loss value.
        """
        # Normalize the embeddings (L2 normalization)
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(anchor, positive)
        
        # If hard negative mining is enabled, modify the similarity matrix
        if self.hard_negative_mining:
            # Generate negative pairs by removing the diagonal (positive pairs)
            negative_similarity = similarity_matrix.clone()
            negative_similarity.fill_diagonal_(-float('inf'))  # Exclude the positive pair
            # Select the hardest negative (maximum similarity) for each anchor
            max_negative_similarity, _ = negative_similarity.max(dim=-1)
            similarity_matrix = torch.cat([similarity_matrix, max_negative_similarity.unsqueeze(1)], dim=1)
            # Labels for hard negatives, now we include the hardest negative as a valid pair
            labels = torch.arange(anchor.size(0)).to(anchor.device)
        else:
            # Create labels for positive pairs
            labels = torch.arange(anchor.size(0)).to(anchor.device)
        
        # Scale similarity scores by temperature
        similarity_matrix /= self.temperature
        
        # Apply CrossEntropyLoss
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss
