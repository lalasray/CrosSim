import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def load_embeddings(directory, suffix):
    """
    Load embeddings from .pt files with a specific suffix in a directory and its subdirectories.

    Args:
        directory (str): The directory to search for .pt files.
        suffix (str): The suffix of the files to load (e.g., '_clip.pt').

    Returns:
        list of (str, torch.Tensor): A list of tuples containing file names and embeddings.
    """
    embeddings = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(suffix):
                file_path = os.path.join(root, file)
                file_name = os.path.basename(file)
                embedding = torch.load(file_path)
                embeddings.append((file_name, embedding))
    return embeddings

def visualize_embeddings(embeddings, title, ax):
    """
    Visualize embeddings using t-SNE and display the file names as labels.

    Args:
        embeddings (list of (str, torch.Tensor)): List of tuples containing file names and embeddings.
        title (str): Title for the subplot.
        ax (matplotlib.axes.Axes): The axes to plot on.
    """
    # Extract file names and embedding tensors
    file_names = [item[0] for item in embeddings]
    embedding_tensors = [item[1].mean(dim=0).cpu().numpy() for item in embeddings]  # Average over batch if needed

    # Convert embeddings to a NumPy array
    embedding_matrix = np.stack(embedding_tensors)

    # Dynamically adjust perplexity
    n_samples = embedding_matrix.shape[0]
    perplexity = min(30, n_samples - 1)  # Perplexity must be less than the number of samples

    # Reduce dimensions using t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=300)
    reduced_embeddings = tsne.fit_transform(embedding_matrix)

    # Plot the embeddings
    for i, (x, y) in enumerate(reduced_embeddings):
        ax.scatter(x, y, label=file_names[i], alpha=0.7)
        ax.text(x + 0.1, y + 0.1, file_names[i], fontsize=9)
    ax.set_title(title)
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")

def main():
    # Directory containing the embeddings
    directory_path = input("Enter the path to the directory: ").strip()

    # Load both CLIP and GTR embeddings
    clip_embeddings = load_embeddings(directory_path, "_clip.pt")
    gtr_embeddings = load_embeddings(directory_path, "_gtr.pt")

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Visualize CLIP embeddings
    visualize_embeddings(clip_embeddings, "CLIP Embeddings", axes[0])

    # Visualize GTR embeddings
    visualize_embeddings(gtr_embeddings, "GTR Embeddings", axes[1])

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
