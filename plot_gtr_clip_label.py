import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def load_embeddings_and_labels(directory, suffix):
    """
    Load embeddings from .pt files and associate them with labels from text files.

    Args:
        directory (str): The directory to search for .pt files.
        suffix (str): The suffix of the files to load (e.g., '_clip.pt').

    Returns:
        list of (str, torch.Tensor): A list of tuples containing text labels and embeddings.
    """
    embeddings = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(suffix):
                base_name = file.replace(suffix, "")
                text_file = os.path.join(root, f"{base_name}.txt")

                # Read label from the corresponding text file
                if os.path.isfile(text_file):
                    with open(text_file, 'r') as f:
                        label = f.read().strip()
                else:
                    label = "Unknown"

                file_path = os.path.join(root, file)
                embedding = torch.load(file_path)
                embeddings.append((label, embedding))
    return embeddings

def visualize_embeddings(embeddings, title, ax):
    """
    Visualize embeddings using t-SNE and display labels.

    Args:
        embeddings (list of (str, torch.Tensor)): List of tuples containing labels and embeddings.
        title (str): Title for the subplot.
        ax (matplotlib.axes.Axes): The axes to plot on.
    """
    # Extract labels and embedding tensors
    labels = [item[0] for item in embeddings]
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
        ax.scatter(x, y, alpha=0.7)
        ax.text(x + 0.1, y + 0.1, labels[i], fontsize=9)
    ax.set_title(title)
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")

def main():
    # Specify the directory path
    directory_path = "test_label"  # Replace with the desired directory path
    print(f"Processing text files from the directory: {directory_path}")

    # Load both CLIP and GTR embeddings along with their labels
    clip_embeddings = load_embeddings_and_labels(directory_path, "_clip.pt")
    gtr_embeddings = load_embeddings_and_labels(directory_path, "_gtr.pt")

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
