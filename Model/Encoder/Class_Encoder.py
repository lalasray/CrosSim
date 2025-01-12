import torch
import torch.nn as nn

class ClassEncoder(nn.Module):
    def __init__(self, num_classes, embedding_dim, hidden_dim, output_dim):
        super(ClassEncoder, self).__init__()
        # Embedding layer to map class labels to continuous representations
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        # MLP to transform the embedding to a higher-level representation
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)  # Shape: [batch_size, embedding_dim]
        x = torch.relu(self.fc1(embedded))  # Shape: [batch_size, hidden_dim]
        x = self.fc2(x)  # Shape: [batch_size, output_dim]
        return x  # Latent representation for each activity class in the batch

def main():
	# Define the class encoder
	num_classes = 5  # 5 activity classes (walking, running, sitting, lying, jumping)
	embedding_dim = 64  # Embedding dimension for each class
	hidden_dim = 128  # Hidden layer size
	output_dim = 512  # Latent space dimension

	# Instantiate the ClassEncoder
	class_encoder = ClassEncoder(num_classes, embedding_dim, hidden_dim, output_dim)

	# Example batch of class labels
	activity_class_labels = torch.tensor([0, 2, 1])  # "walking", "sitting", "running" (batch size of 3)

	# Pass the batch through the encoder
	encoded_classes_batch = class_encoder(activity_class_labels)

	# Print the encoded output for the batch
	print("Encoded Classes Batch: ", activity_class_labels.shape)
	print("Encoded Classes Batch Shape: ", encoded_classes_batch.shape)

if __name__ == "__main__":
    main()
