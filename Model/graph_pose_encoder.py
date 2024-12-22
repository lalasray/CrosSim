import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np

class PoseGraphEncoderTemporal(nn.Module):
    def __init__(self, num_nodes=22, feature_dim=3, embedding_size=512, max_hop=1, dilation=1, temporal_hidden_size=256):
        super(PoseGraphEncoderTemporal, self).__init__()
        
        self.num_nodes = num_nodes
        self.embedding_size = embedding_size
        self.temporal_hidden_size = temporal_hidden_size
        
        # Graph structure
        self.graph = Graph(max_hop=max_hop, dilation=dilation)
        A = self.graph.A.clone().detach().float()
        self.register_buffer('A', A)
        
        # Graph convolution layers
        self.conv1 = GCNConv(in_channels=feature_dim, out_channels=64)
        self.conv2 = GCNConv(in_channels=64, out_channels=128)
        self.conv3 = GCNConv(in_channels=128, out_channels=256)
        self.conv4 = GCNConv(in_channels=256, out_channels=512)
        
        # Temporal modeling using LSTM
        self.lstm = nn.LSTM(input_size=512, hidden_size=temporal_hidden_size, batch_first=True, bidirectional=True)
        
        # Fully connected layer for final embedding
        self.fc = nn.Linear(2 * temporal_hidden_size, self.embedding_size)
    
    def forward(self, data):
        """
        Args:
            data: A tensor of shape (batch_size, time_steps, num_nodes, feature_dim)
        Returns:
            A tensor of shape (batch_size, embedding_size)
        """
        batch_size, time_steps, num_nodes, feature_dim = data.shape
        
        # Reshape and prepare data for graph convolutions
        data = data.view(batch_size * time_steps, num_nodes, feature_dim)
        x = data.permute(1, 0, 2).reshape(-1, feature_dim)  # Node features
        edge_index = self.graph.edge_index.repeat(1, time_steps)  # Repeat edges for all time steps
        
        # Apply graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        
        # Reshape to (batch_size, time_steps, num_nodes, 512)
        x = x.view(batch_size, time_steps, num_nodes, 512)
        x = x.mean(dim=2)  # Aggregate node features across the graph (mean pooling)
        
        # Temporal modeling with LSTM
        x, _ = self.lstm(x)  # Output shape: (batch_size, time_steps, 2 * temporal_hidden_size)
        x = x.mean(dim=1)  # Aggregate across time steps (mean pooling)
        
        # Fully connected layer for the final embedding
        x = self.fc(x)
        return x



# Define the graph structure for pose data
class Graph:
    def __init__(self, max_hop=1, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation
        self.get_edge()
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency()
        self.get_edge_index()  # Create edge_index for PyG

    def get_edge(self):
        self.num_node = 22  # number of body joints/nodes
        self_link = [(i, i) for i in range(self.num_node)]
        neighbor_link = [(1, 0), (2, 1), (3, 2), (4, 3), (5, 0), (6, 5), (7, 6), (8, 7),
                         (9, 0), (10, 9), (11, 10), (12, 11), (13, 12), (14, 11), (15, 14), (16, 15),
                         (17, 16), (18, 11), (19, 18), (20, 19), (21, 20)]
        self.edge = self_link + neighbor_link
        self.center = 0

    def get_adjacency(self):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)
        self.A = torch.tensor(normalize_adjacency, dtype=torch.float32)

    def get_edge_index(self):
        # Convert edge list to edge_index tensor
        edge_index = torch.tensor(self.edge, dtype=torch.long).t()
        self.edge_index = edge_index

def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD

def main():
    import torch

    # Assuming PoseGraphEncoderTemporal is already defined or imported
    # Define a test input tensor
    batch_size = 16
    time_steps = 1000
    num_nodes = 22
    feature_dim = 3
    embedding_size = 512

    # Generate random input data to simulate pose data
    test_input = torch.rand(batch_size, time_steps, num_nodes, feature_dim)

    # Instantiate the model
    model = PoseGraphEncoderTemporal(
        num_nodes=num_nodes,
        feature_dim=feature_dim,
        embedding_size=embedding_size,
        max_hop=1,
        dilation=1,
        temporal_hidden_size=256
    )

    # Perform a forward pass
    output = model(test_input)

    # Print input and output shapes
    print("Input Shape: ", test_input.shape)  # Expected: (batch_size, time_steps, num_nodes, feature_dim)
    print("Output Shape: ", output.shape)    # Expected: (batch_size, embedding_size)


if __name__ == "__main__":
    main()
