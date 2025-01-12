import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np

class TemporalAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Linear(input_dim, 1)
        self.output_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        weights = torch.softmax(self.attention(x), dim=1)  # [batch_size, time_steps, 1]
        x_weighted = (x * weights).sum(dim=1)  # [batch_size, input_dim]
        return self.output_layer(x_weighted)  # [batch_size, output_dim]

class DeepConvGraphEncoderPre(nn.Module):
    def __init__(self, num_nodes, feature_dim, hidden_dim, embedding_dim, window_size=1, stride=1, output_dim=512):
        super(DeepConvGraphEncoderPre, self).__init__()
        self.window_size = window_size
        self.stride = stride
        self.num_nodes = num_nodes
        
        # Graph convolution layers
        self.conv1 = GCNConv(feature_dim, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 256)
        self.conv4 = GCNConv(256, hidden_dim)
        
        # Temporal modeling using LSTM
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        
        # Fully connected layer for final embedding
        self.fc = nn.Linear(hidden_dim * 2, embedding_dim)
        
        # Attention model
        self.attention_model = TemporalAttention(embedding_dim, output_dim)

    def forward(self, data, edge_index):
        batch_size, time_steps, num_nodes, feature_dim = data.shape
        embeddings = []
        
        for i in range(0, time_steps - self.window_size + 1, self.stride):
            window = data[:, i:i+self.window_size, :, :].reshape(-1, num_nodes, feature_dim)
            
            x = F.relu(self.conv1(window, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = F.relu(self.conv3(x, edge_index))
            x = F.relu(self.conv4(x, edge_index))
            
            x = x.view(batch_size, self.window_size, self.num_nodes, -1).mean(dim=2)
            lstm_out, _ = self.lstm(x.view(batch_size, self.window_size, -1))
            embedding = self.fc(lstm_out[:, -1, :])
            embeddings.append(embedding)
        
        embeddings = torch.stack(embeddings, dim=1)
        x_transformed = self.attention_model(embeddings)
        
        return x_transformed

        
class DeepConvGraphEncoderDownstream(nn.Module):
    def __init__(self, num_nodes, feature_dim, hidden_dim, embedding_dim, window_size, stride=2):
        super(DeepConvGraphEncoderDownstream, self).__init__()
        self.window_size = window_size
        self.stride = stride
        self.num_nodes = num_nodes
        
        # Graph convolution layers
        self.conv1 = GCNConv(feature_dim, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 256)
        self.conv4 = GCNConv(256, hidden_dim)
        
        # Temporal modeling using LSTM
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        
        # Fully connected layer for final embedding
        self.fc = nn.Linear(hidden_dim*2, embedding_dim)

    def forward(self, data, edge_index):
        """
        Args:
            data: A tensor of shape (batch_size, time_steps, num_nodes, feature_dim)
            edge_index: The edge index tensor for the graph structure.
        Returns:
            A tensor of shape (batch_size, num_windows, embedding_dim)
        """
        batch_size, time_steps, num_nodes, feature_dim = data.shape
        embeddings = []
        
        # Slide the window over the input sequence with stride
        for i in range(0, time_steps - self.window_size + 1, self.stride):
            window = data[:, i:i+self.window_size, :, :]  # (batch_size, window_size, num_nodes, feature_dim)
            window = window.reshape(-1, num_nodes, feature_dim)  # Reshape for GCN
            
            # Apply graph convolutions
            x = F.relu(self.conv1(window, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = F.relu(self.conv3(x, edge_index))
            x = F.relu(self.conv4(x, edge_index))
            
            # Reshape to (batch_size, window_size, num_nodes, hidden_dim)
            x = x.view(batch_size, self.window_size, self.num_nodes, -1)
            
            # Aggregate node features across the graph (mean pooling)
            x = x.mean(dim=2)
            
            # Temporal modeling with LSTM
            lstm_input = x.view(batch_size, self.window_size, -1)  # (batch_size, window_size, hidden_dim)
            lstm_out, _ = self.lstm(lstm_input)
            lstm_last_out = lstm_out[:, -1, :]  # Use the last output of LSTM for embedding
            
            # Fully connected layer for final embedding
            embedding = self.fc(lstm_last_out)
            embeddings.append(embedding)
        
        return torch.stack(embeddings, dim=1)  # (batch_size, num_windows, embedding_dim)

# Support classes and functions
class IMUGraph:
    def __init__(self, max_hop=1, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation
        self.get_edge()
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency()
        self.get_edge_index()

    def get_edge(self):
        self.num_node = 20  # number of body joints/nodes
        self_link = [(i, i) for i in range(self.num_node)]
        neighbor_link = [(1, 0), (2, 1), (3, 2), (4, 3), (5, 0), (6, 5), (7, 6), (8, 7),
                         (9, 0), (10, 9), (11, 10), (12, 11), (13, 12), (14, 11), (15, 14), (16, 15),
                         (17, 16), (18, 11), (19, 18)]
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
    # Example usage
    graph = IMUGraph(max_hop=1, dilation=1)
    edge_index = graph.edge_index

    encoder = DeepConvGraphEncoderPre(num_nodes=20, feature_dim=6, hidden_dim=128, embedding_dim=64, window_size=1, stride=1)

    # Sample input: (batch_size=16, time_steps=100, num_nodes=22, feature_dim=3)
    sample_input = torch.randn(16, 100, 20, 6)

    # Forward pass
    output = encoder(sample_input, edge_index)

    # Print shapes
    print("Input shape:", sample_input.shape)  # (16, 100, 20, 3)
    print("Output shape:", output.shape)  # (16, num_windows, 64)

