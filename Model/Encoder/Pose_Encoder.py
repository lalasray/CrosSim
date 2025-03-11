import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATv2Conv
import numpy as np
import math

class TemporalAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Linear(input_dim, 1)
        self.output_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        weights = torch.softmax(self.attention(x), dim=1)  # [batch_size, time_steps, 1]
        x_weighted = (x * weights).sum(dim=1)  # [batch_size, input_dim]
        return self.output_layer(x_weighted)  # [batch_size, output_dim]


class NodeSelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(NodeSelfAttention, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5  # Scaling factor for stability

    def forward(self, x):
        """
        x: (batch_size, num_nodes, hidden_dim)
        """
        Q = self.query(x)  # (batch_size, num_nodes, hidden_dim)
        K = self.key(x)    # (batch_size, num_nodes, hidden_dim)
        V = self.value(x)  # (batch_size, num_nodes, hidden_dim)

        # Compute attention scores (batch_size, num_nodes, num_nodes)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Compute weighted sum (batch_size, num_nodes, hidden_dim)
        attention_out = torch.matmul(attention_weights, V)

        # Aggregate node embeddings (batch_size, hidden_dim)
        aggregated_nodes = attention_out.mean(dim=1)

        return aggregated_nodes


class ResidualGCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualGCN, self).__init__()
        self.gcn = GCNConv(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.residual = (in_channels == out_channels)  # Ensure residual connection is valid

    def forward(self, x, edge_index):
        out = self.gcn(x, edge_index)
        out = self.norm(out)  # Apply LayerNorm
        if self.residual:
            out = out + x  # Residual connection
        return F.relu(out)


class NodePositionalEncoding(nn.Module):
    def __init__(self, num_nodes, hidden_dim):
        super(NodePositionalEncoding, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.encoding = self.create_positional_encoding()

    def create_positional_encoding(self):
        """
        Generates a sinusoidal positional encoding for nodes.
        Shape: (num_nodes, hidden_dim)
        """
        position = torch.arange(self.num_nodes).unsqueeze(1).float()  # Shape: (num_nodes, 1)
        div_term = torch.exp(torch.arange(0, self.hidden_dim, 2).float() * 
                             (-math.log(10000.0) / self.hidden_dim))  # Shape: (hidden_dim/2,)
        
        pe = torch.zeros(self.num_nodes, self.hidden_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe

    def forward(self, x):
        """
        Adds positional encoding to the node features.
        x: (batch_size, num_nodes, feature_dim)
        """
        pe = self.encoding.to(x.device)  # Move encoding to the same device as input
        return x + pe.unsqueeze(0)  # Add encoding to all batch elements


class GraphPoseEncoderPre(nn.Module):
    def __init__(self, num_nodes, feature_dim, hidden_dim, embedding_dim, window_size=1, stride=1, output_dim=512, dropout_prob=0.5):
        super(GraphPoseEncoderPre, self).__init__()
        self.window_size = window_size
        self.stride = stride
        self.num_nodes = num_nodes
        
        # Positional Encoding for Nodes
        self.node_pos_encoding = NodePositionalEncoding(num_nodes, feature_dim)
        
        # Graph convolution layers with Residual connections
        self.conv1 = ResidualGCN(feature_dim, 64)
        self.conv2 = ResidualGCN(64, 128)
        self.conv3 = ResidualGCN(128, 256)
        self.conv4 = ResidualGCN(256, hidden_dim)
        
        # LayerNorm applied after each GCN layer
        self.layer_norm1 = nn.LayerNorm(64)
        self.layer_norm2 = nn.LayerNorm(128)
        self.layer_norm3 = nn.LayerNorm(256)
        self.layer_norm4 = nn.LayerNorm(hidden_dim)

        # Node-Level Self-Attention
        self.node_attention = NodeSelfAttention(hidden_dim)
        
        # Transformer Encoder
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, batch_first=True, dim_feedforward=hidden_dim*2, dropout=dropout_prob)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=6)
        
        # Fully connected layer for final embedding
        self.fc = nn.Linear(hidden_dim, embedding_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)
        
        # Attention model
        self.attention_model = TemporalAttention(embedding_dim, output_dim)

    def forward(self, data, edge_index):
        batch_size, time_steps, num_nodes, feature_dim = data.shape
        embeddings = []
        
        for i in range(0, time_steps - self.window_size + 1, self.stride):
            window = data[:, i:i+self.window_size, :, :].reshape(-1, num_nodes, feature_dim)

            # Apply Positional Encoding
            window = self.node_pos_encoding(window)

            x = self.conv1(window, edge_index)
            x = self.layer_norm1(x)  # LayerNorm after conv1
            x = self.conv2(x, edge_index)
            x = self.layer_norm2(x)  # LayerNorm after conv2
            x = self.conv3(x, edge_index)
            x = self.layer_norm3(x)  # LayerNorm after conv3
            x = self.conv4(x, edge_index)
            x = self.layer_norm4(x)  # LayerNorm after conv4
            
            # Reshape to (batch_size, window_size, num_nodes, hidden_dim)
            x = x.view(batch_size, self.window_size, self.num_nodes, -1)
            
            # Apply Node-Level Self-Attention
            x = self.node_attention(x.view(batch_size * self.window_size, num_nodes, -1))  
            x = x.view(batch_size, self.window_size, -1)  # (batch_size, window_size, hidden_dim)
            
            # Pass through Transformer Encoder
            x = x.permute(1, 0, 2)  # Shape: (window_size, batch_size, hidden_dim)
            transformer_out = self.transformer_encoder(x)  # Shape: (window_size, batch_size, hidden_dim)
            transformer_out = transformer_out.permute(1, 0, 2)  # Shape: (batch_size, window_size, hidden_dim)
            
            # Take the last time step embedding (or you can use a pooling operation)
            embedding = transformer_out[:, -1, :]
            embedding = self.fc(embedding)  # (batch_size, embedding_dim)
            embeddings.append(embedding)
        
        embeddings = torch.stack(embeddings, dim=1)  # (batch_size, num_windows, embedding_dim)
        x_transformed = self.attention_model(embeddings)
        
        return x_transformed


# Support classes and functions
class PoseGraph:
    def __init__(self, max_hop=1, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation
        self.get_edge()
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency()
        self.get_edge_index()

    def get_edge(self):
        self.num_node = 24  # number of body joints/nodes
        self_link = [(i, i) for i in range(self.num_node)]
        neighbor_link = [
            # Pelvis
            (0, 1),  # Pelvis - Left_hip
            (0, 2),  # Pelvis - Right_hip

            # Left side of the body
            (1, 4),  # Left_hip - Left_knee
            (4, 7),  # Left_knee - Left_ankle
            (7, 10), # Left_ankle - Left_foot
            (1, 14), # Left_hip - Left_collar
            (14, 16), # Left_collar - Left_shoulder
            (16, 18), # Left_shoulder - Left_elbow
            (18, 20), # Left_elbow - Left_wrist
            (20, 22), # Left_wrist - Left_palm
            
            # Right side of the body
            (2, 5),  # Right_hip - Right_knee
            (5, 8),  # Right_knee - Right_ankle
            (8, 11), # Right_ankle - Right_foot
            (2, 15), # Right_hip - Right_collar
            (15, 17), # Right_collar - Right_shoulder
            (17, 19), # Right_shoulder - Right_elbow
            (19, 21), # Right_elbow - Right_wrist
            (21, 23), # Right_wrist - Right_palm
            
            # Spine and Neck
            (0, 3),  # Pelvis - Spine1
            (3, 6),  # Spine1 - Spine2
            (6, 9),  # Spine2 - Spine3
            (9, 12), # Spine3 - Neck
            (12, 13), # Neck - Head
        ]

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
    graph = PoseGraph(max_hop=1, dilation=1)
    edge_index = graph.edge_index

    encoder = GraphPoseEncoderPre(num_nodes=24, feature_dim=6, hidden_dim=128,
                                                embedding_dim=64, window_size=1, stride=1,
                                                output_dim=768)

    # Sample input: (batch_size=16, time_steps=25, num_nodes=24, feature_dim=3)
    sample_input = torch.randn(16, 25, 24, 6)

    # Forward pass
    output = encoder(sample_input, edge_index)

    # Print shapes
    print("Input shape:", sample_input.shape)  # (16, 25, 24, 3)
    print("Output shape:", output.shape)  # (16, num_windows, 64)

if __name__ == "__main__":
    main()