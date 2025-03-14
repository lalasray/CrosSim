import torch
import torch.nn as nn
from Encoder.Multi_IMU_Encoder import DeepConvGraphEncoderPre, IMUGraph
from Encoder.Gtr_Text_Encoder import EmbeddingEncoder
from Encoder.Pose_Encoder import GraphPoseEncoderPre, PoseGraph
from Decoder.gtr_decoder import SentenceDecoder
from Decoder.pose_decoder import PoseDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to count trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class BiModalIMU(nn.Module):
    def __init__(self, embedding_size=768, pose_joints=24, imu_positions=21, window=1, stride_size=1, hof=3, dilation=1):
        super(BiModalIMU, self).__init__()
        self.text_encoder = EmbeddingEncoder(output_size=embedding_size).to(device)
        self.imu_encoder_grav = DeepConvGraphEncoderPre(num_nodes=imu_positions, feature_dim=6, hidden_dim=128,
                                                        embedding_dim=64, window_size=window * 4, stride=stride_size * 4,
                                                        output_dim=embedding_size).to(device)
        self.IMU_edge_index = IMUGraph(max_hop=hof, dilation=dilation).edge_index.to(device)

    def forward(self, text, imu_grav):
        text_embeddings = self.text_encoder(text)
        imu_embeddings_grav, imuint = self.imu_encoder_grav(imu_grav, self.IMU_edge_index)
        return text_embeddings, imu_embeddings_grav

class BiModalIMUDownText(nn.Module):
    def __init__(self, embedding_size=768, pose_joints=24, imu_positions=21, window=1, stride_size=1, hof=3, dilation=1):
        super(BiModalIMUDownText, self).__init__()
        self.text_encoder = EmbeddingEncoder(output_size=embedding_size).to(device)
        self.imu_encoder_grav = DeepConvGraphEncoderPre(num_nodes=imu_positions, feature_dim=6, hidden_dim=128,
                                                        embedding_dim=64, window_size=window * 4, stride=stride_size * 4,
                                                        output_dim=embedding_size).to(device)
        self.IMU_edge_index = IMUGraph(max_hop=hof, dilation=dilation).edge_index.to(device)
        self.sentence_decoder = SentenceDecoder()

    def forward(self, text, imu_grav):
        text_embeddings = self.text_encoder(text)
        imu_embeddings_grav, imuint = self.imu_encoder_grav(imu_grav, self.IMU_edge_index)
        gtr = self.sentence_decoder(imuint)
        return text_embeddings, imu_embeddings_grav, gtr

class BiModalIMUDownPose(nn.Module):
    def __init__(self, embedding_size=768, pose_joints=24, imu_positions=21, window=1, stride_size=1, hof=3, dilation=1):
        super(BiModalIMUDownPose, self).__init__()
        self.text_encoder = EmbeddingEncoder(output_size=embedding_size).to(device)
        self.imu_encoder_grav = DeepConvGraphEncoderPre(num_nodes=imu_positions, feature_dim=6, hidden_dim=128,
                                                        embedding_dim=64, window_size=window * 4, stride=stride_size * 4,
                                                        output_dim=embedding_size).to(device)
        self.IMU_edge_index = IMUGraph(max_hop=hof, dilation=dilation).edge_index.to(device)
        self.pose_decoder = PoseDecoder()

    def forward(self, text, imu_grav):
        text_embeddings = self.text_encoder(text)
        imu_embeddings_grav, imuint = self.imu_encoder_grav(imu_grav, self.IMU_edge_index)
        pose_emb = self.pose_decoder(imuint)
        return text_embeddings, imu_embeddings_grav, pose_emb
    
class BiModalPose(nn.Module):
    def __init__(self, embedding_size=768, pose_joints=24, imu_positions=21, window=1, stride_size=1, hof=3, dilation=1):
        super(BiModalPose, self).__init__()
        self.pose_encoder = GraphPoseEncoderPre(num_nodes=pose_joints, feature_dim=6, hidden_dim=128,
                                                embedding_dim=64, window_size=window, stride=stride_size,
                                                output_dim=embedding_size).to(device)
        self.imu_encoder_grav = DeepConvGraphEncoderPre(num_nodes=imu_positions, feature_dim=6, hidden_dim=128,
                                                        embedding_dim=64, window_size=window * 4, stride=stride_size * 4,
                                                        output_dim=embedding_size).to(device)
        self.pose_edge_index = PoseGraph(max_hop=hof, dilation=dilation).edge_index.to(device)
        self.IMU_edge_index = IMUGraph(max_hop=hof, dilation=dilation).edge_index.to(device)

    def forward(self, pose, imu_grav):
        pose_embeddings, poseint = self.pose_encoder(pose, self.pose_edge_index)
        imu_embeddings_grav, imuint = self.imu_encoder_grav(imu_grav, self.IMU_edge_index)
        return pose_embeddings,imu_embeddings_grav