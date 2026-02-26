import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
try:
    from .resnet1d import Net1D
except ImportError:
    from resnet1d import Net1D
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class PPGEncoderConfig:
    in_channels: int = 1
    base_filters: int = 64
    ratio: float = 1.0
    filter_list: list = field(default_factory=lambda: [64, 160, 160, 400, 400, 512])
    m_blocks_list: list = field(default_factory=lambda: [2, 2, 2, 3, 3, 1])
    kernel_size: int = 3
    stride: int = 2
    groups_width: int = 16
    use_bn: bool = True
    use_do: bool = True
    verbose: bool = False


class MLP(nn.Module):
    """
    Standard MLP for Projector and Predictor.
    Structure: Linear -> BN -> GELU -> Linear
    """
    def __init__(self, dim_in, dim_hidden, dim_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.BatchNorm1d(dim_hidden),
            nn.GELU(),
            nn.Linear(dim_hidden, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class BYOL(nn.Module):
    def __init__(
        self, 
        ppg_encoder_config: PPGEncoderConfig,
        projection_size: int = 128, 
        hidden_size: int = 512,  
        moving_average_decay: float = 0.99 
    ) -> None:
        super().__init__()
        
        self.moving_average_decay = moving_average_decay
        self.feature_dim = ppg_encoder_config.filter_list[-1]

        self.online_encoder = Net1D(**ppg_encoder_config.__dict__)
        self.online_projector = MLP(self.feature_dim, hidden_size, projection_size)
        self.online_predictor = MLP(projection_size, hidden_size, projection_size) 

        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)

        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_target_network(self):
        """
        Momentum update of the target network parameters.
        theta_target = tau * theta_target + (1 - tau) * theta_online
        Should be called after every optimization step.
        """
        for online_params, target_params in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_params.data = self.moving_average_decay * target_params.data + \
                                 (1 - self.moving_average_decay) * online_params.data
                                 
        for online_params, target_params in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            target_params.data = self.moving_average_decay * target_params.data + \
                                 (1 - self.moving_average_decay) * online_params.data

    def loss_fn(self, x, y):
        """
        Cosine Similarity Loss.
        BYOL minimizes: 2 - 2 * cos(x, y)
        Input vectors must be normalized!
        """
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)

    def forward(
        self, 
        ppg_view1: torch.Tensor,
        ppg_view2: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        
        online_h1 = self.online_encoder(ppg_view1)
        if online_h1.ndim == 3: online_h1 = online_h1.mean(dim=1) 
        
        online_z1 = self.online_projector(online_h1)
        online_pred1 = self.online_predictor(online_z1) 

        with torch.no_grad():
            target_h2 = self.target_encoder(ppg_view2)
            if target_h2.ndim == 3: target_h2 = target_h2.mean(dim=1)
            target_proj2 = self.target_projector(target_h2) 
            
        loss1 = self.loss_fn(online_pred1, target_proj2).mean()
        
        online_h2 = self.online_encoder(ppg_view2)
        if online_h2.ndim == 3: online_h2 = online_h2.mean(dim=1) 
        online_z2 = self.online_projector(online_h2)
        online_pred2 = self.online_predictor(online_z2)
        
        with torch.no_grad():
            target_h1 = self.target_encoder(ppg_view1)
            if target_h1.ndim == 3: target_h1 = target_h1.mean(dim=1)
            target_proj1 = self.target_projector(target_h1)
            
        loss2 = self.loss_fn(online_pred2, target_proj1).mean()
        
        total_loss = (loss1 + loss2) / 2.0

        return {
            "loss": total_loss,
            "online_feat": online_h1
        }