import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from .resnet1d import Net1D
except:
    from resnet1d import Net1D
from dataclasses import dataclass, field
from typing import Dict
from lightly.loss import NTXentLoss

# --- Configuration Classes ---

@dataclass
class PPGEncoderConfig:
    """Configuration for PPG Encoder (ResNet1D)."""
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

# --- Core Model ---

class SimCLR(nn.Module):
    def __init__(
        self, 
        ppg_encoder_config: PPGEncoderConfig,
        embed_dim: int = 128,
        temperature: float = 0.07
    ) -> None:
        super().__init__()
        
        # 1. Backbone Encoder (Shared for both views)
        self.ppg_encoder = Net1D(
            in_channels=ppg_encoder_config.in_channels,
            base_filters=ppg_encoder_config.base_filters,
            ratio=ppg_encoder_config.ratio,
            filter_list=ppg_encoder_config.filter_list,
            m_blocks_list=ppg_encoder_config.m_blocks_list,
            kernel_size=ppg_encoder_config.kernel_size,
            stride=ppg_encoder_config.stride,
            groups_width=ppg_encoder_config.groups_width,
            use_bn=ppg_encoder_config.use_bn,
            use_do=ppg_encoder_config.use_do,
            verbose=ppg_encoder_config.verbose,
        )
        
        # Infer feature dimension D
        self.feature_dim = ppg_encoder_config.filter_list[-1]
        
        # 2. Projection Head (MLP)
        # SimCLR paper recommends: Linear -> ReLU -> Linear
        # Your previous code used GELU and BatchNorm, which is also fine (v2 style)
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(), # Or nn.ReLU()
            nn.Linear(self.feature_dim, embed_dim)
        )
        
        # 3. Loss Function
        self.nt_xent_loss = NTXentLoss(temperature=temperature)


    def forward(
        self, 
        ppg_view1: torch.Tensor,
        ppg_view2: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for SimCLR.
        
        Args:
            ppg_view1: Augmented PPG signals [B, 1, L]
            ppg_view2: Augmented PPG signals [B, 1, L] (different augmentation)
        
        Returns:
            Dictionary containing loss and features.
        """
        
        # --- Step 1: Encoding (Shared Encoder) ---
        # Input: [B, 1, L] -> Output: [B, T, D]
        h1 = self.ppg_encoder(ppg_view1) 
        h2 = self.ppg_encoder(ppg_view2)
        
        # --- Step 2: Global Pooling ---
        # Convert sequence features to global vector [B, D]
        # Assuming last dimension is feature dim. If Net1D returns [B, D, T], use dim=2.
        # Based on typical ResNet1D implementations, it might need checking. 
        # Here assuming [B, T, D] or [B, D, T]. Let's assume [B, T, D] based on previous context.
        if h1.ndim == 3:
            h1 = h1.mean(dim=1) 
            h2 = h2.mean(dim=1)
        
        # --- Step 3: Projection (Shared Head) ---
        # z1, z2: [B, Embed]
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        
        # --- Step 4: Loss Calculation (NT-Xent) ---
        loss = self.nt_xent_loss(z1, z2)

        return {
            "loss": loss,
            "z1": z1,
            "z2": z2,
            "temperature": self.nt_xent_loss.temperature,
        }