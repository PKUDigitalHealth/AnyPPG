import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint 
from dataclasses import dataclass, field
try:
    from loss import CLIPLoss
except:
    from .loss import CLIPLoss

@dataclass
class ECGFounderConfig:
    ckpt_path: str = './checkpoints/ECGFounder/checkpoint.pth'
    in_channels: int = 1
    base_filters: int = 64
    ratio: float = 1.0 
    filter_list: list = field(default_factory=lambda: [64, 160, 160, 400, 400, 1024, 1024])
    m_blocks_list: list = field(default_factory=lambda: [2, 2, 2, 3, 3, 4, 4])
    kernel_size: int = 16
    stride: int = 2
    groups_width: int = 16
    verbose: bool = False
    use_bn: bool = True
    use_do: bool = False
    
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

def create_ecgfounder(cfg: ECGFounderConfig):
    try:
        from backbone.ecgfounder import ECGNet1D
    except:
        from .backbone.ecgfounder import ECGNet1D
    model = ECGNet1D(
        in_channels=cfg.in_channels, 
        base_filters=cfg.base_filters, 
        ratio=cfg.ratio, 
        filter_list=cfg.filter_list, 
        m_blocks_list=cfg.m_blocks_list, 
        kernel_size=cfg.kernel_size, 
        stride=cfg.stride, 
        groups_width=cfg.groups_width,
        verbose=cfg.verbose, 
        use_bn=cfg.use_bn,
        use_do=cfg.use_do
    )
    if cfg.ckpt_path:
        try:
            ckpt = torch.load(cfg.ckpt_path, weights_only=False, map_location='cpu')
            params = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            unmatched_keys = model.load_state_dict(params, strict=False)
            print(f"Loaded ECGFounder from {cfg.ckpt_path}, unmatched keys: {unmatched_keys}")
        except Exception as e:
            print(f"Warning: Failed to load ECGFounder checkpoint: {e}")
    return model

def create_ppgencoder(cfg: PPGEncoderConfig):
    try:
        from backbone.resnet1d import Net1D
    except:
        from .backbone.resnet1d import Net1D
    model = Net1D(
        in_channels=cfg.in_channels, 
        base_filters=cfg.base_filters, 
        ratio=cfg.ratio, 
        filter_list=cfg.filter_list, 
        m_blocks_list=cfg.m_blocks_list, 
        kernel_size=cfg.kernel_size, 
        stride=cfg.stride, 
        groups_width=cfg.groups_width,
        verbose=cfg.verbose, 
        use_bn=cfg.use_bn,
        use_do=cfg.use_do
    )
    return model

class CLIP(nn.Module):
    def __init__(
        self, 
        ppg_cfg: PPGEncoderConfig,
        ecg_cfg: ECGFounderConfig,
        emb_dim: int = 128,
        emb_dropout: float = 0.2,
    ):
        super().__init__()
        
        self.ppg_encoder = create_ppgencoder(ppg_cfg)
        self.ecg_encoder = create_ecgfounder(ecg_cfg)
        
        for param in self.ecg_encoder.parameters():
            param.requires_grad = False
        
        self.ecg_encoder.eval()
        
        ppg_dim = ppg_cfg.filter_list[-1]
        ecg_dim = ecg_cfg.filter_list[-1]
        
        self.emb_dropout = nn.Dropout(emb_dropout)
        
        self.ppg_projector = nn.Sequential(
            nn.Linear(ppg_dim, ppg_dim),
            nn.BatchNorm1d(ppg_dim),
            nn.GELU(),
            nn.Linear(ppg_dim, emb_dim)
        )
        self.ecg_projector = nn.Sequential(
            nn.Linear(ecg_dim, ecg_dim),
            nn.BatchNorm1d(ecg_dim),
            nn.GELU(),
            nn.Linear(ecg_dim, emb_dim)
        )
        
        self.clip_loss = CLIPLoss(temperature_init=0.07)

    def train(self, mode=True):
        super().train(mode)
        self.ecg_encoder.eval()
        return self

    def forward(self, ppg: torch.Tensor, ecg: torch.Tensor):
        
        ppg_feat = self.ppg_encoder(ppg) 
        
        def run_ecg(x):
            return self.ecg_encoder(x)
        ecg_feat = checkpoint(run_ecg, ecg, use_reentrant=False)
        
        if ppg_feat.dim() == 3: ppg_feat = ppg_feat.mean(dim=-1)
        if ecg_feat.dim() == 3: ecg_feat = ecg_feat.mean(dim=-1)
        
        ppg_feat_dropped = self.emb_dropout(ppg_feat)
        
        ppg_clip = self.ppg_projector(ppg_feat_dropped)
        ecg_clip = self.ecg_projector(ecg_feat) 
        
        ppg_clip = F.normalize(ppg_clip, dim=-1)
        ecg_clip = F.normalize(ecg_clip, dim=-1)
        
        loss_clip = self.clip_loss(ppg_clip, ecg_clip)

        with torch.no_grad():
            current_temp = 1.0 / self.clip_loss.logit_scale.exp()

        return {
            "loss_clip": loss_clip,
            "temperature": current_temp,
            "ppg_feat": ppg_feat
        }