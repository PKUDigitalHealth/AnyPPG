import os
import torch
import torch.nn as nn
from typing import Optional 
from pathlib import Path


def get_model(model_name: str, ckpt_path: str, device: Optional[torch.device] = None) -> nn.Module:
    """Get model by name."""
    
    # PaPaGei-S
    if model_name == 'papagei_s':
        from eval_models.papagei_s.papagei_s import ResNet1DMoE
        model = ResNet1DMoE(
            in_channels=1,
            base_filters=32,
            kernel_size=3,
            stride=2,
            groups=1,
            n_block=18,
            n_classes=512,
            n_experts=3
        )
        model_state_dict = torch.load(ckpt_path, map_location='cpu')
        model_state_dict = {
            k.replace("module.", ""): v for k, v in model_state_dict.items()
        }
        model.load_state_dict(model_state_dict)
        return model  
    
    # PaPaGei-P
    if model_name == 'papagei_p':
        from eval_models.papagei_p.papagei_p import ResNet1DMoE
        model = ResNet1DMoE(
            in_channels=1,
            base_filters=32,
            kernel_size=3,
            stride=2,
            groups=1,
            n_block=18,
            n_classes=512,
            n_experts=3
        )
        model_state_dict = torch.load(ckpt_path, map_location='cpu')
        model_state_dict = {
            k.replace("module.", ""): v for k, v in model_state_dict.items()
        }
        model.load_state_dict(model_state_dict)
        return model  
    
    # PulsePPG
    if model_name == 'pulseppg':
        from eval_models.pulseppg.pulseppg import Net
        model = Net(
            in_channels=1,
            base_filters=128,
            kernel_size=11,
            stride=2,
            groups=1,
            n_block=12,
            finalpool="max"
        )
        model_state_dict = torch.load(ckpt_path, map_location='cpu')['net']
        model.load_state_dict(model_state_dict)
        return model  
    
    # CLIP
    if model_name == 'clip':
        from eval_models.clip.framework import (
            CLIP,
            PPGEncoderConfig,
            ECGFounderConfig,
        )
        model = CLIP(
            ppg_cfg=PPGEncoderConfig(),
            ecg_cfg=ECGFounderConfig()
        )
        model_state_dict = torch.load(ckpt_path, map_location='cpu')['model_state_dict']
        model.load_state_dict(model_state_dict)
        return model.ppg_encoder  
    
    # Chronos-2-Series
    if model_name == 'chronos-2' or model_name == 'chronos-2-synth' or model_name == 'chronos-2-small':
        from chronos import Chronos2Pipeline
        class ChronosEncoder(nn.Module):
            def __init__(self, ckpt_path: str, device: torch.device):
                super().__init__()
                self.pipeline: Chronos2Pipeline = Chronos2Pipeline.from_pretrained(
                    pretrained_model_name_or_path=ckpt_path, 
                    device_map=device
                )
                self.pipeline.model.eval()
                
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                embed, _ = self.pipeline.embed(x) 
                embed = torch.cat(embed, dim=0)
                return embed
        ppg_encoder = ChronosEncoder(ckpt_path, device)
        return ppg_encoder
    
    # MOMENT
    if model_name == 'moment':
        from momentfm import MOMENTPipeline
        model = MOMENTPipeline.from_pretrained(
            pretrained_model_name_or_path=ckpt_path, 
            model_kwargs={"task_name": "embedding"},
        )
        model.init()
        return model
    
    # SimCLR
    if model_name == 'simclr':
        from eval_models.simclr.framework import (
            SimCLR,
            PPGEncoderConfig
        )
        model = SimCLR(
            ppg_encoder_config=PPGEncoderConfig()
        )
        model_state_dict = torch.load(ckpt_path, map_location='cpu')['model_state_dict']
        model.load_state_dict(model_state_dict)
        return model.ppg_encoder  
    
    # BYOL
    if model_name == 'byol':
        from eval_models.byol.framework import (
            BYOL,
            PPGEncoderConfig
        )
        model = BYOL(
            ppg_encoder_config=PPGEncoderConfig()
        )
        model_state_dict = torch.load(ckpt_path, map_location='cpu')['model_state_dict']
        model.load_state_dict(model_state_dict)
        return model.online_encoder  
        
    # Unknown model name
    raise ValueError(f"Unknown model name: {model_name}")