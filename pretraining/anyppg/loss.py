import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPLoss(nn.Module):
    def __init__(self, temperature_init=0.07):
        super().__init__()
        # Learnable temperature for Global CLIP loss (init to 0.07 => log(1/0.07))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / temperature_init)))

    def forward(self, features_a, features_b):
        """
        features_a: [B, Embed] (Normalized)
        features_b: [B, Embed] (Normalized)
        """
        logit_scale = self.logit_scale.exp()
        logits_ab = logit_scale * features_a @ features_b.t()
        logits_ba = logits_ab.t()

        labels = torch.arange(features_a.shape[0], device=features_a.device)
        loss = (
            F.cross_entropy(logits_ab, labels) + 
            F.cross_entropy(logits_ba, labels)
        ) / 2
        return loss