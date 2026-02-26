import torch 
from resnet1d import Net1D

model = Net1D(
    in_channels=1,
    base_filters=64,
    ratio=1.0,
    filter_list=[64, 160, 160, 400, 400, 512],
    m_blocks_list=[2, 2, 2, 3, 3, 1],
    kernel_size=3,
    stride=2,
    groups_width=16,
    use_bn=True,
    use_do=True,
    verbose=False,
)

ckpt = torch.load('anyppg_ckpt.pth', map_location='cpu')
model.load_state_dict(ckpt)

data = torch.randn(1, 1, 1000)
print(model(data).shape)