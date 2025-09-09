import torch
from weighted_l1_loss import WeightedL1Loss

# Fake volumes
B, C, D, H, W = 2, 1, 16, 16, 16
input_vol = torch.randn(B, C, D, H, W)
target_vol = torch.randn(B, C, D, H, W)

# Uniform spacing for whole batch (sx, sy, sz)
uniform_spacing = [1.2, 0.8, 2.0]

criterion = WeightedL1Loss()
loss = criterion(input_vol, target_vol, uniform_spacing)
print("Loss (uniform spacing):", float(loss))

# Per-sample spacing [B, 3]
per_sample_spacing = torch.tensor([[1.0, 1.0, 1.0], [0.7, 1.3, 0.9]])
loss2 = criterion(input_vol, target_vol, per_sample_spacing)
print("Loss (per-sample spacing):", float(loss2))