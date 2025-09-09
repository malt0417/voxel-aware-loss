# weighted-l1-loss

Resolution-aware L1 + gradient loss for 3D PyTorch volumes. Supports uniform spacing or per-sample spacing and properly weights by voxel size and axis resolution.

## Features

- Weighted voxel L1 term scaled by voxel volume (sx · sy · sz)
- Gradient difference terms along x/y/z weighted by inverse spacing
- Accepts spacing as `[3]` or `[B, 3]`
- Type hints, validation, tests, and CI-ready

## Installation

```bash
pip install git+https://github.com/yourname/weighted-l1-loss.git
```

Or clone and install locally:

```bash
pip install -e .[dev]
```

## Quick Start

```python
import torch
from weighted_l1_loss import WeightedL1Loss

B, C, D, H, W = 2, 1, 16, 16, 16
input_vol = torch.randn(B, C, D, H, W)
target_vol = torch.randn(B, C, D, H, W)

criterion = WeightedL1Loss()
loss = criterion(input_vol, target_vol, [1.2, 0.8, 2.0])  # (sx, sy, sz)
print(float(loss))
```

Per-sample spacing:

```python
spacing = torch.tensor([[1.0, 1.0, 1.0], [0.7, 1.3, 0.9]])
loss = criterion(input_vol, target_vol, spacing)
```

## Development

- Format and lint: `pre-commit run -a`
- Run tests: `pytest -q`

## License

MIT