import pytest

torch = pytest.importorskip("torch")

from weighted_l1_loss import WeightedL1Loss


def test_loss_uniform_spacing_matches_reference():
    torch.manual_seed(0)
    inp = torch.randn(2, 1, 5, 6, 7)
    tgt = torch.randn(2, 1, 5, 6, 7)
    spacing = torch.tensor([2.0, 1.0, 0.5])

    loss = WeightedL1Loss()
    out = loss(inp, tgt, spacing)

    # Simple sanity checks
    assert out.dim() == 0
    assert out.item() > 0


def test_loss_per_sample_spacing_reduces_mean():
    torch.manual_seed(0)
    inp = torch.randn(3, 2, 4, 4, 4)
    tgt = torch.randn_like(inp)
    spacing = torch.tensor([[1.0, 1.0, 1.0], [2.0, 1.0, 0.5], [0.7, 0.7, 0.7]])

    loss = WeightedL1Loss(reduction="mean")
    out = loss(inp, tgt, spacing)

    assert out.dim() == 0
    assert out.item() > 0


def test_invalid_shapes_raise():
    loss = WeightedL1Loss()
    with pytest.raises(ValueError):
        loss(torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 4), [1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        loss(torch.randn(1, 1, 4, 4, 4), torch.randn(2, 1, 4, 4, 4), [1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        loss(torch.randn(1, 1, 4, 4, 4), torch.randn(1, 1, 4, 4, 4), torch.tensor([1.0, 1.0]))
    with pytest.raises(ValueError):
        loss(
            torch.randn(2, 1, 4, 4, 4),
            torch.randn(2, 1, 4, 4, 4),
            torch.tensor([[1.0, 1.0, 1.0]])
        )