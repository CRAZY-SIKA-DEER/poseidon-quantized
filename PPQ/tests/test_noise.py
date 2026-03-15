import torch

from PPQ.noise import add_quantization_noise


def test_add_quantization_noise_weight_shape():
    w = torch.randn(16, 32)
    step_sizes = torch.ones(16) * 0.01

    w_noisy = add_quantization_noise(w, step_sizes, channel_axis=0)

    assert w_noisy.shape == w.shape
    assert torch.isfinite(w_noisy).all()


def test_add_quantization_noise_activation_shape():
    x = torch.randn(4, 128, 64)
    step_sizes = torch.ones(64) * 0.02

    x_noisy = add_quantization_noise(x, step_sizes, channel_axis=-1)

    assert x_noisy.shape == x.shape
    assert torch.isfinite(x_noisy).all()


def test_add_quantization_noise_respects_bound():
    x = torch.randn(2, 10, 8)
    step_sizes = torch.linspace(0.01, 0.08, 8)

    x_noisy = add_quantization_noise(x, step_sizes, channel_axis=-1)
    noise = (x_noisy - x).abs()

    for c in range(8):
        max_noise_c = noise[:, :, c].max().item()
        expected = 0.5 * step_sizes[c].item()
        assert max_noise_c <= expected + 1e-6