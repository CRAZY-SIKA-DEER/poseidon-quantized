import torch


def add_quantization_noise(tensor, step_sizes, channel_axis):
    """
    Add uniform quantization noise to a tensor, with one step size per channel.

    Args:
        tensor: input tensor
        step_sizes: 1D tensor of shape [num_channels]
        channel_axis: which axis is the channel axis
            - weights: usually 0
            - activations: usually -1

    Returns:
        tensor + uniform noise in [-0.5 * step, 0.5 * step]
    """
    if step_sizes.numel() != tensor.shape[channel_axis]:
        raise ValueError(
            f"step_sizes ({step_sizes.numel()}) does not match "
            f"tensor.shape[channel_axis] ({tensor.shape[channel_axis]})"
        )

    shape = [1] * tensor.dim()
    shape[channel_axis] = tensor.size(channel_axis)

    step_sizes_broadcast = step_sizes.view(shape)
    noise = (torch.rand_like(tensor) - 0.5) * step_sizes_broadcast
    return tensor + noise