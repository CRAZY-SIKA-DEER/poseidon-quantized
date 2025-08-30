import torch
import torch.nn.functional as F
from transformers import Trainer


def denormalize(tensor, constants):
    mean = torch.as_tensor(constants["mean"], dtype=tensor.dtype, device=tensor.device)
    std  = torch.as_tensor(constants["std"],  dtype=tensor.dtype, device=tensor.device)
    for _ in range(tensor.ndim - 1 - mean.ndim):
        mean = mean.unsqueeze(0)
        std  = std.unsqueeze(0)
    return tensor * std + mean

def continuity_loss_torch(u: torch.Tensor, v: torch.Tensor):
    # u,v are (B, H, W)
    dx = 1.0 / (u.size(-1) - 1)
    dy = 1.0 / (u.size(-2) - 1)

    # central differences: shapes → (B, H,   W-2) and (B, H-2, W)
    du_dx = (u[:, :, 2:]   - u[:, :, :-2]) / (2 * dx)
    dv_dy = (v[:, 2:, :]   - v[:, :-2, :]) / (2 * dy)

    # now crop off the first/last rows of du_dx to align heights,
    # and the first/last columns of dv_dy to align widths:
    du_dx = du_dx[:, 1:-1, :]   # → (B, H-2, W-2)
    dv_dy = dv_dy[:, :, 1:-1]   # → (B, H-2, W-2)

    div = du_dx + dv_dy         # now shape matches
    return div.abs().mean()

def momentum_loss_torch(u: torch.Tensor, v: torch.Tensor, p: torch.Tensor=None, Re: float=1.0):
    if u.dim() == 2:
        u = u.unsqueeze(0)
        v = v.unsqueeze(0)
        if p is not None and p.dim() == 2:
            p = p.unsqueeze(0)

    B, H, W = u.shape
    dx, dy = 1/(W-1), 1/(H-1)
    ip, im = slice(2,None), slice(0,-2)
    jp, jm = slice(2,None), slice(0,-2)
    ci, cj = slice(1,-1), slice(1,-1)

    duu_dx = (u[:, ip, cj]**2 - u[:, im, cj]**2) / (2*dx)
    duv_dy = (u[:, ci, jp]*v[:, ci, jp] - u[:, ci, jm]*v[:, ci, jm]) / (2*dy)
    dvu_dx = (v[:, ip, cj]*u[:, ip, cj] - v[:, im, cj]*u[:, im, cj]) / (2*dx)
    dvv_dy = (v[:, ci, jp]**2 - v[:, ci, jm]**2) / (2*dy)

    d2u_dx2 = (u[:, ip, cj] - 2*u[:, ci, cj] + u[:, im, cj]) / (dx*dx)
    d2u_dy2 = (u[:, ci, jp] - 2*u[:, ci, cj] + u[:, ci, jm]) / (dy*dy)
    d2v_dx2 = (v[:, ip, cj] - 2*v[:, ci, cj] + v[:, im, cj]) / (dx*dx)
    d2v_dy2 = (v[:, ci, jp] - 2*v[:, ci, cj] + v[:, ci, jm]) / (dy*dy)

    if p is not None:
        dp_dx = (p[:, ip, cj] - p[:, im, cj]) / (2*dx)
        dp_dy = (p[:, ci, jp] - p[:, ci, jm]) / (2*dy)
    else:
        dp_dx = torch.zeros_like(duu_dx)
        dp_dy = torch.zeros_like(dvv_dy)

    rx = duu_dx + duv_dy - (d2u_dx2 + d2u_dy2)/Re - dp_dx
    ry = dvu_dx + dvv_dy - (d2v_dx2 + d2v_dy2)/Re - dp_dy

    return 0.5 * (rx.abs().mean() + ry.abs().mean())