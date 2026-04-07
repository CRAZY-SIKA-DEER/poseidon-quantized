import torch

try:
    import linklink as link
except ImportError:
    link = None

from BRECQ.quant.quant_layer import QuantModule, StraightThrough, lp_loss
from BRECQ.quant.adaptive_rounding import AdaRoundQuantizer

from poseidon_data_utils import (
    save_inp_oup_data,
    save_grad_data,
    move_block_args_to_device,
    get_reconstruction_output,
)


def poseidon_block_reconstruction(
    model,
    block,
    cali_data,
    batch_size: int = 32,
    iters: int = 20000,
    weight: float = 0.01,
    opt_mode: str = "mse",
    asym: bool = False,
    include_act_func: bool = True,
    b_range: tuple = (20, 2),
    warmup: float = 0.0,
    act_quant: bool = False,
    lr: float = 4e-5,
    p: float = 2.0,
    multi_gpu: bool = False,
):
    """
    Poseidon-specific block reconstruction.

    Supports the same reconstruction loss choices as original BRECQ:
    - mse
    - fisher_diag
    - fisher_full

    Main Poseidon differences:
    - cali_data is a list of batch dicts or one collated batch dict
    - cached block input is a tuple of block forward args
    - block forward is block(*args), not block(x)
    """
    model.set_quant_state(False, False)
    block.set_quant_state(True, act_quant)
    round_mode = "learned_hard_sigmoid"

    org_act_func = None
    if not include_act_func and hasattr(block, "activation_function"):
        org_act_func = block.activation_function
        block.activation_function = StraightThrough()

    if not act_quant:
        # Replace weight quantizer with AdaRoundQuantizer
        for _, module in block.named_modules():
            if isinstance(module, QuantModule):
                module.weight_quantizer = AdaRoundQuantizer(
                    uaq=module.weight_quantizer,
                    round_mode=round_mode,
                    weight_tensor=module.org_weight.data,
                )
                module.weight_quantizer.soft_targets = True

        # Set up optimizer
        opt_params = []
        for _, module in block.named_modules():
            if isinstance(module, QuantModule):
                opt_params.append(module.weight_quantizer.alpha)

        if len(opt_params) == 0:
            raise RuntimeError(
                f"No QuantModule/AdaRound parameters found inside block {block.__class__.__name__}."
            )

        optimizer = torch.optim.Adam(opt_params)
        scheduler = None
    else:
        # Use UniformAffineQuantizer to learn delta
        if hasattr(block.act_quantizer, "delta") and block.act_quantizer.delta is not None:
            opt_params = [block.act_quantizer.delta]
        else:
            opt_params = []

        for _, module in block.named_modules():
            if isinstance(module, QuantModule):
                if (
                    hasattr(module.act_quantizer, "delta")
                    and module.act_quantizer.delta is not None
                ):
                    opt_params.append(module.act_quantizer.delta)

        if len(opt_params) == 0:
            raise RuntimeError(
                f"No activation quantization parameters found inside block {block.__class__.__name__}."
            )

        optimizer = torch.optim.Adam(opt_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=iters, eta_min=0.0
        )

    loss_mode = "none" if act_quant else "relaxation"
    rec_loss = opt_mode

    loss_func = PoseidonLossFunction(
        block,
        round_loss=loss_mode,
        weight=weight,
        max_count=iters,
        rec_loss=rec_loss,
        b_range=b_range,
        decay_start=0,
        warmup=warmup,
        p=p,
    )

    # Save data before optimizing the rounding
    cached_inps, cached_outs = save_inp_oup_data(
        model=model,
        layer=block,
        cali_data=cali_data,
        asym=asym,
        act_quant=act_quant,
        batch_size=batch_size,
    )

    if opt_mode != "mse":
        cached_grads = save_grad_data(
            model=model,
            layer=block,
            cali_data=cali_data,
            act_quant=act_quant,
            batch_size=batch_size,
        )
    else:
        cached_grads = None

    device = next(model.parameters()).device
    num_cached = len(cached_inps)

    for _ in range(iters):
        idx = torch.randint(low=0, high=num_cached, size=(1,)).item()

        cur_inp = move_block_args_to_device(cached_inps[idx], device)
        cur_out = cached_outs[idx].to(device)
        cur_grad = cached_grads[idx].to(device) if opt_mode != "mse" else None

        optimizer.zero_grad()

        out_quant = block(*cur_inp)
        out_quant = get_reconstruction_output(out_quant)

        err = loss_func(out_quant, cur_out, cur_grad)
        err.backward(retain_graph=True)

        if multi_gpu:
            if link is None:
                raise RuntimeError("multi_gpu=True but linklink is not available.")
            for param in opt_params:
                if param.grad is not None:
                    link.allreduce(param.grad)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Finish optimization, use hard rounding.
    for _, module in block.named_modules():
        if isinstance(module, QuantModule):
            module.weight_quantizer.soft_targets = False

    # Reset original activation function
    if org_act_func is not None:
        block.activation_function = org_act_func


class PoseidonLossFunction:
    def __init__(
        self,
        block,
        round_loss: str = "relaxation",
        weight: float = 1.0,
        rec_loss: str = "mse",
        max_count: int = 2000,
        b_range: tuple = (10, 2),
        decay_start: float = 0.0,
        warmup: float = 0.0,
        p: float = 2.0,
    ):
        self.block = block
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p

        self.temp_decay = LinearTempDecay(
            max_count,
            rel_start_decay=warmup + (1 - warmup) * decay_start,
            start_b=b_range[0],
            end_b=b_range[1],
        )
        self.count = 0

    def __call__(self, pred, tgt, grad=None):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy.
        """
        self.count += 1

        if self.rec_loss == "mse":
            rec_loss = lp_loss(pred, tgt, p=self.p)

        elif self.rec_loss == "fisher_diag":
            if grad is None:
                raise ValueError("grad must not be None when rec_loss='fisher_diag'")
            # Matches original BRECQ logic, but generalized to arbitrary tensor dims.
            reduce_dims = tuple(range(1, pred.dim()))
            rec_loss = ((pred - tgt).pow(2) * grad.pow(2)).sum(reduce_dims).mean()

        elif self.rec_loss == "fisher_full":
            if grad is None:
                raise ValueError("grad must not be None when rec_loss='fisher_full'")
            # Matches original BRECQ logic, but generalized to arbitrary tensor dims.
            a = (pred - tgt).abs()
            grad = grad.abs()
            reduce_dims = tuple(range(1, pred.dim()))
            view_shape = (-1,) + (1,) * (pred.dim() - 1)
            batch_dotprod = torch.sum(a * grad, dim=reduce_dims).view(*view_shape)
            rec_loss = (batch_dotprod * a * grad).mean() / 100

        else:
            raise ValueError(f"Not supported reconstruction loss function: {self.rec_loss}")

        b = self.temp_decay(self.count)

        if self.count < self.loss_start or self.round_loss == "none":
            b = 0
            round_loss = 0

        elif self.round_loss == "relaxation":
            round_loss = 0
            for _, module in self.block.named_modules():
                if isinstance(module, QuantModule):
                    round_vals = module.weight_quantizer.get_soft_targets()
                    round_loss += self.weight * (
                        1 - ((round_vals - 0.5).abs() * 2).pow(b)
                    ).sum()
        else:
            raise NotImplementedError

        total_loss = rec_loss + round_loss

        if self.count % 500 == 0:
            print(
                "Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}".format(
                    float(total_loss),
                    float(rec_loss),
                    float(round_loss),
                    b,
                    self.count,
                )
            )

        return total_loss


class LinearTempDecay:
    def __init__(
        self,
        t_max: int,
        rel_start_decay: float = 0.2,
        start_b: int = 10,
        end_b: int = 2,
    ):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        """
        Temperature scheduler for rounding relaxation.
        """
        if t < self.start_decay:
            return self.start_b
        rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
        return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))