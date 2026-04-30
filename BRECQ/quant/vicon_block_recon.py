import torch

try:
    import linklink as link
except ImportError:
    link = None

from BRECQ.quant.quant_layer import QuantModule, StraightThrough, lp_loss
from BRECQ.quant.quant_model import QuantModel
from BRECQ.quant.quant_block import BaseQuantBlock
from BRECQ.quant.adaptive_rounding import AdaRoundQuantizer
from BRECQ.quant.vicon_quant_block import QuantMultiheadAttention
from BRECQ.quant.vicon_data_utils import save_grad_data, save_inp_oup_data


def _replace_quantizer_with_adaround(
    module,
    quantizer_attr: str,
    weight_attr: str,
    org_weight_attr: str,
    round_mode: str,
):
    old_q = getattr(module, quantizer_attr)
    weight = getattr(module, weight_attr)
    org_weight = getattr(module, org_weight_attr)

    device = weight.device
    old_q = old_q.to(device)

    if hasattr(old_q, "delta") and old_q.delta is not None:
        old_q.delta = old_q.delta.to(device)

    if hasattr(old_q, "zero_point") and old_q.zero_point is not None:
        old_q.zero_point = old_q.zero_point.to(device)

    new_q = AdaRoundQuantizer(
        uaq=old_q,
        round_mode=round_mode,
        weight_tensor=org_weight.data.to(device),
    )
    new_q.to(device)
    new_q.soft_targets = True

    setattr(module, quantizer_attr, new_q)


def block_reconstruction(
    model: QuantModel,
    block: BaseQuantBlock,
    cali_data: torch.Tensor,
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
    model.set_quant_state(False, False)
    block.set_quant_state(True, act_quant)
    round_mode = "learned_hard_sigmoid"

    if not include_act_func:
        org_act_func = block.activation_function
        block.activation_function = StraightThrough()

    if not act_quant:
        for name, module in block.named_modules():
            if isinstance(module, QuantModule):
                _replace_quantizer_with_adaround(
                    module=module,
                    quantizer_attr="weight_quantizer",
                    weight_attr="weight",
                    org_weight_attr="org_weight",
                    round_mode=round_mode,
                )

            elif isinstance(module, QuantMultiheadAttention):
                _replace_quantizer_with_adaround(
                    module=module,
                    quantizer_attr="in_proj_weight_quantizer",
                    weight_attr="in_proj_weight",
                    org_weight_attr="org_in_proj_weight",
                    round_mode=round_mode,
                )
                _replace_quantizer_with_adaround(
                    module=module,
                    quantizer_attr="out_proj_weight_quantizer",
                    weight_attr="out_proj_weight",
                    org_weight_attr="org_out_proj_weight",
                    round_mode=round_mode,
                )

        opt_params = []
        for name, module in block.named_modules():
            if isinstance(module, QuantModule):
                opt_params.append(module.weight_quantizer.alpha)

            elif isinstance(module, QuantMultiheadAttention):
                opt_params.append(module.in_proj_weight_quantizer.alpha)
                opt_params.append(module.out_proj_weight_quantizer.alpha)

        optimizer = torch.optim.Adam(opt_params)
        scheduler = None

    else:
        if hasattr(block.act_quantizer, "delta"):
            opt_params = [block.act_quantizer.delta]
        else:
            opt_params = []

        for name, module in block.named_modules():
            if isinstance(module, QuantModule):
                if module.act_quantizer.delta is not None:
                    opt_params.append(module.act_quantizer.delta)

        optimizer = torch.optim.Adam(opt_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=iters, eta_min=0.0
        )

    loss_mode = "none" if act_quant else "relaxation"
    rec_loss = opt_mode

    loss_func = LossFunction(
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

    cached_inps, cached_outs = save_inp_oup_data(
        model, block, cali_data, asym, act_quant, batch_size
    )

    if opt_mode != "mse":
        cached_grads = save_grad_data(
            model, block, cali_data, act_quant, batch_size=batch_size
        )
    else:
        cached_grads = None

    device = "cuda"

    for i in range(iters):
        idx = torch.randperm(cached_inps.size(0))[:batch_size]
        cur_inp = cached_inps[idx].to(device)
        cur_out = cached_outs[idx].to(device)
        cur_grad = cached_grads[idx].to(device) if opt_mode != "mse" else None

        optimizer.zero_grad()
        out_quant = block(cur_inp)

        err = loss_func(out_quant, cur_out, cur_grad)
        err.backward(retain_graph=True)

        if multi_gpu:
            for param in opt_params:
                link.allreduce(param.grad)

        optimizer.step()

        if scheduler:
            scheduler.step()

    torch.cuda.empty_cache()

    for name, module in block.named_modules():
        if isinstance(module, QuantModule):
            module.weight_quantizer.soft_targets = False

        elif isinstance(module, QuantMultiheadAttention):
            module.in_proj_weight_quantizer.soft_targets = False
            module.out_proj_weight_quantizer.soft_targets = False

    if not include_act_func:
        block.activation_function = org_act_func


class LossFunction:
    def __init__(
        self,
        block: BaseQuantBlock,
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
        self.count += 1

        if self.rec_loss == "mse":
            rec_loss = lp_loss(pred, tgt, p=self.p)

        elif self.rec_loss == "fisher_diag":
            rec_loss = ((pred - tgt).pow(2) * grad.pow(2)).sum(1).mean()

        elif self.rec_loss == "fisher_full":
            a = (pred - tgt).abs()
            grad = grad.abs()
            batch_dotprod = torch.sum(a * grad, (1, 2, 3)).view(-1, 1, 1, 1)
            rec_loss = (batch_dotprod * a * grad).mean() / 100

        else:
            raise ValueError(
                f"Not supported reconstruction loss function: {self.rec_loss}"
            )

        b = self.temp_decay(self.count)

        if self.count < self.loss_start or self.round_loss == "none":
            b = 0
            round_loss = 0

        elif self.round_loss == "relaxation":
            round_loss = 0

            for name, module in self.block.named_modules():
                if isinstance(module, QuantModule):
                    round_vals = module.weight_quantizer.get_soft_targets()
                    round_loss += self.weight * (
                        1 - ((round_vals - 0.5).abs() * 2).pow(b)
                    ).sum()

                elif isinstance(module, QuantMultiheadAttention):
                    round_vals = module.in_proj_weight_quantizer.get_soft_targets()
                    round_loss += self.weight * (
                        1 - ((round_vals - 0.5).abs() * 2).pow(b)
                    ).sum()

                    round_vals = module.out_proj_weight_quantizer.get_soft_targets()
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
        if t < self.start_decay:
            return self.start_b

        rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
        return self.end_b + (self.start_b - self.end_b) * max(0.0, 1 - rel_t)