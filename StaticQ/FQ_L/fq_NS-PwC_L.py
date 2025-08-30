# PTF observer
from FQ_ViT.models.ptq.observer.ptf import PtfObserver

# LayerNorm and Softmax quant modules
from FQ_ViT.models.ptq.layers import (
    QIntLayerNorm,
    QIntSoftmax,
    QConv2d,
    QLinear,
    QAct,
)

# If you need the generic builder functions
from FQ_ViT.models.ptq.observer.build   import build_observer
from FQ_ViT.models.ptq.quantizer.build  import build_quantizer



###############################################################################
# 1) LOAD POSEIDON & FQ-VIT CONFIG
###############################################################################
import torch
from scOT.model import ScOT, ScOTConfig

# (If you have a Config class in FQ-ViT to collect flags, import it;
#  otherwise just hard-code booleans below)
from FQ_ViT.config import Config  

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1.a) Build the FQ-VIT config: enable PTF (integer LN), LIS (integer softmax),
#     and choose your activation quant method (‘minmax’, ‘ema’, etc.)
cfg = Config(ptf=True, lis=True, quant_method="minmax")  

# 1.b) Load float Poseidon into this config (unused by ScOT, but needed
#     by QConv2d/QLinear/QAct constructors)
model = ScOT.from_pretrained(FLOAT_MODEL_PATH, cfg=cfg).to(DEVICE).eval()


###############################################################################
# 2) MONKEY-PATCH EVERY MODULE TO THE FQ-VIT COUNTERPART
###############################################################################
def monkey_patch_fqvit(module, cfg):
    from FQ_ViT.models.ptq.quantizer.layers import (
        QConv2d, QLinear, QAct, QIntLayerNorm, QIntSoftmax
    )
    import torch.nn as nn

    for name, child in list(module.named_children()):
        # 2.a) Conv2d → QConv2d
        if isinstance(child, nn.Conv2d):
            fq = QConv2d(
                child.in_channels, child.out_channels,
                child.kernel_size, child.stride, child.padding,
                child.dilation, child.groups,
                bias=(child.bias is not None),
                quant=cfg.quant,
                calibrate=cfg.quant,
                last_calibrate=cfg.quant and cfg.quant_method=="omse",
                bit_type=cfg.bit_type,
                calibration_mode=cfg.calibration_mode,
                observer_str=cfg.quant_method,
                quantizer_str=cfg.quant_method,
            )
            fq.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                fq.bias.data.copy_(child.bias.data)
            setattr(module, name, fq)

        # 2.b) Linear → QLinear
        elif isinstance(child, nn.Linear):
            fq = QLinear(
                child.in_features, child.out_features,
                bias=(child.bias is not None),
                quant=cfg.quant,
                calibrate=cfg.quant,
                last_calibrate=cfg.quant and cfg.quant_method=="omse",
                bit_type=cfg.bit_type,
                calibration_mode=cfg.calibration_mode,
                observer_str=cfg.quant_method,
                quantizer_str=cfg.quant_method,
            )
            fq.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                fq.bias.data.copy_(child.bias.data)
            setattr(module, name, fq)

        # 2.c) Raw Identity / placeholder → QAct
        elif isinstance(child, nn.Identity):
            fq = QAct(
                quant=cfg.quant,
                calibrate=cfg.quant,
                last_calibrate=cfg.quant and cfg.quant_method=="omse",
                bit_type=cfg.bit_type,
                calibration_mode=cfg.calibration_mode,
                observer_str=cfg.quant_method,
                quantizer_str=cfg.quant_method,
            )
            setattr(module, name, fq)

        # 2.d) LayerNorm → QIntLayerNorm
        elif isinstance(child, nn.LayerNorm):
            fqln = QIntLayerNorm(
                child.normalized_shape[0], child.eps,
                elementwise_affine=child.elementwise_affine
            )
            fqln.weight.data.copy_(child.weight.data)
            fqln.bias.data.copy_(child.bias.data)
            fqln.mode = 'int'
            setattr(module, name, fqln)

        # 2.e) Softmax → QIntSoftmax (if present as module)
        elif isinstance(child, nn.Softmax):
            fqsm = QIntSoftmax(
                log_i_softmax=cfg.lis,
                quant=cfg.quant,
                calibrate=cfg.quant,
                last_calibrate=cfg.quant and cfg.quant_method=="omse",
                bit_type=cfg.bit_type,
                calibration_mode=cfg.calibration_mode,
                observer_str=cfg.quant_method,
                quantizer_str=cfg.quant_method,
            )
            setattr(module, name, fqsm)

        else:
            # recurse into child modules
            monkey_patch_fqvit(child, cfg)

# apply it
monkey_patch_fqvit(model, cfg)
print("✔️  Replaced all Conv/Linear/Act/LN/Softmax with FQ-VIT versions.")



# 3.a) Prepare a small “calibration” DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

calib_transform = Compose([ Resize(224), CenterCrop(224),
                            ToTensor(),
                            Normalize(mean, std) ])
calib_ds = ImageFolder(os.path.join(DATA_PATH,"train"), calib_transform)
calib_dl = DataLoader(calib_ds, batch_size=cfg.calib_batchsize,
                      shuffle=True, num_workers=cfg.num_workers)

# 3.b) Open calibration context
model.model_open_calibrate()

# 3.c) Forward a handful of batches
for i,(x,_) in enumerate(calib_dl):
    if i>=cfg.calib_iter: break
    model(x.to(DEVICE))
    # If using OMSE, open last_calibrate on final batch:
    if cfg.quant_method=="omse" and i==cfg.calib_iter-1:
        model.model_open_last_calibrate()

# 3.d) Close & bake in quant parameters
model.model_close_calibrate()
model.model_quant()

print("🛠  Fully quantized model ready (weights, activations, LN, Softmax).")

loss = eval_model_with_model_loss(model)
