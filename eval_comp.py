import os
os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import csv
import re

# ================= 导入 VICON 依赖 =================
from VICON.src import utils
from VICON.src.dataset import all_datasets
import VICON.src.models as models
from VICON.src.trainer import Trainer
from MySVD.convert_attn import convert_attn_to_linear, ExplicitMultiheadAttention
from evaluation.physical_loss import evaluate_physics_metrics
import types


# =====================================================================
# 每个数据集的指标结构
# (display_name, abs_data_key, rel_data_key)
#   - "loss" / "rel_loss" 从 losses / rel_losses dict 读取
#   - 其余从 physics dict 读取
# 每个 entry 生成 4 列: Name, Name_Inc, Rel_Name, Rel_Name_Inc
# =====================================================================
DATASET_METRICS = {
    "NS2D": [
        ("Loss",    "loss",           "rel_loss"),
        ("Sobolev", "sobolev",        "rel_sobolev"),
        ("Div.",    "continuity",     "rel_continuity"),
        ("Vort.",   "vorticity_err",  "rel_vorticity_err"),
    ],
    "COMPRESSIBLE2D": [
        ("Loss",    "loss",           "rel_loss"),
        ("Sobolev", "sobolev",        "rel_sobolev"),
    ],
    "EULER2D": [
        ("Loss",    "loss",           "rel_loss"),
        ("Sobolev", "sobolev",        "rel_sobolev"),
    ],
}

SOBOLEV_ORDER_MAP = {
    "NS2D": 2,
    "COMPRESSIBLE2D": 1,
    "EULER2D": 1
}

# =====================================================================
# Model utilities
# =====================================================================

def disable_fast_path(model):
    def slow_forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = src
        if self.norm_first:
            try:
                sa_out = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            except TypeError:
                sa_out = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + sa_out
            x = x + self._ff_block(self.norm2(x))
        else:
            try:
                sa_out = self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal)
            except TypeError:
                sa_out = self._sa_block(x, src_mask, src_key_padding_mask)
            x = self.norm1(x + sa_out)
            x = self.norm2(x + self._ff_block(x))
        return x
    for module in model.modules():
        if isinstance(module, nn.TransformerEncoderLayer):
            module.forward = types.MethodType(slow_forward, module)
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_linear_params(model):
    """Count total parameters in nn.Linear layers only."""
    total = 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            total += sum(p.numel() for p in m.parameters())
    return total


@torch.inference_mode()
def evaluate_loss(trainer, dataloaders):
    """
    Evaluate MSE loss and relative MSE loss for each dataset.
    Relative loss = MSE / mean(label²) on normalized data.
    Returns: (type_losses, type_rel_losses)
    """
    trainer.model.eval()
    min_ex = trainer.loss_cfg.min_ex
    type_losses = {}
    type_rel_losses = {}
    for dataset_type, loader in dataloaders.items():
        total_loss = 0.0
        total_label_sq = 0.0
        steps = 0
        for batch_cnt, batch in enumerate(tqdm(loader, desc=f"Evaluating {dataset_type}")):
            _, pairs, t_in, t_out, delta_t = batch
            pairs = trainer._move_to_device(pairs)

            loss = trainer.get_loss(pairs)
            total_loss += float(loss)

            # Compute label norm for relative loss
            data, mean, std = trainer._data_preprocess(pairs)
            label = trainer._get_label(data)[:, min_ex:, :-1]  # exclude type channel
            c_mask = data[2][:, :, :-1]  # [bs, 1, c-1, 1, 1]
            masked_label_sq = (label ** 2) * c_mask.float()
            label_sq = float(masked_label_sq.sum() / (c_mask.sum() * label.shape[-1] * label.shape[-2] + 1e-12))
            total_label_sq += label_sq

            steps += 1
        avg_loss = total_loss / steps if steps > 0 else 0
        avg_label_sq = total_label_sq / steps if steps > 0 else 1e-12
        type_losses[dataset_type] = avg_loss
        type_rel_losses[dataset_type] = avg_loss / (avg_label_sq + 1e-12)
    return type_losses, type_rel_losses


def load_compressed_model(model, ckpt_path, dev):
    state_dict = torch.load(ckpt_path, map_location=dev)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    convert_attn_to_linear(model)

    targets = [(name, module) for name, module in model.named_modules()
               if isinstance(module, nn.Linear)]

    for name, module in targets:
        weight_0_key = f"{name}.0.weight"
        weight_1_key = f"{name}.1.weight"
        if weight_0_key in state_dict and weight_1_key in state_dict:
            in_features = state_dict[weight_0_key].shape[1]
            k = state_dict[weight_0_key].shape[0]
            out_features = state_dict[weight_1_key].shape[0]
            has_bias = f"{name}.1.bias" in state_dict
            seq = nn.Sequential(
                nn.Linear(in_features, k, bias=False),
                nn.Linear(k, out_features, bias=has_bias)
            )
            if '.' in name:
                parent_name, attr = name.rsplit('.', 1)
                parent_mod = model.get_submodule(parent_name)
            else:
                parent_mod = model
                attr = name
            setattr(parent_mod, attr, seq)

    model.load_state_dict(state_dict)
    return model


# =====================================================================
# Metric value access helpers
# =====================================================================

def _get_value(entry, ds_name, data_key):
    """Get a metric value from a results entry."""
    if data_key == "loss":
        return entry.get("losses", {}).get(ds_name)
    elif data_key == "rel_loss":
        return entry.get("rel_losses", {}).get(ds_name)
    else:
        return entry.get("physics", {}).get(ds_name, {}).get(data_key)


def _calc_increase(val, orig_val):
    """Calculate percentage increase. Returns None if not computable."""
    if val is None or orig_val is None or orig_val == 0:
        return None
    return (val - orig_val) / (abs(orig_val) + 1e-12) * 100


# =====================================================================
# CSV Read/Write — hierarchical two-row header
# =====================================================================

def _columns_per_dataset(ds):
    """Return list of (col_header, data_key) for a dataset's CSV columns."""
    metrics = DATASET_METRICS.get(ds, [("Loss", "loss", "rel_loss"), ("Sobolev", "sobolev", "rel_sobolev")])
    cols = []
    for display_name, abs_key, rel_key in metrics:
        cols.append((display_name,           abs_key))
        cols.append((f"{display_name}_Inc",  f"_inc_{abs_key}"))   # placeholder key for increase
        cols.append((f"Rel_{display_name}",  rel_key))
        cols.append((f"Rel_{display_name}_Inc", f"_inc_{rel_key}"))
    return cols


def build_csv_header_rows(dataset_names):
    """
    Build two header rows:
      Row 1: Method, Total_Params, Size Red., Linear_Size, Linear_Red., DS1 (spanning), ...
      Row 2: (empty), (empty), (empty), (empty), (empty), Loss, Loss_Inc, ...
    """
    row1 = ["Method", "Total_Params", "Size Red.", "Linear_Size", "Linear_Red."]
    row2 = ["", "", "", "", ""]
    for ds in dataset_names:
        cols = _columns_per_dataset(ds)
        row1.append(ds)
        row1.extend([""] * (len(cols) - 1))
        for col_header, _ in cols:
            row2.append(col_header)
    return row1, row2


def save_csv_results(csv_path, results, dataset_names, orig_params):
    """
    Save results with hierarchical two-row header.
    Separate columns for value and increase%.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    row1, row2 = build_csv_header_rows(dataset_names)

    # Origin reference
    orig_entry = results.get("origin", {}).get("none", {})

    # Row ordering: origin first, then other methods
    method_order = ["origin"] + [m for m in results if m != "origin"]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row1)
        writer.writerow(row2)

        for method in method_order:
            if method not in results:
                continue
            for hp, entry in results[method].items():
                is_origin = (method == "origin")
                total_params = entry.get("params", 0)
                
                # Size reduction
                if is_origin:
                    size_red = "0.00%"
                else:
                    comp_ratio = entry.get("compression_ratio", 0.0)
                    if comp_ratio == 0.0 and orig_params > 0:
                        comp_ratio = (1.0 - total_params / orig_params) * 100
                    size_red = f"{comp_ratio:.2f}%"

                method_display = "Origin" if is_origin else (f"{method}({hp})" if hp != "none" else method)

                # Linear size reduction
                linear_val = entry.get("linear_size", "")
                orig_linear = orig_entry.get("linear_size")
                if is_origin:
                    linear_red = "0.00%"
                elif isinstance(linear_val, (int, float)) and isinstance(orig_linear, (int, float)) and orig_linear > 0:
                    linear_red = f"{(1.0 - linear_val / orig_linear) * 100:.2f}%"
                else:
                    linear_red = "-"

                # 插入总参数量到 Size Red. 前面
                row = [method_display, str(total_params), size_red, str(linear_val), linear_red]
                
                for ds in dataset_names:
                    metrics = DATASET_METRICS.get(ds, [("Loss", "loss", "rel_loss"), ("Sobolev", "sobolev", "rel_sobolev")])
                    for _, abs_key, rel_key in metrics:
                        abs_val = _get_value(entry, ds, abs_key)
                        rel_val = _get_value(entry, ds, rel_key)
                        orig_abs = _get_value(orig_entry, ds, abs_key)
                        orig_rel = _get_value(orig_entry, ds, rel_key)

                        abs_inc = _calc_increase(abs_val, orig_abs) if not is_origin else 0.0
                        rel_inc = _calc_increase(rel_val, orig_rel) if not is_origin else 0.0

                        row.append(f"{abs_val:.6f}" if abs_val is not None else "-")
                        row.append(f"{abs_inc:.2f}%" if abs_inc is not None else "-")
                        row.append(f"{rel_val:.6f}" if rel_val is not None else "-")
                        row.append(f"{rel_inc:.2f}%" if rel_inc is not None else "-")
                writer.writerow(row)


def load_csv_results(csv_path, dataset_names):
    """
    Load hierarchical two-row-header CSV back into the results dict.
    Also supports legacy single-row header format for backward compatibility.
    """
    results = {}
    if not os.path.exists(csv_path):
        return results

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header_row1 = next(reader, None)
        if not header_row1:
            return results

        # Detect format: new format has row2 starting with empty strings
        header_row2 = next(reader, None)
        if not header_row2:
            return results

        is_new_format = (len(header_row2) >= 2 and header_row2[0] == "" and header_row2[1] == "")

        if not is_new_format:
            return _load_csv_legacy(csv_path, dataset_names)

        has_total_params = (len(header_row1) > 1 and header_row1[1] == "Total_Params")

        # ---- New format parsing ----
        for row in reader:
            if not row or len(row) < 2:
                continue

            method_display = row[0].strip()

            # Parse method and hp
            if method_display == "Origin":
                method, hp = "origin", "none"
            else:
                m = re.match(r'^(.+?)\((.+?)\)$', method_display)
                if m:
                    method, hp = m.group(1), m.group(2)
                else:
                    method, hp = method_display, "none"

            if method not in results:
                results[method] = {}

            # Parse Size / Params based on whether Total_Params column exists
            if has_total_params:
                params_str = row[1].strip()
                try:
                    total_params = int(float(params_str))
                except ValueError:
                    total_params = 0
                
                size_str = row[2].strip().replace('%', '')
                try:
                    comp_ratio = float(size_str)
                except ValueError:
                    comp_ratio = 0.0

                if len(header_row1) >= 5 and header_row1[3] == "Linear_Size":
                    linear_str = row[3].strip() if len(row) > 3 else ""
                    try:
                        linear_size = int(float(linear_str)) if linear_str and linear_str != "-" else None
                    except ValueError:
                        linear_size = None
                    col_idx = 5
                else:
                    linear_size = None
                    col_idx = 3
            else:
                # Fallback for old 2-row header without Total_Params
                total_params = 0
                size_str = row[1].strip().replace('%', '')
                try:
                    comp_ratio = float(size_str)
                except ValueError:
                    comp_ratio = 0.0

                if len(header_row1) >= 4 and header_row1[2] == "Linear_Size":
                    linear_str = row[2].strip() if len(row) > 2 else ""
                    try:
                        linear_size = int(float(linear_str)) if linear_str and linear_str != "-" else None
                    except ValueError:
                        linear_size = None
                    col_idx = 4
                else:
                    linear_size = None
                    col_idx = 2

            entry = {
                "params": total_params,
                "linear_size": linear_size,
                "compression_ratio": comp_ratio,
                "losses": {},
                "rel_losses": {},
                "physics": {},
            }

            for ds in dataset_names:
                metrics = DATASET_METRICS.get(ds, [("Loss", "loss", "rel_loss"), ("Sobolev", "sobolev", "rel_sobolev")])
                for _, abs_key, rel_key in metrics:
                    abs_val = _parse_number(row[col_idx]) if col_idx < len(row) else None
                    col_idx += 1
                    col_idx += 1  # skip abs increase
                    rel_val = _parse_number(row[col_idx]) if col_idx < len(row) else None
                    col_idx += 1
                    col_idx += 1  # skip rel increase

                    if abs_val is not None:
                        if abs_key == "loss":
                            entry["losses"][ds] = abs_val
                        else:
                            if ds not in entry["physics"]:
                                entry["physics"][ds] = {}
                            entry["physics"][ds][abs_key] = abs_val

                    if rel_val is not None:
                        if rel_key == "rel_loss":
                            entry["rel_losses"][ds] = rel_val
                        else:
                            if ds not in entry["physics"]:
                                entry["physics"][ds] = {}
                            entry["physics"][ds][rel_key] = rel_val

            results[method][hp] = entry

    return results


def _parse_number(cell_str):
    """Parse a number from a cell. Returns None for '-' or empty."""
    if not cell_str or cell_str.strip() == '-':
        return None
    s = cell_str.strip().replace('%', '')
    try:
        return float(s)
    except ValueError:
        # Try extracting leading number (backward compat with 'value (↑ xx%)' format)
        m = re.match(r'([0-9eE.+-]+)', s)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
    return None


def _load_csv_legacy(csv_path, dataset_names):
    """Load legacy single-row-header CSV format (backward compatibility)."""
    results = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return results

        for row in reader:
            if not row or len(row) < 2:
                continue

            method_display = row[0].strip()
            if method_display == "Origin":
                method, hp = "origin", "none"
            else:
                m = re.match(r'^(.+?)\((.+?)\)$', method_display)
                if m:
                    method, hp = m.group(1), m.group(2)
                else:
                    method, hp = method_display, "none"

            size_str = row[1].strip().replace('%', '')
            try:
                comp_ratio = float(size_str)
            except ValueError:
                comp_ratio = 0.0

            if method not in results:
                results[method] = {}
            entry = {
                "params": 0,
                "compression_ratio": comp_ratio,
                "losses": {},
                "rel_losses": {},
                "physics": {},
            }

            # Legacy columns: DS_MetricName with "value (↑ xx%)" format
            col_idx = 2
            for ds in dataset_names:
                # Old DATASET_COLUMNS style
                old_cols = {
                    "NS2D": [("Loss", "loss"), ("Sobolev", "sobolev"), ("Div.", "continuity"), ("Vort.", "vorticity_err")],
                    "COMPRESSIBLE2D": [("Loss", "loss"), ("Sobolev", "sobolev")],
                    "EULER2D": [("Loss", "loss"), ("Sobolev", "sobolev")],
                }
                cols = old_cols.get(ds, [("Loss", "loss"), ("Sobolev", "sobolev")])
                for _, data_key in cols:
                    if col_idx < len(row):
                        val = _parse_number(row[col_idx])
                        if val is not None:
                            if data_key == "loss":
                                entry["losses"][ds] = val
                            else:
                                if ds not in entry["physics"]:
                                    entry["physics"][ds] = {}
                                entry["physics"][ds][data_key] = val
                    col_idx += 1

            results[method][hp] = entry

    return results


# =====================================================================
# Main
# =====================================================================

@hydra.main(version_base=None, config_path="../VICON/configs/", config_name="default")
def main(cfg: DictConfig):
    rank_ratio = cfg.get("rank_ratio", 0.2)
    dev = cfg.get("DEV", "cuda")

    comp_methods = ["ASVD", "SVDLLM", "SAES", "FWSVD", "DobiSVD", "MySVD"]
    hyper_params = {
        "ASVD": ["cali_2048_asvd_rank"],
        "SVDLLM": ["cali_2048", "cali_2048_single_NS2D"],
        "FWSVD": ["cali2048_order0"],
        "SAES": ["cali2048"],
        "DobiSVD": ["cali_2048"],
        # "DipSVD": ["cali2048"],
        "MySVD": ["cali2048_l1_order2_br10.0_comb",
                  "cali2048_l1_order2_br10.0_trace",
                  "cali2048_l1_br10.0_notrace",
                  "cali2048_l1_br10.0_trace"]
    }

    orig_ckpt_path = "VICON/output/ckpts/train/20260220-085512/200000_params.pth"
    csv_save_path = f"results/vicon_eval_results_{rank_ratio}.csv"

    # ================= 初始化数据集 =================
    print("\n[1/3] 初始化测试数据集 (Split: test)...")
    test_datasets = all_datasets(cfg.datasets, cfg.dataset_workers, cfg.test_seed, "test")
    test_loaders = {
        k: torch.utils.data.DataLoader(v, batch_size=128, num_workers=cfg.dataset_workers, pin_memory=True)
        for k, v in test_datasets.items()
    }
    dataset_names = list(test_loaders.keys())

    # ================= 读取历史结果 =================
    print(f"\n[2/3] 读取历史评测结果 ({csv_save_path})...")
    results = load_csv_results(csv_save_path, dataset_names)

    # ================= Backfill linear_size if missing =================
    _backfill_needed = False
    for _m, _m_res in results.items():
        for _hp, _entry in _m_res.items():
            if "linear_size" not in _entry or _entry.get("linear_size") is None:
                _backfill_needed = True
                break
        if _backfill_needed:
            break

    if _backfill_needed:
        print("\n> Backfilling linear_size for existing results...")
        for _m, _m_res in results.items():
            for _hp, _entry in _m_res.items():
                if "linear_size" in _entry and _entry["linear_size"] is not None:
                    continue
                # Load model to count linear params
                if _m == "origin":
                    _model = models.ICON_CROPPED(cfg.model) if cfg.model.type == "crop" else models.ICON_UNCROPPED(cfg.model)
                    _state = torch.load(orig_ckpt_path, map_location='cpu')
                    if 'model_state_dict' in _state:
                        _model.load_state_dict(_state['model_state_dict'])
                    else:
                        _model.load_state_dict(_state)
                    convert_attn_to_linear(_model)  # expose MHA params as nn.Linear
                else:
                    _ckpt = f"compressed_model/{rank_ratio}/{_m}/{_hp}/compressed_model.pth"
                    if not os.path.exists(_ckpt):
                        print(f"    {_m}({_hp}): checkpoint not found, skipping.")
                        continue
                    _model = models.ICON_CROPPED(cfg.model) if cfg.model.type == "crop" else models.ICON_UNCROPPED(cfg.model)
                    _model = load_compressed_model(_model, _ckpt, dev)
                _entry["linear_size"] = count_linear_params(_model)
                print(f"    {_m}({_hp}): linear_size = {_entry['linear_size']}")
                del _model

        orig_params_bf = results.get("origin", {}).get("none", {}).get("params", 0)
        save_csv_results(csv_save_path, results, dataset_names, orig_params_bf)
        print("  Backfill complete, CSV updated.")

    # ================= 评测原模型 =================
    origin_needs_eval = "origin" not in results or "none" not in results.get("origin", {})
    if not origin_needs_eval and not results["origin"]["none"].get("physics"):
        print("\n> 原模型缺少物理指标，需要重新评测。")
        origin_needs_eval = True
    # Also re-eval if missing rel_losses (upgraded from legacy format)
    if not origin_needs_eval and not results["origin"]["none"].get("rel_losses"):
        print("\n> 原模型缺少相对指标，需要重新评测。")
        origin_needs_eval = True

    if origin_needs_eval:
        print(f"\n> 原模型结果未找到，开始加载并评测: {orig_ckpt_path}")
        model_orig = models.ICON_CROPPED(cfg.model) if cfg.model.type == "crop" else models.ICON_UNCROPPED(cfg.model)

        orig_state = torch.load(orig_ckpt_path, map_location='cpu')
        if 'model_state_dict' in orig_state:
            model_orig.load_state_dict(orig_state['model_state_dict'])
        else:
            model_orig.load_state_dict(orig_state)

        orig_params = count_parameters(model_orig)

        trainer_orig = Trainer(model_orig, cfg.model, cfg.opt, cfg.loss, trainable_mode=cfg.trainable_mode)
        orig_losses, orig_rel_losses = evaluate_loss(trainer_orig, test_loaders)
        orig_physics = evaluate_physics_metrics(trainer_orig, test_loaders, check_gt=True, sobolev_order_map=SOBOLEV_ORDER_MAP)

        if "origin" not in results:
            results["origin"] = {}
        # convert MHA to Linear before counting so linear_size matches compressed models
        convert_attn_to_linear(model_orig)
        results["origin"]["none"] = {
            "params": orig_params,
            "linear_size": count_linear_params(model_orig),
            "compression_ratio": 0.0,
            "losses": orig_losses,
            "rel_losses": orig_rel_losses,
            "physics": orig_physics,
        }
        save_csv_results(csv_save_path, results, dataset_names, orig_params)
    else:
        print("\n> 原模型结果已存在，直接加载使用。")
        orig_params = results["origin"]["none"].get("params", 0)

    # ================= 评测压缩模型 =================
    print("\n[3/3] 开始评估压缩模型...")
    for comp_method in comp_methods:
        if comp_method not in hyper_params:
            continue
        for hp in hyper_params[comp_method]:
            # 跳过已评测
            if comp_method in results and hp in results[comp_method]:
                entry = results[comp_method][hp]
                if entry.get("physics") and entry.get("rel_losses"):
                    # Check if physics has relative metrics
                    has_rel = any(
                        any(k.startswith("rel_") for k in entry["physics"].get(ds, {}))
                        for ds in dataset_names if ds in entry.get("physics", {})
                    )
                    if has_rel:
                        print(f"  >>> 跳过 {comp_method} | {hp}: 已评估。")
                        continue
                    else:
                        print(f"  >>> {comp_method} | {hp}: 缺少相对物理指标，重新评测。")
                elif entry.get("physics") and not entry.get("rel_losses"):
                    print(f"  >>> {comp_method} | {hp}: 缺少相对 Loss，重新评测。")
                else:
                    print(f"  >>> {comp_method} | {hp}: 缺少物理指标，重新评测。")

            comp_ckpt_path = f"compressed_model/{rank_ratio}/{comp_method}/{hp}/compressed_model.pth"
            if not os.path.exists(comp_ckpt_path):
                print(f"  >>> 警告：未找到 {comp_ckpt_path}，跳过。")
                continue

            print(f"\nEvaluating {comp_method} | {hp}")
            model_comp_base = models.ICON_CROPPED(cfg.model) if cfg.model.type == "crop" else models.ICON_UNCROPPED(cfg.model)
            model_comp = load_compressed_model(model_comp_base, comp_ckpt_path, dev)
            model_comp = disable_fast_path(model_comp)

            comp_params = count_parameters(model_comp)
            comp_ratio = (1.0 - comp_params / orig_params) * 100

            trainer_comp = Trainer(model_comp, cfg.model, cfg.opt, cfg.loss, trainable_mode=cfg.trainable_mode)
            comp_losses, comp_rel_losses = evaluate_loss(trainer_comp, test_loaders)
            comp_physics = evaluate_physics_metrics(trainer_comp, test_loaders, sobolev_order_map=SOBOLEV_ORDER_MAP)

            if comp_method not in results:
                results[comp_method] = {}
            results[comp_method][hp] = {
                "params": comp_params,
                "linear_size": count_linear_params(model_comp),
                "compression_ratio": comp_ratio,
                "losses": comp_losses,
                "rel_losses": comp_rel_losses,
                "physics": comp_physics,
            }
            save_csv_results(csv_save_path, results, dataset_names, orig_params)

            # ================= 实时报告 =================
            _print_report(comp_method, hp, comp_ratio, results, dataset_names)

    print(f"\n✅ 所有评估已完成！结果已更新至: {csv_save_path}")


def _print_report(comp_method, hp, comp_ratio, results, dataset_names):
    """Print a formatted per-dataset report to console."""
    orig_entry = results["origin"]["none"]
    comp_entry = results[comp_method][hp]

    print(f"\n{'='*80}")
    print(f"  {comp_method}({hp})  Size Red.: {comp_ratio:.2f}%")
    print(f"{'='*80}")

    for ds in dataset_names:
        metrics = DATASET_METRICS.get(ds, [("Loss", "loss", "rel_loss"), ("Sobolev", "sobolev", "rel_sobolev")])
        if not metrics:
            continue

        print(f"\n  ── {ds} ──")
        header_parts = []
        value_parts = []
        for display_name, abs_key, rel_key in metrics:
            abs_val = _get_value(comp_entry, ds, abs_key)
            rel_val = _get_value(comp_entry, ds, rel_key)
            orig_abs = _get_value(orig_entry, ds, abs_key)
            orig_rel = _get_value(orig_entry, ds, rel_key)

            abs_inc = _calc_increase(abs_val, orig_abs)
            rel_inc = _calc_increase(rel_val, orig_rel)

            abs_str = f"{abs_val:.6f}" if abs_val is not None else "-"
            rel_str = f"{rel_val:.6f}" if rel_val is not None else "-"
            abs_inc_str = f"↑{abs_inc:.1f}%" if abs_inc is not None else ""
            rel_inc_str = f"↑{rel_inc:.1f}%" if rel_inc is not None else ""

            print(f"    {display_name:>8s}: {abs_str} ({abs_inc_str})  |  Rel: {rel_str} ({rel_inc_str})")

    print(f"{'='*80}")


if __name__ == "__main__":
    main()