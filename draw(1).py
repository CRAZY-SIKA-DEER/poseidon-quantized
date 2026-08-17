"""
Auto-regressive multi-step visualization (Scene B).

布局控制：
- TRANSPOSE_LAYOUT = False: 行是 method, 列是 time step (Origin/GT 在最后)
- TRANSPOSE_LAYOUT = True:  行是 time step, 列是 method (Origin/GT 在最后)

推理模式：
- MODE = "rollout":  自回归链式预测, 0→5→10→15, 每步输入是上一步的预测 (误差累积)
- MODE = "oneshot":  始终从 t=0 出发, 分别一步预测 0→5, 0→10, 0→15 (无误差累积)
"""
from poseidon_main.scOT.inference import get_trainer, rollout, get_trajectories
from poseidon_main.scOT.problems.base import get_dataset
from poseidon_main.scOT.problems.fluids.normalization_constants import CONSTANTS
import matplotlib.pyplot as plt
import numpy as np
import os
import csv  # 用于保存 loss 到 CSV 文件

# ============ 配置 ============
ds_name = "fluids.incompressible.PiecewiseConstants"
sample_idx = 0

# 推理模式: "rollout" 或 "oneshot"
MODE = "rollout"

# 控制画图的横竖排版 (False: 竖排, True: 横排)
TRANSPOSE_LAYOUT = True

# AR 参数
AR_STEPS = 1
INITIAL_TIME = 0
FINAL_TIME = 20
rank_ratio = 0.2
DATA_PATH = "poseidon_main/camlab-ethz/down_streams"

assert (FINAL_TIME - INITIAL_TIME) % AR_STEPS == 0, \
    f"(FINAL_TIME - INITIAL_TIME)={FINAL_TIME - INITIAL_TIME} 必须能被 AR_STEPS={AR_STEPS} 整除"

delta_t = (FINAL_TIME - INITIAL_TIME) // AR_STEPS

# 注意：为了计算 loss increase，"Origin" 必须放在第一个，以便优先提取 baseline
models = {
    "Origin":  f"checkpoints/Poseidon-L/no_phyloss/{ds_name}/origin/none",
    # "ASVD":    f"compression/compressed_ft_models/{ds_name}/{rank_ratio}/ASVD/traj64",
    # "FWSVD":   f"compression/compressed_ft_models/{ds_name}/{rank_ratio}/FWSVD/l1_order0_br10.0_traj64",
    # "SVDLLM":  f"compression/compressed_ft_models/{ds_name}/{rank_ratio}/SVDLLM/traj64",
    # "Dobi-SVD": f"compression/compressed_ft_models/{ds_name}/{rank_ratio}/DobiSVD/traj64",
    # "SAES-SVD":f"compression/compressed_ft_models/{ds_name}/{rank_ratio}/SAES_SVD/traj64",
    # "Ours":   f"compression/compressed_ft_models/{ds_name}/{rank_ratio}/MySVD/l1_order2_alpha0.7",
}

# ============ PDE 分类 & 归一化常数 ============
PDE_Class_Map = {
    "fluids.incompressible.BrownianBridge": "NS-BB",
    "fluids.incompressible.PiecewiseConstants": "NS-PwC",
    "fluids.incompressible.VortexSheet": "NS-SVS",
    "fluids.compressible.RichtmyerMeshkov": "CE-RM",
    "fluids.compressible.RiemannKelvinHelmholtz": "CE-RPUI",
    "wave.Gaussians": "Wave-Gauss",
    "wave.Layer": "Wave-Layer",
}

PDE = PDE_Class_Map[ds_name]


def get_channel_config(dataset_name, num_channels):
    """
    根据数据集类型和通道数，返回要可视化的通道配置。
    """
    if "incompressible" in dataset_name:
        if num_channels == 3:
            return {"u": 0, "v": 1}
        else:
            return {"u": 1, "v": 2}
    elif "compressible" in dataset_name:
        return {"rho": 0, "u": 1, "v": 2, "p": 3}
    elif "wave" in dataset_name.lower() or "Wave" in dataset_name:
        return {"u": 0}
    else:
        raise ValueError(f"Unknown dataset type: {dataset_name}")


def denormalize(data, mean, std):
    """反白化: data * std + mean"""
    return data * std + mean


# ============ 目标可视化时刻 ============
target_times = [INITIAL_TIME + (s + 1) * delta_t for s in range(AR_STEPS)]

save_dir = f"results/finetuned_results/{ds_name}/draw_exp_{FINAL_TIME}_{AR_STEPS}"
os.makedirs(save_dir, exist_ok=True)

# ============ GT 轨迹 ============
labels_all_steps = get_trajectories(
    dataset=ds_name, data_path=DATA_PATH,
    ar_steps=AR_STEPS, initial_time=INITIAL_TIME, final_time=FINAL_TIME,
    dataset_kwargs={},
)
if hasattr(labels_all_steps, "cpu"):
    labels_all_steps = labels_all_steps.cpu().numpy()
gt_raw = labels_all_steps


# 用于存储 Loss 以便写入 CSV
csv_rows = []

# =====================================================================
# Rollout 模式
# =====================================================================
if MODE == "rollout":
    test_ds = get_dataset(
        dataset=ds_name, which="test", num_trajectories=1,
        data_path=DATA_PATH,
        fix_input_to_time_step=INITIAL_TIME,
        time_step_size=FINAL_TIME - INITIAL_TIME,
        max_num_time_steps=1,
    )
    constants = test_ds.constants
    mean = np.array(constants['mean']).reshape(-1, 1, 1)
    std  = np.array(constants['std']).reshape(-1, 1, 1)

    all_preds = {}
    origin_loss = None
    
    for name, path in models.items():
        trainer = get_trainer(model_path=path, batch_size=1024, dataset=test_ds, output_all_steps=True)
        preds, _, metrics = rollout(trainer, test_ds, model_path="Poseidon-L",
                              ar_steps=AR_STEPS, output_all_steps=True)
        
        model_loss = metrics.get("_loss", None) if metrics else None
        
        if name == "Origin":
            origin_loss = model_loss
            
        # 计算 Loss Increase (绝对值和百分比)
        loss_inc_abs = (model_loss - origin_loss) if (model_loss is not None and origin_loss is not None) else None
        loss_inc_pct = (loss_inc_abs / origin_loss * 100) if (loss_inc_abs is not None and origin_loss) else None
        
        print(f"Model: {name}, Mode: {MODE}, Loss: {model_loss}")
        
        csv_rows.append({
            "Mode": MODE,
            "Step": "All_Rollout",
            "Model": name,
            "Loss": model_loss,
            "Origin_Loss": origin_loss,
            "Loss_Increase_Abs": loss_inc_abs,
            "Loss_Increase(%)": loss_inc_pct
        })

        if hasattr(preds, "cpu"):
            preds = preds.cpu().numpy()
        all_preds[name] = denormalize(preds, mean, std)

# =====================================================================
# One-shot 模式
# =====================================================================
elif MODE == "oneshot":
    all_preds = {name: [] for name in models}

    for step_i in range(AR_STEPS):
        step_size = (step_i + 1) * delta_t
        print(f"  [oneshot] step {step_i}: fix_input_to_time_step={INITIAL_TIME}, "
              f"time_step_size={step_size}  →  t=0 → t={INITIAL_TIME + step_size}")

        ds_step = get_dataset(
            dataset=ds_name, which="test", num_trajectories=1,
            data_path=DATA_PATH,
            fix_input_to_time_step=INITIAL_TIME,
            time_step_size=step_size,
            max_num_time_steps=1,
        )
        constants = ds_step.constants
        mean = np.array(constants['mean']).reshape(-1, 1, 1)
        std  = np.array(constants['std']).reshape(-1, 1, 1)

        origin_loss = None
        
        for name, path in models.items():
            trainer = get_trainer(model_path=path, batch_size=1024, dataset=ds_step)
            preds, _, metrics = rollout(trainer, ds_step, model_path="Poseidon-L")
            model_loss = metrics.get("_loss", None) if metrics else None
            
            if name == "Origin":
                origin_loss = model_loss
                
            # 计算 Loss Increase (绝对值和百分比)
            loss_inc_abs = (model_loss - origin_loss) if (model_loss is not None and origin_loss is not None) else None
            loss_inc_pct = (loss_inc_abs / origin_loss * 100) if (loss_inc_abs is not None and origin_loss) else None
            
            if model_loss is not None:
                print(f"Model: {name}, Step: {step_i}, Loss: {model_loss:.4e}")

            csv_rows.append({
                "Mode": MODE,
                "Step": step_i,
                "Model": name,
                "Loss": model_loss,
                "Origin_Loss": origin_loss,
                "Loss_Increase_Abs": loss_inc_abs,
                "Loss_Increase(%)": loss_inc_pct
            })

            if hasattr(preds, "cpu"):
                preds = preds.cpu().numpy()
            all_preds[name].append(preds)

    for name in models:
        stacked = np.stack(all_preds[name], axis=1)
        all_preds[name] = denormalize(stacked, mean, std)

else:
    raise ValueError(f"Unknown MODE: {MODE}, must be 'rollout' or 'oneshot'")


# =====================================================================
# 保存 Loss Increase 到 CSV
# =====================================================================
csv_file_path = os.path.join(save_dir, f"loss_increase_{MODE}.csv")
with open(csv_file_path, mode="w", newline="", encoding="utf-8") as f:
    # 字段名更新为包含百分比标识
    fieldnames = ["Mode", "Step", "Model", "Loss", "Origin_Loss", "Loss_Increase_Abs", "Loss_Increase(%)"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_rows)
print(f"\n[+] Loss increase metrics successfully saved to: {csv_file_path}\n")


# ============ 绘图准备 ============
gt = denormalize(gt_raw, mean, std)
num_channels = gt.shape[2]
channels = get_channel_config(ds_name, num_channels)
print(f"Dataset: {ds_name}, num_channels: {num_channels}, visualizing: {channels}")

other_models = [m for m in models.keys() if m != "Origin"]
methods_list = other_models + ["Origin", "GT"]

num_methods = len(methods_list)
num_steps = AR_STEPS

# =====================================================================
# 绘图: 所有 field 合并到一张图
# =====================================================================
field_list = list(channels.items())  # [(field_name, ch), ...]
num_fields = len(field_list)

print(f"Plotting all fields together (Mode: {MODE}, Layout Transpose: {TRANSPOSE_LAYOUT})")
print(f"  Fields: {[f[0] for f in field_list]}, Steps: {num_steps}, Methods: {num_methods}")

if TRANSPOSE_LAYOUT:
    # 行: field × time_step,  列: method
    n_rows = num_fields * num_steps
    n_cols = num_methods
else:
    # 行: method,  列: field × time_step
    n_rows = num_methods
    n_cols = num_fields * num_steps

fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.2 * n_rows))

if n_rows == 1 and n_cols == 1: axes = np.array([[axes]])
elif n_rows == 1: axes = axes[np.newaxis, :]
elif n_cols == 1: axes = axes[:, np.newaxis]

# 预计算每个 field 的 vmin/vmax
field_vlims = {}
for field_name, ch in field_list:
    vmin = np.percentile(gt[sample_idx, :, ch], 1)
    vmax = np.percentile(gt[sample_idx, :, ch], 99)
    field_vlims[field_name] = (vmin, vmax)

# 保存每个 field 最后一个 im，用于各自的 colorbar
field_last_im = {}

for fi, (field_name, ch) in enumerate(field_list):
    gvmin, gvmax = field_vlims[field_name]

    for step_idx in range(num_steps):
        for method_idx in range(num_methods):
            row_name = methods_list[method_idx]

            if TRANSPOSE_LAYOUT:
                r = fi * num_steps + step_idx
                c = method_idx
            else:
                r = method_idx
                c = fi * num_steps + step_idx

            ax = axes[r, c]

            data_field = (gt[sample_idx, step_idx, ch]
                          if row_name == "GT"
                          else all_preds[row_name][sample_idx, step_idx, ch])

            im = ax.imshow(
                data_field.T, origin='lower', cmap='RdBu_r',
                vmin=gvmin, vmax=gvmax,
            )
            field_last_im[field_name] = im

            t_curr = target_times[step_idx]
            is_ours = (row_name == "Ours")
            fw = 'bold' if is_ours else 'normal'

            if TRANSPOSE_LAYOUT:
                # 方法名只在最顶行显示
                if r == 0:
                    ax.set_title(row_name, fontsize=22, fontweight=fw)
                # 左侧: "field, t=X"
                if c == 0:
                    label = f"t = {t_curr}" # if num_fields > 1 else f"t = {t_curr}"
                    ax.set_ylabel(label, fontsize=22)
            else:
                # 方法名只在最左列显示
                if c == 0:
                    ax.set_ylabel(row_name, fontsize=22, fontweight=fw)
                # 顶部: "field, t=X"
                if r == 0:
                    label = f"t = {t_curr}" # if num_fields > 1 else f"t = {t_curr}"
                    ax.set_title(label, fontsize=22)

            ax.set_xticks([]); ax.set_yticks([])

# ============ colorbar: 根据实际 axes 位置对齐 ============
plt.tight_layout(rect=[0, 0, 0.94, 1.0])

# tight_layout 之后，读取真实 axes 位置来定位 colorbar
fig.canvas.draw()  # 强制计算布局

if num_fields == 1:
    # 取第一行和最后一行的 axes 边界
    top_ax = axes[0, -1]
    bot_ax = axes[n_rows - 1, -1]
    top_pos = top_ax.get_position()
    bot_pos = bot_ax.get_position()
    cbar_top = top_pos.y1
    cbar_bot = bot_pos.y0
    cbar_ax = fig.add_axes([0.95, cbar_bot, 0.015, cbar_top - cbar_bot])
    fig.colorbar(field_last_im[field_list[0][0]], cax=cbar_ax)
else:
    for fi, (field_name, _) in enumerate(field_list):
        # 该 field 对应的行范围
        if TRANSPOSE_LAYOUT:
            first_row = fi * num_steps
            last_row  = fi * num_steps + num_steps - 1
        else:
            first_row = 0
            last_row  = n_rows - 1

        top_pos = axes[first_row, -1].get_position()
        bot_pos = axes[last_row, -1].get_position()
        cbar_top = top_pos.y1
        cbar_bot = bot_pos.y0

        cbar_ax = fig.add_axes([0.95, cbar_bot, 0.015, cbar_top - cbar_bot])
        cb = fig.colorbar(field_last_im[field_name], cax=cbar_ax)
        cb.set_label(field_name, fontsize=14)

layout_str = "horiz" if TRANSPOSE_LAYOUT else "vert"
plt.savefig(f"{save_dir}/all_fields_{MODE}_{layout_str}.png",
            dpi=200, bbox_inches='tight')
plt.close(fig)

print(f"\nFinished! Figure saved to: {save_dir}/all_fields_{MODE}_{layout_str}.png")

# """
# Auto-regressive multi-step visualization (Scene B).

# 布局控制：
# - TRANSPOSE_LAYOUT = False: 行是 method, 列是 time step (Origin/GT 在最后)
# - TRANSPOSE_LAYOUT = True:  行是 time step, 列是 method (Origin/GT 在最后)

# 推理模式：
# - MODE = "rollout":  自回归链式预测, 0→5→10→15, 每步输入是上一步的预测 (误差累积)
# - MODE = "oneshot":  始终从 t=0 出发, 分别一步预测 0→5, 0→10, 0→15 (无误差累积)
# """
# from poseidon_main.scOT.inference import get_trainer, rollout, get_trajectories
# from poseidon_main.scOT.problems.base import get_dataset
# from poseidon_main.scOT.problems.fluids.normalization_constants import CONSTANTS
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import csv  # 用于保存 loss 到 CSV 文件

# # ============ 配置 ============
# ds_name = "fluids.incompressible.PiecewiseConstants"
# sample_idx = 0

# # 推理模式: "rollout" 或 "oneshot"
# MODE = "rollout"

# # 控制画图的横竖排版 (False: 竖排, True: 横排)
# TRANSPOSE_LAYOUT = True

# # AR 参数
# AR_STEPS = 3
# INITIAL_TIME = 0
# FINAL_TIME = 15
# rank_ratio = 0.2
# DATA_PATH = "poseidon_main/camlab-ethz/down_streams"

# assert (FINAL_TIME - INITIAL_TIME) % AR_STEPS == 0, \
#     f"(FINAL_TIME - INITIAL_TIME)={FINAL_TIME - INITIAL_TIME} 必须能被 AR_STEPS={AR_STEPS} 整除"

# delta_t = (FINAL_TIME - INITIAL_TIME) // AR_STEPS

# # 注意：为了计算 loss increase，"Origin" 必须放在第一个，以便优先提取 baseline
# models = {
#     "Origin":  f"checkpoints/Poseidon-L/no_phyloss/{ds_name}/origin/none",
#     "ASVD":    f"compression/compressed_ft_models/{ds_name}/{rank_ratio}/ASVD/traj64",
#     "FWSVD":   f"compression/compressed_ft_models/{ds_name}/{rank_ratio}/FWSVD/l1_order0_br10.0_traj64",
#     "SVD-LLM V2":  f"compression/compressed_ft_models/{ds_name}/{rank_ratio}/SVDLLM/traj64",
#     "Dobi-SVD": f"compression/compressed_ft_models/{ds_name}/{rank_ratio}/DobiSVD/traj64",
#     "SAES-SVD":f"compression/compressed_ft_models/{ds_name}/{rank_ratio}/SAES_SVD/traj64",
#     "Ours":   f"compression/compressed_ft_models/{ds_name}/{rank_ratio}/MySVD/l1_order2_alpha0.7",
# }

# # ============ PDE 分类 & 归一化常数 ============
# PDE_Class_Map = {
#     "fluids.incompressible.BrownianBridge": "NS-BB",
#     "fluids.incompressible.PiecewiseConstants": "NS-PwC",
#     "fluids.incompressible.VortexSheet": "NS-SVS",
#     "fluids.compressible.RichtmyerMeshkov": "CE-RM",
#     "fluids.compressible.RiemannKelvinHelmholtz": "CE-RPUI",
#     "wave.Gaussians": "Wave-Gauss",
#     "wave.Layer": "Wave-Layer",
# }

# PDE = PDE_Class_Map[ds_name]


# def get_channel_config(dataset_name, num_channels):
#     """
#     根据数据集类型和通道数，返回要可视化的通道配置。
#     """
#     if "incompressible" in dataset_name:
#         if num_channels == 3:
#             return {"u": 0, "v": 1, "p": 2}
#         else:
#             return {"u": 1, "v": 2, "p": 3}
#     elif "compressible" in dataset_name:
#         return {"rho": 0, "u": 1, "v": 2, "p": 3}
#     elif "wave" in dataset_name.lower() or "Wave" in dataset_name:
#         return {"u": 0}
#     else:
#         raise ValueError(f"Unknown dataset type: {dataset_name}")


# def denormalize(data, mean, std):
#     """反白化: data * std + mean"""
#     return data * std + mean


# # ============ 目标可视化时刻 ============
# target_times = [INITIAL_TIME + (s + 1) * delta_t for s in range(AR_STEPS)]

# save_dir = f"results/finetuned_results/{ds_name}/draw_exp_{FINAL_TIME}_{AR_STEPS}"
# os.makedirs(save_dir, exist_ok=True)

# # ============ GT 轨迹 ============
# labels_all_steps = get_trajectories(
#     dataset=ds_name, data_path=DATA_PATH,
#     ar_steps=AR_STEPS, initial_time=INITIAL_TIME, final_time=FINAL_TIME,
#     dataset_kwargs={},
# )
# if hasattr(labels_all_steps, "cpu"):
#     labels_all_steps = labels_all_steps.cpu().numpy()
# gt_raw = labels_all_steps


# # 用于存储 Loss 以便写入 CSV
# csv_rows = []

# # =====================================================================
# # Rollout 模式
# # =====================================================================
# if MODE == "rollout":
#     test_ds = get_dataset(
#         dataset=ds_name, which="test", num_trajectories=1,
#         data_path=DATA_PATH,
#         fix_input_to_time_step=INITIAL_TIME,
#         time_step_size=FINAL_TIME - INITIAL_TIME,
#         max_num_time_steps=1,
#     )
#     constants = test_ds.constants
#     mean = np.array(constants['mean']).reshape(-1, 1, 1)
#     std  = np.array(constants['std']).reshape(-1, 1, 1)

#     all_preds = {}
#     origin_loss = None
    
#     for name, path in models.items():
#         trainer = get_trainer(model_path=path, batch_size=1024, dataset=test_ds, output_all_steps=True)
#         preds, _, metrics = rollout(trainer, test_ds, model_path="Poseidon-L",
#                               ar_steps=AR_STEPS, output_all_steps=True)
        
#         model_loss = metrics.get("_loss", None) if metrics else None
        
#         if name == "Origin":
#             origin_loss = model_loss
            
#         # 计算 Loss Increase (绝对值和百分比)
#         loss_inc_abs = (model_loss - origin_loss) if (model_loss is not None and origin_loss is not None) else None
#         loss_inc_pct = (loss_inc_abs / origin_loss * 100) if (loss_inc_abs is not None and origin_loss) else None
        
#         print(f"Model: {name}, Mode: {MODE}, Loss: {model_loss}")
        
#         csv_rows.append({
#             "Mode": MODE,
#             "Step": "All_Rollout",
#             "Model": name,
#             "Loss": model_loss,
#             "Origin_Loss": origin_loss,
#             "Loss_Increase_Abs": loss_inc_abs,
#             "Loss_Increase(%)": loss_inc_pct
#         })

#         if hasattr(preds, "cpu"):
#             preds = preds.cpu().numpy()
#         all_preds[name] = denormalize(preds, mean, std)

# # =====================================================================
# # One-shot 模式
# # =====================================================================
# elif MODE == "oneshot":
#     all_preds = {name: [] for name in models}

#     for step_i in range(AR_STEPS):
#         step_size = (step_i + 1) * delta_t
#         print(f"  [oneshot] step {step_i}: fix_input_to_time_step={INITIAL_TIME}, "
#               f"time_step_size={step_size}  →  t=0 → t={INITIAL_TIME + step_size}")

#         ds_step = get_dataset(
#             dataset=ds_name, which="test", num_trajectories=1,
#             data_path=DATA_PATH,
#             fix_input_to_time_step=INITIAL_TIME,
#             time_step_size=step_size,
#             max_num_time_steps=1,
#         )
#         constants = ds_step.constants
#         mean = np.array(constants['mean']).reshape(-1, 1, 1)
#         std  = np.array(constants['std']).reshape(-1, 1, 1)

#         origin_loss = None
        
#         for name, path in models.items():
#             trainer = get_trainer(model_path=path, batch_size=1024, dataset=ds_step)
#             preds, _, metrics = rollout(trainer, ds_step, model_path="Poseidon-L")
#             model_loss = metrics.get("_loss", None) if metrics else None
            
#             if name == "Origin":
#                 origin_loss = model_loss
                
#             # 计算 Loss Increase (绝对值和百分比)
#             loss_inc_abs = (model_loss - origin_loss) if (model_loss is not None and origin_loss is not None) else None
#             loss_inc_pct = (loss_inc_abs / origin_loss * 100) if (loss_inc_abs is not None and origin_loss) else None
            
#             if model_loss is not None:
#                 print(f"Model: {name}, Step: {step_i}, Loss: {model_loss:.4e}")

#             csv_rows.append({
#                 "Mode": MODE,
#                 "Step": step_i,
#                 "Model": name,
#                 "Loss": model_loss,
#                 "Origin_Loss": origin_loss,
#                 "Loss_Increase_Abs": loss_inc_abs,
#                 "Loss_Increase(%)": loss_inc_pct
#             })

#             if hasattr(preds, "cpu"):
#                 preds = preds.cpu().numpy()
#             all_preds[name].append(preds)

#     for name in models:
#         stacked = np.stack(all_preds[name], axis=1)
#         all_preds[name] = denormalize(stacked, mean, std)

# else:
#     raise ValueError(f"Unknown MODE: {MODE}, must be 'rollout' or 'oneshot'")


# # =====================================================================
# # 保存 Loss Increase 到 CSV
# # =====================================================================
# csv_file_path = os.path.join(save_dir, f"loss_increase_{MODE}.csv")
# with open(csv_file_path, mode="w", newline="", encoding="utf-8") as f:
#     # 字段名更新为包含百分比标识
#     fieldnames = ["Mode", "Step", "Model", "Loss", "Origin_Loss", "Loss_Increase_Abs", "Loss_Increase(%)"]
#     writer = csv.DictWriter(f, fieldnames=fieldnames)
#     writer.writeheader()
#     writer.writerows(csv_rows)
# print(f"\n[+] Loss increase metrics successfully saved to: {csv_file_path}\n")


# # ============ 绘图准备 ============
# gt = denormalize(gt_raw, mean, std)
# num_channels = gt.shape[2]
# channels = get_channel_config(ds_name, num_channels)
# print(f"Dataset: {ds_name}, num_channels: {num_channels}, visualizing: {channels}")

# other_models = [m for m in models.keys() if m != "Origin"]
# methods_list = other_models + ["Origin", "GT"]

# num_methods = len(methods_list)
# num_steps = AR_STEPS

# # =====================================================================
# # 绘图循环
# # =====================================================================
# for field_name, ch in channels.items():
#     print(f"Plotting {field_name} (Mode: {MODE}, Layout Transpose: {TRANSPOSE_LAYOUT})")

#     n_rows = num_steps if TRANSPOSE_LAYOUT else num_methods
#     n_cols = num_methods if TRANSPOSE_LAYOUT else num_steps

#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.2 * n_rows))

#     if n_rows == 1 and n_cols == 1: axes = np.array([[axes]])
#     elif n_rows == 1: axes = axes[np.newaxis, :]
#     elif n_cols == 1: axes = axes[:, np.newaxis]

#     global_vmin = np.percentile(gt[sample_idx, :, ch], 1)
#     global_vmax = np.percentile(gt[sample_idx, :, ch], 99)

#     for r in range(n_rows):
#         for c in range(n_cols):
#             ax = axes[r, c]

#             if TRANSPOSE_LAYOUT:
#                 step_idx, method_idx = r, c
#             else:
#                 method_idx, step_idx = r, c

#             row_name = methods_list[method_idx]

#             data_field = (gt[sample_idx, step_idx, ch]
#                           if row_name == "GT"
#                           else all_preds[row_name][sample_idx, step_idx, ch])

#             im = ax.imshow(
#                 data_field.T, origin='lower', cmap='RdBu_r',
#                 vmin=global_vmin, vmax=global_vmax,
#             )

#             t_curr = target_times[step_idx]
#             is_ours = (row_name == "Ours")
#             fw = 'bold' if is_ours else 'normal'

#             if TRANSPOSE_LAYOUT:
#                 if r == 0:
#                     ax.set_title(row_name, fontsize=22, fontweight=fw)
#                 if c == 0:
#                     ax.set_ylabel(f"t = {t_curr}", fontsize=22)
#             else:
#                 if r == 0:
#                     ax.set_title(f"t = {t_curr}", fontsize=22)
#                 if c == 0:
#                     ax.set_ylabel(row_name, fontsize=22, fontweight=fw)

#             ax.set_xticks([]); ax.set_yticks([])

#     plt.tight_layout(rect=[0, 0, 0.94, 1.0])

#     cbar_ax = fig.add_axes([0.95, 0.15, 0.015, 0.7])
#     fig.colorbar(im, cax=cbar_ax)

#     layout_str = "horiz" if TRANSPOSE_LAYOUT else "vert"
#     plt.savefig(f"{save_dir}/field_{field_name}_{MODE}_{layout_str}.png",
#                 dpi=200, bbox_inches='tight')
#     plt.close(fig)

# print(f"\nFinished! All figures saved to: {save_dir}/")