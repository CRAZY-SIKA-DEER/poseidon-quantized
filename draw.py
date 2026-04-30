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

# ============ 配置 ============
ds_name = "fluids.incompressible.PiecewiseConstants"
sample_idx = 0

# 推理模式: "rollout" 或 "oneshot"
MODE = "oneshot"

# 控制画图的横竖排版 (False: 竖排, True: 横排)
TRANSPOSE_LAYOUT = True

# AR 参数
AR_STEPS = 3
INITIAL_TIME = 0
FINAL_TIME = 15
DATA_PATH = "poseidon_main/camlab-ethz/down_streams"

assert (FINAL_TIME - INITIAL_TIME) % AR_STEPS == 0, \
    f"(FINAL_TIME - INITIAL_TIME)={FINAL_TIME - INITIAL_TIME} 必须能被 AR_STEPS={AR_STEPS} 整除"

delta_t = (FINAL_TIME - INITIAL_TIME) // AR_STEPS

models = {
    "Origin":  f"checkpoints/Poseidon-L/no_phyloss/{ds_name}/origin/none",
    "ASVD":    f"compression/compressed_ft_models/{ds_name}/0.2/ASVD/traj64",
    "FWSVD":   f"compression/compressed_ft_models/{ds_name}/0.2/FWSVD/l1_order0_br10.0_traj64",
    "SVDLLM":  f"compression/compressed_ft_models/{ds_name}/0.2/SVDLLM/traj64",
    "Dobi-SVD": f"compression/compressed_ft_models/{ds_name}/0.2/DobiSVD/traj64",
    "SAES-SVD":f"compression/compressed_ft_models/{ds_name}/0.2/SAES_SVD/traj64",
    "Ours":   f"compression/compressed_ft_models/{ds_name}/0.2/MySVD/l1_order2_alpha0.7",
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
    与 fisher_collector.py 中的逻辑一致。
    返回: dict {field_name: channel_index}
    """
    if "incompressible" in dataset_name:
        if num_channels == 3:
            # [u, v, p]
            return {"u": 0, "v": 1, "p": 2}
        else:
            # [rho, u, v, p]
            return {"u": 1, "v": 2, "p": 3}
    elif "compressible" in dataset_name:
        # [rho, u, v, p]
        return {"rho": 0, "u": 1, "v": 2, "p": 3}
    elif "wave" in dataset_name.lower() or "Wave" in dataset_name:
        # wave: 所有通道
        return {"u": 0}
    else:
        raise ValueError(f"Unknown dataset type: {dataset_name}")


def denormalize(data, mean, std):
    """
    反白化: data * std + mean
    data:  [..., C, H, W]
    mean/std: [C, 1, 1]
    """
    return data * std + mean


# ============ 目标可视化时刻 ============
target_times = [INITIAL_TIME + (s + 1) * delta_t for s in range(AR_STEPS)]  # [5, 10, 15]

save_dir = f"results/finetuned_results/{ds_name}/draw_exp"
os.makedirs(save_dir, exist_ok=True)

# ============ GT 轨迹 (两种模式共用，完全相同) ============
labels_all_steps = get_trajectories(
    dataset=ds_name, data_path=DATA_PATH,
    ar_steps=AR_STEPS, initial_time=INITIAL_TIME, final_time=FINAL_TIME,
    dataset_kwargs={},
)
if hasattr(labels_all_steps, "cpu"):
    labels_all_steps = labels_all_steps.cpu().numpy()
gt_raw = labels_all_steps  # [N, AR_STEPS, C, H, W]



# =====================================================================
# Rollout 模式: 0→5→10→15 自回归链式
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
    mean = np.array(constants['mean']).reshape(-1, 1, 1)  # [C, 1, 1]
    std  = np.array(constants['std']).reshape(-1, 1, 1)   # [C, 1, 1]

    all_preds = {}
    for name, path in models.items():
        trainer = get_trainer(model_path=path, batch_size=1024, dataset=test_ds, output_all_steps=True)
        preds, _, _ = rollout(trainer, test_ds, model_path="Poseidon-L",
                              ar_steps=AR_STEPS, output_all_steps=True)
        if hasattr(preds, "cpu"):
            preds = preds.cpu().numpy()
        # 反白化预测
        all_preds[name] = denormalize(preds, mean, std)  # [N, AR_STEPS, C, H, W]

# =====================================================================
# One-shot 模式: 始终从 t=0 出发, 改变 time_step_size 分别预测到 t=5,10,15
# =====================================================================
elif MODE == "oneshot":
    all_preds = {name: [] for name in models}

    for step_i in range(AR_STEPS):
        step_size = (step_i + 1) * delta_t  # 5, 10, 15
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
        mean = np.array(constants['mean']).reshape(-1, 1, 1)  # [C, 1, 1]
        std  = np.array(constants['std']).reshape(-1, 1, 1)   # [C, 1, 1]

        for name, path in models.items():
            trainer = get_trainer(model_path=path, batch_size=1024, dataset=ds_step)
            preds, _, _ = rollout(trainer, ds_step, model_path="Poseidon-L")
            if hasattr(preds, "cpu"):
                preds = preds.cpu().numpy()
            # preds: [N, C, H, W] — 单步预测
            all_preds[name].append(preds)

    # 拼接成 [N, AR_STEPS, C, H, W] 并反白化
    for name in models:
        stacked = np.stack(all_preds[name], axis=1)
        all_preds[name] = denormalize(stacked, mean, std)

else:
    raise ValueError(f"Unknown MODE: {MODE}, must be 'rollout' or 'oneshot'")

# ============ 根据数据集类型确定通道配置 ============
# 反白化 GT
gt = denormalize(gt_raw, mean, std)
num_channels = gt.shape[2]  # C 维度
channels = get_channel_config(ds_name, num_channels)
print(f"Dataset: {ds_name}, num_channels: {num_channels}, visualizing: {channels}")

# ============ 绘图准备 ============
other_models = [m for m in models.keys() if m != "Origin"]
methods_list = other_models + ["Origin", "GT"]

num_methods = len(methods_list)
num_steps = AR_STEPS

# =====================================================================
# 绘图循环
# =====================================================================
for field_name, ch in channels.items():
    print(f"Plotting {field_name} (Mode: {MODE}, Layout Transpose: {TRANSPOSE_LAYOUT})")

    n_rows = num_steps if TRANSPOSE_LAYOUT else num_methods
    n_cols = num_methods if TRANSPOSE_LAYOUT else num_steps

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.2 * n_rows))

    if n_rows == 1 and n_cols == 1: axes = np.array([[axes]])
    elif n_rows == 1: axes = axes[np.newaxis, :]
    elif n_cols == 1: axes = axes[:, np.newaxis]

    # 计算全局统一的 vmin 和 vmax (基于反白化后的 GT)
    global_vmin = np.percentile(gt[sample_idx, :, ch], 1)
    global_vmax = np.percentile(gt[sample_idx, :, ch], 99)

    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]

            if TRANSPOSE_LAYOUT:
                step_idx, method_idx = r, c
            else:
                method_idx, step_idx = r, c

            row_name = methods_list[method_idx]

            data_field = (gt[sample_idx, step_idx, ch]
                          if row_name == "GT"
                          else all_preds[row_name][sample_idx, step_idx, ch])

            im = ax.imshow(
                data_field.T, origin='lower', cmap='RdBu_r',
                vmin=global_vmin, vmax=global_vmax,
            )

            t_curr = target_times[step_idx]
            is_ours = (row_name == "Ours")
            fw = 'bold' if is_ours else 'normal'

            if TRANSPOSE_LAYOUT:
                if r == 0:
                    ax.set_title(row_name, fontsize=14, fontweight=fw)
                if c == 0:
                    ax.set_ylabel(f"t = {t_curr}", fontsize=13)
            else:
                if r == 0:
                    ax.set_title(f"t = {t_curr}", fontsize=13)
                if c == 0:
                    ax.set_ylabel(row_name, fontsize=14, fontweight=fw)

            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"Field: {field_name} ({MODE})", fontsize=15, y=0.995)

    plt.tight_layout(rect=[0, 0, 0.94, 0.985])

    cbar_ax = fig.add_axes([0.95, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    layout_str = "horiz" if TRANSPOSE_LAYOUT else "vert"
    plt.savefig(f"{save_dir}/field_{field_name}_{MODE}_{layout_str}.png",
                dpi=200, bbox_inches='tight')
    plt.close(fig)

print(f"\nFinished! All figures saved to: {save_dir}/")