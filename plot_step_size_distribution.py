import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------
# Config: which files to plot (label -> path)
# Comment out anything you don't want.
# ------------------------------------------------------------------
DYNAMIC_FILES = {
    #"dyn-4": os.path.join(PROJECT_ROOT, "dynamic_stats", "NS-PwC-T-dynamic-stepsizes-4.json"),
    "dyn-8": os.path.join(PROJECT_ROOT, "dynamic_stats", "NS-PwC-T-dynamic-stepsizes-8.json"),
    #"dyn-16": os.path.join(PROJECT_ROOT, "dynamic_stats", "NS-PwC-T-dynamic-stepsizes-16.json"),
}

PPQ_FILES = {
    #"PPQ-0":   os.path.join(PROJECT_ROOT, "ppq_artifacts", "NS-PwC-T", "ppq_step_sizes-0.json"),
    #"PPQ-1.9": os.path.join(PROJECT_ROOT, "ppq_artifacts", "NS-PwC-T", "ppq_step_sizes-1-9.json"),
    "PPQ-conv2d": os.path.join(PROJECT_ROOT, "ppq_artifacts", "NS-PwC-T", "ppq_step_sizes-1-5-conv.json"),
    "PPQ-conv2d-sobolev": os.path.join(PROJECT_ROOT, "ppq_artifacts", "NS-PwC-T", "sobolev", "ppq_step_sizes-1-5-conv-sobolev.json")


}


def load_dynamic_steps(json_path: str) -> np.ndarray:
    """JSON: { "step_sizes": { layer_name: [S_out] } }"""
    with open(json_path, "r") as f:
        data = json.load(f)
    all_steps = []
    for name, steps in data["step_sizes"].items():
        all_steps.extend(steps)
    return np.array(all_steps, dtype=np.float64)


def load_ppq_weight_steps(json_path: str) -> np.ndarray:
    """JSON: { "step_sizes": { layer_name: [[w_steps...], [a_steps...]] } }"""
    with open(json_path, "r") as f:
        data = json.load(f)
    all_steps = []
    for name, pair in data["step_sizes"].items():
        w_list = pair[0]  # first list = weight step sizes
        all_steps.extend(w_list)
    return np.array(all_steps, dtype=np.float64)


def main():
    series = {}  # label -> np.array of log10(step)

    # Dynamic baselines
    for label, path in DYNAMIC_FILES.items():
        if not os.path.exists(path):
            print(f"[WARN] dynamic file not found, skipping: {path}")
            continue
        steps = load_dynamic_steps(path)
        steps = steps[steps > 0]  # avoid log10 problems
        series[label] = np.log10(steps)
        print(f"[INFO] {label}: {steps.size} steps")

    # PPQ runs
    for label, path in PPQ_FILES.items():
        if not os.path.exists(path):
            print(f"[WARN] PPQ file not found, skipping: {path}")
            continue
        steps = load_ppq_weight_steps(path)
        steps = steps[steps > 0]
        series[label] = np.log10(steps)
        print(f"[INFO] {label}: {steps.size} steps")

    if not series:
        print("No data loaded, nothing to plot.")
        return

    # ----------------------------------------------------
    # Plot all distributions on log10(step size) axis
    # ----------------------------------------------------
    plt.figure(figsize=(7, 5))

    # find auto-limits to crop out empty space
    min_log = min(arr.min() for arr in series.values())
    max_log = max(arr.max() for arr in series.values())

    # KDE curves
    for label, log_steps in series.items():
        sns.kdeplot(log_steps, label=label, bw_adjust=0.3)

    plt.xlabel("log10(step size)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.2)

    # crop X range to visible area
    plt.xlim(min_log - 0.3, max_log + 0.3)

    plt.title("Distribution of per-channel weight step sizes")
    plt.tight_layout()

    out_path = os.path.join(PROJECT_ROOT, "step_size_kde_conv2d_sobolev.png")
    plt.savefig(out_path, dpi=250)
    print(f"[INFO] Saved plot to {out_path}")



if __name__ == "__main__":
    main()
