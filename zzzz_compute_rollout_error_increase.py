from pathlib import Path
import pandas as pd

CSV_PATH = Path(
    "/home/u6ey/yiheng.u6ey/poseidon-quantized/"
    "zzzz_plot/NS-BB-L/NS-BB/rollout_correct/NS-BB_rollout_errors.csv"
)

OUT_PATH = CSV_PATH.with_name("NS-BB_rollout_error_increase_over_fp.csv")


def main():
    df = pd.read_csv(CSV_PATH)

    fp = df[df["method"] == "FP"][["step", "l1", "rel_l1_percent"]].rename(
        columns={
            "l1": "fp_l1",
            "rel_l1_percent": "fp_rel_l1_percent",
        }
    )

    out = df.merge(fp, on="step", how="left")

    out["l1_increase"] = out["l1"] - out["fp_l1"]
    out["rel_l1_percent_increase"] = (
        out["rel_l1_percent"] - out["fp_rel_l1_percent"]
    )

    out["l1_increase_percent_over_fp"] = (
        out["l1_increase"] / (out["fp_l1"] + 1e-12) * 100.0
    )

    out["rel_l1_increase_percent_over_fp"] = (
        out["rel_l1_percent_increase"]
        / (out["fp_rel_l1_percent"] + 1e-12)
        * 100.0
    )

    out = out[out["method"] != "FP"]

    out.to_csv(OUT_PATH, index=False)

    print(out.to_string(index=False))
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()