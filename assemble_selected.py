import subprocess
from pathlib import Path

DATA_ROOT = Path("/lus/lfs1aip2/projects/u6ey/yiheng.u6ey/poseidon/dataset")

datasets = [
    "CE-RM",
    "CE-RPUI",
    "Wave-Gauss",
    "Wave-Layer",
    "NS-BB",
]

for name in datasets:
    input_dir = DATA_ROOT / name
    assemble_script = input_dir / "assemble_data.py"
    output_file = input_dir / f"{name}.nc"

    if not input_dir.exists():
        print(f"[Skipped] Missing input dir: {input_dir}")
        continue

    if not assemble_script.exists():
        print(f"[Skipped] Missing assemble script: {assemble_script}")
        continue

    cmd = [
        "python",
        str(assemble_script),
        "--input_dir", str(input_dir),
        "--output_file", str(output_file),
    ]

    print(f"[Assembling] {name}")
    subprocess.run(cmd, check=True)
    print(f"[Completed] {output_file}")