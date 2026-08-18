# Extending PPQ, BRECQ, and SBPQ to New Physical Models

This guide is for adding another model family, such as SUROER or other physical foundation models, without mixing model-specific code into the Poseidon implementation.

## Recommended Repository Layout

Keep the shared SBPQ mathematics in `SBPQ/` and put each model family behind its own adapter folder:

```text
SBPQ/
├── beta_prior.py
├── blocks.py
├── likelihood.py
├── noise.py
├── ranges.py
├── sensitivity.py
├── sobolev.py
├── step_sizes.py
├── trainer.py
├── poseidon/
└── suroer/
```

For a new model, create:

```text
SBPQ/<model_name>/
├── __init__.py
├── config.py
├── blocks.py
├── model_utils.py
├── data_utils.py
├── ranges.py
├── sensitivity.py
├── likelihood.py
├── evaluation.py
├── run_sbpq_<model_name>.py
└── run_fixed_baselines_<model_name>.py
```

The rule is:

```text
General math              -> SBPQ/*.py
Model architecture logic  -> SBPQ/<model_name>/blocks.py
Model input/output logic  -> SBPQ/<model_name>/model_utils.py
Dataset and normalization -> SBPQ/<model_name>/data_utils.py
Metrics                   -> SBPQ/<model_name>/evaluation.py
```

Do not import Poseidon-specific files from `SBPQ/poseidon/` into a new model folder. Use them as references only.

## What Must Be Implemented For Each New Model

Every new model adapter should provide these capabilities.

1. Load the pretrained model.

```python
model, device = load_model(model_path, device)
```

2. Build calibration and validation loaders.

```python
calib_loader, val_loader = build_loaders(...)
```

3. Move a batch to device and expose model inputs/labels.

```python
inputs, labels = prepare_batch(batch, device)
```

4. Run the model and extract prediction tensors.

```python
prediction = model_forward(model, inputs)
```

5. Define quantizable layers.

```python
layer_names = collect_quantizable_layers(model)
```

6. Define structural blocks and layer-to-block mapping.

```python
block_mapping = {
    "encoder_block_0": [...],
    "encoder_block_1": [...],
}
layer_to_block = {...}
```

7. Compute or load ranges.

```python
weight_ranges = compute_weight_ranges(model, layer_names)
activation_ranges = compute_activation_ranges(...)
```

8. Compute sensitivity.

```python
sensitivity = compute_sobolev_aware_sensitivity(...)
```

9. Evaluate FP, fixed baselines, BRECQ, PPQ, and SBPQ with the same metric code.

## Applying The Three Quantization Methods

### Fixed Baselines

Start with fixed 8-bit and fixed 4-bit. These are the sanity checks.

For every quantizable weight tensor, compute per-output-channel step sizes:

```text
step = max_abs(weight_channel) / (2^(bits - 1) - 1)
```

Then evaluate with fake quantization hooks. Fixed 8-bit should usually be close to FP. If it is much better than FP or much worse than expected, check normalization, output extraction, and metric code first.

### PPQ

For PPQ, port only the necessary old probabilistic quantization logic. Keep the implementation model-specific under a PPQ adapter or script if the current PPQ code is tightly coupled to Poseidon.

Minimum requirement:

```text
model loader
data loader
candidate layer names
step-size initialization
likelihood/evaluation
saved step-size format
```

Use the same validation metrics as SBPQ. Otherwise the comparison is not meaningful.

### BRECQ

BRECQ requires a quantized model wrapper and block reconstruction.

For a new model, check:

```text
Can BRECQ wrap the model modules?
Are Linear/Conv layers replaced by QuantModule?
What is a natural reconstruction block?
Does forward(batch) still match the original model API?
Are first/last layers handled consistently?
```

Artifacts should follow:

```text
brecq_artifacts/<ModelTag>/recon/w8/iters10000/adaround_state.pt
brecq_artifacts/<ModelTag>/recon/w8/iters10000/meta.json
brecq_artifacts/<ModelTag>/recon/w4/iters10000/adaround_state.pt
brecq_artifacts/<ModelTag>/recon/w4/iters10000/meta.json
```

The `meta.json` must include:

```json
{
  "model_path": "...",
  "dataset_name": "...",
  "data_path": "...",
  "n_bits_w": 8,
  "iters_w": 10000,
  "num_quant_modules": 0,
  "metrics": {}
}
```

During evaluation, always print:

```text
Loaded BRECQ AdaRound state: loaded=<N>, missing=<M>
```

`missing` should normally be zero.

### SBPQ

SBPQ should reuse the shared method components:

```text
SBPQ/beta_prior.py
SBPQ/noise.py
SBPQ/step_sizes.py
SBPQ/trainer.py
```

The new model folder should only provide the model-specific pieces:

```text
block slicing
batch preparation
model forward
range collection hooks
sensitivity hooks
evaluation metrics
save paths
```

Saved results should use:

```text
SBPQ/artifacts/<model_name>/<ModelTag>/
├── sensitivity/
├── beta_parameters/
└── runs/
    └── network_global_group<run_group>_.../
        ├── metrics.json
        ├── sbpq_trainer_state.pt
        ├── config.json
        └── training_history.json
```

## Metrics To Implement First

Use one common metric file per model family.

For incompressible NS-style models:

```text
L1
RelL1
Sobolev order 2
Divergence
Vorticity error
Average bitwidth
```

For compressible Euler:

```text
L1
RelL1
Sobolev order 1 or 2
Conservation-related metrics if available
Average bitwidth
```

For wave models:

```text
L1
RelL1
Sobolev order 1
Optional wave-speed/channel-specific error
Average bitwidth
```

Always compute physical metrics on denormalized fields. Do not compute divergence or vorticity directly on normalized tensors unless the metric is explicitly defined that way.

## Branch And Conversation Workflow

Use separate git branches or worktrees for independent tasks.

Recommended branches:

```text
main
├── sbpq-ablation
└── sbpq-new-models
```

Use `sbpq-ablation` for:

```text
Gaussian prior ablation
symmetric Beta ablation
no Sobolev sensitivity
unweighted centering
random sensitivity
tables and plots
```

Use `sbpq-new-models` for:

```text
new model adapter folders
new model loaders
new block definitions
new metrics
PPQ/BRECQ/SBPQ porting
new sbatch scripts
```

If two GPT conversations work in parallel, do not let them edit the same files. A clean split is:

```text
Conversation A: ablation
  edits SBPQ/beta_prior.py, SBPQ/trainer.py, SBPQ/poseidon/run scripts

Conversation B: new models
  creates SBPQ/suroer/, new eval scripts, new sbatch scripts
```

If both conversations must touch shared files, merge one branch first, then rebase or manually copy the relevant changes.

## Safer Option: Git Worktrees

If you want two working folders on the same repo:

```bash
git worktree add ../poseidon-quantized-ablation -b sbpq-ablation
git worktree add ../poseidon-quantized-new-models -b sbpq-new-models
```

Then run one GPT conversation in each folder:

```text
/home/.../poseidon-quantized-ablation
/home/.../poseidon-quantized-new-models
```

This avoids file conflicts while jobs are running.

## Suggested Porting Order For A New Model

1. Create `SBPQ/<model_name>/`.
2. Implement model loading and one validation forward pass.
3. Implement dataset loading and denormalization.
4. Implement fixed 8-bit fake quantization.
5. Verify FP vs fixed 8-bit metrics.
6. Define structural blocks.
7. Compute ranges.
8. Compute Sobolev-aware sensitivity.
9. Build Beta parameters.
10. Train SBPQ step sizes for a tiny smoke run.
11. Run a short sweep on one dataset.
12. Add BRECQ wrapper and verify loaded/missing module counts.
13. Add PPQ baseline if needed.
14. Write sbatch sweep scripts only after the smoke tests pass.

## Minimum Smoke Checks

Before launching sweeps:

```text
FP evaluation runs
fixed 8-bit evaluation runs
range collection saves outputs
sensitivity collection saves outputs
SBPQ trains for 2-5 steps
step sizes change after optimization
SBPQ evaluation saves metrics.json
all metrics use denormalized physical fields where needed
```

## Naming Rules

Keep every run folder self-describing:

```text
network_global_group<group>_B8_d2_k150_mc10_lr3p75em05_init8_sob2_cal512_val2_steps800
```

Use `SBPQ_RUN_GROUP` or equivalent environment variables for sweeps:

```text
SBPQ_RUN_GROUP=suroer_first_sweep
SBPQ_RUN_GROUP=suroer_ablation_beta
SBPQ_RUN_GROUP=suroer_all_datasets
```

For Slurm logs:

```text
logs/sbpq_<model>_<dataset>_<group>_<jobid>.out
logs/sbpq_<model>_<dataset>_<group>_<jobid>.err
```

The result folder should be enough to identify the method, dataset, target bits, beta kappa, learning rate, Sobolev order, calibration size, and optimization steps without opening the JSON.

## What To Tell A New GPT Conversation

Start the new conversation with:

```text
Read SBPQ/EXTENDING_TO_NEW_PHYSICAL_MODELS.md first.
Do not modify Poseidon code.
Create a new adapter folder under SBPQ/<model_name>/.
Use SBPQ/poseidon only as reference.
First implement FP and fixed 8-bit evaluation.
Do not submit long sweeps until smoke tests pass.
Keep artifacts under SBPQ/artifacts/<model_name>/.
```

