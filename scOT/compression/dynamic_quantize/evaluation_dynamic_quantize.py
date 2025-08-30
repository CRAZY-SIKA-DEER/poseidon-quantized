import torch
import numpy as np
import random
from transformers import AutoConfig
from scOT.model import ScOT
from scOT.trainer import TrainingArguments, Trainer
from scOT.problems.base import get_dataset, BaseTimeDataset
from scOT.metrics import relative_lp_error, lp_error

# Set random seed
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Paths and parameters
MODEL_NAME = "camlab-ethz/Poseidon-T"  # HuggingFace model name
DATA_PATH = "./datasets/NS-Sines"              # Your local datasets path
DATASET_NAME = "fluids.incompressible.Sines"  # Dataset identifier
BATCH_SIZE = 16                        # Batch size for evaluation

def load_model_from_hf(model_name):
    print(f"[INFO] Loading model {model_name} from Hugging Face...")
    model = ScOT.from_pretrained(model_name)
    model.eval()
    return model

def get_test_dataset(dataset_name, data_path):
    dataset = get_dataset(
        dataset=dataset_name,
        which="test",
        num_trajectories=1,
        data_path=data_path,
        move_to_local_scratch=None
    )
    return dataset

def compute_eval_metrics(predictions, labels):
    rel_error = relative_lp_error(predictions, labels, p=1, return_percent=True)
    l1_error = lp_error(predictions, labels, p=1)
    mean_rel_error = np.mean(rel_error)
    mean_l1_error = np.mean(l1_error)
    return mean_rel_error, mean_l1_error

def main():
    model = load_model_from_hf(MODEL_NAME)
    dataset = get_test_dataset(DATASET_NAME, DATA_PATH)

    args = TrainingArguments(
        output_dir=".",
        per_device_eval_batch_size=BATCH_SIZE,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=args,
    )

    print("[INFO] Running inference...")
    predictions = trainer.predict(dataset)
    
    preds = predictions.predictions
    labels = predictions.label_ids

    mean_rel_error, mean_l1_error = compute_eval_metrics(preds, labels)

    print("\n=== Evaluation Results ===")
    print(f"Mean Relative L1 Error (%): {mean_rel_error:.4f}")
    print(f"Mean L1 Error: {mean_l1_error:.4f}")

if __name__ == "__main__":
    main()