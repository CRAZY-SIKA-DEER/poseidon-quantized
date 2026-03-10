from PPQ.config import PPQConfig
from PPQ.poseidon_utils import load_poseidon_model, build_poseidon_loaders


def main():
    cfg = PPQConfig()

    model, device = load_poseidon_model(cfg.model_path, cfg.device)
    _, _, calib_iter, val_iter = build_poseidon_loaders(
        dataset_name=cfg.dataset_name,
        data_path=cfg.data_path,
        calib_batchsize=cfg.calib_batchsize,
        calib_steps=cfg.calib_steps,
        val_batchsize=cfg.val_batchsize,
        val_steps=cfg.val_steps,
    )


if __name__ == "__main__":
    main()