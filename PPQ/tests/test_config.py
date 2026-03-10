# tests/test_config.py
from PPQ.config import PPQConfig

def test_config_paths():
    cfg = PPQConfig()
    assert cfg.quant_layer_path.name == cfg.quant_layer_file
    assert cfg.lr_dir.name == "lr"