# run_experiments.py
import os

commands = [
    "python train.py --config ./config/bgm/bgm_afdb_1.yaml",
    "python test.py --config ./config/bgm/bgm_afdb_1.yaml",
]

for cmd in commands:
    print(f"Running: {cmd}")

    os.system(cmd)
