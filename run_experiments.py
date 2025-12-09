# run_experiments.py
import os

commands = [
    "python test.py --config ./config/bgm/bgm_afdb_1.yaml",
    "python test.py --config ./config/bgm/bgm_afdb_2.yaml",
    "python test.py --config ./config/bgm/bgm_afdb_3.yaml",
    "python test.py --config ./config/bgm/bgm_afdb_4.yaml",
    "python test.py --config ./config/bgm/bgm_afdb_5.yaml",

]

for cmd in commands:
    print(f"Running: {cmd}")
    os.system(cmd)