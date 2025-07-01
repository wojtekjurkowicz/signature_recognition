import numpy as np
import os
from scripts.models import train_and_evaluate


def test_train_and_evaluate_runs(tmp_path):
    os.makedirs(tmp_path / "models", exist_ok=True)
    os.makedirs(tmp_path / "outputs", exist_ok=True)

    X = np.random.rand(40, 25)
    y = np.array([0]*20 + [1]*20)

    old_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        train_and_evaluate(X, y, model_name="test_run")
    finally:
        os.chdir(old_cwd)
