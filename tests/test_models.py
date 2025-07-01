import numpy as np
from scripts.models import train_and_evaluate


def test_train_and_evaluate_runs():
    X = np.random.rand(40, 25)
    y = np.array([0]*20 + [1]*20)
    train_and_evaluate(X, y, model_name="test_run")
