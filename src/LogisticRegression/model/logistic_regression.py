import numpy as np
import json

from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class LogisticRegressionConfig:
    learning_rate: float
    num_epochs: int
    random_seed: int

class LogisticRegressionModel:

    def __init__(self, config: Optional[LogisticRegressionConfig] = None):
        self.config = config
        self.weight: float = 0.0
        self.bias: float = 0.0
        self.loss_history: List[float] = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        np.random.seed(self.config.random_seed)
        n = X.shape[0]
        eps = 1e-9

        for _ in range(self.config.num_epochs):
            # Predict
            z = self.weight * X + self.bias
            y_pred = self._sigmoid(z)

            # Loss (Binary Cross-Entropy)
            loss = -np.mean(
                y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps)
            )
            self.loss_history.append(loss)

            # Error
            error = y_pred - y

            # Gradient
            dweight = (1/n) * np.sum(error * X)
            db = (1/n) * np.sum(error)

            # Update
            self.weight -= self.config.learning_rate * dweight
            self.bias -= self.config.learning_rate * db

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._sigmoid(self.weight * X + self.bias)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path: str) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump({
                "weight" : self.weight, "bias" : self.bias
            },
            f,
            indent=2)

    def load(self, path: str) -> None:
        with open(path, 'r') as f:
            params = json.load(f)

        self.weight = params['weight']
        self.bias = params['bias']
