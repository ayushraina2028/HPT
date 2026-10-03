import numpy as np 

import json 
from pathlib import Path

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class LinearRegressionConfig:
    learning_rate: float
    num_epochs: int 
    random_seed: int 

class LinearRegressionModel:

    def __init__(self, config: Optional[LinearRegressionConfig] = None):
        self.config = config
        self.weight:float = 0.0
        self.bias:float  = 0.0
        self.loss_history: List[float] = []

    def fit(self, X:np.ndarray, y:np.ndarray) -> None:
        np.random.seed(self.config.random_seed)
        n = X.shape[0]

        for _ in range(self.config.num_epochs):
            # Predict
            y_pred = self.weight * X + self.bias

            # Error
            error = y_pred - y
            loss = np.mean(error ** 2)

            # Record
            self.loss_history.append(loss)

            # Gradient
            dweight = (1/n) * np.sum(error * X)
            db = (1/n) * (np.sum(error))

            # Update
            self.weight -= self.config.learning_rate * dweight
            self.bias -= self.config.learning_rate * db

    def predict(self, X:np.ndarray) -> np.ndarray:
        return self.weight * X + self.bias

    def save(self, path: str) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump({
                "weight" : self.weight, "bias" : self.bias
            },
            f,
            indent=2)

    def load(self, path:str) -> None:
        with open(path, 'r') as f:
            params = json.load(f)

        self.weight = params['weight']
        self.bias = params['bias']

    
