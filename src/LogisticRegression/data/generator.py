import numpy as np
import pandas as pd 

from dataclasses import dataclass
from pathlib import Path

@dataclass
class LogisticDataConfig:
    weight: float
    bias: float
    number_of_points: int
    x_min: int
    x_max: int
    noise_STD: float
    random_seed: int 
    datafile: str


def generate_classification_data(config: LogisticDataConfig) -> pd.DataFrame:
    np.random.seed(config.random_seed)

    # Generation
    X = np.random.uniform(config.x_min, config.x_max, config.number_of_points)
    noise = np.random.uniform(0, config.noise_STD, config.number_of_points)
    z = config.weight * X + config.bias + noise

    # prob and labels
    prob = 1 / (1 + np.exp(-z))
    y = (prob >= 0.5).astype(int)

    # pack
    data = pd.DataFrame({
        "X" : X, "y" : y
    })
    output_path = Path(config.datafile)
    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    data.to_csv(output_path, index=False)
    return data
