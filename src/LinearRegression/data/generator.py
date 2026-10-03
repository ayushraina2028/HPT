import numpy as np 
import pandas as pd 

from dataclasses import dataclass 
from pathlib import Path 

@dataclass
class LinearDataConfig:
    slope: float
    intercept: float
    number_of_points: int 
    x_min: int
    x_max: int
    noise_STD: float
    random_seed: int 
    datafile: str 

def generate_linear_data(config: LinearDataConfig) -> pd.DataFrame:    
    np.random.seed(config.random_seed)

    # Generation
    X = np.random.uniform(config.x_min, config.x_max, config.number_of_points)
    noise = np.random.uniform(0, config.noise_STD, config.number_of_points)
    y = config.slope * X + config.intercept + noise

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
