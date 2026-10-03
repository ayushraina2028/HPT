import sys
from pathlib import Path

# Append path
sys.path.append(
    str(Path(__file__).resolve().parents[1] / "src")
)

from LinearRegression.model.linear_regression import (
    LinearRegressionConfig,
    LinearRegressionModel
)

import yaml 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default='configs/training.yaml'
    )

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Read Dataset
    data = pd.read_csv(config["linear_regression"]["datafile"])
    X = data['X'].to_numpy()
    y = data['y'].to_numpy()

    # Model Config
    model_config = LinearRegressionConfig(
        learning_rate=config['linear_regression']['learning_rate'],
        num_epochs=config['linear_regression']['epochs'],
        random_seed=config['linear_regression']['random_seed']
    )

    # Model
    model = LinearRegressionModel(model_config)
    model.fit(X,y)

    print(f"Learned Weight: {model.weight:.4f}")
    print(f"Learned Bias: {model.bias:.4f}")

    # Save Model
    model.save(config["linear_regression"]["model_file"])
    print(f"Model Saved to: {config['linear_regression']['model_file']}")

    # plot
    plt.figure(figsize=(8, 5))
    plt.plot(model.loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Linear Regression Training Loss")
    plt.tight_layout()
    plt.savefig(
        config["linear_regression"]["loss_plot_file"],
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # Fit-line plot
    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, alpha=0.5, label="Data")

    x_line = np.linspace(X.min(), X.max(), 100)
    y_line = model.predict(x_line)
    plt.plot(x_line, y_line, color="red", linewidth=2, label="Fitted Line")

    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Linear Regression Fit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        config["linear_regression"]["fit_plot_file"],
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

if __name__ == "__main__":
    main()