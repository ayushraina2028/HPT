import sys
from pathlib import Path

# Append path
sys.path.append(
    str(Path(__file__).resolve().parents[1] / "src")
)

from LogisticRegression.model.logistic_regression import (
    LogisticRegressionConfig,
    LogisticRegressionModel
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
    data = pd.read_csv(config["logistic_regression"]["datafile"])
    X = data['X'].to_numpy()
    y = data['y'].to_numpy()

    # Model Config
    model_config = LogisticRegressionConfig(
        learning_rate=config['logistic_regression']['learning_rate'],
        num_epochs=config['logistic_regression']['epochs'],
        random_seed=config['logistic_regression']['random_seed']
    )

    # Model
    model = LogisticRegressionModel(model_config)
    model.fit(X, y)

    predictions = model.predict(X)
    accuracy = (predictions == y).mean()

    print(f"Learned Weight: {model.weight:.4f}")
    print(f"Learned Bias: {model.bias:.4f}")
    print(f"Training Accuracy: {accuracy:.4f}")

    # Save Model
    model.save(config["logistic_regression"]["model_file"])
    print(f"Model Saved to: {config['logistic_regression']['model_file']}")

    # Loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(model.loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.title("Logistic Regression Training Loss")
    plt.tight_layout()
    plt.savefig(
        config["logistic_regression"]["loss_plot_file"],
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # Fit plot (data + learned sigmoid curve)
    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, c=y, cmap='bwr', alpha=0.5, label="Data")

    x_line = np.linspace(X.min(), X.max(), 200)
    y_line = model.predict_proba(x_line)
    plt.plot(x_line, y_line, color="black", linewidth=2, label="Sigmoid Fit")

    plt.xlabel("X")
    plt.ylabel("Probability / Label")
    plt.title("Logistic Regression Fit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        config["logistic_regression"]["fit_plot_file"],
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

if __name__ == "__main__":
    main()
