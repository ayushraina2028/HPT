import argparse

import yaml
import matplotlib.pyplot as plt 

import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1] / "src")
)

from LinearRegression.data.generator import (
    LinearDataConfig,
    generate_linear_data
)

print("import ok")

def main():

    # Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data_generation.yaml"
    )

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Config Parameters written in data_generation.yaml
    data_config = LinearDataConfig(
        slope=config["linear_regression"]["slope"],
        intercept=config["linear_regression"]["intercept"],
        number_of_points=config["linear_regression"]["number_of_points"],
        x_min=config["linear_regression"]["x_min"],
        x_max=config["linear_regression"]["x_max"],
        noise_STD=config["linear_regression"]["noise_STD"],
        random_seed=config["linear_regression"]["random_seed"],
        datafile=config["output_paths_Linear_Regression"]["data_file"],
    )

    # Data Generation
    data = generate_linear_data(data_config)

    # Plot
    plt.figure(figsize=(8,5))
    plt.scatter(data['X'], data['y'])

    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Linear Regression")
    plt.tight_layout()

    plt.savefig(
        config['output_paths_Linear_Regression']["plot_file"],
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    print(f"Data saved to: {config['output_paths_Linear_Regression']['data_file']}")
    print(f"Plot saved to: {config['output_paths_Linear_Regression']['plot_file']}")

if __name__ == "__main__":
    main()

