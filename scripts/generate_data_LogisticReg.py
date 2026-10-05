import argparse

import yaml 
import matplotlib.pyplot as plt

import sys 
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1] / "src")
)

from LogisticRegression.data.generator import (
    LogisticDataConfig,
    generate_classification_data
)
print("Requirements Imported Successfully")

def main():

    #Argument Parsing
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
    data_config = LogisticDataConfig(
        weight=config["logistic_regression"]["weight"],
        bias=config["logistic_regression"]["bias"],
        number_of_points=config["logistic_regression"]["number_of_points"],
        x_min=config["logistic_regression"]["x_min"],
        x_max=config["logistic_regression"]["x_max"],
        noise_STD=config["logistic_regression"]["noise_STD"],
        random_seed=config["logistic_regression"]["random_seed"],
        datafile=config["output_paths_Logistic_Regression"]["data_file"],
    )

    # Data Generation
    data = generate_classification_data(data_config)

    # Plot
    plt.figure(figsize=(8,5))
    plt.scatter(data['X'], data['y'], c=data['y'], cmap='bwr', alpha=0.6)

    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Logistic Regression")
    plt.tight_layout()

    # Create path if not there
    output_path = Path(config['output_paths_Logistic_Regression']["plot_file"])
    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    plt.savefig(
        config['output_paths_Logistic_Regression']["plot_file"],
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    print(f"Data saved to: {config['output_paths_Logistic_Regression']['data_file']}")
    print(f"Plot saved to: {config['output_paths_Logistic_Regression']['plot_file']}")

if __name__ == "__main__":
    main()