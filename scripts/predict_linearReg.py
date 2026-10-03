import argparse
import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1] / "src")
)

from LinearRegression.model.linear_regression import LinearRegressionModel

import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="models/LinearReg/linear_model.json"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="CSV file with an 'X' column"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Where to save predictions (default: <input>_predictions.csv)"
    )

    args = parser.parse_args()

    model = LinearRegressionModel()
    model.load(args.model)

    data = pd.read_csv(args.input)
    data["y_pred"] = model.predict(data["X"].to_numpy())

    output_path = args.output or str(
        Path(args.input).with_name(Path(args.input).stem + "_predictions.csv")
    )
    data.to_csv(output_path, index=False)

    print(f"Predictions saved to: {output_path}")

if __name__ == "__main__":
    main()
