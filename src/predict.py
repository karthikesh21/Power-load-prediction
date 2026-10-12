"""Prediction interface for Power System Load Type Classification.

This module provides a production-ready interface to load the trained Machine
Learning model pipeline and predict the Load_Type ('Light_Load', 'Medium_Load',
or 'Maximum_Load') for single observations or batches of new electrical measurements.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Union

import joblib
import numpy as np
import pandas as pd

# Expected raw input columns (excluding the target column 'Load_Type')
EXPECTED_RAW_COLUMNS = [
    "Date_Time",
    "Usage_kWh",
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
]

# Engineered feature columns used by the trained model pipeline
MODEL_FEATURE_COLUMNS = [
    "Usage_kWh",
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "hour",
    "minute",
    "dayofweek",
    "is_weekend",
    "seconds_from_midnight",
]


def load_model(model_path: Union[str, Path] = "models/final_model.joblib"):
    """Load the trained machine learning pipeline from disk.

    Parameters
    ----------
    model_path : str or Path
        Path to the serialized joblib pipeline.

    Returns
    -------
    sklearn.pipeline.Pipeline
        The loaded scikit-learn pipeline.
    """
    path = Path(model_path)
    if not path.exists():
        # Also attempt looking relative to project root if executed from src/
        alt_path = Path(__file__).resolve().parent.parent / "models" / "final_model.joblib"
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(
                f"Model file not found at '{model_path}' or '{alt_path}'. "
                "Please train and save the model first."
            )
    return joblib.load(path)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Validate inputs and engineer timestamp features consistently with training.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing raw feature columns.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame ready for model input.
    """
    df = df.copy()

    # Verify required columns exist
    missing_cols = [c for c in EXPECTED_RAW_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input is missing required columns: {missing_cols}")

    # Parse Date_Time
    if not pd.api.types.is_datetime64_any_dtype(df["Date_Time"]):
        df["Date_Time"] = pd.to_datetime(
            df["Date_Time"],
            format="%d-%m-%Y %H:%M",
            errors="coerce",
        )
        if df["Date_Time"].isna().any():
            # Flexible parsing fallback
            df["Date_Time"] = pd.to_datetime(df["Date_Time"], errors="coerce")

    # Check for invalid timestamps
    if df["Date_Time"].isna().any():
        raise ValueError("One or more Date_Time values could not be parsed.")

    # Extract time-derived features
    df["hour"] = df["Date_Time"].dt.hour
    df["minute"] = df["Date_Time"].dt.minute
    df["dayofweek"] = df["Date_Time"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["seconds_from_midnight"] = df["hour"] * 3600 + df["minute"] * 60

    # Ensure numeric types for measurement columns
    numeric_cols = [
        "Usage_kWh",
        "Lagging_Current_Reactive.Power_kVarh",
        "Leading_Current_Reactive_Power_kVarh",
        "CO2(tCO2)",
        "Lagging_Current_Power_Factor",
        "Leading_Current_Power_Factor",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df[MODEL_FEATURE_COLUMNS]


def predict_load_type(
    input_data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
    model: Any = None,
    model_path: Union[str, Path] = "models/final_model.joblib",
) -> Union[str, List[str]]:
    """Predict the load type ('Light_Load', 'Medium_Load', 'Maximum_Load') for input observations.

    Parameters
    ----------
    input_data : dict, list of dicts, or pd.DataFrame
        Raw observation(s) containing electrical measurements and Date_Time.
        Does NOT require 'Load_Type'.
    model : Pipeline, optional
        Preloaded pipeline. If None, loaded from `model_path`.
    model_path : str or Path
        Path to load model if `model` is None.

    Returns
    -------
    str or list of str
        The predicted load type label(s).
    """
    is_single_dict = isinstance(input_data, dict)

    if is_single_dict:
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        df = pd.DataFrame(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise TypeError("input_data must be a dict, list of dicts, or pandas DataFrame.")

    if model is None:
        model = load_model(model_path)

    X = engineer_features(df)
    predictions = model.predict(X)

    if is_single_dict:
        return str(predictions[0])
    return [str(p) for p in predictions]


def run_demo():
    """Run built-in demonstration with real test set samples."""
    print("=" * 65)
    print("POWER LOAD TYPE PREDICTION - INFERENCE DEMO")
    print("=" * 65)

    # Example 1: Real test observation (Night time / early morning off-peak)
    sample_obs_1 = {
        "Date_Time": "15-12-2018 03:30",
        "Usage_kWh": 3.82,
        "Lagging_Current_Reactive.Power_kVarh": 0.0,
        "Leading_Current_Reactive_Power_kVarh": 19.8,
        "CO2(tCO2)": 0.0,
        "Lagging_Current_Power_Factor": 100.0,
        "Leading_Current_Power_Factor": 18.9,
    }

    # Example 2: Real test observation (Midday peak industrial operation)
    sample_obs_2 = {
        "Date_Time": "15-12-2018 14:15",
        "Usage_kWh": 92.45,
        "Lagging_Current_Reactive.Power_kVarh": 58.21,
        "Leading_Current_Reactive_Power_kVarh": 0.0,
        "CO2(tCO2)": 0.04,
        "Lagging_Current_Power_Factor": 84.6,
        "Leading_Current_Power_Factor": 100.0,
    }

    # Example 3: Real test observation (High evening industrial activity)
    sample_obs_3 = {
        "Date_Time": "20-12-2018 19:45",
        "Usage_kWh": 118.62,
        "Lagging_Current_Reactive.Power_kVarh": 68.45,
        "Leading_Current_Reactive_Power_kVarh": 0.0,
        "CO2(tCO2)": 0.06,
        "Lagging_Current_Power_Factor": 86.6,
        "Leading_Current_Power_Factor": 100.0,
    }

    samples = [sample_obs_1, sample_obs_2, sample_obs_3]

    print("\nRunning predictions on sample observations (without target column):\n")
    for i, s in enumerate(samples, start=1):
        pred = predict_load_type(s)
        print(f"Sample {i}:")
        print(f"  Timestamp    : {s['Date_Time']}")
        print(f"  Usage (kWh)  : {s['Usage_kWh']}")
        print(f"  Lagging kVarh: {s['Lagging_Current_Reactive.Power_kVarh']}")
        print(f"  => PREDICTED LOAD TYPE: [{pred}]\n")

    print("Batch prediction test:")
    batch_preds = predict_load_type(samples)
    print(f"Batch predictions: {batch_preds}")
    print("\nInference test completed successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Predict power system load type for single observations or batch CSV files."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Path to CSV file with input observations to predict.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save predictions CSV (used with --csv).",
    )
    parser.add_argument(
        "--datetime",
        type=str,
        help="Timestamp of observation, e.g., '15-12-2018 14:15'.",
    )
    parser.add_argument(
        "--usage",
        type=float,
        help="Electricity usage in kWh.",
    )
    parser.add_argument(
        "--lagging-reactive",
        type=float,
        default=0.0,
        help="Lagging reactive power in kVarh (default: 0.0).",
    )
    parser.add_argument(
        "--leading-reactive",
        type=float,
        default=0.0,
        help="Leading reactive power in kVarh (default: 0.0).",
    )
    parser.add_argument(
        "--co2",
        type=float,
        default=0.0,
        help="CO2 in tCO2 (default: 0.0).",
    )
    parser.add_argument(
        "--lagging-pf",
        type=float,
        default=100.0,
        help="Lagging current power factor (default: 100.0).",
    )
    parser.add_argument(
        "--leading-pf",
        type=float,
        default=100.0,
        help="Leading current power factor (default: 100.0).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/final_model.joblib"),
        help="Path to the trained model file.",
    )

    args = parser.parse_args()

    # Case 1: CSV batch prediction
    if args.csv:
        if not args.csv.exists():
            print(f"Error: CSV file not found at {args.csv}", file=sys.stderr)
            sys.exit(1)
        df = pd.read_csv(args.csv)
        preds = predict_load_type(df, model_path=args.model_path)
        df["Predicted_Load_Type"] = preds
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"Predictions saved to {args.output}")
        else:
            print(df[["Date_Time", "Usage_kWh", "Predicted_Load_Type"]].head(10))
            print(f"... Total predicted: {len(df)} rows")
        return

    # Case 2: Single command-line observation
    if args.datetime and args.usage is not None:
        obs = {
            "Date_Time": args.datetime,
            "Usage_kWh": args.usage,
            "Lagging_Current_Reactive.Power_kVarh": args.lagging_reactive,
            "Leading_Current_Reactive_Power_kVarh": args.leading_reactive,
            "CO2(tCO2)": args.co2,
            "Lagging_Current_Power_Factor": args.lagging_pf,
            "Leading_Current_Power_Factor": args.leading_pf,
        }
        pred = predict_load_type(obs, model_path=args.model_path)
        print(f"\nObservation Details:")
        print(f"  Date & Time : {args.datetime}")
        print(f"  Usage (kWh) : {args.usage}")
        print(f"  Lagging kVarh: {args.lagging_reactive}")
        print(f"  => PREDICTED LOAD TYPE: [{pred}]\n")
        return

    # Case 3: Default run built-in demo
    run_demo()


if __name__ == "__main__":
    main()
