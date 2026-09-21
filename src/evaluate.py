import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

from cnn_pipeline import CLASS_NAMES, load_data
from rnn_pipeline import (
    TARGET_COLUMN,
    chronological_split,
    load_frame,
    make_dataset,
    naive_baseline_mae,
    normalize,
)

RNN_CELLS = ("simple_rnn", "gru", "lstm")


def evaluate_cnn(model_path):
    _, _, x_test, y_test = load_data()
    model = keras.models.load_model(model_path, compile=False)
    preds = model.predict(x_test, batch_size=512, verbose=0).argmax(axis=1)

    print(f"\n{model_path}")
    print(classification_report(y_test, preds, target_names=CLASS_NAMES, digits=4))
    print("confusion matrix (rows = true class):")
    print(confusion_matrix(y_test, preds))
    return {"accuracy": float((preds == y_test).mean())}


def evaluate_rnn(model_paths, window, horizon, batch_size, time_features):
    df = load_frame(add_time_features=time_features)
    train_df, val_df, test_df = chronological_split(df)
    _, _, test_n, _, target_std = normalize(train_df, val_df, test_df)
    target_index = list(df.columns).index(TARGET_COLUMN)

    test_ds = make_dataset(test_n, target_index, window, horizon, batch_size, False)
    targets = np.concatenate([y.numpy() for _, y in test_ds])

    baseline = naive_baseline_mae(test_n, target_index, window, horizon, target_std)
    print(f"\nnaive baseline test MAE: {baseline:.4f} degC")

    results = {"naive_baseline": {"mae_degc": baseline}}
    for path in model_paths:
        model = keras.models.load_model(path, compile=False)
        preds = model.predict(test_ds, verbose=0).ravel()
        errors = (preds - targets) * target_std
        results[Path(path).stem] = {
            "mae_degc": float(np.abs(errors).mean()),
            "rmse_degc": float(np.sqrt((errors**2).mean())),
        }
        print(
            f"{Path(path).name}: MAE={results[Path(path).stem]['mae_degc']:.4f} "
            f"RMSE={results[Path(path).stem]['rmse_degc']:.4f} degC"
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["cnn", "rnn", "all"], default="all")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--window", type=int, default=120)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--no-time-features", action="store_true")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    results = {}

    if args.task in ("cnn", "all"):
        results["cnn"] = evaluate_cnn(models_dir / "cnn_best.keras")

    if args.task in ("rnn", "all"):
        paths = [models_dir / f"rnn_{cell}.keras" for cell in RNN_CELLS]
        results["rnn"] = evaluate_rnn(
            [p for p in paths if p.exists()],
            args.window,
            args.horizon,
            args.batch_size,
            not args.no_time_features,
        )

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
