import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

JENA_URL = (
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    "jena_climate_2009_2016.csv.zip"
)
TARGET_COLUMN = "T (degC)"


def download_jena():
    archive = keras.utils.get_file("jena_climate_2009_2016.csv.zip", JENA_URL)
    archive = Path(archive)
    csv_path = archive.parent / "jena_climate_2009_2016.csv"
    if not csv_path.exists():
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(archive.parent)
    return csv_path


def load_frame(sample_every=6, add_time_features=True):
    df = pd.read_csv(download_jena())
    df = df.iloc[sample_every - 1 :: sample_every].reset_index(drop=True)

    timestamps = pd.to_datetime(df.pop("Date Time"), format="%d.%m.%Y %H:%M:%S")
    for column in ("wv (m/s)", "max. wv (m/s)"):
        df[column] = df[column].mask(df[column] < -9000)
    df = df.interpolate(limit_direction="both")

    if not add_time_features:
        return df

    seconds = timestamps.map(pd.Timestamp.timestamp).to_numpy()
    day, year = 24 * 3600, 365.2425 * 24 * 3600
    df["day_sin"] = np.sin(seconds * (2 * np.pi / day))
    df["day_cos"] = np.cos(seconds * (2 * np.pi / day))
    df["year_sin"] = np.sin(seconds * (2 * np.pi / year))
    df["year_cos"] = np.cos(seconds * (2 * np.pi / year))
    return df


def chronological_split(df, train_frac=0.70, val_frac=0.15):
    n = len(df)
    end_train = int(n * train_frac)
    end_val = int(n * (train_frac + val_frac))
    return df[:end_train], df[end_train:end_val], df[end_val:]


def normalize(train, val, test):
    mean = train.mean()
    std = train.std().replace(0, 1.0)
    return (
        (train - mean) / std,
        (val - mean) / std,
        (test - mean) / std,
        float(mean[TARGET_COLUMN]),
        float(std[TARGET_COLUMN]),
    )


def make_dataset(frame, target_index, window, horizon, batch_size, shuffle):
    values = frame.to_numpy(dtype="float32")
    targets = values[window + horizon - 1 :, target_index]
    return keras.utils.timeseries_dataset_from_array(
        data=values[: len(values) - horizon],
        targets=targets,
        sequence_length=window,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def naive_baseline_mae(frame, target_index, window, horizon, target_std):
    values = frame.to_numpy(dtype="float32")[:, target_index]
    current = values[window - 1 : len(values) - horizon]
    future = values[window + horizon - 1 :]
    size = min(len(current), len(future))
    return float(np.abs(current[:size] - future[:size]).mean() * target_std)


def build_model(cell, window, num_features, units, learning_rate, clipnorm):
    cells = {"simple_rnn": layers.SimpleRNN, "lstm": layers.LSTM, "gru": layers.GRU}
    recurrent = cells[cell]

    inputs = keras.Input(shape=(window, num_features))
    x = recurrent(units, return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x)
    x = recurrent(units // 2)(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs, outputs, name=f"jena_{cell}")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate, clipnorm=clipnorm),
        loss="mae",
        metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


def train_cell(cell, datasets, args, target_std, outdir):
    keras.backend.clear_session()
    tf.random.set_seed(args.seed)

    train_ds, val_ds, test_ds, num_features = datasets
    model = build_model(
        cell, args.window, num_features, args.units, args.learning_rate, args.clipnorm
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5
        ),
        keras.callbacks.ModelCheckpoint(
            str(outdir / f"rnn_{cell}.weights.h5"),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    train_mae = model.evaluate(train_ds, verbose=0)[0] * target_std
    val_mae = model.evaluate(val_ds, verbose=0)[0] * target_std
    test_loss, test_rmse = model.evaluate(test_ds, verbose=0)

    model.save(outdir / f"rnn_{cell}.keras")

    return {
        "cell": cell,
        "params": int(model.count_params()),
        "train_mae_degc": float(train_mae),
        "val_mae_degc": float(val_mae),
        "test_mae_degc": float(test_loss * target_std),
        "test_rmse_degc": float(test_rmse * target_std),
        "generalization_gap_degc": float(val_mae - train_mae),
        "epochs_run": len(history.history["loss"]),
    }, {k: [float(v) for v in vals] for k, vals in history.history.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=120)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--units", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--clipnorm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cells", nargs="+", default=["simple_rnn", "gru", "lstm"]
    )
    parser.add_argument("--no-time-features", action="store_true")
    parser.add_argument("--outdir", type=str, default="artifacts/rnn")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    df = load_frame(add_time_features=not args.no_time_features)
    train_df, val_df, test_df = chronological_split(df)
    train_n, val_n, test_n, target_mean, target_std = normalize(
        train_df, val_df, test_df
    )
    target_index = list(df.columns).index(TARGET_COLUMN)
    num_features = df.shape[1]

    datasets = (
        make_dataset(
            train_n, target_index, args.window, args.horizon, args.batch_size, True
        ),
        make_dataset(
            val_n, target_index, args.window, args.horizon, args.batch_size, False
        ),
        make_dataset(
            test_n, target_index, args.window, args.horizon, args.batch_size, False
        ),
        num_features,
    )

    baseline = naive_baseline_mae(
        test_n, target_index, args.window, args.horizon, target_std
    )
    print(f"naive baseline test MAE: {baseline:.4f} degC")

    results, histories = [], {}
    for cell in args.cells:
        metrics, history = train_cell(cell, datasets, args, target_std, outdir)
        results.append(metrics)
        histories[cell] = history
        print(
            f"[{cell}] val={metrics['val_mae_degc']:.4f} "
            f"test={metrics['test_mae_degc']:.4f} degC"
        )

    summary = {
        "config": vars(args),
        "target_mean_degc": target_mean,
        "target_std_degc": target_std,
        "split_sizes": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "naive_baseline_test_mae_degc": baseline,
        "models": results,
    }

    (outdir / "rnn_summary.json").write_text(json.dumps(summary, indent=2))
    (outdir / "rnn_histories.json").write_text(json.dumps(histories, indent=2))


if __name__ == "__main__":
    main()
