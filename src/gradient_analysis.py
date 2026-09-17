import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from rnn_pipeline import (
    TARGET_COLUMN,
    chronological_split,
    load_frame,
    make_dataset,
    normalize,
)


def input_gradient_norms(model, x_batch, y_batch):
    x = tf.convert_to_tensor(x_batch)
    with tf.GradientTape() as tape:
        tape.watch(x)
        preds = model(x, training=False)
        loss = tf.reduce_mean(tf.abs(tf.squeeze(preds) - y_batch))
    grads = tape.gradient(loss, x)
    norms = tf.norm(grads, axis=-1)
    return tf.reduce_mean(norms, axis=0).numpy()


def summarize(norms):
    newest = float(norms[-1])
    oldest = float(norms[0])
    relative = norms / (newest + 1e-12)
    above_1pct = int(np.sum(relative >= 0.01))
    return {
        "norm_newest_step": newest,
        "norm_oldest_step": oldest,
        "decay_ratio_oldest_over_newest": oldest / (newest + 1e-12),
        "steps_above_1pct_of_newest": above_1pct,
        "effective_memory_fraction": above_1pct / len(norms),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--window", type=int, default=120)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--no-time-features", action="store_true")
    parser.add_argument("--outdir", type=str, default="artifacts/gradients")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_frame(add_time_features=not args.no_time_features)
    train_df, val_df, test_df = chronological_split(df)
    _, val_n, _, _, _ = normalize(train_df, val_df, test_df)
    target_index = list(df.columns).index(TARGET_COLUMN)

    dataset = make_dataset(
        val_n, target_index, args.window, args.horizon, args.batch_size, False
    )
    x_batch, y_batch = next(iter(dataset))

    results = {}
    curves = {}
    for path in args.models:
        name = Path(path).stem
        model = keras.models.load_model(path)
        norms = input_gradient_norms(model, x_batch, y_batch)
        curves[name] = [float(v) for v in norms]
        results[name] = summarize(norms)
        print(
            f"[{name}] decay={results[name]['decay_ratio_oldest_over_newest']:.6f} "
            f"memoria efectiva={results[name]['effective_memory_fraction']:.2%}"
        )

    header = "timestep_from_oldest," + ",".join(curves)
    rows = np.column_stack([np.arange(args.window)] + list(curves.values()))
    np.savetxt(
        outdir / "gradient_norms.csv",
        rows,
        delimiter=",",
        header=header,
        comments="",
        fmt="%.8g",
    )
    (outdir / "gradient_summary.json").write_text(
        json.dumps({"config": vars(args), "models": results}, indent=2)
    )


if __name__ == "__main__":
    main()
