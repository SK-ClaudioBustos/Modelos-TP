import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
from tensorflow.keras import layers

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 10


def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    return (
        x_train.astype("float32"),
        y_train.ravel().astype("int32"),
        x_test.astype("float32"),
        y_test.ravel().astype("int32"),
    )


def build_augmenter():
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )


def conv_block(x, filters, dropout_rate, regularized):
    for _ in range(2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=not regularized)(x)
        if regularized:
            x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, strides=2)(x)
    if regularized:
        x = layers.Dropout(dropout_rate)(x)
    return x


def build_model(regularized=True, augment=True, learning_rate=1e-3):
    inputs = keras.Input(shape=INPUT_SHAPE)
    x = layers.Rescaling(1.0 / 255)(inputs)
    if regularized and augment:
        x = build_augmenter()(x)

    x = conv_block(x, 32, 0.20, regularized)
    x = conv_block(x, 64, 0.30, regularized)
    x = conv_block(x, 128, 0.40, regularized)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, use_bias=not regularized)(x)
    if regularized:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if regularized:
        x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="cifar_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_callbacks(checkpoint_path):
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=12, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5
        ),
        keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_accuracy", save_best_only=True
        ),
    ]


def run_cross_validation(args, outdir):
    x_train, y_train, x_test, y_test = load_data()
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    fold_results = []
    histories = {}
    best_score = -1.0
    best_fold = None

    for fold, (tr_idx, va_idx) in enumerate(skf.split(x_train, y_train), start=1):
        if args.max_folds and fold > args.max_folds:
            break

        keras.backend.clear_session()
        tf.random.set_seed(args.seed + fold)

        model = build_model(
            regularized=not args.no_regularization,
            augment=not args.no_augmentation,
            learning_rate=args.learning_rate,
        )
        checkpoint = str(outdir / f"cnn_fold{fold}.weights.h5")

        history = model.fit(
            x_train[tr_idx],
            y_train[tr_idx],
            validation_data=(x_train[va_idx], y_train[va_idx]),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=make_callbacks(checkpoint),
            verbose=2,
        )

        train_loss, train_acc = model.evaluate(
            x_train[tr_idx], y_train[tr_idx], batch_size=512, verbose=0
        )
        val_loss, val_acc = model.evaluate(
            x_train[va_idx], y_train[va_idx], batch_size=512, verbose=0
        )

        fold_results.append(
            {
                "fold": fold,
                "train_accuracy": float(train_acc),
                "val_accuracy": float(val_acc),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "generalization_gap": float(train_acc - val_acc),
                "epochs_run": len(history.history["loss"]),
            }
        )
        histories[f"fold{fold}"] = {
            k: [float(v) for v in vals] for k, vals in history.history.items()
        }

        if val_acc > best_score:
            best_score = val_acc
            best_fold = fold
            model.save(outdir / "cnn_best.keras")

        print(f"[fold {fold}] train={train_acc:.4f} val={val_acc:.4f}")

    return fold_results, histories, best_fold, (x_test, y_test)


def evaluate_on_test(model_path, x_test, y_test, outdir):
    model = keras.models.load_model(model_path)
    probs = model.predict(x_test, batch_size=512, verbose=0)
    preds = probs.argmax(axis=1)

    report = classification_report(
        y_test, preds, target_names=CLASS_NAMES, digits=4, output_dict=True
    )
    matrix = confusion_matrix(y_test, preds)

    np.savetxt(outdir / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    print(classification_report(y_test, preds, target_names=CLASS_NAMES, digits=4))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-regularization", action="store_true")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--outdir", type=str, default="artifacts/cnn")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    fold_results, histories, best_fold, (x_test, y_test) = run_cross_validation(
        args, outdir
    )

    val_scores = np.array([r["val_accuracy"] for r in fold_results])
    gaps = np.array([r["generalization_gap"] for r in fold_results])

    summary = {
        "config": vars(args),
        "folds": fold_results,
        "val_accuracy_mean": float(val_scores.mean()),
        "val_accuracy_std": float(val_scores.std()),
        "generalization_gap_mean": float(gaps.mean()),
        "best_fold": best_fold,
    }

    test_report = evaluate_on_test(outdir / "cnn_best.keras", x_test, y_test, outdir)
    summary["test_report"] = test_report

    (outdir / "cnn_summary.json").write_text(json.dumps(summary, indent=2))
    (outdir / "cnn_histories.json").write_text(json.dumps(histories, indent=2))

    print(
        f"\nCV accuracy: {val_scores.mean():.4f} +/- {val_scores.std():.4f} "
        f"| mean gap: {gaps.mean():.4f}"
    )


if __name__ == "__main__":
    main()
