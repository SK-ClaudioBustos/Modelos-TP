import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {
    "rnn_simple_rnn": "SimpleRNN",
    "rnn_gru": "GRU",
    "rnn_lstm": "LSTM",
}


def plot_gradients(csv_path, outpath):
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    steps = data[data.dtype.names[0]]
    lag = steps.max() - steps

    fig, ax = plt.subplots(figsize=(7, 4))
    for column in data.dtype.names[1:]:
        ax.semilogy(lag, data[column], label=LABELS.get(column, column), linewidth=1.6)

    ax.set_xlabel("Pasos hacia el pasado desde el instante actual (horas)")
    ax.set_ylabel("Norma del gradiente respecto de la entrada")
    ax.set_title("Atenuación del gradiente a lo largo del BPTT")
    ax.invert_xaxis()
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_learning_curves(reg_path, plain_path, outpath):
    reg = json.loads(Path(reg_path).read_text())["fold3"]
    plain = json.loads(Path(plain_path).read_text())["fold1"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, history, title in (
        (axes[0], reg, "Con regularización"),
        (axes[1], plain, "Sin regularización"),
    ):
        ax.plot(history["accuracy"], label="entrenamiento", linewidth=1.6)
        ax.plot(history["val_accuracy"], label="validación", linewidth=1.6)
        ax.set_title(title)
        ax.set_xlabel("Época")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Exactitud")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gradients", default="artifacts/gradients/gradient_norms.csv")
    parser.add_argument("--histories", default="artifacts/cnn/cnn_histories.json")
    parser.add_argument(
        "--histories-plain", default="artifacts/cnn_plain/cnn_histories.json"
    )
    parser.add_argument("--outdir", default="artifacts/figures")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plot_gradients(args.gradients, outdir / "fig_gradientes.png")
    plot_learning_curves(
        args.histories, args.histories_plain, outdir / "fig_curvas_cnn.png"
    )
    print(f"figures written to {outdir}")


if __name__ == "__main__":
    main()
