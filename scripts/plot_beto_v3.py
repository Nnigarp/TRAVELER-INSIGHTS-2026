"""Plot three common-scale matrices from validated saved V3 assessment only."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from traveler_agent.beto_training_v3_analysis import analyze
from traveler_agent.corpus import CorpusError
from traveler_agent.labels import LABELS
from traveler_agent.synthetic_diversity import ignored_target


def plot(root: Path, out: Path):
    """Create new PNG/SVG artifacts; never overwrite assessment or plot files."""
    analysis = analyze(root)
    selected = analysis["selected_model_key"]
    if selected is None:
        raise CorpusError("No hay ajustado seleccionado; no inventar su matriz de confusión.")
    output = ignored_target(out, kinds=("artifacts",))
    targets = [output.with_suffix("." + extension) for extension in ("png", "svg")]
    if any(path.exists() for path in targets):
        raise CorpusError("Gráfica existente; conservarla y usar otro destino.")
    keys = ("lr", "svm", selected)
    metrics = [analysis["paired_family_bootstrap"][key]["classification"] for key in keys]
    matrices = [np.asarray(metric["confusion_matrix"]) for metric in metrics]
    maximum = max(int(matrix.max()) for matrix in matrices)
    titles = ("Regresión logística", "SVM", "BETO ajustado · semilla 42")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.75))
    fig.subplots_adjust(left=.075, right=.89, bottom=.27, top=.78, wspace=.34)
    try:
        for axis, title, matrix, metric in zip(axes, titles, matrices, metrics):
            display = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
            axis.set(xticks=range(3), yticks=range(3), xticklabels=LABELS,
                     yticklabels=LABELS, xlabel="Predicción")
            axis.set_title(f"{title}\nMacro-F1: {metric['macro_f1']:.3f}", pad=12)
            axis.set_xticks(np.arange(-.5, 3, 1), minor=True)
            axis.set_yticks(np.arange(-.5, 3, 1), minor=True)
            axis.grid(which="minor", color="white", linewidth=1.7)
            axis.tick_params(which="both", length=0)
            for spine in axis.spines.values():
                spine.set_visible(False)
            for (i, j), value in np.ndenumerate(matrix):
                axis.text(j, i, str(int(value)), ha="center", va="center", fontsize=14,
                          color="white" if value > maximum * .55 else "#123048")
        axes[0].set_ylabel("Intención sintética (IA)")
        color_axis = fig.add_axes((.92, .305, .015, .43))
        fig.colorbar(display, cax=color_axis, label="Textos · misma escala")
        fig.suptitle(f"BETO V3 · reproducción sintética · n={analysis['assessment_rows']}",
                     fontsize=16, fontweight="bold", x=.49, y=.96)
        fig.text(.075, .125, "Etiquetas y revisión por IA; datos sintéticos, sin gold humano. Cohorte ya observada.",
                 fontsize=10, color="#334155")
        fig.text(.075, .075, "Mismos textos y escala en los tres modelos. Configuración fijada en tune; semilla 42 primaria.",
                 fontsize=9, color="#475569")
        output.parent.mkdir(parents=True, exist_ok=True)
        for path in targets:
            with path.open("xb") as stream:
                fig.savefig(stream, format=path.suffix[1:], dpi=180, facecolor="white")
        svg = targets[1]
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
                       encoding="utf-8")
    finally:
        plt.close(fig)
    return targets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    plot(ignored_target(args.run, kinds=("artifacts",)), args.out)


if __name__ == "__main__":
    main()
