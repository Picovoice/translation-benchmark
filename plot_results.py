import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import ScaledTranslation

from benchmark import Translators
from dataset import Dataset

DEFAULT_RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), "results", "data")
DEFAULT_PLOTS_FOLDER = os.path.join(os.path.dirname(__file__), "results", "plots")


ENGINE_PRINT_NAMES = {
    Translators.ZEBRA: "Picovoice Zebra",
    Translators.HELSINKI: "Helsinki-NLP",
}


ENGINE_COLORS = {
    Translators.ZEBRA: (55 / 255, 125 / 255, 255 / 255),
    Translators.HELSINKI: (119 / 255, 131 / 255, 143 / 255)
}


def plot_bleu(
        results_folder: str,
        output_path: str,
        dataset: Dataset
) -> None:
    scores = {}

    for translator in Translators:
        translator_scores = {}
        for benchmark in dataset.benchmarks():
            path = os.path.join(
                results_folder,
                translator.value,
                f"{dataset.source_language}-{dataset.target_language}",
                f"{benchmark}.json")
            with open(path, "r", encoding="utf-8") as fd:
                data = json.load(fd)
            translator_scores[benchmark] = data["bleu"]
        scores[translator] = translator_scores

    fig, ax = plt.subplots(figsize=(12, 6))

    xticks = np.arange(len(dataset.benchmarks()))
    plt.xticks(xticks, dataset.benchmarks())
    w = 0.8 / len(Translators)

    for i, benchmark in enumerate(dataset.benchmarks()):
        for j, translator in enumerate(Translators):
            ax.bar(
                i + j * w - 0.4 + w / 2,
                scores[translator][benchmark],
                w,
                label=translator.value,
                color=ENGINE_COLORS[translator])

    plt.ylim(0, 100)
    plt.ylabel("BLEU", fontsize=14)
    plt.title("Bilingual Evaluation Understudy")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    for i in range(1, len(dataset.benchmarks())):
        labels = ax.xaxis.get_majorticklabels()
        offset = ScaledTranslation(0, -0.15 * (i % 3), fig.dpi_scale_trans)
        labels[i].set_transform(labels[i].get_transform() + offset)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved plot to `{output_path}`")

    plt.close()


def plot_chrf(
        results_folder: str,
        output_path: str,
        dataset: Dataset
) -> None:
    scores = {}

    for translator in Translators:
        translator_score = 0
        for benchmark in dataset.benchmarks():
            path = os.path.join(
                results_folder,
                translator.value,
                f"{dataset.source_language}-{dataset.target_language}",
                f"{benchmark}.json")
            with open(path, "r", encoding="utf-8") as fd:
                data = json.load(fd)
            translator_score += data["chrf"]
        scores[translator] = translator_score / len(dataset.benchmarks())

    fig, ax = plt.subplots(figsize=(12, 6))

    x = [ENGINE_PRINT_NAMES[x] for x in Translators]
    y = [scores[x] for x in Translators]
    colors = [ENGINE_COLORS[x] for x in Translators]
    ax.bar(x, y, color=colors)

    plt.ylim(0, 1)
    plt.ylabel("chr-F", fontsize=14)
    plt.title("Character Level F-Score")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved plot to `{output_path}`")

    plt.close()


def plot_perf(
        results_folder: str,
        output_path: str,
        dataset: Dataset
) -> None:
    scores = {}

    for translator in Translators:
        words = 0
        seconds = 0
        for benchmark in dataset.benchmarks():
            path = os.path.join(
                results_folder,
                translator.value,
                f"{dataset.source_language}-{dataset.target_language}",
                f"{benchmark}.json")
            with open(path, "r", encoding="utf-8") as fd:
                data = json.load(fd)
            words += data["output_words"]
            seconds += data["translate_duration"]
        scores[translator] = words / seconds

    fig, ax = plt.subplots(figsize=(12, 6))

    x = [ENGINE_PRINT_NAMES[x] for x in Translators]
    y = [scores[x] for x in Translators]
    colors = [ENGINE_COLORS[x] for x in Translators]
    ax.bar(x, y, color=colors)

    plt.title("Words per Seconds")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved plot to `{output_path}`")

    plt.close()


def plot_mem(
        results_folder: str,
        output_path: str,
        dataset: Dataset
) -> None:
    scores = {}

    for translator in Translators:
        memory = 0
        count = 0
        for benchmark in dataset.benchmarks():
            path = os.path.join(
                results_folder,
                translator.value,
                f"{dataset.source_language}-{dataset.target_language}",
                f"{benchmark}.json")
            with open(path, "r", encoding="utf-8") as fd:
                data = json.load(fd)
            memory += data["peak_memory"]
            count += 1
        scores[translator] = (memory / count) / 1024 / 1024

    fig, ax = plt.subplots(figsize=(12, 6))

    x = [ENGINE_PRINT_NAMES[x] for x in Translators]
    y = [scores[x] for x in Translators]
    colors = [ENGINE_COLORS[x] for x in Translators]
    ax.bar(x, y, color=colors)

    plt.title("Average Peak Memory Usage (MB)")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved plot to `{output_path}`")

    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-folder",
        default=DEFAULT_RESULTS_FOLDER,
        help="Path to results folder")
    parser.add_argument(
        "--plots-folder",
        default=DEFAULT_PLOTS_FOLDER,
        help="Path to plots folder")
    parser.add_argument("--source-language", type=str, required=True)
    parser.add_argument("--target-language", type=str, required=True)
    args = parser.parse_args()

    source_language = args.source_language
    target_language = args.target_language
    dataset = Dataset.create(None, source_language, target_language)

    plot_bleu(
        args.results_folder,
        os.path.join(args.plots_folder, f"{source_language}-{target_language}", "bleu.png"),
        dataset
    )

    plot_chrf(
        args.results_folder,
        os.path.join(args.plots_folder, f"{source_language}-{target_language}", "chrf.png"),
        dataset
    )

    plot_perf(
        args.results_folder,
        os.path.join(args.plots_folder, f"{source_language}-{target_language}", "words_per_second.png"),
        dataset
    )

    plot_mem(
        args.results_folder,
        os.path.join(args.plots_folder, f"{source_language}-{target_language}", "peak_memory_usage.png"),
        dataset
    )


if __name__ == "__main__":
    main()
