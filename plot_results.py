import argparse
import json
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import ScaledTranslation

from benchmark import Translators
from dataset import (
    Datasets,
    Dataset
)

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
        output_path: str,
        datasets: Dict[Datasets, Dataset],
        data: dict
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    xticks = np.arange(len(datasets))
    plt.xticks(xticks, [x.value for x in datasets])
    w = 0.8 / len(Translators)

    for i, (pair, dataset) in enumerate(datasets.items()):
        benchmark = dataset.benchmarks()[-1]
        for j, translator in enumerate(Translators):
            ax.bar(
                i + j * w - 0.4 + w / 2,
                data[translator][pair][benchmark]["bleu"],
                w,
                label=translator.value,
                color=ENGINE_COLORS[translator])

    plt.ylim(0, 100)
    plt.ylabel("BLEU")
    plt.title("BLEU on Tatoeba Dataset")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved plot to `{output_path}`")

    plt.close()


def plot_perf(
        output_path: str,
        datasets: Dict[Datasets, Dataset],
        data: dict
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    xticks = np.arange(len(datasets))
    plt.xticks(xticks, [x.value for x in datasets])
    w = 0.8 / len(Translators)

    for i, (pair, dataset) in enumerate(datasets.items()):
        for j, translator in enumerate(Translators):
            words = 0
            seconds = 0
            for benchmark in dataset.benchmarks():
                words += data[translator][pair][benchmark]["output_words"]
                seconds += data[translator][pair][benchmark]["translate_duration"]

            ax.bar(
                i + j * w - 0.4 + w / 2,
                words / seconds,
                w,
                label=translator.value,
                color=ENGINE_COLORS[translator])

    plt.ylabel("Words per Seconds")
    plt.title("Words per Seconds")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved plot to `{output_path}`")

    plt.close()


def plot_mem(
        output_path: str,
        datasets: Dict[Datasets, Dataset],
        data: dict
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    xticks = np.arange(len(datasets))
    plt.xticks(xticks, [x.value for x in datasets])
    w = 0.8 / len(Translators)

    for i, (pair, dataset) in enumerate(datasets.items()):
        for j, translator in enumerate(Translators):
            count = 0
            memory = 0
            for benchmark in dataset.benchmarks():
                count += 1
                memory += data[translator][pair][benchmark]["peak_memory"]

            ax.bar(
                i + j * w - 0.4 + w / 2,
                (memory / count) / 1024 / 1024,
                w,
                label=translator.value,
                color=ENGINE_COLORS[translator])

    plt.ylabel("MB")
    plt.title("Peak Memory Usage (MB)")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved plot to `{output_path}`")

    plt.close()


def compare(
        datasets: Dict[Datasets, Dataset],
        data: dict
) -> None:
    accuracies = []
    for pair, dataset in datasets.items():
        for benchmark in dataset.benchmarks():
            zebra = data[Translators.ZEBRA][pair][benchmark]["bleu"]
            helsinki = data[Translators.HELSINKI][pair][benchmark]["bleu"]
            accuracy = min(1, zebra / helsinki)
            accuracies.append(accuracy * 100)

    speedups = []
    for pair, dataset in datasets.items():
        for benchmark in dataset.benchmarks():
            zebra_words = data[Translators.ZEBRA][pair][benchmark]["output_words"]
            zebra_seconds = data[Translators.ZEBRA][pair][benchmark]["translate_duration"]
            zebra_words_per_second = zebra_words / zebra_seconds
            helsinki_words = data[Translators.HELSINKI][pair][benchmark]["output_words"]
            helsinki_seconds = data[Translators.HELSINKI][pair][benchmark]["translate_duration"]
            helsinki_words_per_second = helsinki_words / helsinki_seconds
            speedups.append(zebra_words_per_second / helsinki_words_per_second)

    ram_usages = []
    for pair, dataset in datasets.items():
        for benchmark in dataset.benchmarks():
            zebra = data[Translators.ZEBRA][pair][benchmark]["peak_memory"]
            helsinki = data[Translators.HELSINKI][pair][benchmark]["peak_memory"]
            ram_usages.append(zebra / helsinki * 100)

    accuracy_mean = np.mean(accuracies)
    accuracy_std = np.std(accuracies)
    print(f"Accuracy = {accuracy_mean:.1f}±{accuracy_std:.0f}%")

    speedup_mean = np.mean(speedups)
    speedup_std = np.std(speedups)
    print(f"Speedup = {speedup_mean:.1f}±{speedup_std:.0f}x")

    ram_usage_mean = np.mean(ram_usages)
    ram_usage_std = np.std(ram_usages)
    print(f"RAM = {ram_usage_mean:.0f}±{ram_usage_std:.0f}%")


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
    parser.add_argument("--language-pairs", type=str, nargs="+", required=True)
    args = parser.parse_args()

    results_folder = args.results_folder
    plots_folder = args.plots_folder
    language_pairs = args.language_pairs

    datasets: Dict[Datasets, Dataset] = {}
    for pair in language_pairs:
        source_language, target_language = pair.split("-")
        datasets[Datasets(pair)] = Dataset.create(None, source_language, target_language)

    data = {}
    for translator in Translators:
        data[translator] = {}
        for pair in datasets.keys():
            data[translator][pair] = {}
            for benchmark in datasets[pair].benchmarks():
                path = os.path.join(
                    results_folder,
                    translator.value,
                    pair.value,
                    f"{benchmark}.json")
                with open(path, "r", encoding="utf-8") as fd:
                    data[translator][pair][benchmark] = json.load(fd)

    plot_bleu(
        os.path.join(plots_folder, "bleu.png"),
        datasets,
        data
    )

    plot_perf(
        os.path.join(plots_folder, "words_per_second.png"),
        datasets,
        data
    )

    plot_mem(
        os.path.join(plots_folder, "peak_memory_usage.png"),
        datasets,
        data
    )

    compare(
        datasets,
        data
    )


if __name__ == "__main__":
    main()
