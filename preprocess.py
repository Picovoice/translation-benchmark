import argparse
import csv
import gzip
import os
import subprocess

import pandas as pd
import regex
from langcodes import Language

from dataset import Dataset

OPUS_REPO_ID = "https://github.com/Helsinki-NLP/OPUS-MT-train"
OPUS_COMMIT_ID = "2805bf49e759a742ac76e2011f1ee8f5eb9b6ab9"

TATOEBA_REPO_ID = "https://github.com/Helsinki-NLP/Tatoeba-Challenge"
TATOEBA_COMMIT_ID = "d34a89ac102fd236503a1911dd1050564bf4e682"


def preprocess_text(text: str) -> str:
    replacements = [
        (r'，', ','),
        (r'。 *', '. '),
        (r'、', ','),
        (r'”', '"'),
        (r'“', '"'),
        (r'∶', ':'),
        (r'：', ':'),
        (r'？', '?'),
        (r'《', '"'),
        (r'》', '"'),
        (r'）', ')'),
        (r'！', '!'),
        (r'（', '('),
        (r'；', ';'),
        (r'１', '1'),
        (r'」', '"'),
        (r'「', '"'),
        (r'０', '0'),
        (r'３', '3'),
        (r'２', '2'),
        (r'５', '5'),
        (r'６', '6'),
        (r'９', '9'),
        (r'７', '7'),
        (r'８', '8'),
        (r'４', '4'),
        (r'． *', '. '),
        (r'～', '~'),
        (r'’', '\''),
        (r'…', '...'),
        (r'━', '-'),
        (r'〈', '<'),
        (r'〉', '>'),
        (r'【', '['),
        (r'】', ']'),
        (r'％', '%'),
        (r' +', ' '),
        (r'^ *', ''),
        (r' *$', ''),
        (r'\p{C}+', ''),
    ]

    for pattern, replacement in replacements:
        text = regex.sub(pattern, replacement, text)
    return text.strip()


def process_opus(
        output_path: str,
        input_path: str,
        dataset: Dataset
) -> None:
    output_path = os.path.join(output_path, f"{dataset.source_language}-{dataset.target_language}")
    input_path = os.path.join(input_path, "testsets", f"{dataset.source_language}-{dataset.target_language}")
    os.makedirs(output_path, exist_ok=True)

    for benchmark in dataset.benchmarks()[:-1]:
        l0_path = os.path.join(input_path, f"{benchmark}.{dataset.source_language}.gz")
        l1_path = os.path.join(input_path, f"{benchmark}.{dataset.target_language}.gz")
        out_path = os.path.join(output_path, f"{benchmark}.{dataset.source_language}.{dataset.target_language}.csv")

        with gzip.open(l0_path, "rt", encoding="utf-8") as fd:
            l0_lines = [preprocess_text(line) for line in fd]

        with gzip.open(l1_path, "rt", encoding="utf-8") as fd:
            l1_lines = [preprocess_text(line) for line in fd]

        d0 = pd.DataFrame(l0_lines, columns=[dataset.source_language])
        d1 = pd.DataFrame(l1_lines, columns=[dataset.target_language])
        data = pd.concat([d0, d1], axis=1)

        data.to_csv(
            out_path,
            index=False
        )


def process_tatoeba(
        output_path: str,
        input_path: str,
        dataset: Dataset
) -> None:
    l0 = Language.get(dataset.source_language).to_alpha3()
    l1 = Language.get(dataset.target_language).to_alpha3()

    data_path = os.path.join(
        input_path,
        "data",
        "test",
        f"{l0}-{l1}",
        "test.txt")
    data_names = ["l0", "l1", dataset.source_language, dataset.target_language]
    if not os.path.exists(data_path):
        l0, l1 = l1, l0
        data_path = os.path.join(
            input_path,
            "data",
            "test",
            f"{l0}-{l1}",
            "test.txt")
        data_names = ["l1", "l0", dataset.target_language, dataset.source_language]

    data = pd.read_csv(
        data_path,
        sep="\t",
        names=data_names,
        on_bad_lines="skip",
        dtype=str,
        quoting=csv.QUOTE_NONE)

    for index, row in data.iterrows():
        data.loc[index, dataset.source_language] = preprocess_text(row[dataset.source_language])
        data.loc[index, dataset.target_language] = preprocess_text(row[dataset.target_language])

    benchmark = dataset.benchmarks()[-1]
    benchmark_folder = f"{dataset.source_language}-{dataset.target_language}"
    benchmark_filename = f"{benchmark}.{dataset.source_language}.{dataset.target_language}.csv"
    benchmark_path = os.path.join(output_path, benchmark_folder, benchmark_filename)

    data[[dataset.source_language, dataset.target_language]].to_csv(
        benchmark_path,
        index=False
    )


def preprocess(
        dataset_path: str,
        tatoeba_path: str,
        opus_path: str,
        source_language: str,
        target_language: str,
) -> None:
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(os.path.dirname(tatoeba_path), exist_ok=True)
    os.makedirs(os.path.dirname(opus_path), exist_ok=True)

    if not os.path.exists(opus_path):
        subprocess.check_call(
            ["git", "clone", "--depth", "1", OPUS_REPO_ID, opus_path])
        subprocess.check_call(
            ["git", "fetch", "--depth", "1", "origin", OPUS_COMMIT_ID],
            cwd=opus_path)
        subprocess.check_call(
            ["git", "checkout", OPUS_COMMIT_ID],
            cwd=opus_path)

    if not os.path.exists(tatoeba_path):
        subprocess.check_call(
            ["git", "clone", "--depth", "1", TATOEBA_REPO_ID, tatoeba_path])
        subprocess.check_call(
            ["git", "fetch", "--depth", "1", "origin", TATOEBA_COMMIT_ID],
            cwd=tatoeba_path)
        subprocess.check_call(
            ["git", "checkout", TATOEBA_COMMIT_ID],
            cwd=tatoeba_path)

    dataset = Dataset.create(None, source_language, target_language)
    process_opus(dataset_path, opus_path, dataset)
    process_tatoeba(dataset_path, tatoeba_path, dataset)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--tatoeba-path", type=str, required=True)
    parser.add_argument("--opus-path", type=str, required=True)
    parser.add_argument("--source-language", type=str, required=True)
    parser.add_argument("--target-language", type=str, required=True)
    args = parser.parse_args()

    dataset_path = os.path.abspath(args.dataset_path)
    tatoeba_path = os.path.abspath(args.tatoeba_path)
    opus_path = os.path.abspath(args.opus_path)
    source_language = args.source_language
    target_language = args.target_language

    preprocess(
        dataset_path,
        tatoeba_path,
        opus_path,
        source_language,
        target_language
    )


if __name__ == "__main__":
    main()
