import argparse
import json
import os
import time
from enum import Enum
from typing import (
    Any,
    Dict
)

import pandas as pd
import psutil
import pvzebra
import sacrebleu
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)

from dataset import Dataset


class Translators(Enum):
    ZEBRA = "zebra"
    HELSINKI = "helsinki"


class Translator:
    def translate(self, input_text: str) -> str:
        raise NotImplementedError(
            f"Method `translate` must be implemented in a subclass of {self.__class__.__name__}")

    @classmethod
    def create(cls, engine: Translators, **kwargs: Any) -> "Translator":
        subclasses = {
            Translators.ZEBRA: ZebraTranslator,
            Translators.HELSINKI: HelsinkiTranslator,
        }

        if engine not in subclasses:
            raise NotImplementedError(f"Cannot create {cls.__name__} of type `{engine.value}`")

        return subclasses[engine](**kwargs)


class ZebraTranslator(Translator):
    def __init__(
            self,
            access_key: str,
            model_path: str,
            device: str,
    ) -> None:
        super().__init__()

        self.model = pvzebra.create(
            access_key=access_key,
            model_path=model_path,
            device=device
        )

    def translate(self, input_text: str) -> str:
        return self.model.translate(input_text)


class HelsinkiTranslator(Translator):
    def __init__(
            self,
            model_path: str,
            num_threads: int
    ) -> None:
        super().__init__()

        torch.set_num_threads(num_threads)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to("cpu")

    def translate(self, input_text: str) -> str:
        input_tokens = self.tokenizer.encode(input_text, return_tensors="pt")
        output_tokens = self.model.generate(input_tokens)
        output_text = "".join(self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True))

        return output_text


def get_translator_init_kwargs(args: argparse.Namespace) -> Dict[str, str]:
    kwargs = dict()
    translator_type = Translators(args.translator)

    kwargs["engine"] = translator_type

    if translator_type is Translators.ZEBRA:
        kwargs["access_key"] = args.picovoice_access_key
        kwargs["model_path"] = args.zebra_model_path
        kwargs["device"] = args.zebra_device

    elif translator_type is Translators.HELSINKI:
        kwargs["model_path"] = f"Helsinki-NLP/opus-mt-{args.source_language}-{args.target_language}"
        kwargs["num_threads"] = args.helsinki_num_threads

    return kwargs


def process(
        dataset: Dataset,
        translator: Translator,
        benchmark: str,
) -> Dict[str, Any]:
    total_sentences = dataset.size(benchmark)
    translations = pd.DataFrame(
        index=range(total_sentences),
        columns=["input", "reference", dataset.target_language]
    )

    input_sentences = 0
    input_words = 0
    output_words = 0
    translate_duration = 0.0

    for index in tqdm(range(total_sentences), total=total_sentences):
        input_text, reference_text = dataset.get(benchmark, index)
        translations.loc[index, "input"] = input_text
        translations.loc[index, "reference"] = reference_text

        t0 = time.perf_counter()
        output_text = translator.translate(input_text)
        t1 = time.perf_counter()
        translations.loc[index, dataset.target_language] = output_text

        input_sentences += 1
        input_words += len(input_text.split())
        output_words += len(output_text.split())
        translate_duration += t1 - t0

    bleu = sacrebleu.corpus_bleu(
        translations[:input_sentences][dataset.target_language],
        [translations[:input_sentences]["reference"]]
    )
    chrf = sacrebleu.corpus_chrf(
        translations[:input_sentences][dataset.target_language],
        translations[:input_sentences]["reference"]
    )

    scores = {
        "input_sentences": input_sentences,
        "input_words": input_words,
        "output_words": output_words,
        "translate_duration": translate_duration,
        "bleu": bleu.score,
        "chrf": chrf.score,
    }

    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to preprocessed dataset created by running `preprocess.py`"
    )
    parser.add_argument("--source-language", type=str, required=True)
    parser.add_argument("--target-language", type=str, required=True)
    parser.add_argument(
        "--engine",
        dest="translator",
        type=str,
        required=True,
        choices=[t.value for t in Translators])
    parser.add_argument("--picovoice-access-key", type=str, default=None)
    parser.add_argument("--zebra-model-path", type=str, default=None)
    parser.add_argument("--zebra-device", type=str, default="cpu:4")
    parser.add_argument("--helsinki-num-threads", type=int, default=4)
    args = parser.parse_args()

    dataset_path = os.path.abspath(args.dataset_path)
    source_language = args.source_language
    target_language = args.target_language

    results_path = os.path.join(
        os.path.dirname(__file__),
        "results",
        "data",
        args.translator,
        f"{source_language}-{target_language}"
    )

    os.makedirs(results_path, exist_ok=True)

    dataset = Dataset.create(dataset_path, source_language, target_language)

    mem_peak = 0
    mem_before = psutil.Process(os.getpid()).memory_info().rss

    translator_init_kwargs = get_translator_init_kwargs(args)
    translator = Translator.create(
        **translator_init_kwargs
    )

    for benchmark in dataset.benchmarks():
        print(benchmark)

        scores = process(
            dataset,
            translator,
            benchmark
        )

        mem_after = psutil.Process(os.getpid()).memory_info().rss
        if mem_after - mem_before > mem_peak:
            mem_peak = mem_after - mem_before

        scores["peak_memory"] = mem_peak

        scores_path = os.path.join(results_path, f"{benchmark}.json")
        with open(scores_path, "w", encoding="utf-8") as fd:
            json.dump(scores, fd, indent=4)


if __name__ == "__main__":
    main()
