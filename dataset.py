import os
from enum import Enum
from typing import (
    List,
    Optional,
    Sequence,
    Tuple
)

import pandas as pd


class Datasets(Enum):
    DE_EN = "de-en"
    DE_ES = "de-es"
    DE_FR = "de-fr"
    DE_IT = "de-it"
    DE_JA = "de-ja"
    DE_KO = "de-ko"
    DE_PT = "de-pt"
    EN_DE = "en-de"
    EN_ES = "en-es"
    EN_FR = "en-fr"
    EN_IT = "en-it"
    EN_JA = "en-ja"
    EN_KO = "en-ko"
    EN_PT = "en-pt"
    ES_DE = "es-de"
    ES_EN = "es-en"
    ES_FR = "es-fr"
    ES_IT = "es-it"
    ES_JA = "es-ja"
    ES_KO = "es-ko"
    ES_PT = "es-pt"
    FR_DE = "fr-de"
    FR_EN = "fr-en"
    FR_ES = "fr-es"
    FR_IT = "fr-it"
    FR_JA = "fr-ja"
    FR_KO = "fr-ko"
    FR_PT = "fr-pt"
    IT_DE = "it-de"
    IT_EN = "it-en"
    IT_ES = "it-es"
    IT_FR = "it-fr"
    IT_JA = "it-ja"
    IT_PT = "it-pt"
    JA_DE = "ja-de"
    JA_EN = "ja-en"
    JA_ES = "ja-es"
    JA_FR = "ja-fr"
    JA_IT = "ja-it"
    JA_KO = "ja-ko"
    JA_PT = "ja-pt"
    KO_DE = "ko-de"
    KO_EN = "ko-en"
    KO_ES = "ko-es"
    KO_FR = "ko-fr"
    KO_JA = "ko-ja"
    PT_DE = "pt-de"
    PT_EN = "pt-en"
    PT_ES = "pt-es"
    PT_FR = "pt-fr"
    PT_IT = "pt-it"
    PT_JA = "pt-ja"


class Dataset(object):
    BENCHMARK_NAMES: List[str]

    def __init__(
            self,
            dataset_path: Optional[str],
            source_language: str,
            target_language: str
    ):
        self.source_language = source_language
        self.target_language = target_language
        self.data = {}

        if dataset_path:
            for benchmark in self.BENCHMARK_NAMES:
                benchmark_folder = f"{self.source_language}-{self.target_language}"
                benchmark_filename = f"{benchmark}.{self.source_language}.{self.target_language}.csv"
                benchmark_path = os.path.join(dataset_path, benchmark_folder, benchmark_filename)
                self.data[benchmark] = pd.read_csv(
                    benchmark_path
                )

    def benchmarks(self) -> Sequence[str]:
        return self.BENCHMARK_NAMES

    def size(self, benchmark: str) -> int:
        return len(self.data[benchmark])

    def get(self, benchmark: str, index: int) -> Tuple[str, str]:
        source_text = self.data[benchmark].loc[index, self.source_language]
        target_text = self.data[benchmark].loc[index, self.target_language]
        return source_text, target_text

    @classmethod
    def create(
            cls,
            dataset_path: Optional[str],
            source_language: str,
            target_language: str
    ) -> 'Dataset':
        datasets = {
            Datasets.DE_EN: DeEnDataset,
            Datasets.DE_FR: DeFrDataset,
            Datasets.EN_DE: EnDeDataset,
            Datasets.EN_ES: EnEsDataset,
            Datasets.EN_FR: EnFrDataset,
            Datasets.EN_IT: EnItDataset,
            Datasets.ES_EN: EsEnDataset,
            Datasets.ES_FR: EsFrDataset,
            Datasets.FR_DE: FrDeDataset,
            Datasets.FR_EN: FrEnDataset,
            Datasets.FR_ES: FrEsDataset,
            Datasets.IT_EN: ItEnDataset,
        }

        key = Datasets(f"{source_language}-{target_language}")

        if key in datasets:
            return datasets[key](dataset_path)
        return TatoebaDataset(dataset_path, source_language, target_language)


class TatoebaDataset(Dataset):
    BENCHMARK_NAMES = [
        "Tatoeba",
    ]

    def __init__(
            self,
            dataset_path: Optional[str],
            source_language: str,
            target_language: str
    ) -> None:
        super().__init__(dataset_path, source_language, target_language)


class DeEnDataset(Dataset):
    BENCHMARK_NAMES = [
        "newssyscomb2009",
        "news-test2008",
        "newstest2009",
        "newstest2010",
        "newstest2011",
        "newstest2012",
        "newstest2013",
        "newstest2014-deen",
        "newstest2015-ende",
        "newstest2016-ende",
        "newstest2017-ende",
        "newstest2018-ende",
        "newstest2019-deen",
        "Tatoeba",
    ]

    def __init__(self, dataset_path: Optional[str]) -> None:
        super().__init__(dataset_path, "de", "en")


class DeFrDataset(Dataset):
    BENCHMARK_NAMES = [
        "euelections_dev2019.de-fr",
        "newssyscomb2009",
        "news-test2008",
        "newstest2009",
        "newstest2010",
        "newstest2011",
        "newstest2012",
        "newstest2013",
        "newstest2019-defr",
        "Tatoeba",
    ]

    def __init__(self, dataset_path: Optional[str]) -> None:
        super().__init__(dataset_path, "de", "fr")


class EnDeDataset(Dataset):
    BENCHMARK_NAMES = [
        "newssyscomb2009",
        "news-test2008",
        "newstest2009",
        "newstest2010",
        "newstest2011",
        "newstest2012",
        "newstest2013",
        "newstest2015-ende",
        "newstest2016-ende",
        "newstest2017-ende",
        "newstest2018-ende",
        "newstest2019-ende",
        "Tatoeba",
    ]

    def __init__(self, dataset_path: Optional[str]) -> None:
        super().__init__(dataset_path, "en", "de")


class EnEsDataset(Dataset):
    BENCHMARK_NAMES = [
        "newssyscomb2009",
        "news-test2008",
        "newstest2009",
        "newstest2010",
        "newstest2011",
        "newstest2012",
        "newstest2013",
        "Tatoeba",
    ]

    def __init__(self, dataset_path: Optional[str]) -> None:
        super().__init__(dataset_path, "en", "es")


class EnFrDataset(Dataset):
    BENCHMARK_NAMES = [
        "newsdiscussdev2015-enfr",
        "newsdiscusstest2015-enfr",
        "newssyscomb2009",
        "news-test2008",
        "newstest2009",
        "newstest2010",
        "newstest2011",
        "newstest2012",
        "newstest2013",
        "Tatoeba",
    ]

    def __init__(self, dataset_path: Optional[str]) -> None:
        super().__init__(dataset_path, "en", "fr")


class EnItDataset(Dataset):
    BENCHMARK_NAMES = [
        "newssyscomb2009",
        "newstest2009",
        "Tatoeba",
    ]

    def __init__(self, dataset_path: Optional[str]) -> None:
        super().__init__(dataset_path, "en", "it")


class EsEnDataset(Dataset):
    BENCHMARK_NAMES = [
        "newssyscomb2009",
        "news-test2008",
        "newstest2009",
        "newstest2010",
        "newstest2011",
        "newstest2012",
        "newstest2013",
        "Tatoeba",
    ]

    def __init__(self, dataset_path: Optional[str]) -> None:
        super().__init__(dataset_path, "es", "en")


class EsFrDataset(Dataset):
    BENCHMARK_NAMES = [
        "newssyscomb2009",
        "news-test2008",
        "newstest2009",
        "newstest2010",
        "newstest2011",
        "newstest2012",
        "newstest2013",
        "Tatoeba",
    ]

    def __init__(self, dataset_path: Optional[str]) -> None:
        super().__init__(dataset_path, "es", "fr")


class FrDeDataset(Dataset):
    BENCHMARK_NAMES = [
        "euelections_dev2019.fr-de",
        "newssyscomb2009",
        "news-test2008",
        "newstest2009",
        "newstest2010",
        "newstest2011",
        "newstest2012",
        "newstest2013",
        "newstest2019-frde",
        "Tatoeba",
    ]

    def __init__(self, dataset_path: Optional[str]) -> None:
        super().__init__(dataset_path, "fr", "de")


class FrEnDataset(Dataset):
    BENCHMARK_NAMES = [
        "newsdiscussdev2015-enfr",
        "newsdiscusstest2015-enfr",
        "newssyscomb2009",
        "news-test2008",
        "newstest2009",
        "newstest2010",
        "newstest2011",
        "newstest2012",
        "newstest2013",
        "newstest2014-fren",
        "Tatoeba",
    ]

    def __init__(self, dataset_path: Optional[str]) -> None:
        super().__init__(dataset_path, "fr", "en")


class FrEsDataset(Dataset):
    BENCHMARK_NAMES = [
        "newssyscomb2009",
        "news-test2008",
        "newstest2009",
        "newstest2010",
        "newstest2011",
        "newstest2012",
        "newstest2013",
        "Tatoeba",
    ]

    def __init__(self, dataset_path: Optional[str]) -> None:
        super().__init__(dataset_path, "fr", "es")


class ItEnDataset(Dataset):
    BENCHMARK_NAMES = [
        "newssyscomb2009",
        "newstest2009",
        "Tatoeba",
    ]

    def __init__(self, dataset_path: Optional[str]) -> None:
        super().__init__(dataset_path, "it", "en")
