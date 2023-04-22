import csv
import re
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"


URL_DIRECT_ASSESMENT = "https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/wmt/2020-da.csv.tar.gz"  # noqa
URL_MQM_BASE = "https://raw.githubusercontent.com/google/wmt-mqm-human-evaluation/7ea75a4a431f8dc74ca2752fc4d728a1214aadfe/newstest2020/ende"  # noqa
URL_MQM_ENDE = URL_MQM_BASE + "/mqm_newstest2020_ende.no-postedits.tsv"
URL_MQM_ENDE_SCORES = URL_MQM_BASE + "/mqm_newstest2020_ende.avg_seg_scores.tsv"


def _cache_url(url: str) -> Path:
    """Loads a url and caches its content on disk (if not already cached).

    Args:
        url (str): the url to be cached

    Returns:
        Path: the path to the cached data
    """
    _, filename = url.rsplit("/", maxsplit=1)
    output_file = DATA_DIR / filename

    DATA_DIR.mkdir(exist_ok=True)

    if not output_file.exists():
        # Create a progress bar and download the file
        with tqdm(unit="B", unit_scale=True, unit_divisor=1024, desc=filename) as pbar:

            def reporthook(chunks: int, chunk_size: int, total: Optional[int]):
                if total is not None:
                    pbar.total = total
                pbar.update(chunks * chunk_size - pbar.n)

            urllib.request.urlretrieve(url, str(output_file), reporthook=reporthook)
    return output_file


def clean_text(text: str) -> str:
    """Removes html tags and leading/trailing spaces from a sentence."""
    # Remove html tags
    # NOTE: the <v> tag is used to highlight errors in the
    text = re.sub(r"</?\w+>", "", text)

    # Remove leading and trailing spaces
    text = text.strip()

    return text


def load_da() -> pd.DataFrame:
    """Loads the direct assesment data"""
    da_gzip_path = _cache_url(URL_DIRECT_ASSESMENT)
    with tarfile.open(da_gzip_path, "r:gz") as da_file:
        return pd.read_csv(da_file.extractfile("2020-da.csv"))  # type: ignore


def load_mqm() -> pd.DataFrame:
    """Loads the mqm data"""
    mqm_path = _cache_url(URL_MQM_ENDE)
    return pd.read_csv(mqm_path, sep="\t", quoting=csv.QUOTE_NONE)


def load_mqm_scores() -> pd.DataFrame:
    """Loads the mqm scores"""
    mqm_scores_path = _cache_url(URL_MQM_ENDE_SCORES)
    return pd.read_csv(mqm_scores_path, sep=" ", quoting=csv.QUOTE_NONE)


def load_data() -> pd.DataFrame:
    cache_file = DATA_DIR / "aggregated.csv"

    if cache_file.exists():
        return pd.read_csv(str(cache_file))

    da = load_da()
    mqm = load_mqm()
    mqm_scores = load_mqm_scores()

    # Extract only the text data from mqm
    data = mqm[["system", "seg_id", "source", "target"]].copy()
    data.target = data.target.map(clean_text)

    # Drop duplicate rows (each sentence pair is repeated once per annotator
    # and per error found in the target sentence)
    data.drop_duplicates(inplace=True)

    # Each (seg_id, system) pair appears in multiple rows, once for each rater
    # and for each error in the text. The source and target texts should be the
    # same in all of these rows, but this was not true in previous versions
    # of the dataset because reviewers had modified their target texts.
    # Here, we check that this is indeed true for our dataset.
    assert (data[["system", "seg_id"]].value_counts() == 1).all()

    # Add the mqm scores
    data = data.merge(mqm_scores, on=["system", "seg_id"], how="outer")

    # Aggregate the da scores
    da = da[da.lp == "en-de"]
    da = (
        da.groupby(["src", "mt"])
        .apply(lambda g: (g["score"] * g["annotators"]).sum() / g["annotators"].sum())
        .rename("da_score")
        .reset_index()
    )

    # Merge the mqm data with the da data
    da.rename({"src": "source", "mt": "target"}, axis=1, inplace=True)
    data = data.merge(da, on=["source", "target"], how="left")

    # Currently each row contains either a reference or a candidate sentence.
    # Here, we remove the rows with reference sentences and add new columns
    # to contain the reference sentences associated with each candidate sentence.
    ref_mask = data.system.str.startswith("Human-")
    ref_systems = data.system[ref_mask].unique().tolist()
    generated_data = data[~ref_mask]

    for system in sorted(ref_systems):
        name = system.removeprefix("Human-").removesuffix(".0")
        ref_data = data.loc[
            data.system == system, ["seg_id", "target", "mqm_avg_score"]
        ]
        ref_data.rename(
            {"mqm_avg_score": f"ref_{name}_mqm_score", "target": f"ref_{name}"},
            axis=1,
            inplace=True,
        )
        generated_data = generated_data.merge(ref_data, on=["seg_id"], how="left")

    generated_data.to_csv(str(cache_file), index=False)

    return generated_data
