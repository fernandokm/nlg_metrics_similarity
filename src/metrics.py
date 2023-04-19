import base64
import functools
import hashlib
import importlib
import json
import os
import warnings
from collections import Counter, defaultdict
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import (
    Callable,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

import evaluate
import numpy as np
import pandas as pd
import transformers
from tqdm import tqdm, trange

DATA_DIR = Path(__file__).parent.parent / "data" / "metrics"
DEFAULT_MODEL = "distilbert-base-multilingual-cased"
DEFAULT_BATCH_SIZE = 256

# only for python 3.8 compatibility
def removeprefix(s: str, prefix: str) -> str:
    if s.startswith(prefix):
        return s[len(prefix) :]
    return s


@lru_cache
def get_model_hidden_layers(model_name):
    model = transformers.pipeline(model=model_name)

    return model.model.config.num_hidden_layers


@contextmanager
def disable_tqdm_and_catch_warnings():
    tqdm_init = tqdm.__init__
    warning_ctx = warnings.catch_warnings()
    try:
        warning_ctx.__enter__()
        warnings.filterwarnings("ignore", module=r"ot\.bregman")

        def new_init(*args, **kwargs):
            kwargs["disable"] = True
            return tqdm_init(*args, **kwargs)

        tqdm.__init__ = new_init
        yield
    finally:
        warning_ctx.__exit__(None, None, None)
        tqdm.__init__ = tqdm_init


def cache_metric_to_disk(fn: Callable[[List[str], List[List[str]]], List[float]]):
    """Enables caching for a metric function.

    This decorator allows a metric function to be cached to disk
    (to the directory data/metrics), so it only has to be computed once
    for any given arguments.
    """

    @functools.wraps(fn)
    def wrapper(
        candidates: List[str],
        references: List[List[str]],
        **kwargs,
    ) -> List[float]:
        # The function is only recomputed if any of the following elements change
        key_dict = {
            "candidates": candidates,
            "references": references,
        }
        key_dict["model"] = kwargs.get("model", DEFAULT_MODEL)
        key_hasher = hashlib.sha256(json.dumps(key_dict).encode("utf-8"))
        key = base64.urlsafe_b64encode(key_hasher.digest()).decode("utf-8")
        cache_file = DATA_DIR / f"{fn.__module__}.{fn.__qualname__}.{key}.json"

        DATA_DIR.mkdir(exist_ok=True, parents=True)

        if cache_file.exists():
            return json.load(cache_file.open("r"))

        result = fn(candidates, references, **kwargs)
        json.dump(result, cache_file.open("w"))

        return result

    return wrapper


T = TypeVar("T")
U = TypeVar("U")


def tqdm_zip(it0: Sequence[T], it1: Sequence[U], *, desc: str) -> Iterable[Tuple[T, U]]:
    return tqdm(zip(it0, it1), desc=desc, total=min(len(it0), len(it1)))


def evaluate_bleu(candidates: List[str], references: List[List[str]]) -> List[float]:
    """
    Compute bleu metric for a batch of sentences.
    params:
        - cands : List[str]
        - refs List[List[str]]

    output:
        - scores : List[float]
    """
    scores = []
    metric = evaluate.load("bleu")
    for pred, refs in tqdm_zip(candidates, references, desc="evaluate_bleu"):
        pred = [pred]
        refs = [refs]
        result = metric.compute(predictions=pred, references=refs)
        assert result is not None
        scores.append(result["bleu"])

    return scores


def evaluate_chrf(candidates: List[str], references: List[List[str]]) -> List[float]:
    """
    Compute chrf metric for a batch of sentences.
    params:
        - cands : List[str]
        - refs List[List[str]]

    output:
        - scores : List[float]
    """
    scores = []
    metric = evaluate.load("chrf")
    for pred, refs in tqdm_zip(candidates, references, desc="evaluate_chrf"):
        pred = [pred]
        refs = [refs]
        result = metric.compute(predictions=pred, references=refs)
        assert result is not None
        scores.append(result["score"])

    return scores


def evaluate_rouge(candidates: List[str], references: List[List[str]]) -> List[float]:
    """
    Compute rouge metric for a batch of sentences.
    params:
        - cands : List[str]
        - refs List[List[str]]

    output:
        - scores : List[float]
    """
    print("evaluate_rouge: running...")
    metric = evaluate.load("rouge")
    result = metric.compute(
        predictions=candidates, references=references, use_aggregator=False
    )
    assert result is not None

    return result["rouge1"]


def evaluate_ter(candidates: List[str], references: List[List[str]]) -> List[float]:
    """
    Compute ter metric for a batch of sentences.
    params:
        - cands : List[str]
        - refs List[List[str]]

    output:
        - scores : List[float]
    """
    scores = []
    metric = evaluate.load("ter")
    for pred, refs in tqdm_zip(candidates, references, desc="evaluate_ter"):
        pred = [pred]
        refs = [refs]
        result = metric.compute(predictions=pred, references=refs)
        assert result is not None
        scores.append(result["score"])

    return scores


def evaluate_meteor(candidates: List[str], references: List[List[str]]) -> List[float]:
    """
    Compute meteor metric for a batch of sentences.
    params:
        - cands : List[str]
        - refs List[List[str]]

    output:
        - scores : List[float]
    """
    scores = []
    metric = evaluate.load("meteor")
    for pred, refs in tqdm_zip(candidates, references, desc="evaluate_meteor"):
        pred = [pred]
        refs = [refs]
        result = metric.compute(predictions=pred, references=refs)
        assert result is not None
        scores.append(result["meteor"])

    return scores


def patch_moverscore_v2(model: str):
    # The moverscore_v2 model is loaded when the module is being imported
    # If moverscore_v2 was previously imported with a different model,
    # we have to forcibly reload it.
    os.environ["MOVERSCORE_MODEL"] = model
    import moverscore_v2

    if moverscore_v2.model_name != model:
        importlib.reload(moverscore_v2)

    def bert_encode(model, x, attention_mask):
        model.eval()
        with moverscore_v2.torch.no_grad():
            result = model(x, attention_mask=attention_mask)
        if moverscore_v2.model_name in (
            "distilbert-base-uncased",
            "distilbert-base-multilingual-cased",
        ):
            return result[1]
        else:
            return result[2]

    moverscore_v2.bert_encode = bert_encode


def flatten_input(
    candidates: List[str],
    references: List[List[str]],
) -> Tuple[List[str], List[str], List[Tuple[int, int]]]:
    """Flattens a set of (candidate, list of references) pairs.

    Args:
        candidates (List[str]): a list of candidate sentences
        references (List[List[str]]): a list containing, for each candidate sentence,
            a list of reference sentences

    Returns: (candidates_flat, references_flat, boundaries)
        candidates_flat (List[str]): a list of candidate sentences.
            Each candidate sentence is repeated once per reference.
        references_flat (List[str]): a list of corresponding reference
            sentences.
        boundaries (List[Tuple[int, int]]): a list of (start, end) pairs
            indicating the indices where each candidate sentence starts
            and ends in candidates_flat.
    """
    candidates_flat = []
    references_flat = []
    boundaries = []
    for cand, refs in zip(candidates, references):
        start = len(candidates_flat)
        end = start + len(refs)
        boundaries.append((start, end))
        candidates_flat += [cand] * len(refs)
        references_flat += refs
    return candidates_flat, references_flat, boundaries


def _run_batched(
    candidates_flat: List[str],
    references_flat: List[str],
    boundaries: List[Tuple[int, int]],
    run_batch: Callable[[List[str], List[str]], List[float]],
    desc: str,
    batch_size: int,
    agg: Callable[[List[float]], float] = max,
) -> List[float]:
    """Batches the input and evaluates the specified metric"""
    flat_scores = []
    batch_iter = trange(0, len(references_flat), batch_size, unit="batch", desc=desc)
    for batch_start in batch_iter:
        flat_scores += run_batch(
            references_flat[batch_start : batch_start + batch_size],
            candidates_flat[batch_start : batch_start + batch_size],
        )

    scores = []
    for start, end in boundaries:
        scores.append(agg(flat_scores[start:end]))

    return scores


def evaluate_moverscore(
    candidates: List[str],
    references: List[List[str]],
    model: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> List[float]:
    """
    Compute moverscore metric for a batch of sentences.
    params:
        - cands : List[str]
        - refs List[List[str]]

    output:
        - scores : List[float]
    """
    if model is None:
        model = DEFAULT_MODEL

    patch_moverscore_v2(model)
    from moverscore_v2 import get_idf_dict, word_mover_score

    candidates_flat, references_flat, boundaries = flatten_input(candidates, references)

    idf_dict_hyp = get_idf_dict(candidates_flat)
    idf_dict_ref = get_idf_dict(references_flat)

    return _run_batched(
        candidates_flat,
        references_flat,
        boundaries,
        run_batch=lambda batch_candidates, batch_references: word_mover_score(
            batch_candidates,
            batch_references,
            idf_dict_ref,
            idf_dict_hyp,
            stop_words=[],
            n_gram=1,
            remove_subwords=False,
        ),
        desc=f"evaluate_moverscore({model})",
        batch_size=batch_size,
    )


def evaluate_bertscore(
    candidates: List[str],
    references: List[List[str]],
    model: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> List[float]:
    """
    Compute bertscore metric for a batch of sentences.
    params:
        - cands : List[str]
        - refs List[List[str]]

    output:
        - scores : List[float]
    """
    from bert_score import score

    if model is None:
        model = DEFAULT_MODEL

    precision, recall, f1 = score(
        candidates,
        references,
        model_type=model,
        verbose=True,
        batch_size=batch_size,
    )
    return f1.numpy().tolist()  # type: ignore


def faster_ref_list_to_idf(input_refs):
    """
    A faster version of BaryScoreMetric.ref_list_to_idf
    """
    idf_count = Counter()
    num_docs = len(input_refs)

    # We use the following instead of:
    #     idf_count.update(sum([list(set(i)) for i in input_refs], []))
    for i in input_refs:
        idf_count.update(set(i))

    idf_dict = defaultdict(lambda: np.log((num_docs + 1) / (1)))
    idf_dict.update(
        {idx: np.log((num_docs + 1) / (c + 1)) for (idx, c) in idf_count.items()}
    )
    return idf_dict


def evaluate_baryscore(
    candidates: List[str],
    references: List[List[str]],
    model: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> List[float]:
    """
    Compute Baryscore for a batch of sentences
    (compute for each reference and take the best score).
    params:
        - cands : List[str]
        - refs List[List[str]]

    output:
        - scores : List[float]
    """
    from nlg_eval_via_simi_measures import BaryScoreMetric

    if model is None:
        model = DEFAULT_MODEL

    candidates_flat, references_flat, boundaries = flatten_input(candidates, references)

    metric_call = BaryScoreMetric(model_name=model)
    metric_call.ref_list_to_idf = faster_ref_list_to_idf
    metric_call.prepare_idfs(candidates_flat, references_flat)

    def run_batch(batch_candidates: List[str], batch_references: List[str]):
        with disable_tqdm_and_catch_warnings():
            result = metric_call.evaluate_batch(
                batch_candidates,
                batch_references,
            )
            return result["baryscore_W"]

    return _run_batched(
        candidates_flat,
        references_flat,
        boundaries,
        run_batch,
        desc=f"evaluate_baryscore({model})",
        batch_size=batch_size,
        agg=min,
    )


def evaluate_depthscore(
    candidates: List[str],
    references: List[List[str]],
    model: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> List[float]:
    """
    Compute DepthScore for a batch of sentences
    (compute for each reference and take the best score).
    params:
        - cands : List[str]
        - refs List[List[str]]

    output:
        - scores : List[float]
    """
    from nlg_eval_via_simi_measures import DepthScoreMetric

    if model is None:
        model = DEFAULT_MODEL
    hidden_layers = get_model_hidden_layers(model)

    candidates_flat, references_flat, boundaries = flatten_input(candidates, references)

    metric_call = DepthScoreMetric(model_name=model, layers_to_consider=hidden_layers)
    metric_call.prepare_idfs(candidates_flat, references_flat)

    def run_batch(batch_candidates: List[str], batch_references: List[str]):
        with disable_tqdm_and_catch_warnings():
            result = metric_call.evaluate_batch(
                batch_candidates,
                batch_references,
            )
            return result["depth_score"]

    return _run_batched(
        candidates_flat,
        references_flat,
        boundaries,
        run_batch,
        desc=f"evaluate_depthscore({model})",
        batch_size=batch_size,
    )


DEFAULT_EVAL_WITHOUT_MODEL = [
    evaluate_bleu,
    evaluate_chrf,
    evaluate_rouge,
    evaluate_ter,
    evaluate_meteor,
]

DEFAULT_EVAL_WITH_MODEL = [
    evaluate_moverscore,
    evaluate_bertscore,
    evaluate_baryscore,
]


def evaluate_all(
    candidates: List[str],
    references: List[List[str]],
    without_model: List[Callable] = DEFAULT_EVAL_WITHOUT_MODEL,
    with_model: List[Callable] = DEFAULT_EVAL_WITH_MODEL,
    models: List[str] = [DEFAULT_MODEL],
    batch_size: int = DEFAULT_BATCH_SIZE,
    cache: bool = True,
) -> pd.DataFrame:
    metrics = {}

    for metric_fn in without_model:
        name = removeprefix(metric_fn.__name__, "evaluate_")
        if cache:
            metric_fn = cache_metric_to_disk(metric_fn)
        metrics[name] = metric_fn(candidates, references)

    for metric_fn in with_model:
        for model in models:
            name = removeprefix(metric_fn.__name__, "evaluate_")
            if len(models) > 1:
                name += "." + model
            if cache:
                metric_fn = cache_metric_to_disk(metric_fn)
            metrics[name] = metric_fn(
                candidates, references, model=model, batch_size=batch_size
            )

    return pd.DataFrame(metrics)
