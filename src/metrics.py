from functools import cache
from typing import Optional

import evaluate
import numpy as np
import transformers

DEFAULT_MODEL = "distilbert-base-multilingual-cased"
hidden_layers = None


@cache
def get_model_hidden_layers(model_name):
    model = transformers.pipeline(model=model_name)

    return model.model.config.num_hidden_layers


def evaluate_bleu(candidates: list[str], references: list[list[str]]) -> list[str]:
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
    for pred, refs in zip(candidates, references):
        pred = [pred]
        refs = [refs]
        result = metric.compute(predictions=pred, references=refs)
        assert result is not None
        scores.append(result["bleu"])

    return scores


def evaluate_chrf(candidates: list[str], references: list[list[str]]) -> list[str]:
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
    for pred, refs in zip(candidates, references):
        pred = [pred]
        refs = [refs]
        result = metric.compute(predictions=pred, references=refs)
        assert result is not None
        scores.append(result["score"])

    return scores


def evaluate_rouge(candidates: list[str], references: list[list[str]]) -> list[str]:
    """
    Compute rouge metric for a batch of sentences.
    params:
        - cands : List[str]
        - refs List[List[str]]

    output:
        - scores : List[float]
    """
    scores = []
    metric = evaluate.load("rouge")
    for pred, refs in zip(candidates, references):
        pred = [pred]
        refs = [refs]
        result = metric.compute(predictions=pred, references=refs)
        assert result is not None
        scores.append(result["rouge1"])

    return scores


def evaluate_ter(candidates: list[str], references: list[list[str]]) -> list[str]:
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
    for pred, refs in zip(candidates, references):
        pred = [pred]
        refs = [refs]
        result = metric.compute(predictions=pred, references=refs)
        assert result is not None
        scores.append(result["score"])

    return scores


def evaluate_meteor(candidates: list[str], references: list[list[str]]) -> list[str]:
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
    for pred, refs in zip(candidates, references):
        pred = [pred]
        refs = [refs]
        result = metric.compute(predictions=pred, references=refs)
        assert result is not None
        scores.append(result["meteor"])

    return scores


def evaluate_moverscore(
    candidates: list[str], references: list[list[str]], model: Optional[str] = None
) -> list[str]:
    """
    Compute moverscore metric for a batch of sentences.
    params:
        - cands : List[str]
        - refs List[List[str]]

    output:
        - scores : List[float]
    """
    from moverscore import defaultdict, word_mover_score

    if model is None:
        model = DEFAULT_MODEL

    list_scores = []

    for pred, refs in zip(candidates, references):
        idf_dict_hyp = defaultdict(lambda: 1.0)
        idf_dict_ref = defaultdict(lambda: 1.0)

        pred = [pred] * len(refs)

        sentence_score = 0

        scores = word_mover_score(
            refs,
            pred,
            idf_dict_ref,
            idf_dict_hyp,
            stop_words=[],
            n_gram=1,
            remove_subwords=False,
        )

        sentence_score = np.mean(scores)

        list_scores.append(sentence_score)

    return list_scores


def evaluate_bertscore(
    candidates: list[str], references: list[list[str]], model: Optional[str] = None
) -> list[str]:
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

    list_scores = []

    for pred, refs in zip(candidates, references):
        pred = [pred]
        refs = [refs]
        sentence_score = score(pred, refs, model_type=model)
        list_scores.append(sentence_score[2].numpy()[0])  # type: ignore

    return list_scores


def evaluate_baryscore(
    candidates: list[str], references: list[list[str]], model: Optional[str] = None
) -> list[str]:
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

    list_scores = []
    metric_call = BaryScoreMetric(model_name=model)
    for pred, refs in zip(candidates, references):
        pred = [pred]
        sentence_scores = []
        for ref in refs:
            ref = [ref]
            metric_call.prepare_idfs(ref, pred)
            sentence_score = metric_call.evaluate_batch(pred, ref)
            sentence_scores.append(sentence_score["baryscore_W"][0])

        list_scores.append(min(sentence_scores))

    return list_scores


def evaluate_depthscore(
    candidates: list[str], references: list[list[str]], model: Optional[str] = None
) -> list[str]:
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

    list_scores = []
    metric_call = DepthScoreMetric(model_name=model, layers_to_consider=hidden_layers)

    for pred, refs in zip(candidates, references):
        pred = [pred]
        sentence_scores = []
        for ref in refs:
            ref = [ref]
            metric_call.prepare_idfs(ref, pred)
            sentence_score = metric_call.evaluate_batch(pred, ref)
            sentence_scores.append(sentence_score["depth_score"][0])

        list_scores.append(min(sentence_scores))

    return list_scores
