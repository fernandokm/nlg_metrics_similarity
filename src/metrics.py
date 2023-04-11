import evaluate
import numpy as np

def evaluate_bleu(candidates : list[list[str]], references : list[str]):
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
        scores.append(result['bleu'])

    return scores


def evaluate_chrf(candidates : list[list[str]], references : list[str]):
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
        scores.append(result['score'])

    return scores


def evaluate_rouge(candidates : list[list[str]], references : list[str]):
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
        scores.append(result['rouge1'])

    return scores


def evaluate_ter(candidates : list[list[str]], references : list[str]):
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
        scores.append(result['score'])

    return scores

def evaluate_meteor(candidates : list[list[str]], references : list[str]):
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
        scores.append(result['meteor'])

    return scores


def evaluate_moverscore(candidates : list[list[str]], references : list[str]):
    """
    Compute moverscore metric for a batch of sentences.
    params:
        - cands : List[str]
        - refs List[List[str]]
    
    output:
        - scores : List[float]
    """
    from moverscore import word_mover_score, defaultdict
    list_scores = []
    
    for pred, refs in zip(candidates, references):

        idf_dict_hyp = defaultdict(lambda: 1.)
        idf_dict_ref = defaultdict(lambda: 1.)
    
        pred = [pred] * len(refs)
    
        sentence_score = 0 

        scores = word_mover_score(refs, pred, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=False)
    
        sentence_score = np.mean(scores)

        list_scores.append(sentence_score)

    return list_scores



