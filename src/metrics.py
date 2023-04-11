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
    bleu = evaluate.load("bleu")
    for pred, refs in zip(candidates, references):
        pred = [pred]
        refs = [refs]
        result = bleu.compute(predictions=pred, references=refs)
        scores.append(result['bleu'])

    return scores


