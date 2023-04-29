# Evaluating Automatic Evaluation Metrics for Machine Translation

This is the source code of the article [Evaluating Automatic Evaluation Metrics for Machine Translation](https://openreview.net/forum?id=EnnsT1uA7D5).

Authors:

- Fernando Kurike Matsumoto
- Victor Felipe Domingues do Amaral

## Installation

To install the dependencies use pip and the requirements.txt in this directory. e.g.

```bash
pip install -r requirements.txt
```

## Files

- `src/metrics.py`: includes the necessary functions to implement an interface to the metrics libraries, compute metric values, and cache them to disk;
- `src/data_utils.py`: Set of functions that can be utilized to load cached results of the metrics;
- `metrics_evaluation.ipynb`: Primary document of the project, which presents all the analyses and plots that were generated during our work.
