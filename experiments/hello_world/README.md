# Hello World — BM25 vs Vector vs Hybrid

Proves that hybrid search (BM25 + Vector with RRF) outperforms either approach alone.

## Run

```bash
PYTHONPATH=src python experiments/hello_world/run.py
```

## What's Inside

- `data/knowledge_base.txt` — 10-chapter AI knowledge base (~19k chars, 40 chunks)
- `data/eval_dataset.json` — 20 queries (10 keyword, 10 semantic) with ground truth
- `run.py` — experiment script
- `reports/` — generated HTML + JSON reports

## Results

| Experiment | F1@5 | MRR | MAP |
|---|---|---|---|
| BM25 Keyword | 38.9% | 85.8% | 63.4% |
| Vector Semantic | 46.9% | 87.2% | 70.1% |
| **Hybrid (BM25+Vector)** | **47.3%** | **87.9%** | **70.1%** |

Hybrid achieves +21.5% F1 lift over BM25 and +0.9% over Vector alone.

## Create Your Own

Duplicate this folder, replace the data, and customize `run.py`. Or start from `experiments/template.py`.
