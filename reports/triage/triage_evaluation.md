# Triage agent evaluation

Model: `claude-opus-5-5` · cases: 60 · fraud share among them: 8.3%. Synthetic data, illustrative costs.

| Strategy | Fraud + friction cost | Per case | Sent to analysts | Accuracy when automated | Fraud approved | Good customers declined |
|---|---|---|---|---|---|---|
| escalate_all (today: every case to an analyst) | $361.69 | $6.03 | 100% | n/a | 0 | 0 |
| model_only (decline if p >= 0.5, else approve) | $819.99 | $13.67 | 0% | 91.7% | 3 | 2 |
| agent | $301.69 | $5.03 | 75% | 100.0% | 0 | 0 |
| oracle (perfect information, lower bound) | $0.00 | $0.00 | 0% | 100.0% | 0 | 0 |

| Agent metric | Value |
|---|---|
| errors | 0 |
| error kinds | [] |
| pattern accuracy on fraud | 0.8 |
| evidence from tools actually called | 1.0 |
| mean tool calls | 5.45 |
| latency p50 s | 13.84 |
| latency p90 s | 18.19 |
| llm cost total usd | 2.8605 |
| llm cost per case usd | 0.0477 |
| net saving vs escalate all usd | 57.14 |

Costs exclude analyst salaries beyond the per-review cost in the config. LLM cost uses list prices in `rde/triage/llm.py`.
