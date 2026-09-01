# Task 1, Financial AI

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dinojan9901/llm-financial-research-agents/blob/main/task1_financial/notebook.ipynb)

LLM-powered equity research assistant.

## Notebook

`notebook.ipynb` runs the full pipeline end to end with outputs left visible.

## Modules

| File | Covers |
|---|---|
| `src/data.py` | OHLCV fetch, two years minimum, relative dates only |
| `src/indicators.py` | SMA50, SMA200, RSI-14, MACD 12/26/9, Bollinger 20/2, all from first principles |
| `src/news.py` | Headline retrieval with fallback sources |
| `src/summary.py` | Summary dictionary and momentum signal |
| `src/prompts.py` | Prompt templates, kept out of business logic |
| `src/analysis.py` | Per-headline sentiment, aggregation, Buy/Hold/Sell reasoning |
| `src/report.py` | Markdown and HTML brief with a matplotlib chart |

No TA-Lib. Every indicator is implemented directly, RSI using Wilder smoothing rather than a simple mean.

## Run

```powershell
python -m task1_financial.src.pipeline --ticker NVDA
```
