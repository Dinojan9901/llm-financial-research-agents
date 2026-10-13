# LLM Financial Research Agents

Three LLM systems for financial research: an equity research assistant that reasons over technical indicators and news sentiment, a QLoRA fine-tuned model that extracts structured risk factors from disclosure text, and a two-agent research pipeline with tool restriction, memory and tracing.

| Task | Domain | Deliverable | Folder | Notebook |
|---|---|---|---|---|
| 1 | Financial AI | LLM-powered equity research assistant | [task1_financial/](task1_financial/) | [notebook](task1_financial/notebook.ipynb) |
| 2 | Generative AI | QLoRA fine-tuning, financial risk clause extraction | [task2_genai/](task2_genai/) | [notebook](task2_genai/notebook.ipynb) |
| 3 | Agentic Workflows | Multi-agent research system with memory and tracing | [task3_agentic/](task3_agentic/) | [notebook](task3_agentic/notebook.ipynb) |

## Task 2 results

Phi-3-mini fine-tuned with QLoRA to extract risk factors as JSON from a closed
twelve-category taxonomy, compared against the untuned base model on the same 12 held-out
examples. Validation loss fell from 0.628 to 0.610 over two epochs.

| Metric | Base | Fine-tuned |
|---|---|---|
| ROUGE-1 | 0.5359 | 0.6383 |
| ROUGE-2 | 0.2808 | 0.3716 |
| ROUGE-L | 0.3588 | 0.4710 |
| BERTScore F1, baseline-rescaled | 0.0334 | 0.5516 |
| Valid JSON, valid against schema | 6/12, 1/12 | 12/12, 12/12 |
| Categories outside the taxonomy | 2 | 0 |
| LLM judge overall, 1 to 5 | 2.17 | 3.42 |
| LLM judge category accuracy, 1 to 5 | 2.17 | 3.50 |
| Hallucinated responses | not measured | 3/12 |

The test set is small: 12 synthetic examples, so one response changing moves the
hallucination rate by 8.3 points. Fine-tuning reliably fixed the output format, but not
semantic accuracy. Full results and caveats are in [task2_genai/README.md](task2_genai/README.md).

Model: [Dino21/financial-risk-extractor-phi3-qlora](https://huggingface.co/Dino21/financial-risk-extractor-phi3-qlora)

## Tests

All tests run offline with no API key and no network access.

```powershell
python -m task1_financial.tests.test_indicators   # verified against Wilder's published RSI figures
python -m task1_financial.tests.test_news
python -m task1_financial.tests.test_analysis
python -m task1_financial.tests.test_report
python -m task3_agentic.tests.test_agentic
```

## Repository layout

```
common/                 shared config, LLM client, Pydantic schemas
task1_financial/        data pipeline, indicators, LLM reasoning, report
task2_genai/            dataset generation, QLoRA training, evaluation
task3_agentic/          tools, single agent, two-agent pipeline, trace logs
docs/SETUP.md           environment setup and troubleshooting
REFLECTION.md           architectural decisions and limitations
```

`common/` is shared deliberately. Task 1 computes the indicators and sentiment that Task 3 exposes as agent tools, so duplicating that logic across folders would let the two drift apart.

## Quick start

```bash
git clone https://github.com/Dinojan9901/llm-financial-research-agents.git
cd llm-financial-research-agents
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env          # then add your keys
```

Get a free Groq key at [console.groq.com](https://console.groq.com/keys). An OpenRouter key is optional and acts as an automatic fallback when Groq rate-limits.

Full instructions, including the Colab path for Task 2, are in [docs/SETUP.md](docs/SETUP.md).

## Model providers

Inference runs on Groq (`openai/gpt-oss-120b`) with OpenRouter as a fallback provider. `common/llm.py` tries Groq first and moves to OpenRouter only when Groq raises, which keeps the multi-agent runs in Task 3 from dying on a single rate-limit response.

Task 2 uses a large **teacher** model for synthetic data generation and fine-tunes a **different, smaller student** model. The teacher was planned as Groq GPT-OSS-120B; Groq's daily token cap was exhausted when generation ran, so the client failed over and MiniMax-M3 via OpenRouter produced all 120 examples. See [task2_genai/README.md](task2_genai/README.md).

## Secrets

No keys are committed. `.env` is gitignored and only `.env.example` is tracked. Every module reads credentials through `common/config.py`.

## Notebooks

Cell outputs are intentionally left in place, so the results of each run can be read without re-executing anything.
