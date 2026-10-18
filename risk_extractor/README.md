# Risk Extractor (Generative AI)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dinojan9901/llm-financial-research-agents/blob/main/risk_extractor/notebook.ipynb)

QLoRA fine-tuning pipeline for financial risk clause extraction.

## Use case

**Input:** a passage of filing-style prose from a company disclosure or risk section.
**Output:** structured JSON listing each distinct risk factor with its category, the trigger described, and the stated or implied impact.
**Correct:** every risk present in the passage is extracted once, categorised from a fixed taxonomy, and no risk is invented.
**Incorrect:** a hallucinated risk, a missed risk, a category outside the taxonomy, or malformed JSON.

This is deliberately not a general chatbot task. A closed taxonomy and an explicit definition of correct output make the fine-tuned model's behaviour measurable.

## Teacher and student

| Role | Model | Reason |
|---|---|---|
| Teacher, data generation | `minimax/minimax-m3:free` via OpenRouter | Strong at structured extraction, 1M context |
| Student, fine-tuned | `microsoft/Phi-3-mini-4k-instruct` | 3.8B fits a T4 under 4-bit, ungated so no licence wait |

The teacher was originally planned as Groq `openai/gpt-oss-120b`. Groq's free tier caps at
200,000 tokens per day and was exhausted when generation ran, so the client failed over to
OpenRouter and MiniMax-M3 produced all 120 examples. Each record carries a
`teacher_provider` field, so the provenance is in the data rather than only in this note.

Teacher and student are different model families.

The full teacher system prompt is at [prompts/teacher_system_prompt.md](prompts/teacher_system_prompt.md)
and is loaded from that file at generation time, so the documented copy cannot drift from
the one actually used.

## Dataset

Generated with `python -m risk_extractor.src.generate_dataset --target 120`.

| | Value |
|---|---|
| Examples | 120 |
| Teacher calls | 30, batched 4 per call |
| Failed calls | 0 |
| Near-duplicates rejected | 0 |
| Risk items | 289, mean 2.41 per example |

Batching matters on a free tier: one call per example would have cost 120 calls, and
generation would not have finished inside a daily allowance.

### Diversity

Run `python -m risk_extractor.src.diversity`. All nine checks pass.

| Measure | Value |
|---|---|
| Passage length | mean 153.7 words, std 30.6, range 70 to 246 |
| Pairwise similarity | mean 0.0001, p95 0.0, max 0.018 |
| Pairs above 0.45 overlap | 0 |
| Vocabulary | 11,320 tokens, 3,261 unique, type-token ratio 0.288 |
| Categories present | 12 of 12, largest share 13.8 percent |
| Sectors | 15 |
| Document styles | 4, evenly distributed |

Diversity is engineered rather than hoped for. Seeds traverse the sector-by-style grid
diagonally, which covers every sector and every style within 30 calls without repeating a
pair, and a Jaccard shingle check rejects near-duplicate passages before they enter the
dataset.

### Split

`python -m risk_extractor.src.format_split`, stratified by document style.

| Split | Examples | Risk items | Categories | Ratio |
|---|---|---|---|---|
| train | 96 | 230 | 12 of 12 | 80% |
| val | 12 | 28 | 9 of 12 | 10% |
| test | 12 | 31 | 11 of 12 | 10% |

Stratifying by document style matters at this size: a random split of 120 can leave a
style out of the 12-example test set entirely, which would make the held-out score a
measurement of luck rather than of the model.

Examples are stored as a `messages` list rather than a pre-rendered string. The chat
template belongs to the tokenizer, so rendering it into the file would bake one model's
special tokens into the dataset.

## Training

QLoRA on a Colab T4: Phi-3-mini loaded in 4-bit NF4 with double quantisation, with
25.2M trainable LoRA parameters. An initial 3-epoch run overfit (validation loss rose at
epoch 3 while training loss kept falling), so the final run uses 2 epochs. It took 6.2
minutes and peaked at 11.89 GB of GPU memory. Every hyperparameter is justified in the
notebook.

| Epoch | Train loss | Val loss |
|---|---|---|
| 1 | 0.7836 | 0.6284 |
| 2 | 0.5728 | 0.6098 |

Validation loss fell from 0.628 to 0.610. The adapters were then merged into a freshly
loaded fp16 base, so the published model needs no PEFT dependency.

## Results

Both models were scored on the same 12 held-out test examples with the same decoding
settings. The base model gets a longer system prompt containing the full category
definitions, so the comparison is not against a strawman.

| Metric | Base | Fine-tuned |
|---|---|---|
| ROUGE-1 | 0.5359 | 0.6383 |
| ROUGE-2 | 0.2808 | 0.3716 |
| ROUGE-L | 0.3588 | 0.4710 |
| BERTScore F1, baseline-rescaled | 0.0334 | 0.5516 |
| Parses as JSON | 6/12 | 12/12 |
| Validates against the schema | 1/12 | 12/12 |
| Categories outside the taxonomy | 2 | 0 |

LLM-as-judge, scored 1 to 5 by `openai/gpt-oss-120b` on Groq. The judge sees the passage,
the reference and the candidate, but not which model produced the candidate.

| Model | Coverage | Precision | Category accuracy | Format | Overall |
|---|---|---|---|---|---|
| Base | 2.75 | 3.25 | 2.17 | 3.00 | 2.17 |
| Fine-tuned | 4.00 | 4.25 | 3.50 | 5.00 | 3.42 |

Hallucination rate of the fine-tuned model, from the judge's verdict on each response:
2 correct, 7 partially correct, 3 hallucinated, so **3/12 (25%)**. Every case is printed
in full in the notebook so the labels can be checked by hand.

**How far these numbers go.** The test set is 12 examples. One response changing label
moves the hallucination rate by 8.3 points, and no confidence intervals are reported.
The test passages and their reference extractions come from the same synthetic teacher
as the training data, so the scores measure agreement with that teacher rather than
with human-annotated filings. Read together, the results show that fine-tuning fixed the
output format reliably (12/12 valid against 1/12) but did not fix semantic accuracy:
category confusion is the weakest dimension, and a quarter of responses contain a risk
the passage does not support.

### RAG fallback

When the fine-tuned model's perplexity over its own output is high, the passage is
re-queried with the two most similar training examples retrieved from a ChromaDB store
(96 passages, `all-MiniLM-L6-v2` embeddings). The notebook shows this on the
highest-perplexity test case. It is a demonstration, not an evaluated component: the
threshold of 3.0 sits below every observed test perplexity (4.15 to 8.33), so as
configured it would fire on every input, and in the case shown the retrieval-augmented
output added a concentration-risk entry that the reference does not contain.

## Model

Merged model on Hugging Face:
[Dino21/financial-risk-extractor-phi3-qlora](https://huggingface.co/Dino21/financial-risk-extractor-phi3-qlora)

## Runs on Colab

Fine-tuning needs a CUDA GPU. See [../docs/SETUP.md](../docs/SETUP.md) and
[../docs/RISK_EXTRACTOR_COLAB.md](../docs/RISK_EXTRACTOR_COLAB.md).
