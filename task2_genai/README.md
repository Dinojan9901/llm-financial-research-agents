# Task 2, Generative AI

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
| Teacher, data generation | Groq `openai/gpt-oss-120b` | Free, strong at structured extraction |
| Student, fine-tuned | a smaller base model, chosen in the notebook | Must differ from the teacher |

The full teacher system prompt is at `prompts/teacher_system_prompt.md`.

## Runs on Colab

Fine-tuning needs a CUDA GPU. See [../docs/SETUP.md](../docs/SETUP.md).

## Model link

Hugging Face: TODO after training.
