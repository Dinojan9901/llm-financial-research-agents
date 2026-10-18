# Setup

## Local environment (research pipeline and agents)

The research pipeline and agents are network-bound and run comfortably on a modest machine. The risk extractor does not run locally, see the Colab section below.

```powershell
git clone https://github.com/Dinojan9901/llm-financial-research-agents.git
cd llm-financial-research-agents
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks the activate script:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## API keys

```powershell
Copy-Item .env.example .env
notepad .env
```

| Key | Required | Where to get it |
|---|---|---|
| `GROQ_API_KEY` | yes | https://console.groq.com/keys, free, no card |
| `OPENROUTER_API_KEY` | optional | https://openrouter.ai/keys, used only as fallback |
| `HF_TOKEN` | Risk extractor only | https://huggingface.co/settings/tokens, needs write scope to push the model |
| `WANDB_API_KEY` | Risk extractor only | https://wandb.ai/authorize, optional if logging manually |

`.env` is gitignored. Confirm before any push:

```powershell
git check-ignore -v .env
```

## Verify the install

```powershell
python -c "from common.llm import get_client; print(get_client().chat('You are terse.', 'Reply with OK'))"
```

## Low-memory machines

The research pipeline and agents were developed on a 4-core laptop with 8 GB of RAM and no CUDA GPU, so modest hardware is enough.

- Close spare browser tabs before running the notebooks. pandas plus yfinance want roughly 400 to 700 MB of free RAM.
- If Python dies with a MemoryError or Windows warns that virtual memory is low, raise the pagefile: System Properties, Advanced, Performance Settings, Advanced, Virtual Memory, Change. Set a system-managed size on a drive with free space.
- Put the virtual environment on whichever drive has the most free space.
- Do not `pip install torch` locally for the research pipeline and agents. It is a multi-gigabyte download, and without CUDA it cannot run the fine-tune anyway.

## Risk extractor on Google Colab

Fine-tuning requires a CUDA GPU, so the risk extractor runs entirely on Colab free tier.

1. Open `risk_extractor/notebook.ipynb` in Colab.
2. Runtime, Change runtime type, T4 GPU.
3. Store `HF_TOKEN` and `GROQ_API_KEY` in Colab Secrets (the key icon in the left sidebar), never inline in a cell.
4. Run all cells and leave the outputs in place before downloading the notebook back into this repo.

Colab free tier disconnects after roughly 90 minutes of inactivity and caps sessions at around 12 hours. Checkpoint to Google Drive so a disconnect does not cost the whole run.
