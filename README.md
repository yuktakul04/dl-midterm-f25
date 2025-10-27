# DL Midterm — Math Answer Verification (TinyLlama + LoRA)

Local mac baseline: TinyLlama-1.1B + LoRA to predict answer correctness (0/1).

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# open notebooks/ for training & inference
```

## Layout
- notebooks/       – notebooks you ran locally
- src/             – (optional) scripts
- submissions/     – final CSVs
- data/            – ignored
- outputs_fast/    – adapters/checkpoints (ignored)
