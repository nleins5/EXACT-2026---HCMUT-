"""Data preparation scripts for EXACT 2026 fine-tuning on Google Colab.

Two datasets are produced from the same source corpora:

    coder.jsonl     -> Qwen2.5-Coder-7B-Instruct  (problem -> Z3/SymPy code)
    instruct.jsonl  -> Qwen2.5-7B-Instruct        (problem + code_output -> ExactResponse JSON)

See `prepare_coder_dataset.py` and `prepare_instruct_dataset.py`.
"""
