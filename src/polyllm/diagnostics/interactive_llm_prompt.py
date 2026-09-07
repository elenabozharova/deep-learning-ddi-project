"""
Manually query the same local LLM (and the same Yes/No scoring method) used
by llm_prompting_experiment.py, for hands-on sanity-checking of specific
drug/side-effect triples.

Reuses, unmodified: load_llm, build_messages, score_yes_no, _first_token_ids,
YES_WORDS, NO_WORDS from llm_prompting_experiment.py -- this asks the model
exactly the same question, in exactly the same format, as the batch
experiment. No training, no batch scoring, no file output -- just a REPL.

Run:
    .venv-polyllm/Scripts/python.exe src/polyllm/diagnostics/interactive_llm_prompt.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_src = Path(__file__).resolve().parents[2]
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from polyllm.diagnostics.llm_prompting_experiment import (  # noqa: E402
    LLM_MODEL_NAME,
    NO_WORDS,
    YES_WORDS,
    _first_token_ids,
    build_messages,
    load_llm,
    score_yes_no,
)


def main() -> None:
    print(f"Loading {LLM_MODEL_NAME} ...")
    tok, model = load_llm()
    yes_ids = _first_token_ids(tok, YES_WORDS)
    no_ids = _first_token_ids(tok, NO_WORDS)
    print("Loaded. Enter a drug pair and a candidate side effect (blank line to quit).\n")

    while True:
        drug_a = input("Drug A: ").strip()
        if not drug_a:
            break
        drug_b = input("Drug B: ").strip()
        side_effect = input("Candidate side effect: ").strip()

        messages = build_messages(None, drug_a, drug_b, side_effect)
        p_yes = score_yes_no(model, tok, messages, yes_ids, no_ids)
        print(f"  p(yes) = {p_yes:.3f}  ->  {'YES' if p_yes >= 0.5 else 'NO'}\n")


if __name__ == "__main__":
    main()
