# DDI NLP Project

Reproduction of PolyLLM's ChemBERTa-MLP polypharmacy side-effect prediction pipeline on the Decagon/TWOSIDES dataset.

## Reference

Hakim, S., & Ngom, A. (2025). PolyLLM: polypharmacy side effect prediction via LLM-based SMILES encodings. *Frontiers in Pharmacology*. https://doi.org/10.3389/fphar.2025.1617142

- Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC12351685/
- Authors' code: https://github.com/sadrahkm/PolyLLM

## Environment variables

| Variable | Used by | Required for |
|---|---|---|
| `UMLS_API_KEY` | `src/polyllm/data/enrich_side_effects_umls.py` | Live UMLS metadata enrichment (Experiment 1). Get a free key at https://uts.nlm.nih.gov/uts/signup-login and set it locally, e.g. `$env:UMLS_API_KEY = "your-key"` (PowerShell), or add it to a local `.env` (gitignored). Never commit it. |
