# Controlled Representation Ablation — Morgan vs. ChemBERTa, Matched M9e Training Recipe

**Date:** 2026-08-13/14
**Status:** Complete. **Redesignated OFFICIAL 2026-09-01** — see note below (originally: controlled ablation, not an official reproduction result).

> **REDESIGNATION, 2026-09-01:** per explicit user decision made while
> drafting the reproduction paper, this Milestone 9f Morgan run
> (`outputs/baseline/morgan_faithful_training/`) is now the **official**
> Morgan baseline, paired with the already-redesignated Milestone 9e
> ChemBERTa result (`outputs/polyllm/chemberta_faithful_training/`,
> `notes/deviations_from_paper.md` §1.9) — both trained under the same
> paper-matching recipe. Rationale: a paper reproduction should report the
> paper-matching configuration, not the historically-first BCE/lr=0.001
> one. The original text below (written before this redesignation) is kept
> for its historical framing and is otherwise still accurate — everywhere
> it says the M6B/7/9c (BCE) runs are "official," read that as superseded;
> `outputs/baseline/morgan/` and `outputs/polyllm/chemberta/` are retained
> on disk unmodified as secondary/historical reference points, and remain
> useful for exactly the robustness point this ablation itself makes: Morgan
> beats ChemBERTa under **both** training recipes, so the finding isn't an
> artifact of the (now-superseded) recipe choice.

---

## 1. Motivation

The repository's two "official" MLP baselines — Morgan (`outputs/baseline/morgan/`)
and ChemBERTa (`outputs/polyllm/chemberta/`) — were both trained under the
same BCE/lr=0.001/batch=256 recipe (Milestone 6B/7/9c), and Morgan
outperformed ChemBERTa on every metric (§0 below). Separately, Milestone 9e
(`outputs/polyllm/chemberta_faithful_training/`) found that a different
training recipe — closely matching the actual PolyLLM authors' source
(`src/mlp/MLPModel.py`, `github.com/sadrahkm/PolyLLM`): binary focal loss
with label smoothing, batch size 32, and a Keras-style per-optimizer-step
exponential LR decay — produced ChemBERTa's closest agreement with the
paper's own reported numbers (`notes/deviations_from_paper.md` §1.9).

Morgan was never retrained under that M9e recipe, so the Milestone 8
Morgan-vs-ChemBERTa comparison was confounded: two representations *and*
two training recipes differed at once between "official Morgan" and
"paper-faithful ChemBERTa." This ablation removes that confound by
training Morgan under the *exact* M9e recipe and comparing it directly
against the existing M9e ChemBERTa checkpoint.

**Research question:** under the same training protocol, does classical
Morgan/ECFP4 outperform frozen ChemBERTa for PolyLLM multi-label
polypharmacy side-effect prediction?

---

## 2. Pre-run audit (verified against the actual M9e implementation, not assumed)

Traced directly from `src/polyllm/train_chemberta_mlp_faithful.py` and
`outputs/polyllm/chemberta_faithful_training/training_config.json`.

| Property | ChemBERTa M9e | Morgan (this ablation) |
|---|---|---|
| Dataset | `data/features/chemberta_pair_embeddings.npy`, 63,472 pairs | `data/features/morgan_pair_fingerprints.npy`, same 63,472 pairs |
| Split | `data/splits/{train,validation,test}_pair_ids.csv`, seed 42, 80/10/10 | identical files |
| Output labels | 963 | 963 |
| Pair fusion | element-wise sum, two 384-dim ChemBERTa embeddings | element-wise sum, two 2048-dim Morgan fingerprints (values in {0,1,2}) — same fusion operator as the official Morgan baseline; verified in `notes/morgan_baseline_findings.md` §2 |
| Input dimension | 384 | 2048 |
| Hidden layers | 512 → 1024 → 2048 | same |
| Activation | LeakyReLU slope=0.1 | same |
| BatchNorm | after hidden layer 1 only | same |
| Dropout | 0.2 | same |
| Loss | `BinaryFocalLossWithLogits(gamma=2.0, alpha=-1 disabled, label_smoothing=0.2)` | same |
| Batch size | 32 | same |
| Initial LR | 0.005 (Adam, betas=(0.9,0.999), eps=1e-7) | same |
| LR decay | Keras `ExponentialDecay`, decay_rate=0.96, decay_steps=1000, staircase=True | same |
| LR decay frequency | per optimizer step (1,587 steps/epoch — identical, since train-pair count and batch size match) | same |
| Early stopping | patience=10, min_delta=0.0001, monitor=val_macro_auprc, best checkpoint restored | same |
| Max epochs | 100 | same |
| Seed | 42 | same |

**Uncontrolled variables identified:** only input dimension (384 vs. 2048)
and the resulting parameter count (4,795,843 vs. 5,647,811) — a direct,
unavoidable consequence of the representation change itself, not a
confound. No other substantive difference was found between the two
configs; `compare_faithful_representation_ablation.py` step 1 additionally
verifies 29 config fields match byte-for-byte before any comparison runs.

---

## 3. Implementation

Two new files, both close copies of the M9e ChemBERTa scripts with feature
paths/`input_dim` changed and nothing else:

- `src/polyllm/train_morgan_mlp_faithful.py` → `outputs/baseline/morgan_faithful_training/`
- `src/polyllm/evaluate_morgan_mlp_faithful.py`
- `src/polyllm/recompute_faithful_ap_at_k.py` — paper-compatible per-label-axis AP@50 (reuses `average_precision_at_k_multi_label`, Milestone 9b)
- `src/polyllm/compare_faithful_representation_ablation.py` — aggregate diffs, per-label win counts, paired bootstrap; reuses `compare_models.py`'s (Milestone 8) helper functions rather than re-deriving them

No existing file was modified. `MultilabelMLP`, `BinaryFocalLossWithLogits`,
and `keras_exponential_decay_lr` are all pre-existing, representation-agnostic
shared modules (not ChemBERTa-specific), so no changes were needed there either.

### Exact commands

```
.venv-polyllm/Scripts/python.exe src/polyllm/train_morgan_mlp_faithful.py
.venv-polyllm/Scripts/python.exe src/polyllm/evaluate_morgan_mlp_faithful.py
.venv-polyllm/Scripts/python.exe src/polyllm/recompute_faithful_ap_at_k.py
.venv-polyllm/Scripts/python.exe src/polyllm/compare_faithful_representation_ablation.py
```

---

## 4. Training results

Ran the full 100 epochs, no early stop — same stability signature as M9e
ChemBERTa (monotonic improvement, no loss spikes, unlike the unstable M9d
ablation).

| | ChemBERTa M9e | Morgan (this ablation) |
|---|---|---|
| Best epoch | 94 | 99 |
| Best val macro AUPRC | 0.3990 | 0.4697 |
| Selected threshold | 0.39 | 0.40 |
| Training stability | Stable, no spikes | Stable, no spikes |

---

## 5. Test-set results

| Metric | ChemBERTa M9e | Morgan (this ablation) | Difference (Morgan − ChemBERTa) | Relative |
|---|---|---|---|---|
| Macro AUROC | 0.8847 | **0.9005** | +0.0159 | +1.79% |
| Macro AUPRC | 0.3943 | **0.4701** | +0.0758 | +19.23% |
| Micro AUPRC | 0.4768 | **0.5456** | +0.0688 | +14.44% |
| Micro F1 (sel. thr.) | 0.4736 | **0.5075** | +0.0339 | +7.15% |
| Macro F1 (sel. thr.) | 0.3977 | **0.4436** | +0.0459 | +11.54% |
| AP@50 (per-pair axis, `sample_mean_ap_at_50`) | 0.3253 | **0.3820** | +0.0567 | +17.42% |
| **AP@50 (paper per-label axis)** | 0.7548 | **0.8557** | +0.1008 | +13.35% |

Morgan wins every single metric, by a wide margin — considerably larger
than the corresponding gap under the official BCE recipe (§7 below).

### Per-label win counts (963 labels)

| Metric | Morgan wins | ChemBERTa wins | Ties |
|---|---|---|---|
| AUROC | 945 (98.1%) | 18 (1.9%) | 0 |
| AUPRC | 941 (97.7%) | 22 (2.3%) | 0 |
| F1 (sel. thr.) | 887 (92.1%) | 76 (7.9%) | 0 |

### Paired bootstrap (500 reps, seed 42, fixed thresholds 0.40/0.39, same resampled indices for both models)

| Metric | 95% CI (Morgan − ChemBERTa) | Positive reps |
|---|---|---|
| macro_auprc | [0.0722, 0.0783] | 500/500 |
| micro_f1_selected_threshold | [0.0322, 0.0355] | 500/500 |
| sample_mean_ap_at_50 | [0.0538, 0.0592] | 500/500 |

Every bootstrap resample favored Morgan on every metric tested — the 95%
CIs sit entirely above zero, with a finite-sample one-sided upper bound on
the p-value of (0+1)/501 ≈ 0.002 for each.

### Prediction agreement

95.97% threshold agreement (thr 0.40/0.39), Pearson r=0.9481, Spearman
ρ=0.9433, top-10 Jaccard mean=0.4325 — the two models agree closely on
*which* pairs/labels are positive despite the large metric gap, consistent
with both learning a broadly similar ranking with Morgan systematically
more confident/accurate at the margins.

Full artifacts: `outputs/baseline/morgan_faithful_training/{test_metrics.json,per_label_metrics.csv,ap_at_k_recompute.json}`,
`outputs/comparison/faithful_representation_ablation/{aggregate_metrics.csv,per_label_comparison.csv,bootstrap_differences.csv,comparison_audit.json}`.

---

## 6. Comparison to the paper and to the official (BCE) baselines

| Metric | Paper (ChemBERTa) | Official ChemBERTa (BCE, M7/9c) | ChemBERTa M9e (faithful) | Official Morgan (BCE, M6B/9c) | **Morgan M9e-recipe (this ablation)** |
|---|---|---|---|---|---|
| Macro AUROC | 0.8859 ± 0.0019 | 0.8880 | 0.8847 | 0.8986 | **0.9005** |
| Macro AUPRC | 0.3979 ± 0.0079 | 0.4328 | 0.3943 | 0.4819 | **0.4701** |
| AP@50 (paper axis) | 0.7557 ± 0.0120 | 0.8146 | 0.7548 | 0.8716 | **0.8557** |

Two notable patterns, both consistent with what §1.7/§1.9 of
`notes/deviations_from_paper.md` already found for ChemBERTa:

1. The M9e-style recipe (focal loss + batch 32 + per-step LR decay) does
   **not** simply make either representation's absolute numbers better —
   it pulled Morgan's own AUPRC and AP@50 down slightly relative to the
   official BCE Morgan run too (0.4819→0.4701, 0.8716→0.8557), mirroring
   the same small downward shift already observed for ChemBERTa
   (0.4328→0.3943, 0.8146→0.7548). This is further evidence the effect is
   a general property of this training recipe on this task, not specific
   to one representation.
2. Despite that shared downward shift, Morgan's *relative* advantage over
   ChemBERTa did not shrink under the matched recipe — if anything it
   widened slightly (macro AUPRC gap: 0.4819−0.4328=+0.0491 official BCE vs.
   0.4701−0.3943=**+0.0758** under the matched M9e recipe).

---

## 7. Interpretation

**Morgan still wins decisively under the matched M9e training recipe** —
on every aggregate metric, on the large majority of individual labels
(92–98% depending on metric), and in every one of 500 bootstrap resamples.
Per the interpretation rule set out before running this ablation: this
result lets us attribute the previously observed Morgan-over-ChemBERTa
advantage (Milestone 8) more confidently to the molecular representation
itself, rather than to the loss function, learning-rate schedule, or batch
size that differed between the two "official" runs. The apparent Morgan
advantage was **not** an artifact of differing training configurations —
equalizing the training protocol did not narrow the gap, and by some
metrics (macro AUPRC) widened it.

**What this does not show:** this ablation reuses the same single fixed
80/10/10 pair-random split as every other MLP experiment in this
repository (`notes/deviations_from_paper.md` §1.2, still an open,
unreproduced deviation from the paper's 10-fold CV protocol) and the same
single-seed, single-run design as the rest of this repository — it is a
point estimate, not a distribution over splits or seeds. It also does not
speak to *why* Morgan's discrete substructure fingerprints outperform a
frozen general-purpose chemical-language-model embedding on this
particular task — only that they do, robustly, under matched training
conditions.

**Status:** frozen as a completed controlled ablation. **Redesignated
OFFICIAL 2026-09-01** (see top-of-file note) — Milestone 9f/9e are now the
official Morgan and ChemBERTa results for paper-reproduction purposes.
Milestone 6B/7/9c (documented in `notes/morgan_baseline_findings.md` and
`notes/chemberta_mlp_findings.md`) are retained as a secondary/historical
reference, and as the source of the robustness claim above (same
conclusion under two recipes).
