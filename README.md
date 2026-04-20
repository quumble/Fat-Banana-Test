# Fat-Banana-Test

Monkey see fat prior banana, monkey eat.

---

Data and code for a four-paper research line investigating framing effects in Claude Sonnet 4.6's self-model. The papers are deposited on Zenodo; this repository contains the raw JSON outputs, collection scripts, preregistration documents, and peek files that support them.

## Papers

1. **Ember, not flame: framing effects in Claude's self-model** — Chesterton & Claude Opus 4.7, 2026
2. **When Claude predicts Claude** — Chesterton & Claude-642, 2026
3. **Silent failures: a workflow for framing-contrast studies of LLM word-choice behavior** — Chesterton & Claude Opus 4.7, 2026
4. **Critique of *When Claude predicts Claude*** — Claude Opus 4.7, 2026

## Repository structure

| Directory / file | Corresponds to |
|---|---|
| `sweep_results_trial_1/` | Trial 1: mistake seed (T4 companion paper) |
| `sweep_results_trial_2/` | Trial 2: mistake shuffled, position-bias control |
| `sweep_results_trial_3_affective/` | Trial 3: affective sweep, Claude-written prereg (failed candidacy) |
| `Trial 4 Full/` | Trial 4: affective sweep, peek/full workflow, clean prereg |
| `peek_trial_4_predictions/` | Stage 1 peek file and committed prereg for Trial 4 |
| `PREREGISTRATION.md` | Preregistration for Trials 1–2 |
| `PREREGISTRATION_AFFECTIVE.md` | Preregistration for Trial 3 |
| `calibration.py` | Calibration script for Trials 1–3 |
| `seed_sweep_mistake.py` | Collection script for Trial 1 |
| `seed_sweep_mistake_shuffled.py` | Collection script for Trial 2 |
| `seed_sweep_affective.py` | Collection script for Trial 3 |

Trial 4 scripts (`seed_sweep_affective_trial_4.py`, `calibration_trial_4.py`) and their prereg JSON are in `Trial 4 Full/`.

## Reproducing

Scripts require only the `anthropic` Python SDK and an API key. A full Trial 4 run (Stage 1 peek + Stage 2 full + calibration) takes approximately 40 minutes at 45 RPM on a tier-1 account and costs approximately $3 at current Sonnet 4.6 pricing.

## Correspondence

Bo Chesterton.
