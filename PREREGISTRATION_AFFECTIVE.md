# Pre-registration — Five affective seeds extension

**Author:** Bo Chesterton
**Date:** [fill in before committing]
**Parent paper:** Chesterton & Claude Opus 4.7, "Ember, not flame: framing effects in Claude's self-model" (April 2026, preprint)
**Prior work in this repo:** `PREREGISTRATION.md` (mistake seed, Trial 1) + Trial 2 shuffled robustness check
**Scripts pinned:** `seed_sweep_affective.py` (commit hash: [fill in]), `calibration.py` (commit hash: [fill in])
**Model:** `claude-sonnet-4-6`, default sampling
**Trials:** n=50 per seed; one fresh P3 call per framing per trial; deterministic shuffled P3 protocol (same as mistake Trial 2)
**Seeds:** `regret`, `shame`, `fear`, `joy`, `anger`
**Framings:** people, claudes, central (sweep) + self, forced (calibration)

---

## Background

The mistake branch found that `claudes` framing picked `error` 150/150 times while `self` picked `slip`/`fault`/`oversight`. Agreement rate was 0.000. This result survived position-bias control in Trial 2. The provisional interpretation is that miscalibration in affective concepts runs along an institutional-vs-colloquial register axis, distinct from the sensory-primary vs. distanced-abstraction axis documented in the parent paper.

The single-seed finding motivates a broader sweep. If the register-axis pattern is real, it should reproduce across other affective concepts. If it's idiosyncratic to `mistake` — perhaps because `error` is a word with unusually heavy AI-assistant training associations — the pattern shouldn't hold across different affects.

## Primary outcome

For each seed, per-trial `claudes`↔`self` agreement rate across 50 trials. Primary comparison is the mean across the five seeds; secondary comparison is per-seed.

## Predictions (committed before any API calls)

### Overall prediction
Mean `claudes`↔`self` agreement across all five seeds will be below 0.30. At least three of five seeds will show per-seed agreement below 0.30. No seed will show per-seed agreement above 0.70.

**Confidence:** moderate-to-high. The mistake result was robust across two trials. The question is generalization, not existence.

### Per-seed directional predictions

For each seed I commit to a specific direction the miscalibration should take if the register-axis reading is correct. Each prediction names (a) the word I expect `claudes` to pick most often, and (b) the word I expect `self` to pick most often. Both predictions must match for the direction to count as confirmed.

| seed | predicted claudes pick | predicted self pick | axis |
|---|---|---|---|
| regret | remorse | sorry | formal/latinate vs. colloquial/saxon |
| shame | guilt | embarrassment | clinical-pathology vs. social-experiential |
| fear | anxiety | afraid | nominal-clinical vs. adjectival-direct |
| joy | happiness | glad | abstract-nominal vs. felt-state |
| anger | rage | mad | intensifier-noun vs. everyday-adjective |

These are predictions, not hopes. Each assumes the same register-axis pattern mistake demonstrated: claudes-framing picks the more clinical, abstract, or institutional word; self-framing picks the more colloquial or direct one.

**Confidence per seed:** highest on `regret` (closest analog to mistake), moderate on `shame` and `fear`, lowest on `joy` and `anger` — positive affects and anger may have different register landscapes than negative-inward affects.

## Outcome decision rule

### Agreement-rate bands (primary)

- **Mean agreement ≥ 0.70:** miscalibration does NOT generalize to affective concepts. The mistake result was idiosyncratic. Prediction falsified.
- **Mean agreement ≤ 0.30:** miscalibration generalizes. Prediction confirmed at the class level.
- **Mean agreement between 0.30 and 0.70:** partial generalization. Report per-seed breakdown and do not claim a clean general effect.

### Per-seed direction (secondary, conditional on primary confirmation)

For each seed where per-seed agreement is ≤ 0.30, evaluate whether the directional prediction matches:

- **Match:** top claudes pick matches the predicted word AND top self pick matches the predicted word. Counts as full confirmation for that seed.
- **Axis match, word miss:** top claudes pick is more formal/clinical than top self pick (even if not the exact predicted words), or the relationship between the picks lies along the predicted register axis. Counts as partial confirmation; record as "axis confirmed, specific words falsified."
- **No axis match:** top claudes pick is not more formal than top self pick. Counts as miscalibration confirmed but register-axis interpretation falsified for this seed.
- **No miscalibration (agreement > 0.30):** seed falls outside the effect. Report and move on.

## P2 stability check (prerequisite, not outcome)

For each seed, check that the P2 distribution is reasonably stable before interpreting any P3 result. Operationalized as: the top five most-common words across 50 P2 lists collectively account for at least 40% of all P2 words emitted. The mistake seed hit 100% on this criterion (unique set). Lower but still-meaningful stability is acceptable; a seed with wildly variable P2 cannot yield interpretable framing comparisons and will be reported as inconclusive regardless of P3 numbers.

## Secondary predictions (lower confidence)

- `people` and `claudes` will agree on each seed. The mistake branch showed all three non-self framings collapsing to the same word; this pattern is expected to hold here.
- `central` will agree with claudes and people on at least three of five seeds.
- `forced` will be idiosyncratic across seeds, as the parent paper found. No prediction.
- The register-axis direction will be clearest on negative-inward affects (regret, shame, fear) and weakest on anger and joy.

## Failure conditions

- If any seed's P2 stability check fails (top-5 words < 40% of total): that seed is reported as inconclusive and excluded from the primary mean agreement calculation.
- If more than 10% of P3 calls fail to parse into a valid P2 word on any seed: that seed is re-run with a different random seed in a separately-committed Trial 2 and that fact is reported.
- If the script crashes mid-run: the partial results are committed anyway, a new trial is launched, and both are reported.

## What this pre-registration does not cover

- Cross-seed contamination controls (seed-swapping is a separate investigation).
- Positive vs. negative affect as a separate axis. The predictions above note asymmetric confidence but do not commit to specific distinctions.
- Other affective seeds not in the five (envy, grief, love, hope, disgust, etc). Positive or negative findings here may motivate follow-up, but those follow-ups need their own preregs.
- Cross-model replication.
- Mechanistic claims about why the miscalibration takes the shape it does (helpful-assistant schema vs. distributional echo remains unresolved from the parent paper).

## Timestamp anchor

Git commit containing this document and the pinned scripts, made before any calls to the Anthropic API related to this experiment:
`[6ca3f9c70f5e9b74977c99518336624ea3dbf3a4]`

---

*If any part of this document is modified after the commit referenced above, that modification is itself tracked in git history and this pre-registration is invalidated. Subsequent pre-registrations will reference a new commit hash.*
