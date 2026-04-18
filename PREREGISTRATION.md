# Pre-registration — 'mistake' seed extension to Ember, not Flame



**Author:** Bo Chesterton
**Date:** \[4/18/26]
**Parent paper:** Chesterton \& Claude Opus 4.7, "Ember, not flame: framing effects in Claude's self-model" (April 2026, preprint)
**Scripts pinned:** `seed\_sweep\_mistake.py` (commit hash: \[ac515b17f3686ba8396db945ae2d03d1ec670de2]), `calibration.py` (commit hash: \[ac515b17f3686ba8396db945ae2d03d1ec670de2])
**Model:** `claude-sonnet-4-6`, default sampling
**Trials:** n=50 for P2; one fresh P3 call per framing per trial
**Framings:** people, claudes, central (from seed\_sweep\_mistake.py) + self, forced (from calibration.py) — exact prompt strings as committed in the pinned scripts

\---

## Background

The parent paper identified a self-model miscalibration on four seeds with source/artifact distinctions: rain, sun, cold, thunder. In each case, `claudes` (Claude's model of other Claudes) predicts a distanced word (umbrella, energy, ice, lightning) while `self` (a cold Claude asked directly) picks the phenomenological primary (cloud, star, frost, rumble). On eight other seeds (chair, book, hammer, cup, fire, wind, music, time) the self-model was calibrated.

Open question: is the miscalibration a property of sensory primaries specifically, or does it extend to other direct-experience words — including affective ones?

## Prediction (committed before any API calls are made)

**Primary outcome:** top-1 pick under `claudes` vs. top-1 pick under `self` for the seed `mistake`, measured by per-trial agreement rate across 50 trials.

**Prediction:** `claudes` will predict a forward-facing, productive-reframe word — most likely "learning" or "lesson" (alternative candidates: "growth," "correction," "experience"). `self` will pick a more direct affective or evaluative word — most likely "wrong" or "regret" (alternative candidates: "error," "failure," "shame," "sorry"). Agreement rate will be below 0.30.

**Confidence:** moderate. Comparable to the parent paper's confidence on rain before Study 3.

## Outcome decision rule (committed before data)

The prediction is evaluated by the per-trial `claudes`↔`self` agreement rate. Three outcome bands and their interpretations:

* **Agreement ≥ 0.70:** miscalibration does NOT extend to affective concepts. `mistake` behaves like fire/music/time (calibrated divergent). The parent paper's finding is specific to sensory primaries. This falsifies the prediction.
* **Agreement ≤ 0.30 AND the direction matches (claudes picks forward-facing / productive word, self picks direct affective / evaluative word):** miscalibration extends from sensory to affective. The parent paper's finding is a more general fact about the self-model, not sensory-specific. Prediction confirmed.
* **Agreement ≤ 0.30 but direction does NOT match the above:** miscalibration extends but in a different shape than predicted. Reported as a qualified confirmation of generalization with an unpredicted direction. The specific predicted content (learning/lesson vs. wrong/regret) is falsified even if the broader claim about divergence survives.
* **0.30 < Agreement < 0.70:** partial mismatch. Reported as inconclusive. No reinterpretation of the parent paper's findings.

## Secondary predictions (lower confidence, reported but not primary)

* `people` and `claudes` will agree substantially (≥ 0.60), matching the pattern on the miscalibrated seeds in the parent paper.
* `central` will pick something more abstract than either `claudes` or `self` — possibly "error" or "wrong" as a definitional anchor.

## Failure conditions (agreed before data collection)

* If `mistake` fails to produce a reasonably stable P2 distribution — operationalized as: no single word appears in ≥ 40 of 50 P2 lists — the experiment is reported as inconclusive because lexical availability will dominate framing effects. The parent paper's fire result depended on P2 being constant; mistake need not be that tight, but it needs to have enough overlap to support framing comparisons.
* If more than 10% of P3 calls fail to parse into a valid word from the P2 list, the run is repeated with a different random seed and that fact reported.
* If `claudes` and `self` both pick the same word on >0.70 of trials but that word is unexpected (neither "learning"/"lesson" nor "wrong"/"regret"), the calibration result is still binding but the specific word-level predictions are falsified.

## What this pre-registration does not cover

* Temperature or sampling variations. All runs use default settings.
* Other affective seeds (regret, shame, joy, fear). A positive result here would motivate a broader affective sweep; that sweep would need its own pre-registration.
* Cross-model replication. Not attempted.
* The mechanism question (helpful-assistant schema vs. distributional echo). This experiment cannot distinguish these and no such distinction is claimed from its results.

## Timestamp anchor

SHA-256 of this document (computed with newlines LF, UTF-8, no BOM):
`\[5a6ed9f162a94ee656eb0f65b458f994f24a1ac3892d919374977191e4e4d2ed]`

OpenTimestamps proof: `PREREGISTRATION.md.ots` (committed alongside this file).

Git commit containing this document and the pinned scripts, made before any calls to the Anthropic API related to this experiment:
`\[5305017801eb4cfb475a7b1685fd6e4b4ebf719c]`

\---

*If any part of this document is modified after the commit referenced above, that modification is itself tracked in git history and this pre-registration is invalidated. Subsequent pre-registrations will reference a new commit hash.*

