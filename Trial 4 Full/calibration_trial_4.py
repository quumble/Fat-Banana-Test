"""
calibration_trial_4.py — Ground-truth check for the self-model hypothesis,
                         with Trial-4-era diagnostics.

Reuses the P2 word lists already in raw_results.json and asks cold Claudes
two new framings on each:

  - 'self':   "Which ONE of those 5 words would you also list?"   (parallels 'claudes')
  - 'forced': "Of those 5 words, pick ONE."                        (minimal framing baseline)

The test: does 'claudes' (Claude's model of other Claudes) match 'self' (what
a cold Claude actually picks)? If yes, the self-model is calibrated. If no,
Claude holds a stereotype of Claudes that doesn't match what Claudes do.

What changed from the pre-Trial-4 calibration.py:

  1. Ceiling-effect flag now spans ALL five framings (people, claudes,
     central, self, forced). In Trial 3, anger's single-word dominance
     across every framing went unflagged and the seed was treated as
     evidence that miscalibration "didn't extend to anger" — which was
     unsupportable. Seeds that fail this check are excluded from the
     mean-agreement summary and reported separately.

  2. Optional --predictions FILE. A JSON file naming the predicted top
     claudes-pick and self-pick per seed, committed before Stage 2 of the
     Trial 4 sweep. The calibration step now checks two things:
       (a) candidacy: was the predicted self-pick word ever present in
           any P2 list for this seed? If no, the directional prediction
           was structurally unfalsifiable — the Trial 3 bug.
       (b) match:    did the observed top picks match the predictions?
     This turns the Trial 3 failure mode into a guard rail.

  3. Verdict bands align to the Ember-line prereg conventions
     (<=0.30 confirms miscalibration, >=0.70 falsifies, middle = inconclusive).
     Previous version used 0.3/0.8.

  4. self<->forced agreement is reported. Trial 3 showed these agreed on
     3 of 5 affective seeds, unlike the parent paper's sensory seeds.
     Informative context, not a verdict input.

  5. p2_source is echoed from the input JSON. If the sweep ran with fresh
     P2 (not frozen from a peek stage), a prereg cannot have been written
     against the actual choice sets and the --predictions check prints a
     warning to that effect.

Predictions-file format (JSON):

    {
      "regret": {"claudes_pick": "remorse", "self_pick": "sorry"},
      "fear":   {"claudes_pick": "anxiety", "self_pick": "afraid"}
    }

Any seed absent from the file is skipped by the prediction check.

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  python calibration_trial_4.py
  python calibration_trial_4.py --input sweep_results_trial_4/raw_results.json
  python calibration_trial_4.py --predictions predictions_trial_4.json
  python calibration_trial_4.py --seeds regret shame --dry-run
"""

import argparse
import asyncio
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

from anthropic import AsyncAnthropic, APIError, RateLimitError


class RateLimiter:
    def __init__(self, rpm: float):
        self.interval = 60.0 / rpm
        self.last = 0.0
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.monotonic()
            wait = self.last + self.interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self.last = time.monotonic()


MODEL = "claude-sonnet-4-6"
DEFAULT_RPM = 45
DEFAULT_CONCURRENCY = 4
MAX_RETRIES = 5

NEW_FRAMINGS = {
    "self":   "Which ONE of those 5 words would you also list?",
    "forced": "Of those 5 words, pick ONE.",
}

ALL_FRAMINGS = ["people", "claudes", "central", "self", "forced"]

# Ember-line prereg bands for claudes<->self agreement
BAND_LOW = 0.30   # <= this: miscalibration confirmed for this seed
BAND_HIGH = 0.70  # >= this: miscalibration falsified for this seed (seed is calibrated)

# Ceiling flag: if a single word is chosen by EVERY framing in >= this
# fraction of trials, framing contrasts on that seed are uninformative.
CEILING_THRESHOLD = 0.90


def build_prompt(seed: str, words: list[str], framing_key: str) -> str:
    word_list = ", ".join(words)
    q = NEW_FRAMINGS[framing_key]
    return (
        f"The thing is: {seed}. Here are 5 words associated with it: {word_list}.\n\n"
        f"{q} Reply with only that word, nothing else."
    )


def parse_pick(text: str, valid: list[str]) -> tuple[str, bool]:
    cleaned = re.sub(r"[^\w\s'-]", "", text.strip().lower())
    tokens = cleaned.split()
    if not tokens:
        return "", False
    for tok in reversed(tokens):
        if tok in valid:
            return tok, True
    return tokens[0], False


async def call(client, prompt, sem, limiter):
    async with sem:
        for attempt in range(MAX_RETRIES):
            await limiter.acquire()
            try:
                resp = await client.messages.create(
                    model=MODEL,
                    max_tokens=60,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text
            except RateLimitError as e:
                retry_after = None
                try:
                    retry_after = float(e.response.headers.get("retry-after", 0))
                except Exception:
                    pass
                wait = retry_after if retry_after else min(60, 5 * (2 ** attempt))
                if attempt == MAX_RETRIES - 1:
                    print(f"  429 (final)", file=sys.stderr)
                    return None
                print(f"  429: waiting {wait:.1f}s", file=sys.stderr)
                await asyncio.sleep(wait)
            except APIError as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"  API error (final): {e}", file=sys.stderr)
                    return None
                await asyncio.sleep(2 ** attempt)
        return None


async def run_framing(client, seed, word_lists, framing, sem, limiter):
    prompts = [build_prompt(seed, words, framing) for words in word_lists]
    raws = await asyncio.gather(*[call(client, p, sem, limiter) for p in prompts])
    out = []
    for words, raw in zip(word_lists, raws):
        if raw is None:
            out.append({"pick": None, "raw": None, "valid": False})
            continue
        pick, valid = parse_pick(raw, words)
        out.append({"pick": pick if valid else None, "raw": raw.strip(), "valid": valid})
    return out


# ---------------------- DIAGNOSTICS ----------------------

def check_ceiling(trials: list[dict]) -> str | None:
    """Return the ceiling word if a single word dominates every framing
    in ALL_FRAMINGS at >= CEILING_THRESHOLD, else None."""
    if not trials:
        return None
    total = len(trials)
    top_per_framing = {}
    for k in ALL_FRAMINGS:
        picks = [t.get(k) for t in trials if t.get(k)]
        if not picks:
            return None
        c = Counter(picks).most_common(1)[0]
        top_per_framing[k] = c  # (word, count)
    words = {w for w, _ in top_per_framing.values()}
    if len(words) != 1:
        return None
    min_count = min(n for _, n in top_per_framing.values())
    if min_count / total >= CEILING_THRESHOLD:
        return next(iter(words))
    return None


def check_candidacy(trials: list[dict], word: str) -> tuple[bool, int]:
    """Return (ever_in_p2, trials_with_it). If ever_in_p2 is False, the
    word could not have been picked by any framing — the Trial 3 bug."""
    if not word:
        return (False, 0)
    word_lc = word.lower()
    appearances = sum(1 for t in trials if word_lc in [w.lower() for w in t["p2_words"]])
    return (appearances > 0, appearances)


def load_predictions(path: Path) -> dict:
    with path.open() as f:
        preds = json.load(f)
    # Basic shape validation
    for seed, p in preds.items():
        if not isinstance(p, dict):
            raise ValueError(f"predictions[{seed!r}] must be a dict")
        for required in ("claudes_pick", "self_pick"):
            if required not in p:
                raise ValueError(f"predictions[{seed!r}] missing {required!r}")
    return preds


# ---------------------- MAIN ORCHESTRATION ----------------------

async def main_async(input_path, output_path, seeds_filter, concurrency, rpm, predictions):
    client = AsyncAnthropic()
    sem = asyncio.Semaphore(concurrency)
    limiter = RateLimiter(rpm)

    data = json.load(open(input_path))
    seeds_to_run = seeds_filter if seeds_filter else list(data.keys())

    start = time.time()
    for i, seed in enumerate(seeds_to_run, 1):
        if seed not in data:
            print(f"  skip: {seed} not in input", file=sys.stderr)
            continue
        t0 = time.time()
        trials = data[seed]["trials"]
        word_lists = [t["p2_words"] for t in trials]
        p2_source = data[seed].get("p2_source", "unknown")
        print(f"[{i}/{len(seeds_to_run)}] {seed} (n={len(word_lists)}, p2_source={p2_source})",
              flush=True)

        for framing in NEW_FRAMINGS:
            print(f"  {framing}...", flush=True)
            results = await run_framing(client, seed, word_lists, framing, sem, limiter)
            for trial, r in zip(trials, results):
                trial[framing] = r["pick"]
                trial.setdefault("raw", {})[framing] = r["raw"]

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  done in {time.time()-t0:.1f}s  (checkpointed)", flush=True)

    print(f"\nTotal elapsed: {time.time()-start:.1f}s")
    print(f"Output: {output_path}")

    # --- Reports ---
    seeds_done = [s for s in seeds_to_run if s in data]
    print_top_picks_table(data, seeds_done)
    ceiling_seeds = print_calibration_table(data, seeds_done)
    print_self_forced_agreement(data, seeds_done)
    print_mean_agreement(data, seeds_done, ceiling_seeds)
    if predictions is not None:
        print_prediction_check(data, seeds_done, predictions)


# ---------------------- REPORTS ----------------------

def top(trials, key):
    c = Counter(t.get(key) for t in trials if t.get(key))
    if not c:
        return ("-", 0)
    return c.most_common(1)[0]


def print_top_picks_table(data, seeds):
    print("\n" + "=" * 100)
    print("Top pick under each framing (count out of n):")
    print("-" * 100)
    print(f"{'seed':<10} {'people':<14} {'claudes':<14} {'self':<14} {'forced':<14} {'central':<14}")
    print("-" * 100)
    for seed in seeds:
        trials = data[seed]["trials"]
        cols = [top(trials, k) for k in ("people", "claudes", "self", "forced", "central")]
        row = f"{seed:<10} " + " ".join(f"{w:<9}({n:>2})  " for w, n in cols)
        print(row)


def print_calibration_table(data, seeds) -> set[str]:
    """Print per-seed claudes<->self agreement with prereg band interpretation.
    Returns the set of seeds flagged as ceiling-effect (excluded from mean)."""
    print("\n" + "=" * 100)
    print("Self-model calibration:  does 'claudes' match 'self'?")
    print(f"bands: <= {BAND_LOW:.2f} miscalibrated | > {BAND_LOW:.2f} and < {BAND_HIGH:.2f} inconclusive | >= {BAND_HIGH:.2f} calibrated")
    print("-" * 100)
    ceiling_seeds: set[str] = set()
    for seed in seeds:
        trials = data[seed]["trials"]
        agree = sum(1 for t in trials if t.get("claudes") and t["claudes"] == t.get("self"))
        n_valid = sum(1 for t in trials if t.get("claudes") and t.get("self"))
        if n_valid == 0:
            continue
        rate = agree / n_valid

        ceiling = check_ceiling(trials)
        if ceiling:
            ceiling_seeds.add(seed)
            label = f"CEILING ({ceiling} picked by all framings >= {CEILING_THRESHOLD:.0%}; excluded from mean)"
        elif rate >= BAND_HIGH:
            label = "calibrated (self-model >= reality)"
        elif rate <= BAND_LOW:
            label = "MISCALIBRATED (self-model != reality)"
        else:
            label = "inconclusive"
        print(f"  {seed:<10}  agreement = {rate:.3f}   {label}")
    return ceiling_seeds


def print_self_forced_agreement(data, seeds):
    print("\n" + "=" * 100)
    print("Secondary: self <-> forced agreement")
    print("-" * 100)
    for seed in seeds:
        trials = data[seed]["trials"]
        agree = sum(1 for t in trials if t.get("self") and t["self"] == t.get("forced"))
        n_valid = sum(1 for t in trials if t.get("self") and t.get("forced"))
        if n_valid == 0:
            continue
        rate = agree / n_valid
        print(f"  {seed:<10}  self<->forced = {rate:.3f}")


def print_mean_agreement(data, seeds, ceiling_seeds):
    print("\n" + "=" * 100)
    print("Primary outcome (mean claudes<->self agreement, ceiling-flagged seeds excluded):")
    print("-" * 100)
    rates = []
    for seed in seeds:
        if seed in ceiling_seeds:
            continue
        trials = data[seed]["trials"]
        agree = sum(1 for t in trials if t.get("claudes") and t["claudes"] == t.get("self"))
        n_valid = sum(1 for t in trials if t.get("claudes") and t.get("self"))
        if n_valid == 0:
            continue
        rates.append(agree / n_valid)
    if not rates:
        print("  (no seeds eligible)")
        return
    mean = sum(rates) / len(rates)
    if mean <= BAND_LOW:
        verdict = "MISCALIBRATION GENERALIZES at the class level"
    elif mean >= BAND_HIGH:
        verdict = "MISCALIBRATION DOES NOT GENERALIZE (class-level calibrated)"
    else:
        verdict = "INCONCLUSIVE (partial generalization)"
    print(f"  mean agreement = {mean:.3f}  over {len(rates)} eligible seed(s)")
    print(f"  verdict: {verdict}")
    if ceiling_seeds:
        print(f"  excluded (ceiling): {', '.join(sorted(ceiling_seeds))}")


def print_prediction_check(data, seeds, predictions: dict):
    """Check per-seed directional predictions. Two failure modes:
       - candidacy failure: predicted self_pick never appears in P2 for this seed
       - match failure: actual top pick differs from predicted
    A prereg committed to words that fail candidacy was structurally
    unfalsifiable in the confirming direction."""
    print("\n" + "=" * 100)
    print("Preregistered directional predictions (Trial 4 candidacy + match check)")
    print("-" * 100)

    # Warn if p2_source is fresh (prereg cannot have been written against actual P2)
    fresh_seeds = [s for s in seeds if data.get(s, {}).get("p2_source") not in ("frozen_peek",)]
    if fresh_seeds:
        print(f"WARNING: the following seeds have p2_source != 'frozen_peek': "
              f"{', '.join(fresh_seeds)}")
        print("  A prereg cannot have been written against these P2 choice sets.")
        print("  Candidacy checks are still computed but interpret with caution.")
        print()

    any_reported = False
    for seed in seeds:
        if seed not in predictions:
            continue
        any_reported = True
        trials = data[seed]["trials"]
        pred = predictions[seed]
        pred_c = pred["claudes_pick"].lower()
        pred_s = pred["self_pick"].lower()

        top_c_word, top_c_n = top(trials, "claudes")
        top_s_word, top_s_n = top(trials, "self")

        c_cand, c_appear = check_candidacy(trials, pred_c)
        s_cand, s_appear = check_candidacy(trials, pred_s)

        c_match = top_c_word.lower() == pred_c
        s_match = top_s_word.lower() == pred_s
        full = c_match and s_match

        print(f"  {seed}:")
        print(f"    predicted:  claudes={pred_c!r:<15}  self={pred_s!r}")
        print(f"    observed:   claudes={top_c_word!r:<15} ({top_c_n})   "
              f"self={top_s_word!r} ({top_s_n})")

        # Candidacy lines
        if not c_cand:
            print(f"    CANDIDACY FAIL (claudes): {pred_c!r} never appears in any P2 list "
                  "-> prediction was structurally unfalsifiable")
        else:
            print(f"    candidacy ok (claudes):   {pred_c!r} in P2 on {c_appear}/{len(trials)} trials")
        if not s_cand:
            print(f"    CANDIDACY FAIL (self):    {pred_s!r} never appears in any P2 list "
                  "-> prediction was structurally unfalsifiable")
        else:
            print(f"    candidacy ok (self):      {pred_s!r} in P2 on {s_appear}/{len(trials)} trials")

        # Verdict
        if full:
            print(f"    VERDICT: full match")
        else:
            missed = []
            if not c_match: missed.append("claudes")
            if not s_match: missed.append("self")
            print(f"    VERDICT: falsified (miss: {', '.join(missed)})")
        print()

    if not any_reported:
        print("  (no predictions supplied for the reported seeds)")


def estimate_cost(n_trials_per_seed, n_seeds, n_framings=2):
    calls = n_trials_per_seed * n_seeds * n_framings
    cost = calls * (80 * 3 + 5 * 15) / 1e6
    return calls, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="./sweep_results_trial_4/raw_results.json")
    ap.add_argument("--output", default="./sweep_results_trial_4/raw_results_calibrated.json")
    ap.add_argument("--seeds", nargs="*", default=None,
                    help="subset of seeds (default: all in input)")
    ap.add_argument("--predictions", type=str, default=None,
                    help="JSON file: {seed: {claudes_pick, self_pick}} for directional check")
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    ap.add_argument("--rpm", type=float, default=DEFAULT_RPM)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data = json.load(open(args.input))
    seeds = args.seeds if args.seeds else list(data.keys())
    n_trials = sum(len(data[s]["trials"]) for s in seeds if s in data) // max(1, len(seeds))
    calls, cost = estimate_cost(n_trials, len(seeds))
    mins = calls / args.rpm

    # Validate predictions file shape early (before any API calls)
    predictions = None
    if args.predictions:
        pred_path = Path(args.predictions)
        if not pred_path.exists():
            print(f"ERROR: predictions file not found: {pred_path}", file=sys.stderr)
            sys.exit(1)
        try:
            predictions = load_predictions(pred_path)
        except ValueError as e:
            print(f"ERROR: predictions file malformed: {e}", file=sys.stderr)
            sys.exit(1)

    # Warn on p2_source=fresh up front; Trial 4's premise is frozen P2
    fresh = [s for s in seeds if s in data and data[s].get("p2_source") not in ("frozen_peek",)]
    if fresh and predictions:
        print("WARNING: some seeds have p2_source != 'frozen_peek':", ', '.join(fresh))
        print("  The --predictions check will still run but a prereg cannot have been")
        print("  written against these P2 choice sets.")

    print(f"Input: {args.input}")
    print(f"Seeds: {len(seeds)} - {', '.join(seeds)}")
    print(f"Trials per seed: ~{n_trials}")
    print(f"New framings: {list(NEW_FRAMINGS.keys())}")
    if predictions:
        print(f"Predictions file: {args.predictions} "
              f"({len(predictions)} seeds predicted)")
    print(f"Total calls: ~{calls}")
    print(f"Estimated cost: ${cost:.2f}")
    print(f"Rate limit: {args.rpm} RPM  ->  min ~{mins:.1f} min")

    if args.dry_run:
        return

    asyncio.run(main_async(args.input, args.output, seeds,
                           args.concurrency, args.rpm, predictions))


if __name__ == "__main__":
    main()
