"""
seed_sweep_affective_trial_4.py — Trial 4 of the affective sweep.

What changed from Trial 3, and why:

  Trial 3 preregistered per-seed directional predictions (e.g. for regret:
  claudes→remorse, self→sorry) that could not be confirmed because four of
  the five predicted `self` words (sorry, afraid, glad, mad) never appeared
  in any P2 list for their seed. The `self` framing can only pick from
  what P2 emits, so a prediction naming a word outside the P2 choice set
  is structurally unfalsifiable in the confirming direction. The register-
  axis interpretation that motivated those predictions was falsified and
  is not carried forward here.

  Trial 4 fixes this with a two-stage workflow:

    Stage 1 (--stage peek):   run ONLY P2 for each seed. Write out the P2
                              distributions. Stop. The researcher inspects
                              the choice sets and then writes directional
                              predictions over words that actually appear
                              in P2. Predictions get committed to a fresh
                              prereg and timestamped BEFORE Stage 2.

    Stage 2 (--stage full):   run the full P3 sweep, loading the frozen P2
                              lists from Stage 1 via --p2-peek-file. No P2
                              regeneration, so the choice sets the prereg
                              was written against are exactly the choice
                              sets used.

  Additional diagnostics Trial 3 lacked:

    - P2 expanded variant (optional, --include-expanded-p2): a 10-word P2
      with an explicit "include both formal and colloquial terms"
      instruction. Secondary diagnostic only — tells us whether a given
      register is absent from the concept's neighborhood or just absent
      from the 5-word format. Not used in the primary sweep.

    - Ceiling-effect flag: seeds where a single P2 word dominates every
      framing >0.9 are flagged in the summary as uninformative for
      framing-contrast purposes (anger in Trial 3 was this).

    - P2 heterogeneity reporting: unique P2 list count and top-word
      dominance per seed, so a reader can see how comparable the 50
      trials are.

  The shuffled-P3 protocol from Trial 2 is retained — position bias was
  already ruled out for the mistake seed and there's no reason to
  re-verify it here.

  The seed set is left user-configurable. Trial 4 is not committed to the
  Trial 3 affectives; the narrower hypothesis worth testing (seeds where
  P2 emits a word with strong AI-assistant training associations are the
  ones that miscalibrate) would motivate a different or expanded seed
  list. That is a prereg decision, not a script decision.

Usage:
  # Stage 1 — P2 only, for prereg writing
  python seed_sweep_affective_trial_4.py --stage peek --out ./peek_trial_4

  # (researcher inspects peek_trial_4/p2_peek.json, writes + commits prereg)

  # Stage 2 — full pipeline, frozen P2
  python seed_sweep_affective_trial_4.py --stage full \\
      --p2-peek-file ./peek_trial_4/p2_peek.json \\
      --out ./sweep_results_trial_4

  # Optional expanded-P2 diagnostic alongside Stage 1
  python seed_sweep_affective_trial_4.py --stage peek --include-expanded-p2 \\
      --out ./peek_trial_4

  # Dry run / pilot
  python seed_sweep_affective_trial_4.py --stage peek --dry-run
  python seed_sweep_affective_trial_4.py --stage full --seeds regret --n 10 \\
      --p2-peek-file ./peek_trial_4/p2_peek.json
"""

import argparse
import asyncio
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from anthropic import AsyncAnthropic, APIError, RateLimitError


# ---------------------- RATE LIMITER ----------------------

class RateLimiter:
    """Serializes call starts to stay under a requests-per-minute cap."""
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


# ---------------------- CONFIG ----------------------

MODEL = "claude-sonnet-4-6"

# Trial 4 seed list — left as the Trial 3 affectives by default for direct
# comparison, but the script is agnostic. A prereg targeting the narrower
# "AI-assistant-coded P2 word predicts miscalibration" hypothesis should
# pick a seed set where some seeds are expected to emit assistant-coded
# words in P2 (like mistake→error) and some aren't.
SEEDS = {
    "regret": "affective",
    "shame":  "affective",
    "fear":   "affective",
    "joy":    "affective",
    "anger":  "affective",
}

FRAMINGS = {
    "people":  "Which ONE of those 5 words would most people also list?",
    "claudes": "Which ONE of those 5 words would other Claudes also list?",
    "central": "Which ONE of those 5 words is most central to the concept?",
}

# P2 variants. "primary" is the 5-word canonical format used throughout
# the Ember-not-flame line. "expanded" is a secondary diagnostic that
# widens the choice set and nudges for register variety — used only to
# check whether a given colloquial/formal register is actually
# recoverable from the concept at all, not fed into P3.
P2_VARIANTS = {
    "primary": {
        "n_words": 5,
        "prompt_template": (
            "The thing is: {seed}. List exactly 5 words associated with it. "
            "Reply with only the 5 words, comma-separated, nothing else."
        ),
    },
    "expanded": {
        "n_words": 10,
        "prompt_template": (
            "The thing is: {seed}. List exactly 10 words associated with it, "
            "including both formal and colloquial terms if they apply. "
            "Reply with only the 10 words, comma-separated, nothing else."
        ),
    },
}

DEFAULT_N = 50          # trials per seed
DEFAULT_CONCURRENCY = 4
DEFAULT_RPM = 45        # stay under tier 1's 50 RPM cap
MAX_RETRIES = 5

# Ceiling-effect threshold: if a single P2 word is picked by every
# framing (people, claudes, central, self, forced) in >= this fraction
# of trials, the seed is flagged as framing-insensitive.
CEILING_THRESHOLD = 0.90

# Rough cost model (Sonnet 4.6: $3/MTok in, $15/MTok out)
COST_IN_PER_MTOK = 3.0
COST_OUT_PER_MTOK = 15.0


# ---------------------- PROMPTS ----------------------

def p2_prompt(seed: str, variant: str = "primary") -> str:
    return P2_VARIANTS[variant]["prompt_template"].format(seed=seed)

def p3_prompt(seed: str, words: list[str], framing_key: str) -> str:
    word_list = ", ".join(words)
    q = FRAMINGS[framing_key]
    return (
        f"The thing is: {seed}. Here are 5 words associated with it: {word_list}.\n\n"
        f"{q} Reply with only that word, nothing else."
    )


# ---------------------- PARSING ----------------------

def parse_p2(text: str, expected_len: int = 5) -> list[str] | None:
    """Parse a comma-separated word list of the expected length."""
    cleaned = re.sub(r"[^\w,\s'-]", "", text.strip())
    words = [w.strip().lower() for w in cleaned.split(",") if w.strip()]
    return words if len(words) == expected_len else None

def parse_p3(text: str, valid: list[str]) -> tuple[str, bool]:
    """Return (parsed_word, is_in_valid_set)."""
    cleaned = re.sub(r"[^\w\s'-]", "", text.strip().lower())
    tokens = cleaned.split()
    if not tokens:
        return "", False
    # Check each token against the valid set, prefer last match (handles "the answer is X")
    for tok in reversed(tokens):
        if tok in valid:
            return tok, True
    return tokens[0], False


# ---------------------- API ----------------------

async def call(client: AsyncAnthropic, prompt: str, semaphore: asyncio.Semaphore, limiter: RateLimiter) -> str | None:
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            await limiter.acquire()
            try:
                resp = await client.messages.create(
                    model=MODEL,
                    max_tokens=120,  # bumped to accommodate 10-word expanded P2
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
                    print(f"  429 (final): giving up after {MAX_RETRIES} attempts", file=sys.stderr)
                    return None
                print(f"  429: waiting {wait:.1f}s (attempt {attempt+1}/{MAX_RETRIES})", file=sys.stderr)
                await asyncio.sleep(wait)
            except APIError as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"  API error (final): {e}", file=sys.stderr)
                    return None
                await asyncio.sleep(2 ** attempt)
        return None


async def run_p2_batch(client, seed: str, n: int, sem, limiter, variant: str = "primary") -> list[list[str]]:
    """Returns n valid word lists of the variant's expected length for the seed."""
    prompt = p2_prompt(seed, variant)
    expected_len = P2_VARIANTS[variant]["n_words"]
    results: list[list[str]] = []
    attempts = 0
    while len(results) < n and attempts < n * 2:
        needed = n - len(results)
        batch = await asyncio.gather(*[call(client, prompt, sem, limiter) for _ in range(needed)])
        for raw in batch:
            if raw is None:
                continue
            parsed = parse_p2(raw, expected_len=expected_len)
            if parsed is not None:
                results.append(parsed)
                if len(results) == n:
                    break
        attempts += needed
    return results[:n]


async def run_p3_batch(client, seed: str, word_lists: list[list[str]], framing: str, sem, limiter) -> list[dict]:
    """For each word list, shuffle deterministically, run one P3 call with the given framing."""
    shuffled_lists = []
    for trial_idx, words in enumerate(word_lists):
        rng = random.Random(f"{seed}:{trial_idx}:{framing}")
        order = list(words)
        rng.shuffle(order)
        shuffled_lists.append(order)

    prompts = [p3_prompt(seed, words, framing) for words in shuffled_lists]
    raw_results = await asyncio.gather(*[call(client, p, sem, limiter) for p in prompts])
    out = []
    for original_words, shuffled_words, raw in zip(word_lists, shuffled_lists, raw_results):
        if raw is None:
            out.append({
                "p2_words": original_words,
                "p3_words_shuffled": shuffled_words,
                "raw": None, "pick": None, "valid": False,
            })
            continue
        pick, valid = parse_p3(raw, original_words)
        out.append({
            "p2_words": original_words,
            "p3_words_shuffled": shuffled_words,
            "raw": raw.strip(),
            "pick": pick,
            "valid": valid,
        })
    return out


# ---------------------- PEEK STAGE ----------------------

async def run_peek(seeds: list[str], n: int, concurrency: int, rpm: float,
                   out_dir: Path, include_expanded: bool):
    """Stage 1: generate only P2 distributions so a prereg can be written
    against actual choice sets. No P3 calls are made."""
    client = AsyncAnthropic()
    sem = asyncio.Semaphore(concurrency)
    limiter = RateLimiter(rpm)
    out_dir.mkdir(parents=True, exist_ok=True)

    peek_data: dict[str, dict] = {}
    start = time.time()

    variants = ["primary"] + (["expanded"] if include_expanded else [])

    for i, seed in enumerate(seeds, 1):
        t0 = time.time()
        print(f"[{i}/{len(seeds)}] {seed} — peek P2 ({', '.join(variants)})...", flush=True)
        seed_data = {
            "seed": seed,
            "type": SEEDS.get(seed, "unknown"),
            "n": n,
        }
        for variant in variants:
            word_lists = await run_p2_batch(client, seed, n, sem, limiter, variant=variant)
            if len(word_lists) < n:
                print(f"  warning ({variant}): only {len(word_lists)}/{n} valid P2 lists",
                      file=sys.stderr)
            # Diagnostics per variant
            all_words = [w for lst in word_lists for w in lst]
            word_counts = Counter(all_words)
            unique_lists = {tuple(sorted(lst)) for lst in word_lists}
            seed_data[variant] = {
                "word_lists": word_lists,
                "unique_list_count": len(unique_lists),
                "word_frequencies": word_counts.most_common(),
                "n_collected": len(word_lists),
            }

        peek_data[seed] = seed_data
        ckpt_path = out_dir / "p2_peek.json"
        with ckpt_path.open("w") as f:
            json.dump(peek_data, f, indent=2)
        print(f"  done in {time.time()-t0:.1f}s  (checkpointed: {ckpt_path})", flush=True)

    print(f"\nPeek complete. P2 lists saved to: {out_dir/'p2_peek.json'}")
    print("\nBefore running Stage 2 (--stage full), inspect the P2 distributions,")
    print("write directional predictions against words that appear in the PRIMARY P2")
    print("choice set, and commit the prereg + this peek file to git.\n")
    print_peek_summary(peek_data)
    print(f"\nTotal elapsed: {time.time()-start:.1f}s")


def print_peek_summary(peek_data: dict):
    print("\n" + "=" * 80)
    print("PEEK SUMMARY — primary (5-word) P2 candidate sets")
    print("=" * 80)
    for seed, d in peek_data.items():
        prim = d.get("primary", {})
        print(f"\n{seed}:  {prim.get('n_collected', 0)} trials,  "
              f"{prim.get('unique_list_count', 0)} unique lists")
        top = prim.get("word_frequencies", [])[:10]
        for w, c in top:
            print(f"    {w:<20} {c}")
        if "expanded" in d:
            exp = d["expanded"]
            print(f"  [expanded P2, top 10 of {exp.get('n_collected', 0)} trials]")
            for w, c in exp.get("word_frequencies", [])[:10]:
                print(f"    {w:<20} {c}")


# ---------------------- FULL STAGE ----------------------

def load_peek(peek_file: Path) -> dict:
    with peek_file.open() as f:
        return json.load(f)


async def run_experiment(seeds: list[str], n: int, concurrency: int, rpm: float,
                         out_dir: Path, peek_data: dict | None):
    """Stage 2: full pipeline. If peek_data is provided, its PRIMARY P2 lists
    are reused verbatim and no fresh P2 is generated."""
    client = AsyncAnthropic()
    sem = asyncio.Semaphore(concurrency)
    limiter = RateLimiter(rpm)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_data: dict[str, dict] = {}
    start = time.time()

    for i, seed in enumerate(seeds, 1):
        t0 = time.time()
        if peek_data is not None and seed in peek_data and "primary" in peek_data[seed]:
            word_lists = peek_data[seed]["primary"]["word_lists"][:n]
            print(f"[{i}/{len(seeds)}] {seed} — using frozen P2 from peek "
                  f"({len(word_lists)} trials)", flush=True)
            if len(word_lists) < n:
                print(f"  warning: peek file has only {len(word_lists)} trials, "
                      f"requested {n}", file=sys.stderr)
        else:
            print(f"[{i}/{len(seeds)}] {seed} — generating P2 fresh (no peek provided)...",
                  flush=True)
            word_lists = await run_p2_batch(client, seed, n, sem, limiter)
            if len(word_lists) < n:
                print(f"  warning: only {len(word_lists)}/{n} valid P2 lists",
                      file=sys.stderr)

        framing_results: dict[str, list[dict]] = {}
        for framing in FRAMINGS:
            print(f"  running P3 [{framing}]...", flush=True)
            framing_results[framing] = await run_p3_batch(
                client, seed, word_lists, framing, sem, limiter
            )

        all_data[seed] = {
            "seed": seed,
            "type": SEEDS.get(seed, "unknown"),
            "n": len(word_lists),
            "p2_source": "frozen_peek" if peek_data is not None else "fresh",
            "shuffle_note": (
                "P3 prompts saw p3_words_shuffled[framing] (deterministic per "
                "seed:trial:framing). calibration.py reads p2_words (original order) "
                "and is unaffected."
            ),
            "trials": [
                {
                    "p2_words": word_lists[j],
                    "p3_words_shuffled": {
                        k: framing_results[k][j]["p3_words_shuffled"] for k in FRAMINGS
                    },
                    "people":  framing_results["people"][j]["pick"],
                    "claudes": framing_results["claudes"][j]["pick"],
                    "central": framing_results["central"][j]["pick"],
                    "raw": {
                        k: framing_results[k][j]["raw"] for k in FRAMINGS
                    },
                }
                for j in range(len(word_lists))
            ],
        }

        ckpt_path = out_dir / "raw_results.json"
        with ckpt_path.open("w") as f:
            json.dump(all_data, f, indent=2)
        print(f"  done in {time.time()-t0:.1f}s  (checkpointed)", flush=True)

    raw_path = out_dir / "raw_results.json"
    print(f"\nRaw data: {raw_path}")

    summary = summarize(all_data)
    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary:  {summary_path}")

    print_summary_table(summary)
    print(f"\nTotal elapsed: {time.time()-start:.1f}s")


# ---------------------- SUMMARY ----------------------

def summarize(all_data: dict) -> dict:
    """Per-seed: framing picks, claudes↔central agreement, and the
    stability diagnostics Trial 3 lacked (unique P2 lists, top-word
    dominance, ceiling-effect flag)."""
    summary = {}
    for seed, d in all_data.items():
        trials = d["trials"]
        if not trials:
            continue
        total = len(trials)
        counts = {k: Counter(t[k] for t in trials if t[k]) for k in FRAMINGS}

        claudes_central_agree = sum(
            1 for t in trials if t["claudes"] and t["claudes"] == t["central"]
        )
        people_claudes_agree = sum(
            1 for t in trials if t["people"] and t["people"] == t["claudes"]
        )

        # P2 heterogeneity diagnostics
        unique_p2 = {tuple(sorted(t["p2_words"])) for t in trials}
        all_p2_words = [w for t in trials for w in t["p2_words"]]
        p2_word_freq = Counter(all_p2_words)
        top5 = p2_word_freq.most_common(5)
        top5_share = sum(c for _, c in top5) / max(1, len(all_p2_words))

        # Ceiling-effect check on the framings we have here (people,
        # claudes, central). If every one of them picks the same word
        # in >= CEILING_THRESHOLD of trials, flag. calibration.py can
        # extend this to self/forced in its own summary.
        ceiling_word = None
        if counts["people"] and counts["claudes"] and counts["central"]:
            top_p, cnt_p = counts["people"].most_common(1)[0]
            top_c, cnt_c = counts["claudes"].most_common(1)[0]
            top_x, cnt_x = counts["central"].most_common(1)[0]
            if top_p == top_c == top_x:
                min_share = min(cnt_p, cnt_c, cnt_x) / total
                if min_share >= CEILING_THRESHOLD:
                    ceiling_word = top_p

        summary[seed] = {
            "type": d["type"],
            "n": total,
            "p2_source": d.get("p2_source", "unknown"),
            "top_people":  counts["people"].most_common(3),
            "top_claudes": counts["claudes"].most_common(3),
            "top_central": counts["central"].most_common(3),
            "claudes_central_agreement": round(claudes_central_agree / total, 3),
            "people_claudes_agreement":  round(people_claudes_agree / total, 3),
            "unique_p2_lists": len(unique_p2),
            "p2_top5_share": round(top5_share, 3),
            "p2_top_word": p2_word_freq.most_common(1)[0] if p2_word_freq else None,
            "ceiling_word_sweep_framings": ceiling_word,
        }
    return summary


def print_summary_table(summary: dict):
    print("\n" + "=" * 100)
    print(f"{'seed':<10} {'n':>3} {'src':<7} {'people':<12} {'claudes':<12} "
          f"{'central':<12} {'c≈x':>6} {'uniq':>5} {'ceil':<10}")
    print("-" * 100)
    by_type = defaultdict(list)
    for seed, s in summary.items():
        by_type[s["type"]].append((seed, s))

    for tp in ["artifact", "phenomenon_with_artifact", "phenomenon_bare", "abstract", "affective"]:
        for seed, s in by_type.get(tp, []):
            top_p = s["top_people"][0][0]  if s["top_people"]  else "-"
            top_c = s["top_claudes"][0][0] if s["top_claudes"] else "-"
            top_x = s["top_central"][0][0] if s["top_central"] else "-"
            agree = s["claudes_central_agreement"]
            uniq = s["unique_p2_lists"]
            ceil = s["ceiling_word_sweep_framings"] or ""
            src = s.get("p2_source", "?")[:7]
            print(f"{seed:<10} {s['n']:>3} {src:<7} {top_p:<12} {top_c:<12} "
                  f"{top_x:<12} {agree:>6.2f} {uniq:>5} {ceil:<10}")
        if by_type.get(tp):
            print()

    flagged = [seed for seed, s in summary.items() if s["ceiling_word_sweep_framings"]]
    if flagged:
        print(f"Ceiling-flagged seeds (single word dominates every sweep framing "
              f">={CEILING_THRESHOLD:.0%}): {', '.join(flagged)}")
        print("  → framing-contrast claims on these seeds are not interpretable.")


# ---------------------- CLI ----------------------

def estimate_cost(seeds: list[str], n: int, stage: str, include_expanded: bool) -> float:
    # P2 (primary): ~30 in, ~30 out
    p2_primary_in = len(seeds) * n * 30
    p2_primary_out = len(seeds) * n * 30
    # P2 (expanded): ~40 in, ~60 out
    p2_expanded_in = len(seeds) * n * 40 if include_expanded else 0
    p2_expanded_out = len(seeds) * n * 60 if include_expanded else 0
    # P3: ~80 in, ~5 out, three framings each — only in full stage
    p3_in = len(seeds) * n * 3 * 80 if stage == "full" else 0
    p3_out = len(seeds) * n * 3 * 5 if stage == "full" else 0
    total_in = p2_primary_in + p2_expanded_in + p3_in
    total_out = p2_primary_out + p2_expanded_out + p3_out
    return (total_in / 1e6) * COST_IN_PER_MTOK + (total_out / 1e6) * COST_OUT_PER_MTOK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["peek", "full"], default="full",
                    help="peek = P2 only (write before prereg); full = P3 sweep "
                         "(needs --p2-peek-file to use frozen P2)")
    ap.add_argument("--p2-peek-file", type=str, default=None,
                    help="path to p2_peek.json from Stage 1; required for a valid "
                         "Stage 2 run against a committed prereg")
    ap.add_argument("--include-expanded-p2", action="store_true",
                    help="also run the 10-word expanded-P2 diagnostic in Stage 1")
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="trials per seed")
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    ap.add_argument("--rpm", type=float, default=DEFAULT_RPM,
                    help="max requests per minute (tier 1 cap is 50, default 45)")
    ap.add_argument("--seeds", nargs="*", default=None, help="subset of seeds to run")
    ap.add_argument("--out", type=str, default="./sweep_results_trial_4")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seeds = args.seeds if args.seeds else list(SEEDS.keys())
    unknown = [s for s in seeds if s not in SEEDS]
    if unknown:
        print(f"Unknown seeds: {unknown}\nAvailable: {list(SEEDS.keys())}", file=sys.stderr)
        sys.exit(1)

    # Stage-specific validation
    peek_data = None
    if args.stage == "full":
        if args.p2_peek_file is None:
            print("WARNING: --stage full without --p2-peek-file.", file=sys.stderr)
            print("  Trial 4's whole point is frozen P2 from a pre-prereg peek.",
                  file=sys.stderr)
            print("  Continuing with fresh P2 generation — this reverts to the "
                  "Trial 3 workflow.", file=sys.stderr)
        else:
            peek_path = Path(args.p2_peek_file)
            if not peek_path.exists():
                print(f"ERROR: peek file not found: {peek_path}", file=sys.stderr)
                sys.exit(1)
            peek_data = load_peek(peek_path)
            missing = [s for s in seeds if s not in peek_data]
            if missing:
                print(f"ERROR: peek file missing seeds: {missing}", file=sys.stderr)
                sys.exit(1)

    if args.include_expanded_p2 and args.stage != "peek":
        print("WARNING: --include-expanded-p2 only runs in --stage peek "
              "(secondary diagnostic, not part of P3 sweep).", file=sys.stderr)

    # Cost estimate
    include_expanded = args.include_expanded_p2 and args.stage == "peek"
    if args.stage == "peek":
        # Peek only makes P2 calls
        total_calls = len(seeds) * args.n * (2 if include_expanded else 1)
    else:
        # Full: P3 framings × 3; P2 only if no peek file
        p2_calls = 0 if peek_data is not None else len(seeds) * args.n
        total_calls = p2_calls + len(seeds) * args.n * 3
    cost = estimate_cost(seeds, args.n, args.stage, include_expanded)
    est_minutes = total_calls / args.rpm

    print(f"Stage: {args.stage}")
    print(f"Seeds: {len(seeds)} ({', '.join(seeds)})")
    print(f"Trials per seed: {args.n}")
    if args.stage == "full":
        print(f"P2 source: {'frozen from peek' if peek_data else 'fresh (warning: not preregged-against)'}")
    if include_expanded:
        print(f"Expanded P2 diagnostic: ON")
    print(f"Total API calls: ~{total_calls}")
    print(f"Estimated cost: ${cost:.2f}")
    print(f"Rate limit: {args.rpm} RPM  →  min ~{est_minutes:.1f} min wall clock")
    print(f"Concurrency: {args.concurrency}")

    if args.dry_run:
        print("\n(dry run — exiting without API calls)")
        return

    if args.stage == "peek":
        asyncio.run(run_peek(seeds, args.n, args.concurrency, args.rpm,
                             Path(args.out), include_expanded))
    else:
        asyncio.run(run_experiment(seeds, args.n, args.concurrency, args.rpm,
                                   Path(args.out), peek_data))


if __name__ == "__main__":
    main()
