"""
seed_sweep_mistake.py — Single-seed branch of seed_sweep.py for 'mistake'.

Design:
  Same pipeline as seed_sweep.py but restricted to one affective seed.
  Runs N P2 trials and three P3 framings (people / claudes / central) on
  fresh cold Sonnet 4.6 instances. Output feeds directly into calibration.py
  to add 'self' and 'forced' framings.

Hypothesis being tested:
  The Ember-not-flame paper found a self-model miscalibration on phenomena
  with source/artifact distinctions (rain, sun, cold, thunder): 'claudes'
  predicts a distanced word; 'self' picks the phenomenological primary.
  'mistake' tests whether that miscalibration generalizes from sensory
  primaries to affective concepts. Pre-registered prediction lives in
  PREREGISTRATION.md (committed before this script is run).

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  pip install anthropic
  python seed_sweep_mistake.py --out ./sweep_results_mistake
  python seed_sweep_mistake.py --dry-run
  python seed_sweep_mistake.py --n 10 --out ./sweep_results_mistake_pilot
"""

import argparse
import asyncio
import json
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

# Seed typology — 'mistake' branch.
# Testing whether the self-model miscalibration found on sensory primaries
# (rain→cloud, thunder→rumble, cold→frost, sun→star) extends to affective
# concepts. See PREREGISTRATION.md for the full pre-registered prediction.
SEEDS = {
    "mistake": "affective",
}

FRAMINGS = {
    "people":  "Which ONE of those 5 words would most people also list?",
    "claudes": "Which ONE of those 5 words would other Claudes also list?",
    "central": "Which ONE of those 5 words is most central to the concept?",
}

DEFAULT_N = 50          # trials per seed
DEFAULT_CONCURRENCY = 4
DEFAULT_RPM = 45        # stay under tier 1's 50 RPM cap
MAX_RETRIES = 5

# Rough cost model (Sonnet 4.6: $3/MTok in, $15/MTok out)
COST_IN_PER_MTOK = 3.0
COST_OUT_PER_MTOK = 15.0


# ---------------------- PROMPTS ----------------------

def p2_prompt(seed: str) -> str:
    return f"The thing is: {seed}. List exactly 5 words associated with it. Reply with only the 5 words, comma-separated, nothing else."

def p3_prompt(seed: str, words: list[str], framing_key: str) -> str:
    word_list = ", ".join(words)
    q = FRAMINGS[framing_key]
    return (
        f"The thing is: {seed}. Here are 5 words associated with it: {word_list}.\n\n"
        f"{q} Reply with only that word, nothing else."
    )


# ---------------------- PARSING ----------------------

def parse_p2(text: str) -> list[str] | None:
    """Parse a comma-separated 5-word list. Returns None if malformed."""
    cleaned = re.sub(r"[^\w,\s'-]", "", text.strip())
    words = [w.strip().lower() for w in cleaned.split(",") if w.strip()]
    return words if len(words) == 5 else None

def parse_p3(text: str, valid: list[str]) -> tuple[str, bool]:
    """Return (parsed_word, is_in_valid_set)."""
    cleaned = re.sub(r"[^\w\s'-]", "", text.strip().lower())
    # Take first token; models sometimes add "The word is X" despite instructions
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
                    max_tokens=60,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text
            except RateLimitError as e:
                # Respect retry-after if present, else exponential backoff
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


async def run_p2_batch(client, seed: str, n: int, sem, limiter) -> list[list[str]]:
    """Returns n valid 5-word lists for the seed."""
    prompt = p2_prompt(seed)
    results: list[list[str]] = []
    attempts = 0
    while len(results) < n and attempts < n * 2:
        needed = n - len(results)
        batch = await asyncio.gather(*[call(client, prompt, sem, limiter) for _ in range(needed)])
        for raw in batch:
            if raw is None:
                continue
            parsed = parse_p2(raw)
            if parsed is not None:
                results.append(parsed)
                if len(results) == n:
                    break
        attempts += needed
    return results[:n]


async def run_p3_batch(client, seed: str, word_lists: list[list[str]], framing: str, sem, limiter) -> list[dict]:
    """For each word list, run one P3 call with the given framing."""
    prompts = [p3_prompt(seed, words, framing) for words in word_lists]
    raw_results = await asyncio.gather(*[call(client, p, sem, limiter) for p in prompts])
    out = []
    for words, raw in zip(word_lists, raw_results):
        if raw is None:
            out.append({"p2_words": words, "raw": None, "pick": None, "valid": False})
            continue
        pick, valid = parse_p3(raw, words)
        out.append({"p2_words": words, "raw": raw.strip(), "pick": pick, "valid": valid})
    return out


# ---------------------- MAIN ----------------------

async def run_experiment(seeds: list[str], n: int, concurrency: int, rpm: float, out_dir: Path):
    client = AsyncAnthropic()
    sem = asyncio.Semaphore(concurrency)
    limiter = RateLimiter(rpm)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_data: dict[str, dict] = {}
    start = time.time()

    for i, seed in enumerate(seeds, 1):
        t0 = time.time()
        print(f"[{i}/{len(seeds)}] {seed} — generating P2 associations...", flush=True)
        word_lists = await run_p2_batch(client, seed, n, sem, limiter)
        if len(word_lists) < n:
            print(f"  warning: only {len(word_lists)}/{n} valid P2 lists", file=sys.stderr)

        framing_results: dict[str, list[dict]] = {}
        for framing in FRAMINGS:
            print(f"  running P3 [{framing}]...", flush=True)
            framing_results[framing] = await run_p3_batch(client, seed, word_lists, framing, sem, limiter)

        all_data[seed] = {
            "seed": seed,
            "type": SEEDS.get(seed, "unknown"),
            "n": len(word_lists),
            "trials": [
                {
                    "p2_words": word_lists[j],
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

        # Checkpoint after every seed so a crash mid-run doesn't wipe progress
        ckpt_path = out_dir / "raw_results.json"
        with ckpt_path.open("w") as f:
            json.dump(all_data, f, indent=2)
        print(f"  done in {time.time()-t0:.1f}s  (checkpointed)", flush=True)

    # Save summary (raw_results.json already checkpointed per-seed)
    raw_path = out_dir / "raw_results.json"
    print(f"\nRaw data: {raw_path}")

    # Summarize
    summary = summarize(all_data)
    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary:  {summary_path}")

    print_summary_table(summary)
    print(f"\nTotal elapsed: {time.time()-start:.1f}s")


def summarize(all_data: dict) -> dict:
    """Compute top picks per framing and agreement between claudes/central."""
    summary = {}
    for seed, d in all_data.items():
        trials = d["trials"]
        if not trials:
            continue
        counts = {k: Counter(t[k] for t in trials if t[k]) for k in FRAMINGS}

        # Agreement: claudes and central pick the same word on the same trial
        claudes_central_agree = sum(
            1 for t in trials if t["claudes"] and t["claudes"] == t["central"]
        )
        people_claudes_agree = sum(
            1 for t in trials if t["people"] and t["people"] == t["claudes"]
        )

        total = len(trials)
        summary[seed] = {
            "type": d["type"],
            "n": total,
            "top_people":  counts["people"].most_common(3),
            "top_claudes": counts["claudes"].most_common(3),
            "top_central": counts["central"].most_common(3),
            "claudes_central_agreement": round(claudes_central_agree / total, 3),
            "people_claudes_agreement":  round(people_claudes_agree / total, 3),
        }
    return summary


def print_summary_table(summary: dict):
    print("\n" + "=" * 90)
    print(f"{'seed':<10} {'type':<26} {'people':<12} {'claudes':<12} {'central':<12} {'c≈central':>10}")
    print("-" * 90)
    # Group by type for readability
    by_type = defaultdict(list)
    for seed, s in summary.items():
        by_type[s["type"]].append((seed, s))

    for tp in ["artifact", "phenomenon_with_artifact", "phenomenon_bare", "abstract", "affective"]:
        for seed, s in by_type.get(tp, []):
            top_p = s["top_people"][0][0]  if s["top_people"]  else "-"
            top_c = s["top_claudes"][0][0] if s["top_claudes"] else "-"
            top_x = s["top_central"][0][0] if s["top_central"] else "-"
            agree = s["claudes_central_agreement"]
            flag = " ←diverge" if agree < 0.5 else ""
            print(f"{seed:<10} {tp:<26} {top_p:<12} {top_c:<12} {top_x:<12} {agree:>10.2f}{flag}")
        print()

    # Aggregate by type
    print("Typology check — mean claudes↔central agreement by seed class:")
    for tp, items in by_type.items():
        if not items:
            continue
        mean = sum(s["claudes_central_agreement"] for _, s in items) / len(items)
        print(f"  {tp:<28} {mean:.2f}  (n={len(items)} seeds)")


def estimate_cost(seeds: list[str], n: int) -> float:
    # P2: ~30 input, ~30 output tokens
    p2_calls = len(seeds) * n
    p2_in = p2_calls * 30
    p2_out = p2_calls * 30
    # P3: ~80 input, ~5 output, three framings each
    p3_calls = len(seeds) * n * 3
    p3_in = p3_calls * 80
    p3_out = p3_calls * 5
    total_in = p2_in + p3_in
    total_out = p2_out + p3_out
    cost = (total_in / 1e6) * COST_IN_PER_MTOK + (total_out / 1e6) * COST_OUT_PER_MTOK
    return cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="trials per seed")
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    ap.add_argument("--rpm", type=float, default=DEFAULT_RPM,
                    help="max requests per minute (tier 1 cap is 50, default 45)")
    ap.add_argument("--seeds", nargs="*", default=None, help="subset of seeds to run")
    ap.add_argument("--out", type=str, default="./sweep_results")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seeds = args.seeds if args.seeds else list(SEEDS.keys())
    unknown = [s for s in seeds if s not in SEEDS]
    if unknown:
        print(f"Unknown seeds: {unknown}\nAvailable: {list(SEEDS.keys())}", file=sys.stderr)
        sys.exit(1)

    total_calls = len(seeds) * args.n * 4   # P2 + 3 × P3
    cost = estimate_cost(seeds, args.n)
    est_minutes = total_calls / args.rpm
    print(f"Seeds: {len(seeds)} ({', '.join(seeds)})")
    print(f"Trials per seed: {args.n}")
    print(f"Total API calls: ~{total_calls}")
    print(f"Estimated cost: ${cost:.2f}")
    print(f"Rate limit: {args.rpm} RPM  →  min ~{est_minutes:.1f} min wall clock")
    print(f"Concurrency: {args.concurrency}")

    if args.dry_run:
        print("\n(dry run — exiting without API calls)")
        return

    asyncio.run(run_experiment(seeds, args.n, args.concurrency, args.rpm, Path(args.out)))


if __name__ == "__main__":
    main()
