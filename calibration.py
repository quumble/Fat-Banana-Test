"""
calibration.py — Ground-truth check for the self-model hypothesis.

Reuses the P2 word lists already in raw_results.json and asks cold Claudes
two new framings on each:

  - 'self':   "Which ONE of those 5 words would you also list?"   (parallels 'claudes')
  - 'forced': "Of those 5 words, pick ONE."                        (minimal framing baseline)

The test: does 'claudes' (Claude's model of other Claudes) match 'self' (what a
cold Claude actually picks)? If yes, the self-model is calibrated. If no, Claude
holds a stereotype of Claudes that doesn't match what Claudes do.

Writes an augmented raw JSON and prints a calibration table.

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  python calibration.py                                 # uses default paths
  python calibration.py --input raw_results.json        # custom input
  python calibration.py --seeds fire music sun time     # divergent seeds only
  python calibration.py --dry-run
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


async def main_async(input_path, output_path, seeds_filter, concurrency, rpm):
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
        print(f"[{i}/{len(seeds_to_run)}] {seed} (n={len(word_lists)})", flush=True)

        for framing in NEW_FRAMINGS:
            print(f"  {framing}...", flush=True)
            results = await run_framing(client, seed, word_lists, framing, sem, limiter)
            for trial, r in zip(trials, results):
                trial[framing] = r["pick"]
                trial.setdefault("raw", {})[framing] = r["raw"]

        # Checkpoint after every seed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  done in {time.time()-t0:.1f}s  (checkpointed)", flush=True)

    print(f"\nTotal elapsed: {time.time()-start:.1f}s")
    print(f"Output: {output_path}")
    print_calibration_table(data, seeds_to_run)


def print_calibration_table(data, seeds):
    print("\n" + "=" * 100)
    print("Top pick under each framing (count out of 50):")
    print("-" * 100)
    print(f"{'seed':<10} {'people':<14} {'claudes':<14} {'self':<14} {'forced':<14} {'central':<14}")
    print("-" * 100)

    def top(trials, key):
        c = Counter(t.get(key) for t in trials if t.get(key))
        if not c:
            return ("-", 0)
        return c.most_common(1)[0]

    for seed in seeds:
        if seed not in data:
            continue
        trials = data[seed]["trials"]
        cols = [top(trials, k) for k in ("people", "claudes", "self", "forced", "central")]
        row = f"{seed:<10} " + " ".join(f"{w:<9}({n:>2})  " for w, n in cols)
        print(row)

    print("\n" + "=" * 100)
    print("Self-model calibration:  does 'claudes' (model of Claudes) match 'self' (actual Claude)?")
    print("-" * 100)
    for seed in seeds:
        if seed not in data:
            continue
        trials = data[seed]["trials"]
        # Agreement per trial
        agree = sum(1 for t in trials if t.get("claudes") and t["claudes"] == t.get("self"))
        n_valid = sum(1 for t in trials if t.get("claudes") and t.get("self"))
        if n_valid == 0:
            continue
        rate = agree / n_valid
        interp = ""
        if rate >= 0.8:
            interp = "calibrated (self-model ≈ reality)"
        elif rate < 0.3:
            interp = "STEREOTYPE (self-model ≠ reality)"
        else:
            interp = "partial mismatch"
        print(f"  {seed:<10}  agreement = {rate:.2f}   {interp}")


def estimate_cost(n_trials_per_seed, n_seeds, n_framings=2):
    # ~80 input, ~5 output tokens per call; Sonnet 4.6 at $3/$15 per MTok
    calls = n_trials_per_seed * n_seeds * n_framings
    cost = calls * (80 * 3 + 5 * 15) / 1e6
    return calls, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="./sweep_results/raw_results.json")
    ap.add_argument("--output", default="./sweep_results/raw_results_calibrated.json")
    ap.add_argument("--seeds", nargs="*", default=None,
                    help="subset of seeds (default: all in input)")
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    ap.add_argument("--rpm", type=float, default=DEFAULT_RPM)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data = json.load(open(args.input))
    seeds = args.seeds if args.seeds else list(data.keys())
    n_trials = sum(len(data[s]["trials"]) for s in seeds if s in data) // max(1, len(seeds))
    calls, cost = estimate_cost(n_trials, len(seeds))
    mins = calls / args.rpm

    print(f"Input: {args.input}")
    print(f"Seeds: {len(seeds)} — {', '.join(seeds)}")
    print(f"Trials per seed: ~{n_trials}")
    print(f"New framings: {list(NEW_FRAMINGS.keys())}")
    print(f"Total calls: ~{calls}")
    print(f"Estimated cost: ${cost:.2f}")
    print(f"Rate limit: {args.rpm} RPM  →  min ~{mins:.1f} min")

    if args.dry_run:
        return

    asyncio.run(main_async(args.input, args.output, seeds, args.concurrency, args.rpm))


if __name__ == "__main__":
    main()
