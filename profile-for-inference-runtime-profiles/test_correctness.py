#!/usr/bin/env python3
"""Correctness and cache gates for the Qwen3.8-27B EXL3 two-Spark TP2 profile.

Gates:
  1. Greedy determinism            - same prompt twice, byte-identical output
  2. Warm prefix reuse (APC)       - repeat of a long prompt is materially faster,
                                     byte-identical (the historical hybrid-model
                                     silent-corruption scenario)
  3. Shared-prefix divergence      - two requests sharing a long prefix but
                                     diverging at the end each answer their own
                                     question (catches stale recurrent state)
  4. Post-engine-restart replay    - operator restarts the engine, then this
                                     verifies the external tier served the prefix
  5. Heartbeat window              - a replay issued >4 minutes after engine start
                                     still hits. Stock lmcache 0.5.2 fails here
                                     while passing everything above, because the
                                     scheduler-side heartbeat never starts and the
                                     servers reap the engine's registration after
                                     ~2.5 minutes; lookups then silently return 0.

Usage:
    python3 test_correctness.py --url http://RANK0_LAN_ADDR:8000/v1
    python3 test_correctness.py --url ... --stage replay      # after an engine restart
    python3 test_correctness.py --url ... --stage heartbeat   # >4 min after start
    python3 test_correctness.py --url ... --stage all
"""

import argparse
import sys
import time

import requests

MODEL_NAME = "qwen38"
BLOCK = 1600  # mamba block / lmcache chunk size


def completion(url, prompt, max_tokens=32, model=MODEL_NAME, timeout=600):
    """Send a greedy completion; return (text, wall_seconds, prompt_tokens)."""
    t0 = time.monotonic()
    r = requests.post(
        f"{url}/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        },
        timeout=timeout,
    )
    r.raise_for_status()
    elapsed = time.monotonic() - t0
    body = r.json()
    return body["choices"][0]["text"], elapsed, body["usage"]["prompt_tokens"]


def long_prompt(approx_tokens, seed_text="The quick brown fox jumps over the lazy dog near the riverbank at dawn. "):
    """Build a prompt of roughly approx_tokens tokens (~4 chars/token heuristic)."""
    repeats = max(1, approx_tokens // 16)
    return (seed_text * repeats).strip()


def report(name, passed, details):
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {details}")
    return passed


def gate_determinism(url):
    p = "Compute 17 * 23 and reply with only the number."
    a, _, _ = completion(url, p, max_tokens=16)
    b, _, _ = completion(url, p, max_tokens=16)
    return report("greedy determinism", a == b, f"{a.strip()!r} vs {b.strip()!r}")


def gate_warm_prefix(url):
    p = long_prompt(9000)
    cold_text, cold_s, ntok = completion(url, p)
    warm_text, warm_s, _ = completion(url, p)
    ok = cold_text == warm_text and warm_s < cold_s * 0.75
    return report(
        "warm prefix reuse",
        ok,
        f"{ntok} prompt tokens, {cold_s:.1f}s cold -> {warm_s:.1f}s warm "
        f"({cold_s / warm_s:.1f}x), identical={cold_text == warm_text}",
    )


def gate_shared_prefix(url):
    """Two requests share a long prefix, then ask different questions.

    A stack that resumes on stale recurrent state answers the wrong question or
    emits token salad here while looking healthy on the gates above.
    """
    prefix = long_prompt(6000)
    a_text, _, _ = completion(url, prefix + "\n\nQ: What is 8 plus 5? Reply with only the number.\nA:", max_tokens=8)
    b_text, _, _ = completion(url, prefix + "\n\nQ: What is 20 minus 3? Reply with only the number.\nA:", max_tokens=8)
    ok = "13" in a_text and "17" in b_text
    return report("shared-prefix divergence", ok, f"a={a_text.strip()!r} b={b_text.strip()!r}")


def gate_replay(url, reference_seed):
    """Run after an engine restart: the prefix must come from the external tier.

    Populate first (same seed, before the restart), then restart the engine, then
    run this stage. Check the cache-server logs for
    'Prefetch request completed (L1+L2)' and 'Retrieved N tokens'.
    """
    p = long_prompt(18000, seed_text=reference_seed)
    text, elapsed, ntok = completion(url, p)
    ok = elapsed < 6.0
    return report(
        "post-restart replay",
        ok,
        f"{ntok} prompt tokens in {elapsed:.1f}s "
        f"({'external hit' if ok else 'looks like a full recompute'}); "
        f"tail < {BLOCK} tokens always recomputes",
    )


def gate_heartbeat(url, reference_seed):
    """Same as replay, but meaningful only >4 minutes after engine start."""
    print("  note: only valid if the engine has been up for more than 4 minutes")
    return gate_replay(url, reference_seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="e.g. http://RANK0_LAN_ADDR:8000/v1")
    ap.add_argument("--stage", default="basic", choices=["basic", "replay", "heartbeat", "all"])
    ap.add_argument(
        "--replay-seed",
        default="Seven silver spacecraft slipped silently across the glowing harbor skyline. ",
        help="prompt seed text; use the same value to populate and to replay",
    )
    args = ap.parse_args()

    results = []
    if args.stage in ("basic", "all"):
        results.append(gate_determinism(args.url))
        results.append(gate_warm_prefix(args.url))
        results.append(gate_shared_prefix(args.url))
    if args.stage in ("replay", "all"):
        results.append(gate_replay(args.url, args.replay_seed))
    if args.stage == "heartbeat":
        results.append(gate_heartbeat(args.url, args.replay_seed))

    print(f"\n{sum(results)}/{len(results)} gates passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
