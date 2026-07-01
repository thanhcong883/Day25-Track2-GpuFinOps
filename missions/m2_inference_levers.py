"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}

# Extension 3: writing a cache entry carries a premium over a normal input token
# (illustrative — Anthropic-style ephemeral cache write ~1.25x the base input price).
CACHE_WRITE_MULTIPLIER = 1.25

# Extension 4: cap reasoning traffic to this share of requests when estimating savings.
REASONING_TRAFFIC_CAP = 0.10


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    base_cost = opt_cost = 0.0
    total_tokens = 0

    # Extension 3 bookkeeping: reused-prefix reads, grouped by (team, project) — a proxy
    # for "how many requests reuse the same cached system prompt".
    cache_reads_by_key = defaultdict(int)

    # Extension 4 bookkeeping: cost/energy split for is_reasoning vs normal traffic.
    reasoning = {"n": 0, "cost": 0.0, "tokens": 0, "wh": 0.0}
    normal = {"n": 0, "cost": 0.0, "tokens": 0, "wh": 0.0}

    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        is_reasoning = bool(int(num(r.get("is_reasoning", 0))))
        total_tokens += inp + out
        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)
        # OPTIMIZED: cascade (route_tier), prompt caching, batch API
        pin, pout = MODEL_PRICES[r["route_tier"]]
        req_cost = pricing.request_cost(inp, out, pin, pout, cached_in=cached, batch=is_batch)
        opt_cost += req_cost

        if cached > 0:
            cache_reads_by_key[(r["team"], r["project"])] += 1

        bucket = reasoning if is_reasoning else normal
        bucket["n"] += 1
        bucket["cost"] += req_cost
        bucket["tokens"] += inp + out
        bucket["wh"] += sustainability.wh_per_query(inp + out, is_reasoning=is_reasoning)

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    # --- Extension 3: is prompt caching actually worth it at our reuse rate? ---
    avg_cache_reads = (
        sum(cache_reads_by_key.values()) / len(cache_reads_by_key) if cache_reads_by_key else 0.0
    )
    cache_verdict = {}
    for tier, (pin, _pout) in MODEL_PRICES.items():
        write_cost = pin * CACHE_WRITE_MULTIPLIER
        breakeven = pricing.cache_breakeven_reads(write_cost, pin)
        cache_verdict[tier] = {
            "breakeven_reads": round(breakeven, 1),
            "worth_it": pricing.cache_is_worth_it(avg_cache_reads, write_cost, pin),
        }

    # --- Extension 4: reasoning traffic's share of $ and Wh, and a capped-traffic estimate ---
    total_n = reasoning["n"] + normal["n"]
    total_cost_all = reasoning["cost"] + normal["cost"]
    reasoning_traffic_pct = reasoning["n"] / total_n * 100 if total_n else 0.0
    reasoning_cost_pct = reasoning["cost"] / total_cost_all * 100 if total_cost_all else 0.0
    reasoning_wh_pct = (
        reasoning["wh"] / (reasoning["wh"] + normal["wh"]) * 100
        if (reasoning["wh"] + normal["wh"]) else 0.0
    )
    target_n = int(total_n * REASONING_TRAFFIC_CAP)
    cap_cost_saved = cap_wh_saved = 0.0
    if reasoning["n"] > target_n and reasoning["n"] > 0:
        avg_r_cost, avg_r_wh = reasoning["cost"] / reasoning["n"], reasoning["wh"] / reasoning["n"]
        avg_n_cost = normal["cost"] / normal["n"] if normal["n"] else 0.0
        avg_n_wh = normal["wh"] / normal["n"] if normal["n"] else 0.0
        excess_n = reasoning["n"] - target_n
        cap_cost_saved = excess_n * max(0.0, avg_r_cost - avg_n_cost)
        cap_wh_saved = excess_n * max(0.0, avg_r_wh - avg_n_wh)

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")

        print("\n-- Extension 3: cache_is_worth_it() --")
        print(f"avg cache reads per (team,project): {avg_cache_reads:.1f}")
        for tier, v in cache_verdict.items():
            print(f"  {tier:6} break-even reads={v['breakeven_reads']:>5}  worth_it={v['worth_it']}")

        print("\n-- Extension 4: reasoning budget --")
        print(f"reasoning traffic: {reasoning_traffic_pct:.1f}% of requests, "
              f"{reasoning_cost_pct:.1f}% of $ cost, {reasoning_wh_pct:.1f}% of Wh")
        if reasoning["n"] > target_n:
            print(f"if capped to {REASONING_TRAFFIC_CAP:.0%} of requests: "
                  f"save ${cap_cost_saved:.4f}/day, {cap_wh_saved:.1f} Wh/day")
        else:
            print(f"already under the {REASONING_TRAFFIC_CAP:.0%} cap ({reasoning_traffic_pct:.1f}% < "
                  f"{REASONING_TRAFFIC_CAP:.0%}) -> nothing to trim on request count, but its "
                  f"{reasoning_wh_pct:.1f}% share of Wh vs {reasoning_traffic_pct:.1f}% share of "
                  f"requests means growth here is disproportionately expensive in energy")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "avg_cache_reads": round(avg_cache_reads, 1), "cache_verdict": cache_verdict,
        "reasoning_traffic_pct": round(reasoning_traffic_pct, 1),
        "reasoning_cost_pct": round(reasoning_cost_pct, 1),
        "reasoning_wh_pct": round(reasoning_wh_pct, 1),
        "reasoning_cap_savings_usd_day": round(cap_cost_saved, 4),
        "reasoning_cap_savings_wh_day": round(cap_wh_saved, 1),
    }


if __name__ == "__main__":
    run()
