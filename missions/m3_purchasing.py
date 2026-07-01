"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing

DAYS = 30


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = 0.0
    recs = []
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours * od

        tier = pricing.recommend_tier(hpd, interruptible)
        if tier == "spot":
            sim = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)
            opt_cost = sim["spot_cost"]
        elif tier == "reserved":
            opt_cost = gpu_hours * num(c["reserved_3yr_hr"])
        else:
            opt_cost = on_demand_cost

        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": tier,
                     "on_demand": round(on_demand_cost), "optimized": round(opt_cost)})

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0

    # --- Extension 1: recommend_tier() v2 — per-GPU interrupt rate + 1yr/3yr comparison ---
    v2_optimized_monthly = 0.0
    v2_recs = []
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        job_days = num(j["days"])
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])

        tier2 = pricing.recommend_tier(hpd, interruptible, gpu_type=gtype,
                                        job_days=job_days, period_days=DAYS)
        if tier2 == "spot":
            sim = pricing.spot_checkpoint_cost(
                gpu_hours, num(c["spot_hr"]), od,
                interrupt_rate=pricing.INTERRUPT_RATE_BY_GPU.get(gtype, 0.05),
            )
            opt2 = sim["spot_cost"]
        elif tier2 == "reserved_3yr":
            opt2 = gpu_hours * num(c["reserved_3yr_hr"])
        elif tier2 == "reserved_1yr":
            opt2 = gpu_hours * num(c["reserved_1yr_hr"])
        else:
            opt2 = gpu_hours * od
        v2_optimized_monthly += opt2
        v2_recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": tier2, "optimized": round(opt2)})

    v2_savings_pct = ((on_demand_monthly - v2_optimized_monthly) / on_demand_monthly * 100
                       if on_demand_monthly else 0.0)

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':7}{'tier':11}{'on-demand':>12}{'optimized':>12}")
        for r in recs:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['tier']:11}${r['on_demand']:>11,}${r['optimized']:>11,}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")

        print("\n-- Extension 1: recommend_tier() v2 (interrupt rate by GPU + 1yr/3yr compare) --")
        print(f"{'job':18}{'gpu':7}{'v1 tier':11}{'v2 tier':13}{'v2 optimized':>13}")
        for r1, r2 in zip(recs, v2_recs):
            print(f"{r1['job_id']:18}{r1['gpu_type']:7}{r1['tier']:11}{r2['tier']:13}${r2['optimized']:>12,}")
        print(f"v1 savings: {savings_pct:.1f}%   v2 savings: {v2_savings_pct:.1f}%   "
              f"(delta {v2_savings_pct - savings_pct:+.1f}pp)")

    return {"recommendations": recs, "on_demand_monthly": round(on_demand_monthly),
            "optimized_monthly": round(optimized_monthly), "savings_pct": round(savings_pct, 1),
            "recommendations_v2": v2_recs, "optimized_monthly_v2": round(v2_optimized_monthly),
            "savings_pct_v2": round(v2_savings_pct, 1)}


if __name__ == "__main__":
    run()
