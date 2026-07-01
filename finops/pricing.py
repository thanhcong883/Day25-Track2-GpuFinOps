"""Pricing & purchasing economics — measure in $/1M-token, not $/GPU-hr.

Figures are June-2026 as-of snapshots from the deck's RESEARCH dossier; treat
live prices as fast-moving (re-baseline before each cohort).
"""
from __future__ import annotations


def request_cost(
    input_tok: int,
    output_tok: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_in: int = 0,
    cache_discount: float = 0.10,   # Anthropic cached-read ~0.1x (=-90%)
    batch: bool = False,
    batch_discount: float = 0.50,   # Batch API ~ -50%
) -> float:
    """USD cost of a single request. Cached input billed at cache_discount x price."""
    cached_in = min(max(0, cached_in), input_tok)
    uncached_in = input_tok - cached_in
    cost = (
        (uncached_in / 1e6) * price_in_per_m
        + (cached_in / 1e6) * price_in_per_m * cache_discount
        + (output_tok / 1e6) * price_out_per_m
    )
    if batch:
        cost *= batch_discount
    return cost


def dollars_per_million(total_cost_usd: float, total_tokens: int) -> float:
    """Aggregate unit economics: $ per 1,000,000 tokens served."""
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd / (total_tokens / 1e6)


def discount_stack(
    batch: bool = False,
    cache_hit_frac: float = 0.0,
    batch_discount: float = 0.50,
    cache_discount: float = 0.10,
) -> float:
    """Effective fraction of the naive bill after stacking discounts (input-heavy view).

    Discounts MULTIPLY: cache applies to the cached share of input, batch to the
    whole bill. batch + 100% cache-hit -> 0.5 * 0.1 = 0.05 (~95% off).
    """
    cache_mult = cache_hit_frac * cache_discount + (1.0 - cache_hit_frac)
    batch_mult = batch_discount if batch else 1.0
    return cache_mult * batch_mult


def break_even_utilization(discount_frac: float) -> float:
    """Utilization at which a commitment pays off ~= 1 - discount.

    A 45% reserved discount needs ~55% utilization (~13.2h/day) to beat on-demand.
    """
    return max(0.0, min(1.0, 1.0 - discount_frac))


def recommend_tier(
    hours_per_day: float,
    interruptible: bool,
    reserved_discount: float = 0.45,
    gpu_type: str | None = None,
    job_days: float | None = None,
    period_days: float = 30.0,
    reserved_discount_1yr: float = 0.25,
) -> str:
    """Pick a purchasing tier from a workload's duty cycle + interruptibility.

    Base v1 policy (default — used by the graded M3 pipeline, `gpu_type`/`job_days` omitted):
      - interruptible & not 24/7  -> 'spot'      (checkpoint and ride the discount)
      - duty cycle >= break-even  -> 'reserved'  (steady, high utilization)
      - otherwise                 -> 'on_demand' (spiky / low duty)

    Extension 1 policy (pass `gpu_type` and/or `job_days`) adds two factors v1 ignores:
      - interruption rate varies by GPU type (`INTERRUPT_RATE_BY_GPU`) — H100 spot is rarely
        reclaimed (high demand, deep pools); A10G/L4 spot is reclaimed far more often, so
        "interruptible" alone isn't enough to justify spot on those GPUs.
      - a job only active a fraction of the billing period (`job_days` out of `period_days`,
        e.g. a 2-week training burst within a 30-day month) isn't steady enough to amortize a
        3yr lock-in even at a high hourly duty cycle while it runs — it's compared against the
        1yr reserved discount instead.
    """
    duty = max(0.0, hours_per_day) / 24.0
    be_3yr = break_even_utilization(reserved_discount)

    if gpu_type is None and job_days is None:
        if interruptible and hours_per_day < 24:
            return "spot"
        if duty >= be_3yr:
            return "reserved"
        return "on_demand"

    irate = INTERRUPT_RATE_BY_GPU.get(gpu_type, 0.05)
    be_1yr = break_even_utilization(reserved_discount_1yr)
    safe_spot = interruptible and hours_per_day < 24 and irate <= 0.08

    # A safely-interruptible job on a low-reclaim GPU beats reserved at any duty cycle —
    # spot (~40-60% off) undercuts even a 3yr reserved commit (45% off), so check it first.
    if safe_spot:
        return "spot"
    steady = job_days is None or period_days <= 0 or (job_days / period_days) >= 0.85
    if duty >= be_3yr:
        if not steady:
            return "reserved_1yr" if duty >= be_1yr else "on_demand"
        return "reserved_3yr"
    if duty >= be_1yr:
        return "reserved_1yr"
    return "on_demand"


# Extension 1: illustrative per-hour reclaim probability by GPU type. H100/H200 are the
# capacity everyone wants for training, so cloud providers reclaim spot slices on them
# less often; A10G/L4 are cheap "leftover capacity" tiers reclaimed much more readily.
INTERRUPT_RATE_BY_GPU = {"H100": 0.03, "H200": 0.03, "A100": 0.05, "A10G": 0.10, "L4": 0.12}


def cache_breakeven_reads(write_cost_per_m: float, price_in_per_m: float,
                          read_discount: float = 0.10) -> float:
    """Minimum reuse count of a cached prefix before caching pays for itself.

    Writing a cache costs `write_cost_per_m` $/1M tokens once (a premium over normal input
    price). Each later read then costs `read_discount x price_in_per_m` instead of the full
    price, saving `(1 - read_discount) * price_in_per_m` per read. Break-even is the read
    count where cumulative savings cover the one-time write premium.
    """
    saved_per_read = price_in_per_m * (1.0 - read_discount)
    if saved_per_read <= 0:
        return float("inf")
    return write_cost_per_m / saved_per_read


def cache_is_worth_it(avg_cache_reads: float, write_cost_per_m: float,
                      price_in_per_m: float, read_discount: float = 0.10) -> bool:
    """True if a cached prefix is reused often enough to cover its write cost.

    Extension 3: prompt caching is only a net saving once `avg_cache_reads` clears
    `cache_breakeven_reads()` — a prefix read once or twice may cost more than it saves.
    """
    breakeven = cache_breakeven_reads(write_cost_per_m, price_in_per_m, read_discount)
    return avg_cache_reads >= breakeven


def spot_checkpoint_cost(
    job_hours: float,
    spot_hr: float,
    on_demand_hr: float,
    interrupt_rate: float = 0.05,      # per-hour chance (H100 spot ~<5%)
    ckpt_overhead_frac: float = 0.03,  # steady cost of writing checkpoints
    rework_hours_per_interrupt: float = 0.5,
) -> dict:
    """Effective cost of running a checkpointable job on spot vs on-demand.

    Interruptions waste the compute since the last checkpoint (rework); checkpointing
    adds a small steady overhead. Spot still wins for interruptible jobs.
    """
    expected_interrupts = job_hours * interrupt_rate
    rework_hours = expected_interrupts * rework_hours_per_interrupt
    effective_hours = job_hours * (1.0 + ckpt_overhead_frac) + rework_hours
    spot_cost = effective_hours * spot_hr
    on_demand_cost = job_hours * on_demand_hr
    savings_pct = (1.0 - spot_cost / on_demand_cost) * 100.0 if on_demand_cost > 0 else 0.0
    return {
        "spot_effective_hours": round(effective_hours, 2),
        "spot_cost": round(spot_cost, 2),
        "on_demand_cost": round(on_demand_cost, 2),
        "savings_pct": round(savings_pct, 1),
    }
