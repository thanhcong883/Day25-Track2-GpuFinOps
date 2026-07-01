"""Extra tests for the "Your Turn" extensions (not part of the graded 15).

These do not modify any graded test file — they cover the new functions added for
Extension 1 (recommend_tier v2) and Extension 3 (cache_is_worth_it).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finops import pricing


def test_recommend_tier_v1_unchanged_without_extension_args():
    # Same three cases graded test_pricing.py checks — must still hold with new kwargs present.
    assert pricing.recommend_tier(2, True) == "spot"
    assert pricing.recommend_tier(24, False) == "reserved"
    assert pricing.recommend_tier(4, False) == "on_demand"


def test_recommend_tier_v2_respects_gpu_interrupt_rate():
    # A10G reclaim rate (0.10) is above the 0.08 safe-spot bar, so a short, low-duty,
    # interruptible A10G job should NOT get spot even though v1 would recommend it.
    v1 = pricing.recommend_tier(8, True)
    v2 = pricing.recommend_tier(8, True, gpu_type="A10G", job_days=22, period_days=30)
    assert v1 == "spot"
    assert v2 == "on_demand"

    # H100 (low reclaim rate) keeps the spot recommendation under the same duty cycle.
    v2_h100 = pricing.recommend_tier(8, True, gpu_type="H100", job_days=22, period_days=30)
    assert v2_h100 == "spot"


def test_recommend_tier_v2_short_job_avoids_3yr_lockin():
    # High duty cycle but only active 5 of 30 days this period -> not steady enough for 3yr.
    tier = pricing.recommend_tier(20, False, gpu_type="H100", job_days=5, period_days=30)
    assert tier in ("reserved_1yr", "on_demand")
    assert tier != "reserved_3yr"

    # Same duty cycle, active nearly every day -> steady enough for the 3yr commitment.
    tier_steady = pricing.recommend_tier(20, False, gpu_type="H100", job_days=30, period_days=30)
    assert tier_steady == "reserved_3yr"


def test_cache_breakeven_reads_and_worth_it():
    breakeven = pricing.cache_breakeven_reads(write_cost_per_m=3.75, price_in_per_m=3.00)
    assert breakeven > 0
    assert pricing.cache_is_worth_it(breakeven + 1, write_cost_per_m=3.75, price_in_per_m=3.00) is True
    assert pricing.cache_is_worth_it(0, write_cost_per_m=3.75, price_in_per_m=3.00) is False


def test_cache_is_worth_it_zero_discount_edge_case():
    # read_discount=1.0 means cached reads cost the same as uncached -> never worth it.
    assert pricing.cache_is_worth_it(1000, write_cost_per_m=1.0, price_in_per_m=3.0,
                                      read_discount=1.0) is False
