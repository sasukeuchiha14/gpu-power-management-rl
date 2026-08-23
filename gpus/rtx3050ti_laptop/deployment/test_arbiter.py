"""
Plain-assert tests for arbiter.py -- no pytest dependency, matches this
repo's existing lightweight style (no test framework used elsewhere).
Run directly: python3 test_arbiter.py

These test pure decision logic with synthetic telemetry sequences, so they
need no GPU, no NVML, no torch -- they can (and should) run on any machine,
unlike the rest of the deployment daemon.
"""
from arbiter import DoNoHarmArbiter

CORE_MAX = 1900


def test_disabled_never_overrides():
    a = DoNoHarmArbiter(enabled=False)
    for _ in range(10):
        target, overridden, tripped = a.decide(sm_util_pct=99, tracked_core_mhz=600,
                                                 core_max_mhz=CORE_MAX, requested_core_mhz=500)
        assert target == 500 and not overridden and not tripped
    print("PASS: test_disabled_never_overrides")


def test_low_util_never_trips():
    a = DoNoHarmArbiter(util_threshold=85, patience=3, cooldown=5)
    for _ in range(20):
        target, overridden, tripped = a.decide(sm_util_pct=10, tracked_core_mhz=600,
                                                 core_max_mhz=CORE_MAX, requested_core_mhz=500)
        assert target == 500 and not overridden and not tripped
    print("PASS: test_low_util_never_trips")


def test_high_util_at_max_clock_never_trips():
    # already at max clock -- nothing to "do harm" with, must not trip
    a = DoNoHarmArbiter(util_threshold=85, patience=3, cooldown=5)
    for _ in range(20):
        target, overridden, tripped = a.decide(sm_util_pct=99, tracked_core_mhz=CORE_MAX,
                                                 core_max_mhz=CORE_MAX, requested_core_mhz=CORE_MAX)
        assert target == CORE_MAX and not overridden and not tripped
    print("PASS: test_high_util_at_max_clock_never_trips")


def test_trips_after_exactly_patience_consecutive_bad_ticks():
    a = DoNoHarmArbiter(util_threshold=85, patience=3, cooldown=5)
    # ticks 1, 2: bad but not yet at patience
    for i in range(2):
        target, overridden, tripped = a.decide(sm_util_pct=95, tracked_core_mhz=600,
                                                 core_max_mhz=CORE_MAX, requested_core_mhz=500)
        assert not overridden and not tripped, f"should not trip on bad tick {i+1}/3"
    # tick 3: hits patience, must trip now
    target, overridden, tripped = a.decide(sm_util_pct=95, tracked_core_mhz=600,
                                             core_max_mhz=CORE_MAX, requested_core_mhz=500)
    assert target == CORE_MAX and overridden and tripped, "must trip on the 3rd consecutive bad tick"
    print("PASS: test_trips_after_exactly_patience_consecutive_bad_ticks")


def test_non_consecutive_bad_ticks_do_not_accumulate():
    a = DoNoHarmArbiter(util_threshold=85, patience=3, cooldown=5)
    pattern = [95, 95, 10, 95, 95, 10, 95, 95]  # never 3 bad in a row
    for util in pattern:
        target, overridden, tripped = a.decide(sm_util_pct=util, tracked_core_mhz=600,
                                                 core_max_mhz=CORE_MAX, requested_core_mhz=500)
        assert not tripped, "must never trip -- no 3 consecutive bad ticks in this pattern"
    print("PASS: test_non_consecutive_bad_ticks_do_not_accumulate")


def test_cooldown_holds_max_for_exactly_cooldown_ticks_then_releases():
    a = DoNoHarmArbiter(util_threshold=85, patience=3, cooldown=5)
    # drive it to trip
    for _ in range(3):
        target, overridden, tripped = a.decide(sm_util_pct=95, tracked_core_mhz=600,
                                                 core_max_mhz=CORE_MAX, requested_core_mhz=500)
    assert tripped and target == CORE_MAX, "setup: should have tripped on 3rd tick"

    # the trip tick itself is cooldown tick 1 of 5 -- 4 more forced-max ticks follow
    for i in range(4):
        target, overridden, tripped = a.decide(sm_util_pct=95, tracked_core_mhz=CORE_MAX,
                                                 core_max_mhz=CORE_MAX, requested_core_mhz=500)
        assert overridden and target == CORE_MAX and not tripped, \
            f"cooldown tick {i+2}/5 should still hold max, not re-trip"

    # cooldown now exhausted -- control returns to the policy's request
    target, overridden, tripped = a.decide(sm_util_pct=10, tracked_core_mhz=CORE_MAX,
                                             core_max_mhz=CORE_MAX, requested_core_mhz=500)
    assert not overridden and not tripped and target == 500, \
        "after cooldown elapses, arbiter must hand control back to the policy"
    print("PASS: test_cooldown_holds_max_for_exactly_cooldown_ticks_then_releases")


def test_bad_ticks_during_cooldown_do_not_extend_it_or_immediately_retrip():
    a = DoNoHarmArbiter(util_threshold=85, patience=3, cooldown=5)
    for _ in range(3):
        a.decide(sm_util_pct=95, tracked_core_mhz=600, core_max_mhz=CORE_MAX, requested_core_mhz=500)
    # 4 more cooldown ticks, all still "bad" util -- must not re-trip mid-cooldown
    # or silently extend cooldown length
    for _ in range(4):
        target, overridden, tripped = a.decide(sm_util_pct=95, tracked_core_mhz=CORE_MAX,
                                                 core_max_mhz=CORE_MAX, requested_core_mhz=500)
        assert overridden and not tripped
    # cooldown exhausted -- since util is STILL bad and tracked clock is about
    # to drop below max again via the policy's own request, the very next
    # tick's bad-streak counter starts fresh at 1, not pre-loaded
    target, overridden, tripped = a.decide(sm_util_pct=95, tracked_core_mhz=CORE_MAX,
                                             core_max_mhz=CORE_MAX, requested_core_mhz=500)
    assert not overridden and not tripped, "control returns to policy; re-tripping needs a fresh patience count"
    print("PASS: test_bad_ticks_during_cooldown_do_not_extend_it_or_immediately_retrip")


if __name__ == "__main__":
    test_disabled_never_overrides()
    test_low_util_never_trips()
    test_high_util_at_max_clock_never_trips()
    test_trips_after_exactly_patience_consecutive_bad_ticks()
    test_non_consecutive_bad_ticks_do_not_accumulate()
    test_cooldown_holds_max_for_exactly_cooldown_ticks_then_releases()
    test_bad_ticks_during_cooldown_do_not_extend_it_or_immediately_retrip()
    print("\nALL ARBITER TESTS PASSED")
