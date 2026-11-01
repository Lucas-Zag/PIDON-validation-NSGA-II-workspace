"""Physical-unit cure-cycle checks; no imports from sibling projects."""
import math

VARIABLES = ("r1", "ht1", "hd1", "r2", "ht2", "hd2")


def cycle_timing(case, initial_C, total_s):
    """Return heating/holding end and remaining cooling time in seconds.

    Existing XPIDON uses fixed total_s; changing it is not supported here.
    """
    values = [float(case[k]) for k in VARIABLES]
    r1, ht1, hd1, r2, ht2, hd2 = values
    if not all(math.isfinite(v) for v in values + [initial_C, total_s]):
        raise ValueError("Cycle values must be finite")
    if r1 <= 0 or r2 <= 0 or hd1 < 0 or hd2 < 0:
        raise ValueError("Ramp rates must be positive and hold durations non-negative")
    if not initial_C <= ht1 <= ht2 or total_s <= 0:
        raise ValueError("Invalid heating temperatures or total time")
    heating_hold_end_s = 60 * ((ht1-initial_C)/r1 + hd1 + (ht2-ht1)/r2 + hd2)
    cooling_s = total_s - heating_hold_end_s
    return {
        "heating_hold_end_s": heating_hold_end_s,
        "cooling_s": cooling_s,
        "complete_cycle_s": total_s,
        "positive_cooling_interval": cooling_s > 0,
    }
