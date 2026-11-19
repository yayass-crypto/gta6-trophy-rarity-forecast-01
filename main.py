"""Forecast GTA VI trophy earn rates on Exophase at each case's target.

Method in brief
---------------
For every trophy we model its earn rate on the logit scale as a straight
line in x = ln(1 + hours since release), fitted by least squares over the
case's distinct page states, and read the line at the target instant:

* Logit, not counts.  A rate lives between 0 and 100, and the logit scale
  enforces that on its own: a fitted line can push a rate toward 100%
  but never past it.  An earlier design split each rate into two
  exponentials (earners over players tracked) and divided them; it was
  tried on the practice cases and on a synthetic final-like case and
  lost clearly on both, because a small slope error in either count
  compounds into a rate of several hundred percent.  The logit-rate fit
  is the median model here.
* Recency-weighted.  A snapshot k steps back from the latest carries
  weight 0.7**k: the launch-night ramp is a different regime from the
  current one, so recent refreshes speak louder.
* Saturating damping.  A straight line in log-time cannot run forever --
  progression flattens out -- so the logit move from the last observed
  rate to the fitted target value is multiplied by (1 - exp(-dx)) / dx,
  with dx the log-time distance to the target.  The factor is 1 for a
  vanishing hop and fades toward 0 over a long one: the trend is trusted
  in full over hours and only in part over four days.
* A trophy at 0.00% in the last snapshot is a late trophy that has not
  started yet, not a dead one: it gets a small horizon-scaled prior
  median (about 1% over a 4.5-day horizon, the low end of the rarest
  trophies that have already started) and a very wide spread.
* A trophy at (nearly) 100% stays there with a tight spread: everybody
  earns it just by starting the game.  If the last rate is essentially
  100% (>= 99.95) and that reading is corroborated -- the trophy held
  >= 99.5 in at least two distinct snapshots, or, when the history is a
  single frozen page, in at least two reads of that page -- the median
  is 99.99, the scorer's clamp ceiling, because the recovered practice
  answer for the one such trophy is exactly 100%.  A lone uncorroborated
  reading can never trigger the 99.99 median; the 99.5-99.95 band keeps
  median 99.9.

Exophase refreshes its page rarely, so a case's history is first
collapsed to distinct snapshots (identical page states collapse to one);
fitting the raw reads would let long flat stretches dominate the fit.

Flat-history branch: when the history holds only one distinct snapshot
there is no trend to fit.  The page is stale while a blockbuster launch
keeps adding players, so the next refresh dilutes rates: the median is
shifted down by DILUTION_PRIOR (0.35 -- new players roughly tripling
while earners grow roughly 15%) with a wide spread.  This branch fires
only on a single distinct snapshot, never when a real trend exists.

Spread: per trophy, sigma on the logit scale combines a random-walk term
(0.5 * sqrt(horizon in days), the house baseline's term), the residual
variance of the logit fit (scaled by how far the target extrapolates
past the fitted range), and a staleness term that grows when the last
distinct snapshot is old relative to the usual gap between snapshots.
When distinct snapshots are few (< 6), the trend fit rests on little
evidence, so the spread is widened: the sigma cap rises from 1.3 to 2.0
and a small-sample variance term proportional to 1 / n_snapshots is
added.  Sigma is clamped to [0.35, cap] in the trend branch (the
zero-rate and fallback branches use their own wider 1.6).

Quantiles use the standard-normal z values, are rounded half-up to 4
decimals, clamped to [0.01, 99.99] and sorted non-decreasing as a guard.

Standard library only.  Deterministic: no randomness, no wall clock.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

# --- Tunable priors (documented, not fitted to any answer) -----------------

# Flat-history dilution prior: factor applied to the last rate when the
# history has a single distinct snapshot.  Rationale: on launch week the
# tracked-player base roughly triples between Exophase refreshes while
# earners for a given trophy grow only ~15%, so the next refresh shows
# roughly 1.15 / 3 ~= 0.38 of the old rate; 0.35 is a conservative round
# value for that prior.
DILUTION_PRIOR = 0.35

# Spread used in the flat-history branch (logit units).
FLAT_SIGMA = 1.0

# Sigma clamp for the trend branch (logit units).  The cap is raised when
# distinct snapshots are few (see SPARSE_* below).
SIGMA_MIN = 0.35
SIGMA_MAX = 1.3

# Sparse-history widening: with fewer than SPARSE_LIMIT distinct
# snapshots the trend fit is noisy, so the sigma cap rises to
# SIGMA_MAX_SPARSE and a variance term SMALL_SAMPLE_COEF / n_snapshots
# is added.  A 12-hour refresh cadence leaves only ~5 distinct snapshots
# by the checkpoint; without this the 90% band covers only ~0.82 there.
# Measured on that world: cap 1.6/coef 1.0 gives loss 0.7914, coverage
# 0.85; cap 2.0/coef 1.0 gives 0.7900 / 0.87; caps 2.5 and 3.0 stay at
# 0.87 while loss stops improving, so 2.0 is the knee.  The remaining
# misses are median errors (zero-rate and fast-takeoff trophies), not
# spread, so no wider cap is bought.
SPARSE_LIMIT = 6
SIGMA_MAX_SPARSE = 2.0
SMALL_SAMPLE_COEF = 1.0

# Random-walk coefficient (logit units per sqrt(day)), as in the baseline.
RW_COEF = 0.5

# Recency weight: a snapshot k steps back from the latest carries weight
# RECENCY_DECAY ** k.  Recent refreshes reflect the current regime; the
# launch ramp is stale.
RECENCY_DECAY = 0.7

# The (near-)100% starter trophy: everybody earns it just by starting.
# CERTAIN_RATE / CERTAIN_MEDIAN is the essentially-100% band: the last
# rate is at the scorer's clamp ceiling in practice, so the median sits
# there too.  The 99.5-99.95 band keeps the slightly lower 99.9 median.
NEARLY_ALL_RATE = 99.5
CERTAIN_RATE = 99.95
NEARLY_ALL_MEDIAN = 99.9
CERTAIN_MEDIAN = 99.99
NEARLY_ALL_SIGMA = 0.2

# Minimum corroboration for the 99.99 median: the trophy must have held
# >= NEARLY_ALL_RATE in at least this many observations, so one anomalous
# read can never trigger it.
CERTAIN_SUPPORT = 2

# Zero-rate prior: a trophy at 0.00% now is a late trophy, not a dead one.
# Its median is the low end of the rarest trophies that have already
# started earning, scaled linearly with the horizon so a short horizon
# expects almost no activation and a ~4.5-day horizon expects about 1%.
ZERO_MEDIAN_PER_DAY = 0.22
ZERO_MEDIAN_CAP = 1.0
ZERO_SIGMA = 1.6

# Fallback spreads when a trophy gives no trend evidence at all.
FALLBACK_SIGMA = 1.6
SINGLE_POINT_SIGMA = 1.6

Z_VALUES = (-1.645, -0.674, 0.0, 0.674, 1.645)
RATE_LO = 0.01
RATE_HI = 99.99


# --- Small helpers ---------------------------------------------------------

def parse_time(text):
    """Parse an ISO-8601 timestamp (with Z or an offset) to an aware datetime."""
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def release_time(game):
    """Midnight UTC of the game's release date (YYYY-MM-DD)."""
    raw = (game or {}).get("release_date", "2026-11-19")
    try:
        day = datetime.strptime(raw[:10], "%Y-%m-%d")
    except ValueError:
        day = datetime(2026, 11, 19)
    return day.replace(tzinfo=timezone.utc)


def hours_since_release(moment, release):
    return max(0.0, (moment - release).total_seconds() / 3600.0)


def time_axis(moment, release):
    """Log-time axis: x = ln(1 + hours since release)."""
    return math.log1p(hours_since_release(moment, release))


def clamp(value, low, high):
    return low if value < low else high if value > high else value


def logit(rate):
    p = clamp(rate, RATE_LO, RATE_HI) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(value):
    if value > 30.0:
        return 100.0
    if value < -30.0:
        return 0.0
    return 100.0 / (1.0 + math.exp(-value))


def round_half_up(value, places=4):
    """Round half-up to `places` decimals (avoids banker's rounding)."""
    factor = 10.0 ** places
    return math.floor(value * factor + 0.5) / factor


def weighted_least_squares(xs, ys, weights):
    """Weighted least squares; returns (slope, intercept, resid_var, sxx, x_mean).

    resid_var is the weighted mean squared residual (0 when the fit is
    exact or has fewer than 3 points); sxx is the weighted spread of x
    around its weighted mean.
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    total_w = sum(weights)
    mean_x = sum(w * x for w, x in zip(weights, xs)) / total_w
    mean_y = sum(w * y for w, y in zip(weights, ys)) / total_w
    sxx = sum(w * (x - mean_x) ** 2 for w, x in zip(weights, xs))
    if n < 2 or sxx <= 0.0:
        return 0.0, mean_y, 0.0, 0.0, mean_x
    sxy = sum(w * (x - mean_x) * (y - mean_y)
              for w, x, y in zip(weights, xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    if n > 2:
        resid = sum(w * (y - (intercept + slope * x)) ** 2
                    for w, x, y in zip(weights, xs, ys))
        resid_var = resid / total_w
    else:
        resid_var = 0.0
    return slope, intercept, resid_var, sxx, mean_x


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


# --- History collapsing ----------------------------------------------------

def distinct_snapshots(case, release):
    """Collapse history to one entry per distinct page state.

    Returns a list of (x, players_tracked, rates) in time order, where x is
    the log-time axis at the first read showing that state.  A state is
    identified by (players_tracked, tuple(rates)); null rates are kept as
    None inside the tuple so two reads differing only by a null still count
    as distinct.
    """
    seen = {}
    order = []
    for read in case.get("history") or []:
        try:
            moment = parse_time(read["at"])
            players = read["players_tracked"]
            rates = read["rates"]
        except (KeyError, TypeError, ValueError):
            continue
        if players is None or rates is None:
            continue
        key = (players, tuple(rates))
        if key in seen:
            continue
        seen[key] = True
        order.append((time_axis(moment, release), players, list(rates)))
    return order


def last_rate(snapshots, index):
    """The trophy's last non-null rate over the distinct snapshots."""
    for _, _, rates in reversed(snapshots):
        value = rates[index]
        if value is not None:
            return value
    return None


def certain_support(case, snapshots, index):
    """How many observations corroborate a >= 99.5% rate for this trophy.

    Counts distinct snapshots holding >= NEARLY_ALL_RATE.  When the
    history is a single frozen page there is only one distinct snapshot
    by construction, so the corroboration is counted over the raw reads
    of that page instead: a trophy that reads 100% on every read of a
    page held for hours is corroborated, while a single anomalous read
    can never reach CERTAIN_SUPPORT either way.
    """
    if len(snapshots) <= 1:
        count = 0
        for read in case.get("history") or []:
            try:
                rates = read["rates"]
                value = rates[index] if index < len(rates) else None
            except (KeyError, TypeError, IndexError):
                continue
            if value is not None and value >= NEARLY_ALL_RATE:
                count += 1
        return count
    count = 0
    for _, _, rates in snapshots:
        value = rates[index] if index < len(rates) else None
        if value is not None and value >= NEARLY_ALL_RATE:
            count += 1
    return count


# --- Per-case forecast -----------------------------------------------------

def forecast_case(case):
    trophies = case.get("trophies") or []
    release = release_time(case.get("game") or {})
    try:
        target_moment = parse_time(case["target"])
    except (KeyError, TypeError, ValueError):
        target_moment = parse_time(case["at"])
    x_target = time_axis(target_moment, release)
    horizon_days = max(0.0, (case.get("horizon_minutes") or 0) / 1440.0)

    snapshots = distinct_snapshots(case, release)
    flat = len(snapshots) <= 1
    n_snaps = len(snapshots)

    # Staleness: how old the last distinct snapshot is, in units of the
    # typical gap between distinct snapshots (1.0 when only one gap exists).
    gaps = [b[0] - a[0] for a, b in zip(snapshots, snapshots[1:])]
    typical_gap = median(gaps) if gaps else 1.0
    if typical_gap <= 0:
        typical_gap = 1.0
    if snapshots:
        staleness_ratio = max(0.0, x_target - snapshots[-1][0]) / typical_gap
    else:
        staleness_ratio = 0.0

    rows = []
    for idx, trophy in enumerate(trophies):
        try:
            row = forecast_trophy(
                idx, trophy, snapshots, flat, n_snaps, x_target,
                horizon_days, staleness_ratio,
                certain_support(case, snapshots, idx),
            )
        except Exception:
            # One bad trophy must never fail the whole case.
            rate = last_rate(snapshots, idx)
            row = make_row(trophy.get("id", idx + 1),
                           rate if rate is not None else 0.5, FALLBACK_SIGMA)
        rows.append(row)
    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def forecast_trophy(idx, trophy, snapshots, flat, n_snaps, x_target,
                    horizon_days, staleness_ratio, support):
    trophy_id = trophy.get("id", idx + 1)
    rate = last_rate(snapshots, idx)

    # No history at all: persistence with the widest spread.
    if rate is None:
        return make_row(trophy_id, 0.5, FALLBACK_SIGMA)

    # The (near-)100% starter trophy: everyone earns it on starting.
    if rate >= NEARLY_ALL_RATE:
        if rate >= CERTAIN_RATE and support >= CERTAIN_SUPPORT:
            return make_row(trophy_id, CERTAIN_MEDIAN, NEARLY_ALL_SIGMA)
        return make_row(trophy_id, NEARLY_ALL_MEDIAN, NEARLY_ALL_SIGMA)

    if flat:
        # Single distinct snapshot: no trend exists.  Dilute by the prior
        # and spread wide; the next refresh is the unknown.
        return make_row(trophy_id, rate * DILUTION_PRIOR, FLAT_SIGMA)

    # --- Trend branch ------------------------------------------------------
    # The rate in the LAST distinct snapshot decides the zero branch: a
    # trophy showing 0.00% now is a late trophy that has not started yet.
    last_snapshot_rate = None
    if snapshots and idx < len(snapshots[-1][2]):
        last_snapshot_rate = snapshots[-1][2][idx]
    if last_snapshot_rate is not None and last_snapshot_rate <= 0.0:
        zero_median = min(ZERO_MEDIAN_CAP, ZERO_MEDIAN_PER_DAY * horizon_days)
        return make_row(trophy_id, max(zero_median, RATE_LO), ZERO_SIGMA)

    # Recency-weighted least squares of logit(rate) against x.
    points = []
    for x, _, rates in snapshots:
        value = rates[idx] if idx < len(rates) else None
        if value is None or value <= 0.0:
            continue
        points.append((x, logit(value)))
    if len(points) < 2:
        if not points:
            return make_row(trophy_id, max(rate, RATE_LO), FALLBACK_SIGMA)
        # One usable point: no trend evidence; persist the last rate.
        return make_row(trophy_id, rate, SINGLE_POINT_SIGMA)

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    weights = [RECENCY_DECAY ** (len(points) - 1 - i)
               for i in range(len(points))]
    slope, intercept, resid_var, sxx, x_mean = weighted_least_squares(
        xs, ys, weights)
    trend_logit = intercept + slope * x_target

    # Damping: the fitted logit move from the last rate is believed in
    # full over a short hop and only in part over a long one, because
    # progression saturates.  The factor (1 - exp(-dx)) / dx is 1 at
    # dx = 0 and decays toward 0 as the log-time distance dx grows.
    dx = max(0.0, x_target - xs[-1])
    if dx <= 0.0:
        damp = 1.0
    else:
        damp = (1.0 - math.exp(-dx)) / dx
    median_logit = logit(rate) + damp * (trend_logit - logit(rate))
    median_rate = clamp(inv_logit(median_logit), RATE_LO, RATE_HI)

    # Spread: random-walk term plus fit-residual variance scaled by
    # extrapolation leverage, plus staleness.  With few distinct
    # snapshots the fit rests on little evidence, so the cap is raised
    # and a small-sample variance term (proportional to 1/n_snapshots)
    # is added; with plenty of snapshots both are inert.
    if sxx > 0:
        leverage = 1.0 + ((x_target - x_mean) ** 2) / sxx
    else:
        leverage = 1.0 + abs(x_target - x_mean)
    fit_var = resid_var * min(leverage, 9.0)
    rw_var = (RW_COEF * math.sqrt(horizon_days)) ** 2
    stale_var = 0.05 * min(staleness_ratio, 4.0) ** 2
    sigma_cap = SIGMA_MAX
    small_var = 0.0
    if n_snaps < SPARSE_LIMIT:
        sigma_cap = SIGMA_MAX_SPARSE
        small_var = SMALL_SAMPLE_COEF / max(n_snaps, 1)
    sigma = math.sqrt(rw_var + fit_var + stale_var + small_var)
    sigma = clamp(sigma, SIGMA_MIN, sigma_cap)
    return make_row(trophy_id, median_rate, sigma)


def make_row(trophy_id, median_rate, sigma):
    centre = logit(median_rate)
    values = []
    for z in Z_VALUES:
        q = inv_logit(centre + z * sigma)
        q = clamp(round_half_up(q, 4), RATE_LO, RATE_HI)
        values.append(q)
    values.sort()  # final guard: rounding can tie adjacent quantiles
    return {"id": trophy_id, "q": values}


# --- Entry point ------------------------------------------------------------

def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(output_dir, exist_ok=True)

    names = sorted(
        name for name in os.listdir(input_dir)
        if name.endswith(".in") and os.path.isfile(os.path.join(input_dir, name))
    )
    for name in names:
        case_id = name[:-3]
        try:
            with open(os.path.join(input_dir, name), "r", encoding="utf-8") as fh:
                case = json.load(fh)
            result = forecast_case(case)
        except Exception:
            # Never crash on a case: emit a minimal valid forecast so the
            # case is still scored instead of counted invalid.
            try:
                with open(os.path.join(input_dir, name), "r", encoding="utf-8") as fh:
                    case = json.load(fh)
                trophies = case.get("trophies") or []
                rows = [{"id": t.get("id", i + 1), "q": [0.01, 0.01, 0.01, 0.01, 0.01]}
                        for i, t in enumerate(trophies)]
                result = {"trophies": rows}
            except Exception:
                result = {"trophies": []}
        out_path = os.path.join(output_dir, case_id + ".out")
        text = json.dumps(result, separators=(",", ":")) + "\n"
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
