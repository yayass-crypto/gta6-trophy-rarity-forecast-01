"""Forecast GTA VI trophy earn rates on Exophase at a case's target.

Approach
--------
Each trophy's earn rate is modelled on the logit scale against
x = ln(1 + hours since release), fitted by least squares over the case's
distinct snapshots (see below), and read at the target:

* The logit scale is the right scale for a rate: it is self-damping over
  a long horizon (a rate can approach 100% but never pass it), where
  dividing two separately extrapolated exponential counts proved
  fragile -- small slope errors in earners and in players tracked
  compound into rates of several hundred percent.  Both shapes were
  tested on the practice cases and on a synthetic final-like case; the
  logit-rate fit won clearly on both, so it is the median model here.
* The fit is recency-weighted (weights 0.7 per snapshot of age): the
  launch ramp is a different regime from the current one, so recent
  refreshes carry more weight.
* The fitted move is damped toward persistence.  A linear trend in x
  cannot hold forever -- progression saturates -- so the logit move from
  the last rate to the fitted target value is multiplied by
  (1 - exp(-dx)) / dx, where dx is the log-time distance to the target.
  The factor is 1 for a vanishing horizon and decays toward 0 for a very
  long one: the trend is believed in full over short hops and only in
  part over four days.
* A trophy showing 0.00% in the last snapshot is a late trophy that has
  not started yet, not a dead one: it gets a small horizon-scaled prior
  median (about 1% over a 4.5-day horizon, the low end of the rarest
  trophies that have already started) with a very wide spread.
* A trophy whose last rate is (nearly) 100% keeps median 99.9 with a
  tight spread: everyone earns it on starting.

Exophase refreshes its page infrequently, so a case's history is first
collapsed to distinct snapshots (identical page states collapse to one);
fitting the raw reads would overweight long flat stretches.

Flat-history branch: when the history holds only one distinct snapshot
there is no trend to fit.  The page is stale while a blockbuster launch
keeps adding players, so the next refresh dilutes rates: the median is
shifted down by the DILUTION_PRIOR (0.35 -- new players roughly tripling
while earners grow roughly 15%) and a wide spread is used.  This branch
fires only on a single distinct snapshot, never when a real trend exists.

Spread: per trophy, sigma on the logit scale combines a random-walk term
(0.5 * sqrt(horizon in days), the house baseline's term), the residual
variance of the logit fit (scaled by how far the target extrapolates
past the fitted range), and a staleness term that grows when the last
distinct snapshot is old relative to the usual gap between snapshots.
Sigma is clamped to [0.35, 1.3] in the trend branch (the zero-rate and
fallback branches use their own wider 1.6).

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

# Sigma clamp for the trend branch (logit units).
SIGMA_MIN = 0.35
SIGMA_MAX = 1.3

# Random-walk coefficient (logit units per sqrt(day)), as in the baseline.
RW_COEF = 0.5

# Recency weight: a snapshot k steps back from the latest carries weight
# RECENCY_DECAY ** k.  Recent refreshes reflect the current regime; the
# launch ramp is stale.
RECENCY_DECAY = 0.7

# Median for a trophy whose last rate is essentially 100%: everyone earns
# it just by starting, so it stays (nearly) 100% with a tight spread.
NEARLY_ALL_RATE = 99.5
NEARLY_ALL_MEDIAN = 99.9
NEARLY_ALL_SIGMA = 0.2

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
                idx, trophy, snapshots, flat, x_target, horizon_days,
                staleness_ratio,
            )
        except Exception:
            # One bad trophy must never fail the whole case.
            rate = last_rate(snapshots, idx)
            row = make_row(trophy.get("id", idx + 1),
                           rate if rate is not None else 0.5, FALLBACK_SIGMA)
        rows.append(row)
    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def forecast_trophy(idx, trophy, snapshots, flat, x_target, horizon_days,
                    staleness_ratio):
    trophy_id = trophy.get("id", idx + 1)
    rate = last_rate(snapshots, idx)

    # No history at all: persistence with the widest spread.
    if rate is None:
        return make_row(trophy_id, 0.5, FALLBACK_SIGMA)

    # The (near-)100% starter trophy: everyone earns it on starting.
    if rate >= NEARLY_ALL_RATE:
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
    # extrapolation leverage, plus staleness.
    if sxx > 0:
        leverage = 1.0 + ((x_target - x_mean) ** 2) / sxx
    else:
        leverage = 1.0 + abs(x_target - x_mean)
    fit_var = resid_var * min(leverage, 9.0)
    rw_var = (RW_COEF * math.sqrt(horizon_days)) ** 2
    stale_var = 0.05 * min(staleness_ratio, 4.0) ** 2
    sigma = math.sqrt(rw_var + fit_var + stale_var)
    sigma = clamp(sigma, SIGMA_MIN, SIGMA_MAX)
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
