# GTA VI Trophy Rarity Forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/yayass-crypto/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/yayass-crypto/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Entry for the [make.fun bounty gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01): forecast, for every Grand Theft Auto VI trophy on PlayStation 5, its earn rate on Exophase at the case's target time.

## How it works

The program reads one case (the house's reads of Exophase's trophy page before the checkpoint) and, for each of the 55 trophies, writes five quantiles (5%, 25%, 50%, 75%, 95%) of its earn rate at the target. It first collapses the history to distinct page states, since Exophase refreshes infrequently and long flat stretches would otherwise dominate any fit; then it fits each trophy's logit earn rate against log-time since release with a recency-weighted least squares (recent refreshes count more than the launch ramp), damps the fitted move toward persistence with a factor that shrinks as the forecast horizon grows (progression saturates, so a four-day trend is believed only in part), and wraps the median in a spread built from a random-walk term, the fit's residuals, and the staleness of the last snapshot. Special cases are handled explicitly: a history with a single distinct snapshot has no trend, so the median is diluted by a prior (new players roughly triple between refreshes while earners grow ~15%) with a wide spread; a trophy at 0.00% is a late trophy that has not started yet, so it gets a small horizon-scaled prior median with a very wide spread; and a trophy at (nearly) 100% stays there with a tight spread. Everything is deterministic, standard library only, and runs in well under a second per case.

## Run it locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<dir-with-case-.in-files> MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Each `<id>.in` in the input directory produces one `<id>.out` in the output directory: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`, one row per trophy, quantiles non-decreasing and within 0–100.

## Validation

- Practice cases p00/p01: mean pinball loss **0.3693** (house baseline 1.5084).
- Synthetic final-like case (generated locally with a different process than the forecaster assumes): loss **0.6770** vs the house baseline's 1.8446, with 90% coverage 0.93.
- Output is byte-identical across repeated runs, and each case runs in ~0.05 s.

## License

MIT — see [LICENSE](LICENSE).
