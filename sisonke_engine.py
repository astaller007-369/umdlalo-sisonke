"""
sisonke_engine.py
==================

All the math for the Sisonke Football Predictive Terminal, kept
completely separate from the Streamlit UI so it can be tested directly
with plain Python (no Streamlit needed to verify the numbers are right).

DESIGN DECISIONS WORTH KNOWING ABOUT (documented up front because this
is a real-money-adjacent tool and every assumption should be visible,
not buried):

1. HOME ADVANTAGE IS NOT DOUBLE-COUNTED. Host stats are computed ONLY
   from a team's home matches, and visitor stats ONLY from their away
   matches - so whatever real home boost a team gets is already baked
   into their host numbers. No separate home-advantage multiplier is
   applied anywhere in this file.

2. TEAM STRENGTH NEVER COMES FROM A TEAM'S OWN RAW GOALS. The
   attack/defense STRENGTH RATIOS that differentiate one team from
   another are built entirely from territory metrics (big chances,
   shots on target, box touches) - never from a team's own win/loss
   record or their own average goals, which is exactly the "luck"
   signal the spec says to avoid. The one place raw goals appear at all
   is as a single LEAGUE-WIDE average goals figure used as a shared
   baseline unit (the same number for every team in that league) to
   convert relative territory strength into an actual expected-goals
   scale for the Poisson math - since "big chances" has no absolute
   goals unit on its own. That's a scale anchor, not a team signal.

3. PROBABILITIES ARE ALWAYS COMPUTED, NEVER FIXED. Every tactical
   multiplier in Section 6 only ever adjusts the INPUT attack/defense
   rates that feed the Dixon-Coles and Monte Carlo engines. The actual
   market probabilities always come out of real Poisson math or a real
   10,000-run simulation - a slider never directly sets a probability
   or an EV number by formula shortcut.

4. RHO (the Dixon-Coles low-score correlation parameter) is fitted from
   the league's own historical low-score frequencies, not a fixed
   constant - see fit_rho().

5. The half-life for time-decay weighting is chosen by actually
   backtesting candidate half-lives against real past results with a
   Brier score (see optimize_half_life()) - not asserted as a fixed
   number, unless you tick "Freeze Decay" for a fixed 45-day window.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import poisson as scipy_poisson
from scipy.optimize import minimize as _minimize

try:
    import requests as _requests
except ImportError:  # Telegram sending degrades gracefully if requests isn't installed
    _requests = None

MIN_SAMPLE_ROWS = 5          # the "5-match sample safety rail"
FROZEN_HALF_LIFE_DAYS = 45   # fallback when "Freeze Decay" is ticked
# Decay search grid, 15 days to ONE YEAR. Fine steps where short memory matters, wider steps for the long end (a league
# whose results stay informative for most of a year needs the 240-365 range; the old grid stopped at 180).
HALF_LIFE_CANDIDATES = [15, 30, 45, 60, 75, 90, 105, 120, 150, 180, 210, 240, 300, 365]
HALF_LIFE_MAX_DAYS = max(HALF_LIFE_CANDIDATES)
GOAL_CAP = 10                # max goals per side considered in the Poisson grid
MC_ITERATIONS = 10_000
DISPERSION_ADJUST_TRIGGER = 0.85
DISPERSION_ADJUST_FACTOR = 1.15

BACKTEST_MAX_EVALUATED = 1200  # app default: score the most recent N matches (history still uses everything)
ENGINE_VERSION = 10            # bump when any change would make cached backtest parameters stale
SEASON_CARRYOVER = 0.4        # weight of each earlier season in league baselines (1.0 = pool all seasons, 0.0 = current season only)
DUEL_MULT_FLOOR = 0.85        # duel-resilience multiplier on a team's defence ratio is clamped to this band
DUEL_MULT_CEILING = 1.15
EV_ENGINE_MODE = "min"        # "min" = edge BOTH engines support (default), "mean" = average of the two, "max" = most generous engine (old behaviour)


def set_season_carryover(value: float) -> None:
    """Read at call time by season_row_weights(), so the app can change it from a slider."""
    global SEASON_CARRYOVER
    SEASON_CARRYOVER = float(min(max(value, 0.0), 1.0))

TERRITORY_STATS = ["big_chances", "shots_on_target", "box_touches"]

# ---------------------------------------------------------------------------
# Section 3: Column standardisation, division/fixture parsing
# ---------------------------------------------------------------------------

REQUIRED_BASE_COLUMNS = [
    "date", "home_team", "away_team", "home_goals", "away_goals",
    "home_shots_on_target", "away_shots_on_target",
    "home_big_chances", "away_big_chances",
    "home_box_touches", "away_box_touches",
]
DIVISION_COLUMN_CANDIDATES = ["league_country", "league", "competition"]


# One consistent name for the long-pass stats. Older files (and older saved data) call the
# volume column "accurate_long_balls"; it is renamed to "accurate_long_passes" so it matches
# "successful_long_passes_percentage". Never overwrites a column that already has the new name.
COLUMN_RENAMES = {"accurate_long_balls": "accurate_long_passes"}


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strips whitespace, lowercases, and replaces spaces with
    underscores in every column name (e.g. 'Home Box Touches' becomes
    'home_box_touches'), then applies COLUMN_RENAMES (home_/away_ prefixed names)."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    existing = set(df.columns)
    renames = {}
    for prefix in ("home_", "away_"):
        for old, new in COLUMN_RENAMES.items():
            if f"{prefix}{old}" in existing and f"{prefix}{new}" not in existing:
                renames[f"{prefix}{old}"] = f"{prefix}{new}"
    return df.rename(columns=renames) if renames else df


def find_division_column(df: pd.DataFrame) -> str | None:
    for candidate in DIVISION_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return None


# Same flexible-detection pattern as the division column, for the same
# reason: different CSV exports call this column different things, and
# blindly assuming "date" exists crashes anything downstream that reads
# it (half-life decay, backtesting, the fixture picker) with a cryptic
# AttributeError rather than a clear, actionable message.
DATE_COLUMN_CANDIDATES = [
    "date", "match_date", "fixture_date", "game_date", "kickoff",
    "kickoff_date", "kickoff_time", "date_time", "match_datetime",
    "match_time", "played_on", "utc_date",
]


def find_date_column(df: pd.DataFrame) -> str | None:
    for candidate in DATE_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return None


SEASON_COLUMN_CANDIDATES = [
    "season", "Season", "SEASON", "season_id", "Season_ID", "campaign",
    "Campaign", "div_season", "Div_Season", "year_season",
]


def find_season_column(df: pd.DataFrame) -> str | None:
    for candidate in SEASON_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return None


def infer_season_label(match_date) -> "str | None":
    """European-convention season boundary: a match in July or later
    belongs to the season starting that year (e.g. Aug 2025 -> '2025-26');
    a match from January-June belongs to the season that started the
    PREVIOUS year (e.g. Mar 2026 -> '2025-26'). This is a heuristic, not
    a guarantee - a handful of leagues run calendar-year seasons instead
    (MLS, Brazil, J-League) where this would mislabel things, but it's a
    reasonable default absent an explicit season column, and it's only
    ever the SECOND fallback (see derive_season_series below) - an
    explicit season column always wins when one exists."""
    if match_date is None or pd.isna(match_date):
        return None
    year, month = match_date.year, match_date.month
    return f"{year}-{str(year + 1)[-2:]}" if month >= 7 else f"{year - 1}-{str(year)[-2:]}"


def derive_season_series(df: pd.DataFrame) -> tuple[pd.Series, str]:
    """The fallback sequence requested: (1) an explicit season column in
    the CSV, used verbatim - the most trustworthy source, since it's
    whatever the data provider actually means by 'season', correctly
    handling calendar-year leagues this function's own date heuristic
    would get wrong; (2) if no such column exists but dates do, infer a
    season label per match via the Aug-June convention above; (3) if
    neither is available, fall back to a single constant label spanning
    the whole dataset - the original, pre-season-aware behaviour, so
    nothing breaks for a dataset with no season info and no usable dates.

    Returns (season_series, method) where method is 'column', 'inferred',
    or 'fallback' - callers can surface which path was used rather than
    silently guessing."""
    season_col = find_season_column(df)
    if season_col is not None:
        return df[season_col].astype(str), "column"

    if "date" in df.columns and df["date"].notna().any():
        return df["date"].apply(infer_season_label), "inferred"

    return pd.Series(["All Matches"] * len(df), index=df.index), "fallback"


# Keyword fragments that flag a competition as a CUP/TOURNAMENT rather
# than a standard home-and-away league - this model's whole points/table
# framework (xPts, season simulation, "played N times") assumes a
# standard league fixture list, which doesn't hold for single/double-leg
# knockout cup football (extra time, penalties, one-off ties, no table).
CUP_TOURNAMENT_KEYWORDS = [
    "cup", "trophy", "shield", "playoff", "play-off", "knockout",
    "copa", "coupe", "pokal", "taça", "taca", "supercup", "super cup",
    "champions league", "europa league", "conference league", "libertadores",
    "sudamericana", "afcon", "world cup", "euros", "european championship",
    "nations league", "friendly", "friendlies", "qualifier", "qualifying",
]


def is_cup_or_tournament(division_text: str) -> bool:
    """Keyword-based flag, not a guarantee - a league that happens to
    have 'Cup' in its sponsor name would still need a manual override,
    but this catches the overwhelming majority of real knockout
    competitions without needing per-competition metadata the CSV
    doesn't provide."""
    if not division_text:
        return False
    text_lower = str(division_text).lower()
    return any(kw in text_lower for kw in CUP_TOURNAMENT_KEYWORDS)


def filter_to_standard_leagues(divisions: list[str]) -> tuple[list[str], list[str]]:
    """Splits a list of division names into (standard_leagues, excluded).
    Use this to keep cup/tournament competitions out of the workspace
    dropdown entirely, per the model being strictly for standard league
    play."""
    standard = [d for d in divisions if not is_cup_or_tournament(d)]
    excluded = [d for d in divisions if is_cup_or_tournament(d)]
    return standard, excluded


def normalize_name_casing(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Makes 'Chelsea' and 'chelsea' resolve to ONE team instead of two
    different ones. Builds a SINGLE canonical-casing map from the
    combined values across ALL the given columns together (not one map
    per column) - otherwise 'Chelsea' could end up resolving to a
    different casing in home_team than in away_team, which would still
    silently split the same real team into two. Each variant is mapped to
    whichever ORIGINAL casing appears most often across all those columns
    combined (ties broken by first-seen), which preserves correct
    capitalization for names like 'PSV' or 'AS Roma' that a blind
    .title() would mangle into 'Psv' or 'As Roma'. Columns that don't
    exist are skipped."""
    df = df.copy()
    existing_columns = [c for c in columns if c in df.columns]
    if not existing_columns:
        return df

    # Whitespace hygiene FIRST: 'Chelsea' and 'Chelsea ' (trailing space) or 'Man  City'
    # (double space) are otherwise treated as different teams, silently splitting one
    # club's history in two.
    for col in existing_columns:
        df[col] = df[col].apply(
            lambda v: re.sub(r"\s+", " ", str(v)).strip() if pd.notna(v) else v
        )

    combined = pd.concat([df[c].dropna().astype(str) for c in existing_columns], ignore_index=True)
    if combined.empty:
        return df

    canonical_map = combined.groupby(combined.str.lower()).agg(
        lambda s: s.value_counts().idxmax()
    ).to_dict()

    for col in existing_columns:
        df[col] = df[col].apply(
            lambda v: canonical_map.get(str(v).lower(), v) if pd.notna(v) else v
        )
    return df


def is_unplayed(home_goals_val, away_goals_val) -> bool:
    """A match counts as unplayed if either goals cell is blank/NaN, or
    contains a comma (covers both conventions described in the spec -
    a genuinely empty cell, or a combined 'x,y' placeholder string some
    spreadsheets use for an unplayed fixture)."""
    for val in (home_goals_val, away_goals_val):
        if val is None:
            return True
        if isinstance(val, float) and math.isnan(val):
            return True
        text = str(val).strip()
        if text == "" or text.lower() in {"nan", "none"}:
            return True
        if "," in text:
            return True
        try:
            float(text)
        except (TypeError, ValueError):
            return True
    return False


def split_played_unplayed(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (settled_df, upcoming_df) for a division's fixtures."""
    unplayed_mask = df.apply(
        lambda r: is_unplayed(r.get("home_goals"), r.get("away_goals")), axis=1
    )
    return df[~unplayed_mask].copy(), df[unplayed_mask].copy()


def coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def parse_dates(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    """Parses the date column - and, critically, GUARANTEES the returned
    dataframe always has a `col` column afterward, even if the source CSV
    used a different name (or no date column at all). Everything
    downstream (decay weighting, backtesting, the fixture picker) reads
    this column directly off dataframe rows, so silently having no such
    column crashes deep in an unrelated tab with a cryptic
    'Pandas object has no attribute date' AttributeError instead of a
    clear message - this is what that bug looked like in practice."""
    df = df.copy()
    if col not in df.columns:
        detected = find_date_column(df)
        if detected is not None:
            df = df.rename(columns={detected: col})
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    else:
        # No date-like column found anywhere - fill with NaT rather than
        # leaving the column missing, so `row.date` always resolves to
        # something (safely treated as "unknown date") instead of raising.
        df[col] = pd.NaT
    return df


# ---------------------------------------------------------------------------
# Section 4 & Core Parameter A: time-decay weighted territory vectors
# ---------------------------------------------------------------------------

def decay_weights(dates: pd.Series, reference_date: pd.Timestamp, half_life_days: float) -> np.ndarray:
    """Weight = exp(-ln(2) * days_elapsed / half_life). More recent
    matches (smaller days_elapsed) get a weight closer to 1.0."""
    days_elapsed = (reference_date - dates).dt.days.to_numpy(dtype=float)
    days_elapsed = np.where(np.isnan(days_elapsed), np.nan, np.maximum(days_elapsed, 0.0))
    weights = np.exp(-math.log(2) * days_elapsed / half_life_days)
    # A match with an unreadable date gets weight 0 (excluded) - it must never turn the whole
    # team profile into NaN, and must never be silently treated as "recent" either.
    return np.where(np.isnan(weights), 0.0, weights)


# Used only when "Freeze Decay" is on AND index-based weighting is
# requested - see decay_weights_by_index below for why this exists.
FROZEN_HALF_LIFE_MATCHES = 8.0  # a reasonable, editable index-based analog
                                 # to the 45 calendar-day freeze setting -
                                 # NOT a precise conversion (that depends on
                                 # fixture density, which varies by league)


def decay_weights_by_index(n_matches: int, half_life_matches: float) -> np.ndarray:
    """Weight = exp(-ln(2) * matches_ago / half_life_matches), where
    matches_ago counts backward from the most recent match (0) regardless
    of the ACTUAL CALENDAR GAP between matches.

    Why this exists: calendar-day decay has a blind spot. If a team's
    last domestic match was right before a long international break or
    the summer off-season, EVERY one of their matches - including the
    handful right before the break, which are still the most relevant
    form reference available - ends up heavily time-decayed just because
    a lot of calendar days happened to pass, not because the team's form
    is actually stale. Counting by match INDEX instead of days sidesteps
    that: the team's most recent match is always weight 1.0, their
    second-most-recent is next, and so on, regardless of how many
    calendar days sit between them and the upcoming fixture."""
    if n_matches <= 0:
        return np.array([])
    matches_ago = np.arange(n_matches, dtype=float)  # 0 = most recent (rows must be sorted newest-first)
    return np.exp(-math.log(2) * matches_ago / max(half_life_matches, 1e-6))


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """NaN-aware: a missing value is EXCLUDED from the average (its weight is dropped too),
    never counted as zero. Returns NaN when no value at all is usable, so callers can tell
    'no data' apart from 'a real zero'."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ok = np.isfinite(values) & np.isfinite(weights)
    if not ok.any():
        return float("nan")
    v, w = values[ok], weights[ok]
    total_weight = w.sum()
    if total_weight <= 0:
        return float(np.mean(v))
    return float(np.sum(v * w) / total_weight)


def effective_sample_size(weights: np.ndarray) -> float:
    """ESS = (sum(w))^2 / sum(w^2) - how many EQUALLY-weighted matches
    this decay-weighted sample is actually worth, not just how many
    matches exist. A team with 30 matches on file but a short half-life
    might have an ESS of only ~8 if the weighting is concentrated on
    the most recent few - ESS is the number that should drive "is this
    enough data" checks, not raw match count. This is the same
    "effective sample size, not total match count" caution that applies
    when a half-life search hits the edge of its candidate range (see
    half_life_at_search_boundary below) - a real fix needs to check ESS
    per team per venue, since that's the sample the decay curve is
    actually working with.

    With uniform weights of 1.0, ESS reduces to len(weights) exactly -
    e.g. a rolling-window profile where every in-window match counts
    equally has ESS == the number of matches in the window."""
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    sq_total = np.sum(weights ** 2)
    if sq_total <= 0:
        return 0.0
    return float((total ** 2) / sq_total)


# ---------------------------------------------------------------------------
# Missing-data-safe helpers + season-aware weighting
# ---------------------------------------------------------------------------

def season_row_weights(df: pd.DataFrame, carryover: float | None = None) -> np.ndarray:
    """One weight per row for building LEAGUE BASELINES: matches from the most recent season
    in `df` get weight 1.0, the season before that `carryover`, two seasons back
    `carryover**2`, and so on. So the baseline adapts as a new season starts (its own scoring
    environment takes over as matches accumulate) instead of one fixed average across every
    season on file. carryover=1.0 reproduces the old 'pool everything' behaviour.

    Season labels come from derive_season_series() (explicit season column first, then an
    Aug-June date inference). With no season information at all every row gets weight 1.0.
    'Most recent' is judged from the rows passed in, so a walk-forward backtest (which only
    ever passes in matches before the one being predicted) sees the right 'current' season at
    every step - no look-ahead. (Numpy-only: this runs once per backtest step.)"""
    n = len(df)
    if n == 0:
        return np.array([])
    c = SEASON_CARRYOVER if carryover is None else float(carryover)
    if c >= 0.999:
        return np.ones(n)
    try:
        seasons, method = derive_season_series(df)
    except Exception:
        return np.ones(n)
    if method == "fallback":
        return np.ones(n)
    codes, uniques = pd.factorize(np.asarray(seasons, dtype=object))
    n_u = len(uniques)
    if n_u <= 1:
        return np.ones(n)
    nat = np.iinfo(np.int64).min
    latest = np.full(n_u, nat, dtype=np.int64)
    if "date" in df.columns:
        dates = df["date"] if pd.api.types.is_datetime64_any_dtype(df["date"]) else pd.to_datetime(df["date"], errors="coerce")
        d = dates.to_numpy(dtype="datetime64[ns]").astype("int64")
        valid = (d != nat) & (codes >= 0)
        if valid.any():
            np.maximum.at(latest, codes[valid], d[valid])
    if (latest == nat).all():
        latest = np.arange(n_u, dtype=np.int64)  # no dates: the label that appeared last is the most recent
    order = np.argsort(-latest.astype(float), kind="stable")
    rank = np.empty(n_u, dtype=float)
    rank[order] = np.arange(n_u)
    ranks = np.where(codes >= 0, rank[np.clip(codes, 0, None)], float(n_u))
    return np.power(c, ranks)


def _wcol_mean(df: pd.DataFrame, col: str, weights: np.ndarray) -> float:
    if col not in df.columns or len(df) == 0:
        return float("nan")
    return weighted_mean(df[col].to_numpy(dtype=float), weights)


def _ratio_sums(numerator: np.ndarray, denominator: np.ndarray, weights: np.ndarray) -> float:
    """sum(num*w) / sum(den*w) over ONLY the rows where both are present. NaN if unusable."""
    ok = np.isfinite(numerator) & np.isfinite(denominator) & np.isfinite(weights)
    denom = float(np.sum(denominator[ok] * weights[ok]))
    if denom <= 0:
        return float("nan")
    return float(np.sum(numerator[ok] * weights[ok]) / denom)


def _shrink(raw: float, weight: float, league_value: float) -> float:
    if raw is None or not np.isfinite(raw):
        return league_value  # no usable own data -> sit on the league average
    return weight * raw + (1 - weight) * league_value


def _pooled_ratio(df: pd.DataFrame, pairs: list) -> float:
    """Pooled sum(num)/sum(den) over rows where BOTH columns are present (pairs = [(num_col, den_col), ...])."""
    num_total = den_total = 0.0
    for num_col, den_col in pairs:
        num = df[num_col].to_numpy(dtype=float)
        den = df[den_col].to_numpy(dtype=float)
        ok = np.isfinite(num) & np.isfinite(den)
        num_total += float(num[ok].sum())
        den_total += float(den[ok].sum())
    return num_total / den_total if den_total > 0 else float("nan")


def _nanaverage(ratios, weights, neutral: float = 1.0) -> float:
    r = np.asarray(ratios, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(r) & (w > 0)
    if not ok.any():
        return neutral
    return float(np.average(r[ok], weights=w[ok]))


@dataclass
class TerritoryProfile:
    """The three advanced territory metrics for one team, one venue role
    (host or visitor), for both what they generate ('for') and what they
    concede ('against')."""
    n_matches: int
    effective_sample_size: float
    big_chances_for: float
    big_chances_against: float
    shots_on_target_for: float
    shots_on_target_against: float
    box_touches_for: float
    box_touches_against: float


def team_territory_profile(
    league_df: pd.DataFrame, team: str, venue: str, half_life_days: float, reference_date: pd.Timestamp,
    use_match_index: bool = False, half_life_matches: float = FROZEN_HALF_LIFE_MATCHES,
) -> TerritoryProfile | None:
    """venue is 'home' or 'away'. Only ever looks at rows where the team
    played in that exact venue role - this is the "strict venue-isolated
    split" from Section 3. Returns None if there's no data at all for
    this team/venue (caller applies the 5-match safety rail).

    use_match_index=True switches the decay basis from calendar days to
    match recency index (see decay_weights_by_index) - this is what
    "Freeze Decay" now uses, specifically to avoid a summer break or
    international window fictitiously flattening a team's recent form
    just because a lot of calendar days happened to pass."""
    if venue == "home":
        rows = league_df[league_df["home_team"] == team]
        for_prefix, against_prefix = "home_", "away_"
    else:
        rows = league_df[league_df["away_team"] == team]
        for_prefix, against_prefix = "away_", "home_"

    if rows.empty:
        return None

    if use_match_index:
        rows = rows.sort_values("date", ascending=False)  # index 0 = most recent
        w = decay_weights_by_index(len(rows), half_life_matches)
    else:
        w = decay_weights(rows["date"], reference_date, half_life_days)

    def wmean(col):
        vals = rows[col].to_numpy(dtype=float)
        return weighted_mean(vals, w)

    # Only rows that actually carry at least one territory stat count toward the sample -
    # a match whose stats are all missing says nothing about this team.
    stat_cols = [f"{p}{m}" for p in (for_prefix, against_prefix) for m in TERRITORY_STATS]
    usable = rows[stat_cols].notna().any(axis=1).to_numpy()
    if not usable.any():
        return None
    w_effective = np.where(usable, w, 0.0)

    return TerritoryProfile(
        n_matches=int(usable.sum()),
        effective_sample_size=round(effective_sample_size(w_effective), 1),
        big_chances_for=wmean(f"{for_prefix}big_chances"),
        big_chances_against=wmean(f"{against_prefix}big_chances"),
        shots_on_target_for=wmean(f"{for_prefix}shots_on_target"),
        shots_on_target_against=wmean(f"{against_prefix}shots_on_target"),
        box_touches_for=wmean(f"{for_prefix}box_touches"),
        box_touches_against=wmean(f"{against_prefix}box_touches"),
    )


def rolling_window_weights(days_elapsed: np.ndarray, window_days: float) -> np.ndarray:
    """The classic alternative to exponential time-decay: a HARD cutoff.
    Full weight (1.0) for any match within the window, ZERO outside it -
    unlike decay_weights, which never fully zeroes out an old match, just
    fades it smoothly toward (but never reaching) zero. A rolling window
    is a step function; exponential decay is a smooth curve that a
    rolling window is a special, discontinuous case of."""
    days_elapsed = np.asarray(days_elapsed, dtype=float)
    return np.where(days_elapsed <= window_days, 1.0, 0.0)


def team_territory_profile_rolling(
    league_df: pd.DataFrame, team: str, venue: str, window_days: float, reference_date: pd.Timestamp,
) -> "TerritoryProfile | None":
    """Rolling-window counterpart to team_territory_profile - same venue
    isolation and aggregation, but a hard cutoff instead of exponential
    decay. Deliberately kept as a fully separate function (not a mode
    flag on the live one) so running this comparison can never accidentally
    change what the live prediction path actually uses."""
    if venue == "home":
        rows = league_df[league_df["home_team"] == team]
        for_prefix, against_prefix = "home_", "away_"
    else:
        rows = league_df[league_df["away_team"] == team]
        for_prefix, against_prefix = "away_", "home_"

    if rows.empty:
        return None

    days_elapsed = (reference_date - rows["date"]).dt.days.clip(lower=0).to_numpy(dtype=float)
    days_elapsed = np.where(np.isnan(days_elapsed), np.inf, days_elapsed)  # unreadable date -> outside every window
    w = rolling_window_weights(days_elapsed, window_days)
    if w.sum() <= 0:
        return None  # every match for this team/venue fell outside the window

    def wmean(col):
        vals = rows[col].to_numpy(dtype=float)
        return weighted_mean(vals, w)

    return TerritoryProfile(
        n_matches=int((w > 0).sum()),  # only count matches actually inside the window
        effective_sample_size=round(effective_sample_size(w), 1),  # == n_matches here (0/1 weights), kept for API consistency with the decay-weighted profile
        big_chances_for=wmean(f"{for_prefix}big_chances"),
        big_chances_against=wmean(f"{against_prefix}big_chances"),
        shots_on_target_for=wmean(f"{for_prefix}shots_on_target"),
        shots_on_target_against=wmean(f"{against_prefix}shots_on_target"),
        box_touches_for=wmean(f"{for_prefix}box_touches"),
        box_touches_against=wmean(f"{against_prefix}box_touches"),
    )


SOT_CONVERSION_SHRINKAGE_K = 15.0  # "pseudo-matches" of league-average prior
                                    # strength - same empirical-Bayes shrinkage
                                    # spirit used elsewhere for thin samples:
                                    # shrink_weight = ESS / (ESS + K), so a team
                                    # with ESS well below K sits close to the
                                    # league average, and one with ESS well
                                    # above K is trusted almost entirely on its
                                    # own data.


@dataclass
class SOTConversionProfile:
    """How efficiently a team turns shots on target into goals (FOR),
    and how efficiently opponents turn shots on target into goals
    AGAINST this team (i.e. this team's own defending/goalkeeping at
    preventing SOT from becoming goals) - separate from raw SOT volume,
    which team_territory_profile already covers. Both conversion rates
    are shrunk toward the league average based on effective sample size,
    since a handful of matches can produce a wildly noisy conversion
    rate (a single deflected shot or an outstanding goalkeeping display
    swings it hard) - the raw, un-shrunk numbers are kept alongside for
    transparency."""
    n_matches: int
    effective_sample_size: float
    sot_conversion_for: float
    sot_conversion_against: float
    raw_sot_conversion_for: float
    raw_sot_conversion_against: float
    league_avg_conversion: float
    shrinkage_weight: float  # 0 = entirely league average, 1 = entirely this team's own data


def league_average_sot_conversion(settled_df: pd.DataFrame) -> float:
    """Aggregate goals / aggregate shots-on-target across the WHOLE
    league (home and away pooled together, since conversion efficiency
    isn't a venue-specific concept the way attack/defense strength is) -
    the shrinkage target every team's own conversion rate is pulled
    toward when its own sample is thin."""
    if settled_df.empty:
        return 0.3  # a reasonable generic football fallback - roughly matches typical SOT conversion rates
    pooled = _pooled_ratio(settled_df, [("home_goals", "home_shots_on_target"), ("away_goals", "away_shots_on_target")])
    return pooled if np.isfinite(pooled) else 0.3


def team_sot_conversion_profile(
    league_df: pd.DataFrame, team: str, venue: str, half_life_days: float, reference_date: pd.Timestamp,
    league_avg_conversion: float, use_match_index: bool = False, half_life_matches: float = FROZEN_HALF_LIFE_MATCHES,
    shrinkage_k: float = SOT_CONVERSION_SHRINKAGE_K,
) -> "SOTConversionProfile | None":
    """Same venue-isolated, decay-weighted approach as
    team_territory_profile, but for goals-per-SOT (for) and
    goals-conceded-per-SOT-faced (against). Uses the ratio of
    decay-weighted SUMS (sum(goals*w) / sum(sot*w)) rather than
    averaging each match's own ratio - this avoids a single low-SOT,
    lucky-goal match producing a wild per-match ratio that then gets
    treated as equally informative as a normal match; it's the same
    "aggregate the raw counts, then divide" principle odds_calibration_table
    uses when it aggregates predictions into bands rather than averaging
    per-match ratios."""
    if venue == "home":
        rows = league_df[league_df["home_team"] == team]
        for_prefix, against_prefix = "home_", "away_"
    else:
        rows = league_df[league_df["away_team"] == team]
        for_prefix, against_prefix = "away_", "home_"

    if rows.empty:
        return None

    if use_match_index:
        rows = rows.sort_values("date", ascending=False)
        w = decay_weights_by_index(len(rows), half_life_matches)
    else:
        w = decay_weights(rows["date"], reference_date, half_life_days)

    ess = effective_sample_size(w)

    goals_for = rows[f"{for_prefix}goals"].to_numpy(dtype=float)
    sot_for = rows[f"{for_prefix}shots_on_target"].to_numpy(dtype=float)
    goals_against = rows[f"{against_prefix}goals"].to_numpy(dtype=float)
    sot_against = rows[f"{against_prefix}shots_on_target"].to_numpy(dtype=float)

    raw_for = _ratio_sums(goals_for, sot_for, w)
    raw_against = _ratio_sums(goals_against, sot_against, w)

    shrink_weight = ess / (ess + shrinkage_k) if (ess + shrinkage_k) > 0 else 0.0
    shrunk_for = _shrink(raw_for, shrink_weight, league_avg_conversion)
    shrunk_against = _shrink(raw_against, shrink_weight, league_avg_conversion)

    return SOTConversionProfile(
        n_matches=len(rows),
        effective_sample_size=round(ess, 1),
        sot_conversion_for=round(shrunk_for, 3),
        sot_conversion_against=round(shrunk_against, 3),
        raw_sot_conversion_for=round(raw_for, 3),
        raw_sot_conversion_against=round(raw_against, 3),
        league_avg_conversion=round(league_avg_conversion, 3),
        shrinkage_weight=round(shrink_weight, 3),
    )


# --- xGOT-based conversion: a sharper upgrade to SOT-conversion above -----
#
# SOT-conversion (goals / shots on target) treats every shot on target as
# equally hard to score - a point-blank tap-in and a fingertip-save-worthy
# top-corner strike both just count as "1 SOT". xGOT (expected goals on
# target - post-shot quality, conditioned on the shot already being on
# frame: placement and power, not pre-shot location/angle like regular xG)
# is the real, analyst-grade version of that same "how clinical is this
# team's finishing" question. goals / xGOT is a materially sharper ratio
# than goals / SOT wherever xGOT data exists.
#
# NOT OFFERED FOR EVERY LEAGUE by most providers (unlike shots_on_target,
# which is near-universal) - every function below checks for the columns
# and returns None rather than guessing when they're missing, so a league
# without xGOT data falls back to the existing SOT-based conversion
# untouched, with nothing else in the pipeline affected.

def league_average_xgot_conversion(settled_df: pd.DataFrame) -> "float | None":
    if "home_xgot" not in settled_df.columns or "away_xgot" not in settled_df.columns:
        return None
    pooled = _pooled_ratio(settled_df, [("home_goals", "home_xgot"), ("away_goals", "away_xgot")])
    return pooled if np.isfinite(pooled) else None


def team_xgot_conversion_profile(
    league_df: pd.DataFrame, team: str, venue: str, half_life_days: float, reference_date: pd.Timestamp,
    league_avg_conversion: float, use_match_index: bool = False, half_life_matches: float = FROZEN_HALF_LIFE_MATCHES,
    shrinkage_k: float = SOT_CONVERSION_SHRINKAGE_K,
) -> "SOTConversionProfile | None":
    """Identical shape and shrinkage to team_sot_conversion_profile, with
    goals/xGOT in place of goals/SOT wherever home_xgot/away_xgot exist
    in league_df. Returns None (not a fallback guess) when the columns
    aren't present for this league - the caller decides whether to show
    the SOT-based version instead, never silently substituting one for
    the other."""
    if "home_xgot" not in league_df.columns or "away_xgot" not in league_df.columns:
        return None
    if venue == "home":
        rows = league_df[league_df["home_team"] == team]
        for_prefix, against_prefix = "home_", "away_"
    else:
        rows = league_df[league_df["away_team"] == team]
        for_prefix, against_prefix = "away_", "home_"
    if rows.empty:
        return None

    if use_match_index:
        rows = rows.sort_values("date", ascending=False)
        w = decay_weights_by_index(len(rows), half_life_matches)
    else:
        w = decay_weights(rows["date"], reference_date, half_life_days)

    ess = effective_sample_size(w)

    goals_for = rows[f"{for_prefix}goals"].to_numpy(dtype=float)
    xgot_for = rows[f"{for_prefix}xgot"].to_numpy(dtype=float)
    goals_against = rows[f"{against_prefix}goals"].to_numpy(dtype=float)
    xgot_against = rows[f"{against_prefix}xgot"].to_numpy(dtype=float)

    raw_for = _ratio_sums(goals_for, xgot_for, w)
    raw_against = _ratio_sums(goals_against, xgot_against, w)

    shrink_weight = ess / (ess + shrinkage_k) if (ess + shrinkage_k) > 0 else 0.0
    shrunk_for = _shrink(raw_for, shrink_weight, league_avg_conversion)
    shrunk_against = _shrink(raw_against, shrink_weight, league_avg_conversion)

    return SOTConversionProfile(
        n_matches=len(rows),
        effective_sample_size=round(ess, 1),
        sot_conversion_for=round(shrunk_for, 3),
        sot_conversion_against=round(shrunk_against, 3),
        raw_sot_conversion_for=round(raw_for, 3),
        raw_sot_conversion_against=round(raw_against, 3),
        league_avg_conversion=round(league_avg_conversion, 3),
        shrinkage_weight=round(shrink_weight, 3),
    )


@dataclass
class LeagueBaseline:
    """League-wide averages used as the shared, non-team-specific
    scale anchor - see design note #2 at the top of this file."""
    avg_home_goals: float
    avg_away_goals: float
    home_big_chances_for: float
    home_big_chances_against: float
    home_sot_for: float
    home_sot_against: float
    home_box_for: float
    home_box_against: float
    away_big_chances_for: float
    away_big_chances_against: float
    away_sot_for: float
    away_sot_against: float
    away_box_for: float
    away_box_against: float


def compute_league_baseline(settled_df: pd.DataFrame, season_carryover: float | None = None) -> LeagueBaseline:
    """League-wide averages, SEASON-AWARE: the most recent season in `settled_df` counts fully
    and each earlier season counts `season_carryover` times less than the one after it (see
    season_row_weights). The baseline therefore follows the scoring environment of the current
    season as it develops, instead of one fixed average across every season on file.
    Missing stats are excluded from each average (never counted as 0). A stat missing
    everywhere gets a NaN baseline, and any ratio built on it is simply left out of the
    strength blend rather than being dragged towards 0."""
    w = season_row_weights(settled_df, season_carryover)

    def goals_mean(col):
        v = _wcol_mean(settled_df, col, w)
        return v if np.isfinite(v) and v > 0 else 1.0

    def stat_mean(col):
        v = _wcol_mean(settled_df, col, w)
        return v if np.isfinite(v) and v > 0 else float("nan")

    return LeagueBaseline(
        avg_home_goals=goals_mean("home_goals"),
        avg_away_goals=goals_mean("away_goals"),
        home_big_chances_for=stat_mean("home_big_chances"),
        home_big_chances_against=stat_mean("away_big_chances"),
        home_sot_for=stat_mean("home_shots_on_target"),
        home_sot_against=stat_mean("away_shots_on_target"),
        home_box_for=stat_mean("home_box_touches"),
        home_box_against=stat_mean("away_box_touches"),
        away_big_chances_for=stat_mean("away_big_chances"),
        away_big_chances_against=stat_mean("home_big_chances"),
        away_sot_for=stat_mean("away_shots_on_target"),
        away_sot_against=stat_mean("home_shots_on_target"),
        away_box_for=stat_mean("away_box_touches"),
        away_box_against=stat_mean("home_box_touches"),
    )


def _safe_ratio(numerator: float, denominator: float) -> float:
    """NaN in -> NaN out (callers leave NaN ratios out of their blends). A zero/negative
    denominator keeps the old neutral 1.0."""
    if numerator is None or denominator is None:
        return float("nan")
    if not (np.isfinite(numerator) and np.isfinite(denominator)):
        return float("nan")
    if denominator <= 0:
        return 1.0
    return numerator / denominator


# The 3 territory ratios are (big_chances, shots_on_target, box_touches),
# in that fixed order, wherever a `weights` tuple appears. Equal weighting
# is the historical default (matches the model's original behavior
# unchanged); optimize_territory_weights() below can find a better-fitting
# blend per league instead.
TERRITORY_WEIGHTS_DEFAULT = (1 / 3, 1 / 3, 1 / 3)

# A modest, deliberately small grid for the weight-calibration search -
# each candidate here means re-running a walk-forward backtest, so this
# stays a manageable size rather than an exhaustive continuous search.
_TERRITORY_WEIGHT_STEPS = [0.15, 0.25, 1 / 3, 0.45, 0.55]


def _territory_weight_candidates():
    candidates = []
    for big_chances_w in _TERRITORY_WEIGHT_STEPS:
        for sot_w in _TERRITORY_WEIGHT_STEPS:
            box_w = 1.0 - big_chances_w - sot_w
            if 0.05 <= box_w <= 0.6:
                candidates.append((round(big_chances_w, 4), round(sot_w, 4), round(box_w, 4)))
    return candidates


def attack_strength(profile, baseline: LeagueBaseline, venue: str, weights: tuple = TERRITORY_WEIGHTS_DEFAULT) -> float:
    """Composite attack strength = weighted average of the 3 territory
    ratios vs the league baseline for that venue role (big chances, shots
    on target, box touches - in that order in `weights`). Defaults to
    equal weighting (1/3 each), matching the original behavior; a
    per-league calibrated weighting can be passed in instead - see
    optimize_territory_weights(). Falls back to a neutral 1.0 if the
    sample is too small (Section 3's 5-match safety rail)."""
    if profile is None or profile.n_matches < MIN_SAMPLE_ROWS:
        return 1.0
    if venue == "home":
        ratios = [
            _safe_ratio(profile.big_chances_for, baseline.home_big_chances_for),
            _safe_ratio(profile.shots_on_target_for, baseline.home_sot_for),
            _safe_ratio(profile.box_touches_for, baseline.home_box_for),
        ]
    else:
        ratios = [
            _safe_ratio(profile.big_chances_for, baseline.away_big_chances_for),
            _safe_ratio(profile.shots_on_target_for, baseline.away_sot_for),
            _safe_ratio(profile.box_touches_for, baseline.away_box_for),
        ]
    return _nanaverage(ratios, weights)


def defense_strength(profile, baseline: LeagueBaseline, venue: str, weights: tuple = TERRITORY_WEIGHTS_DEFAULT) -> float:
    """Composite defense strength (how much this team ALLOWS relative to
    league baseline, in the same venue role) - values BELOW 1.0 mean a
    better-than-average defense. Same weighting contract as
    attack_strength() above. Falls back to neutral 1.0 on a too-small
    sample."""
    if profile is None or profile.n_matches < MIN_SAMPLE_ROWS:
        return 1.0
    if venue == "home":
        ratios = [
            _safe_ratio(profile.big_chances_against, baseline.home_big_chances_against),
            _safe_ratio(profile.shots_on_target_against, baseline.home_sot_against),
            _safe_ratio(profile.box_touches_against, baseline.home_box_against),
        ]
    else:
        ratios = [
            _safe_ratio(profile.big_chances_against, baseline.away_big_chances_against),
            _safe_ratio(profile.shots_on_target_against, baseline.away_sot_against),
            _safe_ratio(profile.box_touches_against, baseline.away_box_against),
        ]
    return _nanaverage(ratios, weights)


def expected_goals(
    home_attack: float, away_defense: float, away_attack: float, home_defense: float, baseline: LeagueBaseline,
) -> tuple[float, float]:
    """Home advantage is NOT re-applied here - it's already inside
    home_attack/home_defense because those numbers only ever came from
    the team's own home-position rows (see design note #1)."""
    lambda_home = baseline.avg_home_goals * home_attack * away_defense
    lambda_away = baseline.avg_away_goals * away_attack * home_defense
    return max(lambda_home, 0.05), max(lambda_away, 0.05)


def team_expected_goals_against(defense: float, baseline: LeagueBaseline, venue: str) -> float:
    """A team's own STANDALONE 'expected goals against' rating - how
    many goals they'd be expected to concede against a LEAGUE-AVERAGE
    opponent (attack_strength held at the neutral 1.0), using only
    their own defense_strength rating for that venue. This is exactly
    the same formula expected_goals() uses to build a specific
    fixture's lambda, just with the opponent's attack fixed at neutral
    instead of a real opponent's rating - so the number reflects THIS
    team's own defense alone, not muddied by whoever they're about to
    play. Useful as a standalone team-profile stat (same spirit as the
    SOT conversion against rating above), independent of any one
    fixture - e.g. for comparing two teams' defensive quality directly,
    or for a league-wide "best defense" ranking.

    venue is the TEAM's own venue role ('home' or 'away') - a home
    team's expected goals against uses the AWAY-goals league baseline
    (what an average away side scores), since that's who they're
    notionally defending against; an away team's uses the HOME-goals
    baseline, mirroring exactly how expected_goals() pairs lambda_away
    with home_defense and lambda_home with away_defense."""
    opponent_baseline_goals = baseline.avg_away_goals if venue == "home" else baseline.avg_home_goals
    return max(opponent_baseline_goals * defense, 0.05)


# ---------------------------------------------------------------------------
# Core Parameter A: dynamic half-life optimisation via Brier score backtest
# ---------------------------------------------------------------------------

def _quick_lambda_for_backtest(
    history_df: pd.DataFrame, home_team: str, away_team: str, half_life_days: float, as_of: pd.Timestamp,
    weights: tuple = TERRITORY_WEIGHTS_DEFAULT, full_model: bool = False, reliabilities: "dict | None" = None,
    engine_weights: "dict | None" = None, duel_strength: float = 1.0, style_out: "dict | None" = None,
):
    """full_model=False (default) is the FAST territorial-only approximation used purely to PICK
    a half-life / territory weights (high-multiplicity searches). full_model=True runs the exact
    same compute_match_strengths() pipeline the live app uses (engine blend + duel + momentum),
    which is what walk_forward_backtest() now uses so the validation describes the real model."""
    if len(history_df) < MIN_SAMPLE_ROWS * 2:
        return None
    baseline = compute_league_baseline(history_df)
    if full_model:
        ms = compute_match_strengths(
            history_df, home_team, away_team, half_life_days, as_of, weights,
            reliabilities=reliabilities, baseline=baseline, engine_weights=engine_weights, duel_strength=duel_strength,
        )
        if style_out is not None:
            style_out["home"], style_out["away"] = ms.home_style, ms.away_style
        return expected_goals(ms.home_attack, ms.away_defense, ms.away_attack, ms.home_defense, baseline)
    home_profile = team_territory_profile(history_df, home_team, "home", half_life_days, as_of)
    away_profile = team_territory_profile(history_df, away_team, "away", half_life_days, as_of)
    ha = attack_strength(home_profile, baseline, "home", weights)
    hd = defense_strength(home_profile, baseline, "home", weights)
    aa = attack_strength(away_profile, baseline, "away", weights)
    ad = defense_strength(away_profile, baseline, "away", weights)
    return expected_goals(ha, ad, aa, hd, baseline)


def _quick_corner_lambda_for_backtest(
    history_df: pd.DataFrame, home_team: str, away_team: str, half_life_days: float, as_of: pd.Timestamp,
) -> "tuple | None":
    """Same walk-forward, no-lookahead shape as _quick_lambda_for_backtest
    above, just for corners - only ever called from walk_forward_backtest
    when include_all_markets=True (the 1X2-only backtest paths used by
    half-life/rho optimisation don't need this extra cost), and only
    when corner_kicks columns actually exist in this dataset (checked via
    corners_league_baseline returning None otherwise)."""
    if "home_corner_kicks" not in history_df.columns or "away_corner_kicks" not in history_df.columns:
        return None
    if len(history_df) < MIN_SAMPLE_ROWS * 2:
        return None
    baseline = corners_league_baseline(history_df)
    home_profile = corners_team_profile(history_df, home_team, "home", half_life_days, as_of)
    away_profile = corners_team_profile(history_df, away_team, "away", half_life_days, as_of)
    return expected_corners(home_profile, away_profile, baseline)


def _outcome_probs_from_lambdas(lam_home: float, lam_away: float) -> tuple[float, float, float]:
    home_win = draw = away_win = 0.0
    for x in range(GOAL_CAP + 1):
        px = scipy_poisson.pmf(x, lam_home)
        for y in range(GOAL_CAP + 1):
            py = scipy_poisson.pmf(y, lam_away)
            p = px * py
            if x > y:
                home_win += p
            elif x == y:
                draw += p
            else:
                away_win += p
    total = home_win + draw + away_win
    if total <= 0:
        return 1 / 3, 1 / 3, 1 / 3
    return home_win / total, draw / total, away_win / total


def optimize_half_life(settled_df: pd.DataFrame, max_matches_evaluated: int = 150, weights: tuple = TERRITORY_WEIGHTS_DEFAULT, candidates: list = None):
    """Backtests each candidate half-life on real past results using a
    Brier score, and returns the one with the lowest average error. Only
    evaluates the most recent `max_matches_evaluated` matches for
    performance - this is a real backtest, not a fixed guess, but a
    league's full season history doesn't need to be replayed dozens of
    times over to get a stable answer.

    `candidates` defaults to HALF_LIFE_CANDIDATES (15 to 365 days) - the live
    app always uses this default. It exists as a parameter so a diagnostic
    can pass in a different range."""
    candidates = candidates if candidates is not None else HALF_LIFE_CANDIDATES
    df = settled_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if len(df) < MIN_SAMPLE_ROWS * 3:
        return FROZEN_HALF_LIFE_DAYS, {"reason": "not enough settled matches to backtest - using frozen default"}

    eval_start = max(MIN_SAMPLE_ROWS * 2, len(df) - max_matches_evaluated)
    scores = {}
    for half_life in candidates:
        brier_terms = []
        for i in range(eval_start, len(df)):
            row = df.iloc[i]
            history = df.iloc[:i]
            lambdas = _quick_lambda_for_backtest(
                history, row["home_team"], row["away_team"], half_life, row["date"], weights
            )
            if lambdas is None:
                continue
            p_home, p_draw, p_away = _outcome_probs_from_lambdas(*lambdas)
            actual = (
                (1, 0, 0) if row["home_goals"] > row["away_goals"]
                else (0, 1, 0) if row["home_goals"] == row["away_goals"]
                else (0, 0, 1)
            )
            brier_terms.append(
                (p_home - actual[0]) ** 2 + (p_draw - actual[1]) ** 2 + (p_away - actual[2]) ** 2
            )
        if brier_terms:
            scores[half_life] = float(np.mean(brier_terms))

    if not scores:
        return FROZEN_HALF_LIFE_DAYS, {"reason": "backtest produced no evaluable matches - using frozen default"}

    best = min(scores, key=scores.get)
    return best, {"brier_scores_by_half_life": scores, "chosen": best}


EXTENDED_HALF_LIFE_CANDIDATES = HALF_LIFE_CANDIDATES  # the normal grid now reaches 365 days, so there is no longer a separate 'extended' range


def extended_half_life_diagnostic(settled_df: pd.DataFrame, max_matches_evaluated: int = 150, weights: tuple = TERRITORY_WEIGHTS_DEFAULT) -> dict:
    """Diagnostic-only: re-runs the half-life search with candidates
    extending well past the normal 180-day ceiling (up to 365 days). Does
    NOT change what the live app uses - optimize_half_life's default
    candidate list is untouched, this is purely for checking whether a
    180-day boundary result reflects the score genuinely still improving
    past 180 (the normal grid really was capping something real), or
    whether it plateaus/worsens past 180 (180 was already close to a
    genuine optimum, and hitting the boundary was closer to noise between
    neighboring candidates than a sign of anything being cut off)."""
    best, info = optimize_half_life(
        settled_df, max_matches_evaluated=max_matches_evaluated, weights=weights,
        candidates=EXTENDED_HALF_LIFE_CANDIDATES,
    )
    if "brier_scores_by_half_life" not in info:
        return {"chosen": best, "still_improving_past_180": None, "reason": info.get("reason")}

    scores = info["brier_scores_by_half_life"]
    within_180 = {hl: s for hl, s in scores.items() if hl <= 180}
    beyond_180 = {hl: s for hl, s in scores.items() if hl > 180}
    still_improving = None
    if within_180 and beyond_180:
        best_within = min(within_180.values())
        best_beyond = min(beyond_180.values())
        still_improving = best_beyond < best_within
    return {
        "chosen": best,
        "brier_scores_by_half_life": scores,
        "still_improving_past_180": still_improving,
    }


def half_life_at_search_boundary(half_life: float, candidates=HALF_LIFE_CANDIDATES) -> str | None:
    """Returns a warning string if the optimizer picked the exact minimum
    or maximum of the tested half-life range - not necessarily wrong, but
    a real diagnostic flag: it means either the TRUE optimum lies outside
    the range that was actually searched (the grid artificially capped
    it), or the backtest sample was too thin/noisy for the Brier search to
    meaningfully discriminate between candidates, and it landed on an edge
    somewhat arbitrarily. Returns None when the chosen half-life is
    comfortably inside the range (no warning needed)."""
    if half_life is None or (isinstance(half_life, float) and math.isnan(half_life)):
        return None
    lo, hi = min(candidates), max(candidates)
    if half_life <= lo:
        return (
            f"⚠️ The optimizer selected {half_life:.0f} days - the SHORTEST option tested "
            f"({lo}-{hi} day range). The true optimum may be even faster-decaying than what "
            "was searched, or the backtest sample may be too thin to distinguish clearly "
            "between short half-lives yet."
        )
    if half_life >= hi and hi >= HALF_LIFE_MAX_DAYS:
        return (
            f"ℹ️ The optimizer selected {half_life:.0f} days - the LONGEST option tested "
            f"({lo}-{hi} day range, the one-year ceiling). Matches up to a year old still carry real weight here: "
            "either this league's teams change very little from one season to the next, or the sample is too "
            "thin for the search to tell longer memories apart. It is a boundary result, not necessarily a "
            "wrong one - check how many settled matches each team has at each venue."
        )
    if half_life >= hi:
        return (
            f"⚠️ The optimizer selected {half_life:.0f} days - the LONGEST option tested "
            f"({lo}-{hi} day range). The true optimum may need MORE memory than what was "
            "searched (the grid capped it here), or the backtest sample may be too thin/noisy "
            "for the search to confidently distinguish between longer half-lives. Worth "
            "checking how many settled matches this league actually has, especially per team "
            "per venue, since that's the more relevant sample size than the total match count."
        )
    return None


def safety_rail_dilution_report(settled_df: pd.DataFrame, max_matches_evaluated: int = 150) -> dict:
    """Reports what share of the matches the half-life search actually
    evaluates have AT LEAST ONE side under the 5-match safety rail at
    that point in time - meaning that side contributes a flat, neutral
    1.0 (see attack_strength/defense_strength) rather than real signal.
    This directly explains a half-life search landing on a grid boundary
    somewhat arbitrarily: a "diluted" match scores IDENTICALLY no matter
    which half-life candidate is tested (the neutral side's contribution
    doesn't change with decay speed), so it adds noise to the search
    without adding any real discriminating signal. Uses the exact same
    walk-forward window (eval_start, per-match eligibility) as
    optimize_half_life, so the percentage reported here reflects what
    that search actually saw - not a separately-defined approximation.

    Deliberately independent of which half-life is chosen: n_matches is
    just a raw row count, unaffected by decay weighting, so this doesn't
    need to be recomputed per half-life candidate."""
    df = settled_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if len(df) < MIN_SAMPLE_ROWS * 3:
        return {"total_evaluated": 0, "diluted_count": 0, "diluted_pct": float("nan"),
                "reason": "not enough settled matches to backtest"}

    eval_start = max(MIN_SAMPLE_ROWS * 2, len(df) - max_matches_evaluated)
    total = 0
    diluted = 0
    for i in range(eval_start, len(df)):
        row = df.iloc[i]
        history = df.iloc[:i]
        if len(history) < MIN_SAMPLE_ROWS * 2:
            continue  # matches _quick_lambda_for_backtest's own eligibility check exactly
        home_profile = team_territory_profile(history, row["home_team"], "home", FROZEN_HALF_LIFE_DAYS, row["date"])
        away_profile = team_territory_profile(history, row["away_team"], "away", FROZEN_HALF_LIFE_DAYS, row["date"])
        total += 1
        home_thin = home_profile is None or home_profile.n_matches < MIN_SAMPLE_ROWS
        away_thin = away_profile is None or away_profile.n_matches < MIN_SAMPLE_ROWS
        if home_thin or away_thin:
            diluted += 1

    if total == 0:
        return {"total_evaluated": 0, "diluted_count": 0, "diluted_pct": float("nan"),
                "reason": "no evaluable matches"}
    return {
        "total_evaluated": total,
        "diluted_count": diluted,
        "diluted_pct": round(100 * diluted / total, 1),
    }


def optimize_territory_weights(settled_df: pd.DataFrame, half_life_days: float, max_matches_evaluated: int = 150):
    """Same idea as optimize_half_life(), but searches how much weight to
    put on big chances vs shots on target vs box touches (instead of
    searching decay speed), holding half-life FIXED at whatever was
    already chosen for this league. Different leagues' playing styles can
    genuinely differ in which territory metric best predicts goals - a
    possession-heavy league might lean more on big chances, a
    counter-attacking one more on shots on target - so a single universal
    equal-weighted blend isn't guaranteed to fit every league equally
    well. This is a real backtest search, not a guess: each candidate
    weighting is scored with the same walk-forward Brier score as
    everywhere else in this app, and whichever scores best is returned.

    Deliberately NOT run automatically on every prediction (that would
    nest an expensive grid-search backtest inside every single fixture
    load) - call this explicitly (e.g. from a 'Calibrate weights for this
    league' button) and cache the result per league instead."""
    df = settled_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if len(df) < MIN_SAMPLE_ROWS * 3:
        return TERRITORY_WEIGHTS_DEFAULT, {"reason": "not enough settled matches to calibrate - using equal weights"}

    eval_start = max(MIN_SAMPLE_ROWS * 2, len(df) - max_matches_evaluated)
    candidates = _territory_weight_candidates()
    scores = {}
    for weights in candidates:
        brier_terms = []
        for i in range(eval_start, len(df)):
            row = df.iloc[i]
            history = df.iloc[:i]
            lambdas = _quick_lambda_for_backtest(
                history, row["home_team"], row["away_team"], half_life_days, row["date"], weights
            )
            if lambdas is None:
                continue
            p_home, p_draw, p_away = _outcome_probs_from_lambdas(*lambdas)
            actual = (
                (1, 0, 0) if row["home_goals"] > row["away_goals"]
                else (0, 1, 0) if row["home_goals"] == row["away_goals"]
                else (0, 0, 1)
            )
            brier_terms.append(
                (p_home - actual[0]) ** 2 + (p_draw - actual[1]) ** 2 + (p_away - actual[2]) ** 2
            )
        if brier_terms:
            scores[weights] = float(np.mean(brier_terms))

    if not scores:
        return TERRITORY_WEIGHTS_DEFAULT, {"reason": "backtest produced no evaluable matches - using equal weights"}

    best = min(scores, key=scores.get)
    # Don't look up the equal-weight baseline by exact dict-key equality -
    # TERRITORY_WEIGHTS_DEFAULT uses unrounded 1/3 floats while candidates
    # are rounded to 4dp, so they'd never be bit-identical even though one
    # of the candidates IS effectively equal weighting. Find the closest
    # candidate instead.
    equal_weight_key = min(
        scores.keys(),
        key=lambda w: sum(abs(a - b) for a, b in zip(w, TERRITORY_WEIGHTS_DEFAULT)),
    )
    equal_weight_score = scores[equal_weight_key]
    return best, {
        "brier_scores_by_weights": scores,
        "chosen": best,
        "equal_weight_score": equal_weight_score,
        "best_score": scores[best],
        "improved_over_equal_weight": scores[best] < equal_weight_score,
    }


# ---------------------------------------------------------------------------
# Full-dataset walk-forward backtest, Brier Skill Score, and accuracy
# ---------------------------------------------------------------------------

# MULTI SCORE (grouped correct score) - replaces the old single-scoreline Correct Score markets, none of which
# ever reached a usable probability on its own. Each market covers several exact scorelines (home goals, away
# goals) in ONE bet and wins if the final score is any of them. The groups plus "Other" (every scoreline not
# listed: 4+ goals for a side, 4-4 and above, ...) are mutually exclusive and exhaustive, so they sum to 1.
# THIS LIST IS THE ONLY PLACE the groups are defined - edit it to change them; the grid, the simulation, the real
# result check, calibration and the market list all follow automatically. A scoreline may appear in only one group.
MULTI_SCORE_GROUPS = [
    ("Multi Score Home 1-0/2-0/2-1", [(1, 0), (2, 0), (2, 1)]),
    ("Multi Score Home 3-0/3-1/3-2", [(3, 0), (3, 1), (3, 2)]),
    ("Multi Score Draw 0-0/1-1", [(0, 0), (1, 1)]),
    ("Multi Score Draw 2-2/3-3", [(2, 2), (3, 3)]),
    ("Multi Score Away 0-1/0-2/1-2", [(0, 1), (0, 2), (1, 2)]),
    ("Multi Score Away 0-3/1-3/2-3", [(0, 3), (1, 3), (2, 3)]),
]
MULTI_SCORE_OTHER = "Multi Score Other"
MULTI_SCORE_MARKETS = [name for name, _ in MULTI_SCORE_GROUPS] + [MULTI_SCORE_OTHER]
_multi_score_cells = [cell for _, cells in MULTI_SCORE_GROUPS for cell in cells]
assert len(_multi_score_cells) == len(set(_multi_score_cells)), "a scoreline is listed in two Multi Score groups"

CORNERS_LINES = [8.5, 9.5, 10.5]
CORNERS_MARKETS = (
    [f"Corners Over {line}" for line in CORNERS_LINES]
    + [f"Corners Under {line}" for line in CORNERS_LINES]
    + ["Most Corners Home", "Most Corners Away", "Most Corners Equal"]
)

# Combo markets - all built from the SAME Dixon-Coles grid / MC simulation
# already used for everything else (no new data, no new engine). The
# "+Total Goals" combos are scoped to the single 2.5 line only (the line
# bookmakers actually offer these combos at) rather than all three O/U
# lines, which would triple the market count for combos nobody books at
# 1.5 or 3.5 anyway.
# DE-DUPLICATION. These 11 markets were EXACTLY the same bet (identical scorelines) as a market that already exists
# under another name, so they added nothing but a second, separately calibrated copy of the same probability. They
# are no longer offered; the value is the market that is kept. Every complete-set group (calibration partitions) that
# contained one of them now points at the kept equivalent, so each set still sums to exactly 1.
REDUNDANT_WRAPPERS = {
    "Under 1.5 & BTTS No": "Under 1.5 Goals",
    "Over 1.5 & BTTS Yes": "BTTS - Yes",
    "1X & BTTS No": "Home Clean Sheet",
    "X2 & BTTS No": "Away Clean Sheet",
    "Home & BTTS No": "Home Win to Nil",
    "Away & BTTS No": "Away Win to Nil",
    "Draw & Under 2.5": "Multi Score Draw 0-0/1-1",
    "Draw & Under 3.5": "Multi Score Draw 0-0/1-1",
    "Draw & Over 3.5": "Draw & Over 2.5",
    "Draw & Over 1.5": "Draw & BTTS Yes",
    "Draw & Under 1.5": "Draw & BTTS No",
}


def canonical_market(name: str) -> str:
    """The market that is kept in place of a removed duplicate (a market that is not a duplicate maps to itself)."""
    return REDUNDANT_WRAPPERS.get(name, name)


def _canon_partition(names) -> tuple:
    """A complete set of markets (sums to 1) with any removed duplicate swapped for its kept equivalent."""
    return tuple(dict.fromkeys(canonical_market(n) for n in names))


DOUBLE_CHANCE_BTTS_MARKETS = [
    "1X & BTTS Yes", "12 & BTTS Yes", "12 & BTTS No", "X2 & BTTS Yes",
]
DOUBLE_CHANCE_TOTAL_GOALS_MARKETS = [
    "1X & Over 2.5", "1X & Under 2.5", "12 & Over 2.5", "12 & Under 2.5", "X2 & Over 2.5", "X2 & Under 2.5",
]
MATCH_RESULT_TOTAL_GOALS_MARKETS = [
    "Home & Over 2.5", "Home & Under 2.5", "Draw & Over 2.5", "Away & Over 2.5", "Away & Under 2.5",
]
TOTAL_GOALS_BTTS_MARKETS = [
    "Over 2.5 & BTTS Yes", "Over 2.5 & BTTS No", "Under 2.5 & BTTS Yes", "Under 2.5 & BTTS No",
]
GOAL_RANGE_MARKETS = ["Goal Range 0-1", "Goal Range 2-3", "Goal Range 4-5", "Goal Range 6+"]

# Total-goals combos on the 1.5 and 3.5 lines (the 2.5 line keeps its original lists above). Same three
# families as 2.5 - double chance + total goals, 1X2 + total goals, total goals + BTTS - generated from ONE
# spec so the grid, the simulation and the real-result check can never drift apart.
# "Under 1.5 & BTTS Yes" is impossible (BTTS needs 2+ goals, Under 1.5 allows at most 1), so it is not offered.
EXTRA_COMBO_LINES = [1.5, 3.5]


def _extra_combo_specs() -> list:
    """[(market name, (condition keys...))]; a key starting with '~' means 'not'."""
    specs = []
    for line in EXTRA_COMBO_LINES:
        over, under = f"over{line}", f"~over{line}"
        for label, key in (("1X", "dc_1x"), ("12", "dc_12"), ("X2", "dc_x2")):
            specs += [(f"{label} & Over {line}", (key, over)), (f"{label} & Under {line}", (key, under))]
        for label, key in (("Home", "home"), ("Draw", "draw"), ("Away", "away")):
            specs += [(f"{label} & Over {line}", (key, over)), (f"{label} & Under {line}", (key, under))]
        specs += [(f"Over {line} & BTTS Yes", (over, "btts")), (f"Over {line} & BTTS No", (over, "~btts")),
                  (f"Under {line} & BTTS No", (under, "~btts"))]
        if int(line) >= 2:
            specs.append((f"Under {line} & BTTS Yes", (under, "btts")))
    return specs


EXTRA_LINE_COMBO_SPECS = [spec for spec in _extra_combo_specs() if spec[0] not in REDUNDANT_WRAPPERS]
EXTRA_LINE_COMBO_MARKETS = [name for name, _ in EXTRA_LINE_COMBO_SPECS]


def _extra_line_combo_values(conds: dict, total, reduce) -> dict:
    """Evaluate every extra-line combo. `conds` maps home/draw/away/dc_1x/dc_12/dc_x2/btts to boolean masks (or
    plain bools); `total` is total goals; `reduce` turns a mask into a probability (matrix sum / simulation mean)
    or a yes/no (actual result)."""
    masks = dict(conds)
    for line in EXTRA_COMBO_LINES:
        masks[f"over{line}"] = total > line
    out = {}
    for name, keys in EXTRA_LINE_COMBO_SPECS:
        parts = [np.logical_not(masks[k[1:]]) if k.startswith("~") else masks[k] for k in keys]
        out[name] = reduce(np.logical_and(parts[0], parts[1]))
    return out

# Goal-structure markets that replaced the duplicates above. Each is a plain function of the final score, so one
# definition drives the grid, the simulation and the real-result check (they cannot drift apart).
GOAL_STRUCTURE_MARKETS = [
    "Total Goals Odd", "Total Goals Even", "Exactly 2 Goals", "Exactly 3 Goals", "Exactly 4 Goals",
    "Home Team Over 1.5", "Home Team Under 1.5", "Away Team Over 1.5", "Away Team Under 1.5",
    "Home Wins by Exactly 1", "Away Wins by Exactly 1",
]


def _goal_structure_values(home, away, reduce) -> dict:
    """`home`/`away` are goals (scalars, simulation arrays or grids); `reduce` turns a mask into a probability / yes-no."""
    total, margin = home + away, home - away
    return {
        "Total Goals Odd": reduce(total % 2 == 1), "Total Goals Even": reduce(total % 2 == 0),
        "Exactly 2 Goals": reduce(total == 2), "Exactly 3 Goals": reduce(total == 3), "Exactly 4 Goals": reduce(total == 4),
        "Home Team Over 1.5": reduce(home >= 2), "Home Team Under 1.5": reduce(home <= 1),
        "Away Team Over 1.5": reduce(away >= 2), "Away Team Under 1.5": reduce(away <= 1),
        "Home Wins by Exactly 1": reduce(margin == 1), "Away Wins by Exactly 1": reduce(margin == -1),
    }


# 1X2 + BTTS: 3 results x 2 BTTS states = a genuine mutually-exclusive-and-exhaustive partition (sums to 1).
MATCH_RESULT_BTTS_MARKETS = [
    "Home & BTTS Yes", "Draw & BTTS Yes", "Draw & BTTS No", "Away & BTTS Yes",
]


def actual_market_outcomes(
    home_goals: float, away_goals: float, home_corners: "float | None" = None, away_corners: "float | None" = None,
) -> dict:
    """The real yes/no result for each market, given a match's actual
    final score - thresholds mirror market_probs_from_matrix exactly, so
    "did the model's predicted side for THIS market actually happen"
    comparisons are apples-to-apples with what the live sheet shows.

    Draw No Bet markets return None (not True/False) when the match was
    actually a draw - a Draw No Bet stake is VOID on a draw, not lost,
    so it must never be silently scored as a wrong prediction. Every
    consumer of this dict (walk_forward_backtest, per_market_backtest_
    accuracy, the calibration fitters) treats None as "exclude this row
    from this market's sample" rather than "incorrect".

    home_corners/away_corners are optional - when omitted (the default),
    every CORNERS_MARKETS key comes back None too, same "exclude, don't
    guess" treatment as Draw No Bet on a draw; only pass real corner
    counts when this match's data actually includes them."""
    total = home_goals + away_goals
    home_win = home_goals > away_goals
    draw = home_goals == away_goals
    away_win = home_goals < away_goals
    margin = home_goals - away_goals
    result = {
        "Home Win": home_win, "Draw": draw, "Away Win": away_win,
        "Double Chance 1X": home_win or draw,
        "Double Chance 12": home_win or away_win,
        "Double Chance X2": draw or away_win,
        "Draw No Bet Home": (home_win if not draw else None),
        "Draw No Bet Away": (away_win if not draw else None),
        "Over 1.5 Goals": total > 1.5, "Under 1.5 Goals": total < 1.5,
        "Over 2.5 Goals": total > 2.5, "Under 2.5 Goals": total < 2.5,
        "Over 3.5 Goals": total > 3.5, "Under 3.5 Goals": total < 3.5,
        "BTTS - Yes": home_goals >= 1 and away_goals >= 1,
        "BTTS - No": not (home_goals >= 1 and away_goals >= 1),
        "Home Clean Sheet": away_goals == 0, "Away Clean Sheet": home_goals == 0,
        "Home Win to Nil": home_win and away_goals == 0,
        "Away Win to Nil": away_win and home_goals == 0,
        "Asian Handicap Home -1.5": margin > 1.5, "Asian Handicap Away +1.5": not (margin > 1.5),
        "Asian Handicap Home +1.5": not ((-margin) > 1.5), "Asian Handicap Away -1.5": (-margin) > 1.5,
    }

    btts = home_goals >= 1 and away_goals >= 1
    over_25 = total > 2.5
    dc_1x, dc_12, dc_x2 = (home_win or draw), (home_win or away_win), (draw or away_win)
    result.update({
        "1X & BTTS Yes": dc_1x and btts,
        "12 & BTTS Yes": dc_12 and btts, "12 & BTTS No": dc_12 and not btts,
        "X2 & BTTS Yes": dc_x2 and btts,
        "1X & Over 2.5": dc_1x and over_25, "1X & Under 2.5": dc_1x and not over_25,
        "12 & Over 2.5": dc_12 and over_25, "12 & Under 2.5": dc_12 and not over_25,
        "X2 & Over 2.5": dc_x2 and over_25, "X2 & Under 2.5": dc_x2 and not over_25,
        "Home & Over 2.5": home_win and over_25, "Home & Under 2.5": home_win and not over_25,
        "Draw & Over 2.5": draw and over_25,
        "Away & Over 2.5": away_win and over_25, "Away & Under 2.5": away_win and not over_25,
        "Over 2.5 & BTTS Yes": over_25 and btts, "Over 2.5 & BTTS No": over_25 and not btts,
        "Under 2.5 & BTTS Yes": (not over_25) and btts, "Under 2.5 & BTTS No": (not over_25) and not btts,
        "Home & BTTS Yes": home_win and btts,
        "Draw & BTTS Yes": draw and btts, "Draw & BTTS No": draw and not btts,
        "Away & BTTS Yes": away_win and btts,
        "Goal Range 0-1": 0 <= total <= 1, "Goal Range 2-3": 2 <= total <= 3,
        "Goal Range 4-5": 4 <= total <= 5, "Goal Range 6+": total >= 6,
    })
    result.update(_extra_line_combo_values(
        {"home": home_win, "draw": draw, "away": away_win, "dc_1x": dc_1x, "dc_12": dc_12, "dc_x2": dc_x2, "btts": btts},
        total, lambda m: bool(m)))
    result.update(_goal_structure_values(home_goals, away_goals, lambda m: bool(m)))

    named_hit = False
    for name, cells in MULTI_SCORE_GROUPS:
        hit = any(home_goals == h and away_goals == a for h, a in cells)
        result[name] = hit
        named_hit = named_hit or hit
    result[MULTI_SCORE_OTHER] = not named_hit

    if home_corners is None or away_corners is None or pd.isna(home_corners) or pd.isna(away_corners):
        for market in CORNERS_MARKETS:
            result[market] = None
    else:
        total_corners = home_corners + away_corners
        for line in CORNERS_LINES:
            result[f"Corners Over {line}"] = total_corners > line
            result[f"Corners Under {line}"] = total_corners < line
        result["Most Corners Home"] = home_corners > away_corners
        result["Most Corners Away"] = home_corners < away_corners
        result["Most Corners Equal"] = home_corners == away_corners

    return result


def walk_forward_backtest(
    settled_df: pd.DataFrame, half_life_days: float = FROZEN_HALF_LIFE_DAYS,
    weights: tuple = TERRITORY_WEIGHTS_DEFAULT, include_all_markets: bool = False,
    full_model: bool = True, reliability_refresh_every: int = 25, mc_iterations: int = 4000,
    engine_weights: "dict | None" = None, duel_strength: float = 1.0, mismatch_strength: float = 1.0,
    max_evaluated: "int | None" = None,
) -> pd.DataFrame:
    """max_evaluated: only the most recent N matches are scored (every one still sees ALL earlier matches as
    history). Bounds the cost on long multi-season files; None scores everything.

    full_model=True (default) backtests the SAME pipeline the live app runs (see
    compute_match_strengths); engine reliabilities are recomputed from history only, every
    `reliability_refresh_every` matches. With include_all_markets=True the Monte Carlo engine is
    ALSO backtested (stored as mcprob__{market}) so it can get its own calibrators. Manual
    situational toggles (injuries, derby, weather...) cannot be backtested and stay off.

    Walks through EVERY settled match in chronological order (not just
    the last 10) and, for each one with enough PRIOR history, predicts it
    using ONLY data from before that match's date - a genuine
    out-of-sample backtest across the whole dataset, no lookahead bias.

    Uses a single fixed half-life for the whole sweep rather than
    re-running the expensive half-life optimizer separately for every
    match (that would nest one already-expensive backtest inside
    another - pass in whichever half-life the live projection is
    currently using so this reflects the same settings). Same idea for
    `weights` - pass in a per-league calibrated territory weighting (see
    optimize_territory_weights()) to backtest THAT weighting choice,
    rather than always assuming equal weights.

    The core Home/Draw/Away probabilities now fit rho and build the real
    Dixon-Coles-adjusted score matrix for every historical match, then
    read Home Win/Draw/Away Win straight off it via
    market_probs_from_matrix() - matching EXACTLY what the live
    prediction path does (build_score_matrix + market_probs_from_matrix),
    not an approximation of it. This replaced an earlier version that
    used a plain rho=0 Poisson shortcut for speed; that shortcut is what
    optimize_half_life() below still uses internally (a much
    higher-multiplicity search - up to 12 half-life candidates times up
    to 150 matches each - where fitting rho at every single step would
    be prohibitively slow for comparatively little gain in THAT specific
    search), so the two are now intentionally NOT the same computation -
    this function mirrors live prediction exactly; the half-life search
    still uses a fast approximation purely to pick a half-life, which is
    a coarser decision than a full probability estimate.

    include_all_markets=False (the default, used everywhere BSS/accuracy/
    the reliability badge/dilution report/half-life search already
    depend on this function) only keeps the 1X2 outcome, still on the
    real rho-adjusted matrix - the matrix is now always built regardless
    of this flag, since Home/Draw/Away needs it anyway. Passing True
    additionally stores every OTHER market's own predicted probability
    and ground-truth outcome (needed for per-market accuracy and
    per-market calibration fitting) - a real extra cost per row (mostly
    the market_probs_from_matrix() call itself and storing 2×len(MARKET_LIST)
    extra columns), so it's opt-in rather than always paid."""
    df = settled_df.dropna(subset=["date", "home_goals", "away_goals"]).sort_values("date").reset_index(drop=True)
    rows = []
    rel_cache = None
    rel_cache_at = -10**9
    ledger = MismatchLedger()  # learns tactical-mismatch multipliers from past matches only
    first_scored = 0 if max_evaluated is None else max(0, len(df) - int(max_evaluated))
    for i in range(first_scored, len(df)):
        row = df.iloc[i]
        history = df.iloc[:i]
        if full_model and len(history) >= MIN_SAMPLE_ROWS * 2 and (rel_cache is None or i - rel_cache_at >= reliability_refresh_every):
            rel_cache = compute_engine_reliabilities(history, weights)  # history only: no look-ahead
            rel_cache_at = i
        styles = {}
        lambdas = _quick_lambda_for_backtest(
            history, row["home_team"], row["away_team"], half_life_days, row["date"], weights,
            full_model=full_model, reliabilities=rel_cache, engine_weights=engine_weights,
            duel_strength=duel_strength, style_out=styles,
        )
        if lambdas is None:
            continue
        lam_home_base, lam_away_base = lambdas
        mm_home = mm_away = 1.0
        if styles:  # same automatic mismatch step the live app applies, from matches BEFORE this one
            raw_h, raw_a = ledger.raw_multipliers(styles["home"], styles["away"])
            mm_home, mm_away = 1.0 + mismatch_strength * (raw_h - 1.0), 1.0 + mismatch_strength * (raw_a - 1.0)
            lambdas = (max(0.05, lam_home_base * mm_home), max(0.05, lam_away_base * mm_away))
        rho = fit_rho(history) if len(history) >= MIN_SAMPLE_ROWS else 0.0
        matrix = build_score_matrix(lambdas[0], lambdas[1], rho)
        corner_lambdas = (
            _quick_corner_lambda_for_backtest(history, row["home_team"], row["away_team"], half_life_days, row["date"])
            if include_all_markets else None
        )
        predicted_probs = market_probs_from_matrix(matrix, corner_lambdas=corner_lambdas)
        mc_probs_bt = None
        if include_all_markets:
            vol_bt = compute_volatility_profile(history).vol_dampener if len(history) >= MIN_SAMPLE_ROWS else 1.0
            hg_bt, ag_bt = monte_carlo_simulate(
                lambdas[0], lambdas[1], volatility_dampener=vol_bt, iterations=mc_iterations,
                rng=np.random.default_rng(10_000 + i),
            )
            mc_probs_bt = market_probs_from_simulation(hg_bt, ag_bt, corner_lambdas=corner_lambdas)
        p_home, p_draw, p_away = predicted_probs["Home Win"], predicted_probs["Draw"], predicted_probs["Away Win"]
        actual = "H" if row["home_goals"] > row["away_goals"] else ("D" if row["home_goals"] == row["away_goals"] else "A")
        predicted_pick = max([("H", p_home), ("D", p_draw), ("A", p_away)], key=lambda kv: kv[1])[0]
        result_row = {
            "date": row["date"], "home_team": row["home_team"], "away_team": row["away_team"],
            "home_goals": row["home_goals"], "away_goals": row["away_goals"],
            "goal_difference": row["home_goals"] - row["away_goals"],
            "p_home": round(p_home, 4), "p_draw": round(p_draw, 4), "p_away": round(p_away, 4),
            "actual": actual, "predicted_pick": predicted_pick,
            "correct": predicted_pick == actual,
            "lam_home_base": lam_home_base, "lam_away_base": lam_away_base,
            "home_style": styles.get("home"), "away_style": styles.get("away"),
            "mismatch_home": mm_home, "mismatch_away": mm_away,
        }
        if styles:
            ledger.add(styles["home"], styles["away"], row["home_goals"], lam_home_base, row["away_goals"], lam_away_base)
        if include_all_markets:
            home_corners_actual = row.get("home_corner_kicks") if "home_corner_kicks" in df.columns else None
            away_corners_actual = row.get("away_corner_kicks") if "away_corner_kicks" in df.columns else None
            actual_outcomes = actual_market_outcomes(row["home_goals"], row["away_goals"], home_corners_actual, away_corners_actual)
            for market in MARKET_LIST:
                actual_val = actual_outcomes[market]
                predicted_val = predicted_probs.get(market)
                mc_val = mc_probs_bt.get(market) if mc_probs_bt is not None else None
                result_row[f"mcprob__{market}"] = round(mc_val, 4) if mc_val is not None else None
                if predicted_val is None:
                    # corner_lambdas wasn't available for this match (thin history, or no
                    # corner_kicks data at all) - this market simply wasn't predicted here,
                    # excluded from both accuracy and calibration for this row rather than
                    # crashing on a missing key or silently treating it as 0% confidence.
                    result_row[f"prob__{market}"] = None
                    result_row[f"actual__{market}"] = None
                    continue
                result_row[f"prob__{market}"] = round(predicted_val, 4)
                # Ground truth for THIS market, independent of any 50% threshold - this is
                # what per-market CALIBRATION fitting needs (see fit_all_market_calibrators),
                # since a market's calibration should be checked against "did this actually
                # happen", not "did the model's own >=50% pick happen to be right".
                result_row[f"actual__{market}"] = actual_val
                if actual_val is None:
                    # Not applicable for this match (e.g. Draw No Bet when the match itself
                    # WAS a draw - void, not a loss) - excluded from accuracy/calibration,
                    # never silently counted as wrong.
                    result_row[f"correct__{market}"] = None
                else:
                    predicted_side = predicted_probs[market] >= 0.5
                    result_row[f"correct__{market}"] = predicted_side == actual_val
        rows.append(result_row)
    return pd.DataFrame(rows)


CALL_LEVEL_CAP = 0.50      # a market is never asked for more than 50% to count as 'backed' (the usual call)
CALL_LEVEL_QUANTILE = 0.80  # ...but a market the model rarely rates that high is judged against its OWN top 20%


def market_call_levels(backtest_df: pd.DataFrame) -> dict:
    """The probability at which the model counts as genuinely BACKING each market, set per market from how high the
    model's own probability for it actually gets: min(50%, the level its top 20% most confident matches reach).
    So a market the model is often confident about (Home Win,
    Over 2.5) keeps the usual 50%, while a rare one (an exact score, 'Away & Over 3.5') is backed at the level it
    reaches in its own most confident matches - e.g. about 11% for an exact score - instead of a 50% it can never
    reach. Every market therefore has calls to judge, and 'lift on calls' (hit rate / how often it happens) says
    whether those calls beat the market's normal frequency. Descriptive: derived from the backtest itself, it
    labels how to read each market; it is not a betting threshold."""
    levels = {}
    for market in MARKET_LIST:
        prob_col, actual_col = f"prob__{market}", f"actual__{market}"
        level = CALL_LEVEL_CAP
        if backtest_df is not None and prob_col in backtest_df.columns and actual_col in backtest_df.columns:
            probs = pd.to_numeric(backtest_df[prob_col], errors="coerce").dropna()
            actual = pd.to_numeric(backtest_df[actual_col].astype(float), errors="coerce").dropna()
            if len(probs) >= 20 and len(actual):
                level = float(min(CALL_LEVEL_CAP, probs.quantile(CALL_LEVEL_QUANTILE)))
        levels[market] = level
    return levels


def per_market_backtest_accuracy(backtest_df: pd.DataFrame, min_confidence_pct: float = 0.0) -> dict:
    """Accuracy % AND sample size (N) for each market in MARKET_LIST
    individually, aggregated across the same walk-forward sweep.
    Requires backtest_df to have been built with
    walk_forward_backtest(..., include_all_markets=True) - returns an
    empty dict otherwise rather than guessing or crashing.

    min_confidence_pct=0 (the default) matches the ORIGINAL behaviour:
    every settled match counts, regardless of how confident the model
    was in that specific market - so N is the same for every market
    (the full backtest length) unless a higher floor is set here.
    min_confidence_pct=X is a per-market confidence level, where 50 means "each market's own normal call level"
    (see market_call_levels). Between 0 and 50 the cutoff is call_level_m * X / 50 (so a rare market such as an
    exact score is not asked to reach 50%, which it essentially never does). Between 50 and 100 the cutoff climbs
    through the market's own most confident matches: 50 = its call level (about its top 20%), 75 = about its top 11%,
    100 = about its top 2%.
    N therefore differs market to market once X is raised above 0, which is what makes a high accuracy % on a
    market like Draw or Win-to-Nil auditable rather than a number with no visible N behind it.
    Whatever the slider says, happened_pct (the baseline) is ALWAYS the market's frequency over the whole backtest,
    n_calls / call_hit_rate_pct describe the matches at or above max(call level, slider cutoff), and call_level_pct is
    that cutoff - so lift is a real comparison at every slider setting (it rises with the slider when the model's
    confidence means something, and stays near 1.0 when it does not).

    Returns {market: {"accuracy_pct", "n", "happened_pct", "n_calls", "call_hit_rate_pct", "call_level_pct",
    "lift_on_calls"}} -
    callers that only want the accuracy value can do `result[market]["accuracy_pct"]`."""
    if backtest_df is None or backtest_df.empty:
        return {}
    result = {}
    levels = market_call_levels(backtest_df)
    for market in MARKET_LIST:
        correct_col = f"correct__{market}"
        prob_col = f"prob__{market}"
        if correct_col not in backtest_df.columns:
            continue
        applicable = backtest_df[backtest_df[correct_col].notna()]  # drop not-applicable rows (e.g. Draw No Bet on a drawn match) rather than counting them as wrong
        subset = applicable
        level = levels[market]
        if min_confidence_pct <= 0:
            slider_cutoff = 0.0
        elif min_confidence_pct <= 50 or prob_col not in applicable.columns:
            slider_cutoff = min(0.95, level * min_confidence_pct / 50.0)       # 0-50: scale the market's own call level down
        else:
            # 50-100: climb through the model's OWN most confident matches for this market - 50 = its call level (about its
            # top 20%), 75 = about its top 11%, 100 = about its top 2% - so the slider never asks for a probability the model
            # cannot reach and the rows do not simply run out.
            q = 0.80 + 0.18 * (min(min_confidence_pct, 100) - 50) / 50.0
            slider_cutoff = max(level, float(pd.to_numeric(applicable[prob_col], errors="coerce").quantile(q)))
        if slider_cutoff > 0 and prob_col in subset.columns:
            subset = subset[subset[prob_col] >= slider_cutoff]
        call_cutoff = max(level, slider_cutoff)  # a "call" is at least the market's own call level, and at least what the slider asks for
        n = int(len(subset))
        accuracy = round(subset[correct_col].astype(bool).mean() * 100, 1) if n > 0 else float("nan")
        # Context that makes the accuracy % auditable. "Accuracy" counts a market as right whenever the model's
        # >=50% call matches the outcome, so a market the model never backs (prob always < 50%) still scores
        # highly just because it rarely happens. These three numbers expose that:
        #   happened_pct       how often the outcome occurred over ALL matches (the full backtest, never the
        #                      confidence-gated subset - otherwise raising the slider to 50 would make every call's
        #                      hit rate equal its own baseline and force every lift to exactly 1.00)
        #   n_calls            how many times the model backed it (probability >= max(call level, slider cutoff))
        #   call_hit_rate_pct  how often it happened when the model backed it (NaN when it never did)
        happened, n_calls, call_hit = float("nan"), 0, float("nan")
        actual_col = f"actual__{market}"
        if actual_col in applicable.columns:
            base_actual = pd.to_numeric(applicable[actual_col].astype(float), errors="coerce").dropna()
            if len(base_actual):
                happened = round(float(base_actual.mean()) * 100, 1)
        if n > 0 and actual_col in subset.columns and prob_col in subset.columns:
            actual = pd.to_numeric(subset[actual_col].astype(float), errors="coerce")
            known = actual.notna()
            if known.any():
                called = known & (subset[prob_col] >= call_cutoff)
                n_calls = int(called.sum())
                call_hit = round(float(actual[called].mean()) * 100, 1) if n_calls else float("nan")
        lift = round(call_hit / happened, 2) if n_calls and np.isfinite(happened) and happened > 0 else float("nan")
        result[market] = {"accuracy_pct": accuracy, "n": n, "happened_pct": happened,
                          "n_calls": n_calls, "call_hit_rate_pct": call_hit,
                          "call_level_pct": round(call_cutoff * 100, 1), "lift_on_calls": lift}
    return result


def _actual_vector(actual: str) -> tuple[int, int, int]:
    return (1, 0, 0) if actual == "H" else ((0, 1, 0) if actual == "D" else (0, 0, 1))


def multiclass_brier_score(backtest_df: pd.DataFrame) -> float:
    """Standard 3-class Brier score: mean squared error between the
    predicted probability vector and the one-hot actual outcome, summed
    across the 3 classes. 0 is a perfect forecaster, 2 is the worst
    possible (fully confident and always wrong)."""
    if backtest_df.empty:
        return float("nan")
    terms = []
    for _, r in backtest_df.iterrows():
        a0, a1, a2 = _actual_vector(r["actual"])
        terms.append((r["p_home"] - a0) ** 2 + (r["p_draw"] - a1) ** 2 + (r["p_away"] - a2) ** 2)
    return float(np.mean(terms))


def climatology_brier_score(backtest_df: pd.DataFrame) -> float:
    """The 'no-skill' reference forecast Brier Skill Score is measured
    against: always predicting the dataset's OVERALL historical
    home/draw/away frequency for every match, regardless of who's
    playing."""
    if backtest_df.empty:
        return float("nan")
    freq_home = float((backtest_df["actual"] == "H").mean())
    freq_draw = float((backtest_df["actual"] == "D").mean())
    freq_away = float((backtest_df["actual"] == "A").mean())
    terms = []
    for _, r in backtest_df.iterrows():
        a0, a1, a2 = _actual_vector(r["actual"])
        terms.append((freq_home - a0) ** 2 + (freq_draw - a1) ** 2 + (freq_away - a2) ** 2)
    return float(np.mean(terms))


def brier_skill_score(backtest_df: pd.DataFrame) -> float:
    """BSS = 1 - (model Brier / climatology Brier). Positive means the
    model beats blindly guessing the league's overall historical outcome
    split; 0 means no better than that naive baseline; negative means
    WORSE than just guessing the league average."""
    model_bs = multiclass_brier_score(backtest_df)
    ref_bs = climatology_brier_score(backtest_df)
    if not ref_bs or math.isnan(ref_bs) or ref_bs == 0:
        return float("nan")
    return 1 - (model_bs / ref_bs)


# Thresholds are deliberately conservative: football is genuinely
# high-variance at the match level, so even a real, usable edge tends to
# show up as a modest positive BSS rather than a dramatic one - this
# isn't a "grade the model out of 100" scale, it's a rough traffic light
# for "is there evidence this model beats guessing this league's own
# historical average, given what's been backtested so far."
def league_reliability_badge(bss: float) -> tuple[str, str]:
    """Returns (emoji, label) for a league's overall Brier Skill Score.
    NaN/None (not enough backtest data yet) gets its own neutral badge -
    it is NOT the same thing as a negative BSS and shouldn't look like a
    warning."""
    if bss is None or (isinstance(bss, float) and math.isnan(bss)):
        return "⚪", "Insufficient Backtest Data"
    if bss > 0.02:
        return "🟢", "Reliable"
    if bss >= -0.02:
        return "🟡", "Marginal"
    return "🔴", "Unreliable"


# ---------------------------------------------------------------------------
# Continuous league reliability score: replaces a hand-curated "which
# leagues are good/bad for territorial metrics" list with a score derived
# directly from THIS league's own data. Combines three independently-
# computable signals - see compute_league_reliability_score() docstring
# for exactly what each one means and, importantly, what this does NOT
# capture (set-piece share, direct-play index, referee tendencies, and
# fixture congestion aren't in this dataset, so they aren't in the score;
# extending it later if that data becomes available is straightforward).
# ---------------------------------------------------------------------------

def compute_dispersion_index(settled_df: pd.DataFrame) -> float:
    """Ratio of actual combined-goals variance to what a true Poisson
    process with the same mean would produce (variance == mean, for a
    real Poisson variable). 1.0 = perfectly Poisson-behaved; above 1.0
    means this league's scorelines are MORE spread out than Poisson
    predicts (blowouts and 0-0s both more common than a tidy Poisson
    model expects - a "chaotic" league); below 1.0 means tighter,
    more predictable scorelines than Poisson expects.

    This is NOT the same thing as rho (fit_rho below), which corrects
    specifically for LOW-SCORE correlation (0-0/1-0/0-1/1-1 being more
    common than independent Poisson implies) - dispersion_index looks at
    overall scoreline variance, a different and complementary signal."""
    goals = pd.concat([settled_df["home_goals"], settled_df["away_goals"]]).dropna().to_numpy(dtype=float)
    if len(goals) < MIN_SAMPLE_ROWS or goals.mean() <= 0:
        return 1.0
    return float(goals.var(ddof=1) / goals.mean())


def xg_proxy_correlation(settled_df: pd.DataFrame, weights: tuple = TERRITORY_WEIGHTS_DEFAULT) -> float:
    """Pearson correlation between a simple match-level territorial
    differential (big chances / shots on target / box touches, home
    minus away, each standardised within this league so the three raw
    metrics are on a comparable scale before blending with `weights`)
    and the ACTUAL goal difference. This is a direct, model-independent
    check of "does dominating these territorial metrics actually
    translate into goals in THIS league" - the exact question this
    engine's whole design has been chasing since the very first
    La-Liga-vs-Championship discussion, now computed straight from the
    data instead of argued from football-culture priors.

    Returns NaN (not 0) when there isn't enough data or no variance to
    correlate against - NaN reads as "unknown", 0 would wrongly read as
    "measured and found to be zero relationship"."""
    needed = ["home_big_chances", "away_big_chances", "home_shots_on_target",
              "away_shots_on_target", "home_box_touches", "away_box_touches",
              "home_goals", "away_goals"]
    present = [c for c in needed if c in settled_df.columns]
    df = settled_df.dropna(subset=present).copy()
    if len(df) < MIN_SAMPLE_ROWS or len(present) < len(needed):
        return float("nan")

    def z(series):
        s = series.std(ddof=0)
        return (series - series.mean()) / s if s > 0 else series * 0.0

    bc_diff = z(df["home_big_chances"] - df["away_big_chances"])
    sot_diff = z(df["home_shots_on_target"] - df["away_shots_on_target"])
    box_diff = z(df["home_box_touches"] - df["away_box_touches"])
    territorial_diff = weights[0] * bc_diff + weights[1] * sot_diff + weights[2] * box_diff

    goal_diff = df["home_goals"] - df["away_goals"]
    if territorial_diff.std(ddof=0) == 0 or goal_diff.std(ddof=0) == 0:
        return float("nan")
    return float(np.corrcoef(territorial_diff, goal_diff)[0, 1])


@dataclass
class LeagueReliabilityScore:
    bss: float
    dispersion_index: float
    xg_proxy_correlation: float
    composite_score: float  # 0-1, higher = territorial/xG-proxy signal is more trustworthy for this league
    n_matches: int


def compute_league_reliability_score(
    settled_df: pd.DataFrame, backtest_df: pd.DataFrame = None, weights: tuple = TERRITORY_WEIGHTS_DEFAULT,
) -> LeagueReliabilityScore:
    """ONE continuous 0-1 score for how much this league's territorial/
    xG-proxy metrics can be trusted, built entirely from this league's
    own data:

      - BSS (if a walk-forward backtest is supplied) - the most direct
        empirical evidence: does the model actually predict results
        well here, full stop.
      - xg_proxy_correlation - does the territorial differential line
        up with the real goal difference in this league.
      - dispersion_index - how close this league's goal-scoring
        variance sits to a well-behaved Poisson process; heavily
        over- or under-dispersed leagues are the harder-to-model ones.

    Each component is normalised to roughly [0, 1] before averaging, so
    no single component with a wider natural range can silently dominate
    the composite. SCOPE NOTE: this one score is built from results and
    territorial/xG-proxy numbers (goals, big chances, SOT, box touches)
    only. It is not the whole model - the other effects are handled
    elsewhere: set-piece share by the set-piece and xG set-play engines
    (each with its own reliability score), direct/transition style by the
    transition engine, fixture congestion by compute_fixture_congestion()
    (calculated for every fixture from the league's own schedule), and
    referee tendency by the Referee Strictness setting."""
    n = len(settled_df)
    bss = brier_skill_score(backtest_df) if backtest_df is not None and not backtest_df.empty else float("nan")
    dispersion = compute_dispersion_index(settled_df)
    xg_corr = xg_proxy_correlation(settled_df, weights)

    # 0 BSS -> ~0.33 normalised, 0.10 BSS (a genuinely strong league) -> 1.0.
    # Missing BSS scores as a neutral 0.5 rather than dragging the composite down for a reason unrelated to reliability.
    bss_norm = float(np.clip((bss + 0.05) / 0.15, 0, 1)) if not math.isnan(bss) else 0.5
    # |r|=0.1 (weak/no real relationship) -> 0, |r|=0.5 (strong relationship) -> 1.0
    xg_corr_norm = float(np.clip((abs(xg_corr) - 0.1) / 0.4, 0, 1)) if not math.isnan(xg_corr) else 0.5
    # dispersion == 1.0 (perfectly Poisson) -> 1.0, degrading as it drifts either direction
    dispersion_norm = float(np.clip(1.0 - abs(dispersion - 1.0), 0, 1))

    composite = float(np.mean([bss_norm, xg_corr_norm, dispersion_norm]))

    return LeagueReliabilityScore(
        bss=round(bss, 4) if not math.isnan(bss) else float("nan"),
        dispersion_index=round(dispersion, 3),
        xg_proxy_correlation=round(xg_corr, 3) if not math.isnan(xg_corr) else float("nan"),
        composite_score=round(composite, 3),
        n_matches=n,
    )


# ---------------------------------------------------------------------------
# Odds/probability calibration: "2.00 odds means an implied 50% chance -
# does the team the model backs at that price actually win about half the
# time?" Uses the walk-forward backtest's own historical predictions, so
# it needs no real bookmaker odds data at all - it's checking whether the
# MODEL's own probabilities are trustworthy, not comparing against a
# specific bookmaker's pricing.
# ---------------------------------------------------------------------------

ODDS_BAND_EDGES = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, float("inf")]
ODDS_BAND_LABELS = ["1.01-1.50", "1.51-2.00", "2.01-2.50", "2.51-3.00", "3.01-4.00", "4.01-6.00", "6.01+"]


def implied_probability(odds: float) -> float:
    """The plain reverse of fair_odds: 2.00 odds implies a 50% chance,
    3.00 implies 33.3%, etc. - exactly the relationship the odds
    calibration table below checks against real outcomes."""
    if odds <= 0:
        return 0.0
    return 1.0 / odds


def odds_calibration_table(backtest_df: pd.DataFrame) -> pd.DataFrame:
    """For every historical walk-forward prediction, takes the model's
    own probability for whichever side it actually picked, converts that
    to an implied odds price (1 / probability), buckets those into
    standard odds bands, and checks: within each band, how often did the
    picked side actually win? A well-calibrated model should show an
    actual win rate close to the band's implied probability - e.g.
    picks priced around 2.00 (50% implied) should win roughly half the
    time, not 30% or 70% of the time."""
    if backtest_df.empty:
        return pd.DataFrame(columns=[
            "Odds Band", "Predictions", "Avg Model-Implied Odds",
            "Model-Implied Win Rate %", "Actual Win Rate %", "Calibration Gap (pts)",
        ])

    pick_prob = backtest_df[["p_home", "p_draw", "p_away"]].max(axis=1)
    implied_odds = pick_prob.apply(lambda p: (1.0 / p) if p > 0 else float("inf"))
    bands = pd.cut(implied_odds, bins=ODDS_BAND_EDGES, labels=ODDS_BAND_LABELS, right=True, include_lowest=True)

    working = pd.DataFrame({
        "band": bands, "pick_prob": pick_prob, "implied_odds": implied_odds, "correct": backtest_df["correct"],
    })

    rows = []
    for label in ODDS_BAND_LABELS:
        bucket = working[working["band"] == label]
        if bucket.empty:
            continue
        avg_implied_odds = float(bucket["implied_odds"].replace(np.inf, np.nan).mean())
        model_implied_win_rate = float(bucket["pick_prob"].mean() * 100)
        actual_win_rate = float(bucket["correct"].mean() * 100)
        rows.append({
            "Odds Band": label,
            "Predictions": int(len(bucket)),
            "Avg Model-Implied Odds": round(avg_implied_odds, 2) if not math.isnan(avg_implied_odds) else None,
            "Model-Implied Win Rate %": round(model_implied_win_rate, 1),
            "Actual Win Rate %": round(actual_win_rate, 1),
            "Calibration Gap (pts)": round(actual_win_rate - model_implied_win_rate, 1),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Section 7.5: Probability calibration (Platt scaling / isotonic regression)
#
# odds_calibration_table() above can reveal the raw model probabilities are
# systematically off in one of two different shapes:
#   - a smooth, ONE-DIRECTIONAL drift (e.g. increasingly overconfident as
#     odds lengthen) - Platt scaling fits this well with just 2 parameters.
#   - a NON-MONOTONIC, zig-zagging gap (well-calibrated in some bands, off
#     in others, no consistent direction) - isotonic regression fits this
#     better, since it makes no assumption about the correction's shape
#     beyond "a higher raw probability must still map to a higher (or
#     equal) corrected probability".
#
# Rather than a human eyeballing the calibration table to guess which shape
# a league has, select_calibrator() fits BOTH methods on the same historical
# rows and keeps whichever one actually scores better on genuinely held-out
# folds (k-fold cross-validation - never scored on the same rows it was fit
# on). "No correction at all" is included as a third candidate, so a league
# whose calibration gaps are just noise doesn't get a correction fit to
# that noise - see the Bundesliga vs. La Liga gap shapes discussed when this
# was designed: one drifted in a single direction, the other zig-zagged
# with no gap ever really clearing its own sampling noise.
#
# Fit PER OUTCOME CLASS (Home/Draw/Away each get their own independently-
# selected method), not on whichever side the model happened to pick - this
# is what lets "is a 35%-probability draw actually ~35%" be checked
# directly against every match where 35% draw probability occurred, not
# just the rare matches where draw happened to be the model's top pick
# (recall predicted_pick is H or A in the overwhelming majority of
# matches - draw is almost never the argmax - so a pick-only calibration
# would have almost no draw data to learn from).
#
# Isotonic regression is implemented directly via the Pool-Adjacent-
# Violators Algorithm (PAVA) rather than pulling in scikit-learn as a new
# dependency just for this one function - same "no shortcuts" spirit as
# market_probs_from_matrix() above, which sums the real Dixon-Coles grid
# cells instead of approximating.
# ---------------------------------------------------------------------------

CALIBRATION_MIN_SAMPLES = 40   # below this, select_calibrator() returns "none" outright - too few rows for either method to be trustworthy, same safety-rail spirit as MIN_SAMPLE_ROWS elsewhere in this file
CALIBRATION_CV_FOLDS = 5
CALIBRATION_EPS = 1e-6         # keeps logit()/log-loss finite at p=0 or p=1


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), CALIBRATION_EPS, 1 - CALIBRATION_EPS)
    return np.log(p / (1 - p))


def _sigmoid(z) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def _binary_log_loss(p, y) -> float:
    p = np.clip(np.asarray(p, dtype=float), CALIBRATION_EPS, 1 - CALIBRATION_EPS)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


@dataclass
class PlattCalibrator:
    """sigmoid(a * logit(p) + b) - the standard Platt scaling
    formulation applied to logit(p) rather than a raw classifier score,
    since our 'raw score' already IS a probability. a=1, b=0 is a
    no-op (returns the input unchanged)."""
    a: float
    b: float

    def predict(self, probs) -> np.ndarray:
        return _sigmoid(self.a * _logit(probs) + self.b)


def fit_platt(probs, outcomes) -> PlattCalibrator:
    """Fits (a, b) by maximum likelihood - minimising binary log-loss -
    via Nelder-Mead (dependency-free: no gradient needed, and this is a
    tiny 2-parameter problem where a derivative-free optimiser is more
    than fast enough)."""
    x = _logit(probs)
    y = np.asarray(outcomes, dtype=float)

    def loss(params):
        a, b = params
        return _binary_log_loss(_sigmoid(a * x + b), y)

    result = _minimize(loss, x0=np.array([1.0, 0.0]), method="Nelder-Mead")
    a, b = result.x
    return PlattCalibrator(a=float(a), b=float(b))


@dataclass
class IsotonicCalibrator:
    """A monotonically non-decreasing step function fit by PAVA, stored
    as (x, fitted_y) breakpoints and evaluated on new inputs via linear
    interpolation between them (queries outside the training range clip
    to the nearest endpoint - numpy's default np.interp behaviour, and
    the standard way isotonic regression handles out-of-range inputs)."""
    thresholds: np.ndarray
    values: np.ndarray

    def predict(self, probs) -> np.ndarray:
        return np.interp(np.asarray(probs, dtype=float), self.thresholds, self.values)


def fit_isotonic(probs, outcomes) -> IsotonicCalibrator:
    """Pool-Adjacent-Violators Algorithm: sort by raw probability, then
    repeatedly merge any adjacent block whose average outcome rate would
    otherwise be LOWER than the block before it, until the whole
    sequence is non-decreasing. Every original point keeps the fitted
    value of whichever final block it belongs to."""
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    order = np.argsort(probs, kind="mergesort")
    x_sorted = probs[order]
    y_sorted = outcomes[order]

    block_sum = list(y_sorted)
    block_count = [1] * len(y_sorted)
    block_start = list(range(len(y_sorted)))
    block_end = list(range(len(y_sorted)))

    i = 0
    while i < len(block_sum) - 1:
        avg_i = block_sum[i] / block_count[i]
        avg_next = block_sum[i + 1] / block_count[i + 1]
        if avg_i > avg_next:
            block_sum[i] += block_sum[i + 1]
            block_count[i] += block_count[i + 1]
            block_end[i] = block_end[i + 1]
            del block_sum[i + 1]
            del block_count[i + 1]
            del block_start[i + 1]
            del block_end[i + 1]
            if i > 0:
                i -= 1  # merging may have created a new violation with the PREVIOUS block too
        else:
            i += 1

    fitted = np.empty(len(x_sorted))
    for s, c, start, end in zip(block_sum, block_count, block_start, block_end):
        fitted[start:end + 1] = s / c

    return IsotonicCalibrator(thresholds=x_sorted, values=fitted)


@dataclass
class CalibrationResult:
    """method is 'platt', 'isotonic', or 'none' - whichever won the
    cross-validated comparison. cv_log_loss holds ALL candidates' scores
    (not just the winner's) so the choice is auditable rather than a
    black box - see calibration_summary_table() below for a display-
    ready version of this."""
    method: str
    calibrator: object | None
    cv_log_loss: dict
    n_samples: int


def _kfold_indices(n: int, k: int, seed: int = 42) -> list[np.ndarray]:
    rng = np.random.RandomState(seed)
    return np.array_split(rng.permutation(n), k)


def select_calibrator(
    probs, outcomes, n_folds: int = CALIBRATION_CV_FOLDS, seed: int = 42,
) -> CalibrationResult:
    """Fits Platt scaling AND isotonic regression on the same (raw
    probability, actual 0/1 outcome) pairs, scores BOTH via k-fold
    cross-validation - never evaluated on the rows used to fit it - and
    returns whichever wins on held-out log-loss, alongside the
    uncorrected baseline ('none'), so a league whose gaps are just noise
    isn't force-fit a correction that would only add variance. The
    winning method is then refit on the FULL dataset (not just the
    folds) for the calibrator actually returned - cross-validation is
    only used to CHOOSE the method, not to build the deployed one, so
    the deployed calibrator gets the benefit of every available row."""
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    n = len(probs)

    if n < CALIBRATION_MIN_SAMPLES:
        return CalibrationResult(method="none", calibrator=None, cv_log_loss={}, n_samples=n)

    folds = _kfold_indices(n, n_folds, seed=seed)
    losses = {"none": [], "platt": [], "isotonic": []}

    for k in range(n_folds):
        test_idx = folds[k]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        if len(train_idx) < 10 or len(test_idx) < 5:
            continue  # a fold this thin would just add noise to the comparison rather than resolve it

        p_train, y_train = probs[train_idx], outcomes[train_idx]
        p_test, y_test = probs[test_idx], outcomes[test_idx]

        losses["none"].append(_binary_log_loss(p_test, y_test))
        losses["platt"].append(_binary_log_loss(fit_platt(p_train, y_train).predict(p_test), y_test))
        losses["isotonic"].append(_binary_log_loss(fit_isotonic(p_train, y_train).predict(p_test), y_test))

    mean_losses = {method: float(np.mean(vals)) for method, vals in losses.items() if vals}
    if not mean_losses:
        return CalibrationResult(method="none", calibrator=None, cv_log_loss={}, n_samples=n)

    best_method = min(mean_losses, key=mean_losses.get)

    if best_method == "platt":
        calibrator = fit_platt(probs, outcomes)
    elif best_method == "isotonic":
        calibrator = fit_isotonic(probs, outcomes)
    else:
        calibrator = None

    return CalibrationResult(method=best_method, calibrator=calibrator, cv_log_loss=mean_losses, n_samples=n)


def build_calibration_training_data(backtest_df: pd.DataFrame, outcome_class: str) -> tuple[np.ndarray, np.ndarray]:
    """outcome_class is 'H', 'D', or 'A'. Returns that class's raw
    walk-forward-backtested probability and a binary 0/1 array for
    whether that class was the actual result, across every backtested
    match - the (probs, outcomes) pair select_calibrator() expects."""
    col = {"H": "p_home", "D": "p_draw", "A": "p_away"}[outcome_class]
    probs = backtest_df[col].to_numpy(dtype=float)
    outcomes = (backtest_df["actual"] == outcome_class).to_numpy(dtype=float)
    return probs, outcomes


def fit_1x2_calibrators(backtest_df: pd.DataFrame) -> dict:
    """{'H': CalibrationResult, 'D': CalibrationResult, 'A': CalibrationResult}
    - an independently-selected calibrator per outcome class, since
    there's no reason the home-win, draw, and away-win probabilities
    would need the same correction shape. Requires walk_forward_backtest()
    output (any include_all_markets setting works - only p_home/p_draw/
    p_away/actual are used here)."""
    return {cls: select_calibrator(*build_calibration_training_data(backtest_df, cls)) for cls in ("H", "D", "A")}


def apply_1x2_calibration(p_home: float, p_draw: float, p_away: float, calibrators: dict) -> tuple[float, float, float]:
    """Applies each class's own selected calibrator (if any) to the raw
    live probability, then renormalises the three corrected numbers so
    they still sum to 1. Independent per-class correction followed by
    renormalisation is the standard way to calibrate a multiclass
    probability vector without needing one joint 3-way model. Classes
    with method='none' (not enough backtest data, or no correction beat
    the uncorrected baseline) pass straight through unchanged."""
    raw = {"H": p_home, "D": p_draw, "A": p_away}
    corrected = {}
    for cls, p in raw.items():
        result = calibrators.get(cls)
        if result is None or result.calibrator is None:
            corrected[cls] = p
        else:
            corrected[cls] = float(result.calibrator.predict(np.array([p]))[0])

    total = sum(corrected.values())
    if total <= 0:
        return p_home, p_draw, p_away  # degenerate fallback - never divide by zero
    return corrected["H"] / total, corrected["D"] / total, corrected["A"] / total


def calibration_summary_table(calibrators: dict) -> pd.DataFrame:
    """Display-ready audit of what fit_1x2_calibrators() actually chose
    and why - which method won each class, its held-out log-loss versus
    the alternatives it beat, and the sample size behind the decision.
    Meant to be shown in the UI next to the odds calibration table itself,
    so which correction (if any) is currently active per league is
    always visible rather than a silent, unlogged choice."""
    rows = []
    for cls, label in (("H", "Home Win"), ("D", "Draw"), ("A", "Away Win")):
        result = calibrators.get(cls)
        if result is None:
            continue
        rows.append({
            "Outcome": label,
            "Method Selected": result.method,
            "N (backtest rows)": result.n_samples,
            "CV Log-Loss (none)": round(result.cv_log_loss.get("none"), 4) if "none" in result.cv_log_loss else None,
            "CV Log-Loss (platt)": round(result.cv_log_loss.get("platt"), 4) if "platt" in result.cv_log_loss else None,
            "CV Log-Loss (isotonic)": round(result.cv_log_loss.get("isotonic"), 4) if "isotonic" in result.cv_log_loss else None,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Calibration persistence - JSON-safe serialisation so a league's fitted
# calibrators (and the other expensive-to-backtest parameters alongside
# them) can be saved to disk once and reloaded instantly on every later
# run, instead of being refit from a fresh walk-forward backtest every
# time a fixture or league is selected in the UI.
# ---------------------------------------------------------------------------

def serialize_calibration_result(result: CalibrationResult) -> dict:
    """Plain-dict, JSON-safe representation of a fitted CalibrationResult."""
    out = {"method": result.method, "cv_log_loss": result.cv_log_loss, "n_samples": result.n_samples}
    if result.method == "platt" and result.calibrator is not None:
        out["params"] = {"a": result.calibrator.a, "b": result.calibrator.b}
    elif result.method == "isotonic" and result.calibrator is not None:
        out["params"] = {
            "thresholds": result.calibrator.thresholds.tolist(),
            "values": result.calibrator.values.tolist(),
        }
    else:
        out["params"] = None
    return out


def deserialize_calibration_result(data: dict) -> CalibrationResult:
    method = data.get("method", "none")
    params = data.get("params")
    calibrator = None
    if method == "platt" and params:
        calibrator = PlattCalibrator(a=params["a"], b=params["b"])
    elif method == "isotonic" and params:
        calibrator = IsotonicCalibrator(
            thresholds=np.array(params["thresholds"], dtype=float),
            values=np.array(params["values"], dtype=float),
        )
    return CalibrationResult(
        method=method, calibrator=calibrator,
        cv_log_loss=data.get("cv_log_loss", {}), n_samples=data.get("n_samples", 0),
    )


def serialize_1x2_calibrators(calibrators: dict) -> dict:
    return {cls: serialize_calibration_result(result) for cls, result in calibrators.items()}


def deserialize_1x2_calibrators(data: dict) -> dict:
    return {cls: deserialize_calibration_result(d) for cls, d in data.items()}


# ---------------------------------------------------------------------------
# ALL-MARKET calibration - the same select_calibrator() machinery above
# (fit Platt AND isotonic, cross-validate, keep whichever wins, or "none"
# if neither beats the raw baseline), extended from just Home/Draw/Away to
# every market in MARKET_LIST. Requires backtest_df built
# with walk_forward_backtest(..., include_all_markets=True), which stores
# prob__{market} (the model's raw probability) and actual__{market} (the
# real ground truth, independent of any 50% threshold - see that
# function's docstring) for exactly this purpose.
# ---------------------------------------------------------------------------

def build_market_calibration_training_data(backtest_df: pd.DataFrame, market: str, prob_prefix: str = "prob") -> tuple[np.ndarray, np.ndarray]:
    """Returns (probs, outcomes) for ONE market, dropping any row where
    the market didn't apply to that match (actual__{market} is None -
    e.g. Draw No Bet on a match that was actually a draw). This is the
    market-general version of build_calibration_training_data() above."""
    prob_col, actual_col = f"{prob_prefix}__{market}", f"actual__{market}"
    if prob_col not in backtest_df.columns or actual_col not in backtest_df.columns:
        return np.array([]), np.array([])
    subset = backtest_df[[prob_col, actual_col]].dropna()
    if subset.empty:
        return np.array([]), np.array([])
    probs = subset[prob_col].to_numpy(dtype=float)
    outcomes = subset[actual_col].astype(bool).to_numpy(dtype=float)
    return probs, outcomes


def fit_all_market_calibrators(backtest_df: pd.DataFrame, prob_prefix: str = "prob") -> dict:
    """{market_name: CalibrationResult} for EVERY market in MARKET_LIST,
    each independently cross-validated (Platt vs. isotonic vs. none) -
    the market-general version of fit_1x2_calibrators() above. Requires
    walk_forward_backtest(..., include_all_markets=True) output."""
    # prob_prefix="prob" fits calibrators for the Dixon-Coles probabilities, "mcprob" fits SEPARATE ones for
    # the Monte Carlo engine (its jittered distribution differs from DC, so DC-fitted calibrators do not apply to it).
    return {market: select_calibrator(*build_market_calibration_training_data(backtest_df, market, prob_prefix)) for market in MARKET_LIST}


# Groups of markets that must stay internally consistent (sum to 1, or to
# each other) AFTER calibration - each market inside a group is calibrated
# independently first, then the WHOLE group is renormalised together, so
# calibration can correct each market's own bias without breaking the
# logical relationship between markets in the same group. Markets with no
# complementary partner (Clean Sheet, Win to Nil) are simply left as
# independently-calibrated, standalone probabilities.
CALIBRATION_COMPLEMENTARY_GROUPS = [
    ("Home Win", "Draw", "Away Win"),
    ("Over 1.5 Goals", "Under 1.5 Goals"),
    ("Over 2.5 Goals", "Under 2.5 Goals"),
    ("Over 3.5 Goals", "Under 3.5 Goals"),
    ("BTTS - Yes", "BTTS - No"),
    ("Draw No Bet Home", "Draw No Bet Away"),
    ("Asian Handicap Home -1.5", "Asian Handicap Away +1.5"),
    ("Asian Handicap Home +1.5", "Asian Handicap Away -1.5"),
    tuple(MULTI_SCORE_MARKETS),
    # Goal Range, 1X2+Total Goals, and Total Goals+BTTS are genuine mutually-
    # exclusive-and-exhaustive partitions (every match falls into exactly one
    # bucket in each group) - see the combo_probs comment in
    # market_probs_from_matrix for why DC+BTTS and DC+Total Goals are
    # deliberately NOT here: 1X/12/X2 overlap each other, so those two
    # groups' six markets don't sum to 1 and must never be forced to.
    ("Goal Range 0-1", "Goal Range 2-3", "Goal Range 4-5", "Goal Range 6+"),
    _canon_partition(("Home & Over 2.5", "Home & Under 2.5", "Draw & Over 2.5", "Draw & Under 2.5", "Away & Over 2.5", "Away & Under 2.5")),
    ("Over 2.5 & BTTS Yes", "Over 2.5 & BTTS No", "Under 2.5 & BTTS Yes", "Under 2.5 & BTTS No"),
    _canon_partition(("Home & BTTS Yes", "Home & BTTS No", "Draw & BTTS Yes", "Draw & BTTS No", "Away & BTTS Yes", "Away & BTTS No")),
    ("Total Goals Odd", "Total Goals Even"),
    ("Home Team Over 1.5", "Home Team Under 1.5"),
    ("Away Team Over 1.5", "Away Team Under 1.5"),
]
for _line in EXTRA_COMBO_LINES:  # per extra line: 1X2 x (Over/Under) and (Over/Under) x BTTS are each a full partition
    CALIBRATION_COMPLEMENTARY_GROUPS.append(_canon_partition(f"{r} & {side} {_line}" for r in ("Home", "Draw", "Away") for side in ("Over", "Under")))
    CALIBRATION_COMPLEMENTARY_GROUPS.append(_canon_partition(
        [f"Over {_line} & BTTS Yes", f"Over {_line} & BTTS No", f"Under {_line} & BTTS No"]
        + ([f"Under {_line} & BTTS Yes"] if int(_line) >= 2 else [])))
# NOTE: "1X & BTTS Yes" + "1X & BTTS No" is NOT a sum-to-1 pair, even though
# it looks like one - it sums to P(Double Chance 1X), which is itself < 1
# (Away Win still has positive probability). An earlier pass here briefly
# added these as complementary groups on that false assumption, which would
# have forcibly inflated both legs to sum to 1.0 - verified wrong via direct
# computation (P(1X)/P(12)/P(X2) landed at 0.59/0.76/0.65 in testing, not
# 1.0) and reverted. These 12 markets are correctly left independently
# calibrated with no forced-sum constraint; the true constraint they'd need
# (summing to the CALIBRATED value of "Double Chance 1X", not a fixed 1.0)
# would require a different mechanism than the fixed-target-1.0 renormalise
# below - not built, since it adds real complexity for a cosmetic-only gap
# (both legs still move in the correct calibrated direction individually).

# Markets that are never independently calibrated - always re-derived from
# the (already-calibrated) markets they're built from, since they're just
# sums of other rows on the sheet, not their own independent read of the
# match.
DERIVED_MARKETS = {
    "Double Chance 1X": ("Home Win", "Draw"),
    "Double Chance 12": ("Home Win", "Away Win"),
    "Double Chance X2": ("Draw", "Away Win"),
}


def apply_all_market_calibration(predicted_probs: dict, calibrators: dict) -> dict:
    """Applies every market's own independently-selected calibrator
    (Platt/isotonic/none) to a live prediction's raw probabilities, then:
      1. Re-normalises every CALIBRATION_COMPLEMENTARY_GROUPS group so
         calibration never breaks a pair/set that has to sum to 1 (Over/
         Under, BTTS Yes/No, Draw No Bet, the two AH line-pairs, the full
         Home/Draw/Away split, and the Multi Score set).
      2. Re-derives the 3 Double Chance markets from the now-calibrated
         Home/Draw/Away, since they were never independently calibrated
         in the first place (see DERIVED_MARKETS).

    This is the SINGLE function that must run on dc_probs/mc_probs before
    either reaches build_valuation_sheet() - calibration has to be
    applied and resolved before the summary table is built, not after,
    so every EV/convergence/verdict calculation downstream is already
    working from calibrated numbers rather than needing a second pass."""
    corrected = dict(predicted_probs)
    for market in MARKET_LIST:
        if market in DERIVED_MARKETS:
            continue  # filled in at the end, from the calibrated values of what they're derived from
        result = calibrators.get(market)
        if result is not None and result.calibrator is not None and market in predicted_probs:
            corrected[market] = float(result.calibrator.predict(np.array([predicted_probs[market]]))[0])

    # A kept equivalent can belong to more than one complete set (e.g. "Draw & BTTS Yes" is in the 1X2 + BTTS set AND in
    # the 1.5-line 1X2 + total-goals set), so a single pass of normalising can undo an earlier set. Repeat until every set
    # sums to 1. Sets that share no market (the original situation) are already exact after the first pass.
    for _ in range(300):
        worst = 0.0
        for group in CALIBRATION_COMPLEMENTARY_GROUPS:
            present = [m for m in group if m in corrected]
            total = sum(corrected[m] for m in present)
            if total > 0:
                worst = max(worst, abs(total - 1.0))
                for m in present:
                    corrected[m] = corrected[m] / total
        if worst < 1e-12:
            break

    for derived_market, components in DERIVED_MARKETS.items():
        if all(c in corrected for c in components):
            corrected[derived_market] = sum(corrected[c] for c in components)

    return corrected


def all_market_calibration_summary_table(calibrators: dict) -> pd.DataFrame:
    """Display-ready audit of fit_all_market_calibrators() - one row per
    market, which method won, and its held-out log-loss versus the
    alternatives it beat. The market-general version of
    calibration_summary_table() above."""
    rows = []
    for market in MARKET_LIST:
        result = calibrators.get(market)
        if result is None:
            continue
        rows.append({
            "Market": market,
            "Method Selected": result.method,
            "N (backtest rows)": result.n_samples,
            "CV Log-Loss (none)": round(result.cv_log_loss.get("none"), 4) if "none" in result.cv_log_loss else None,
            "CV Log-Loss (platt)": round(result.cv_log_loss.get("platt"), 4) if "platt" in result.cv_log_loss else None,
            "CV Log-Loss (isotonic)": round(result.cv_log_loss.get("isotonic"), 4) if "isotonic" in result.cv_log_loss else None,
        })
    return pd.DataFrame(rows)


def serialize_all_market_calibrators(calibrators: dict) -> dict:
    return {market: serialize_calibration_result(result) for market, result in calibrators.items()}


def deserialize_all_market_calibrators(data: dict) -> dict:
    return {market: deserialize_calibration_result(d) for market, d in data.items()}


def backtest_accuracy_pct(backtest_df: pd.DataFrame) -> float:
    """% of matches where the model's highest-probability pick (H/D/A)
    matched the actual result."""
    if backtest_df.empty:
        return float("nan")
    return float(100 * backtest_df["correct"].mean())


def apply_manual_override(p_home: float, p_draw: float, p_away: float, override_pct: float) -> tuple[float, float, float]:
    """A manual calibration nudge for sensitivity testing: shifts the
    home-win probability by override_pct percentage points (+/-) and
    rebalances draw/away proportionally so all three still sum to 1.
    override_pct=0 returns the inputs completely unchanged."""
    if override_pct == 0:
        return p_home, p_draw, p_away
    shift = override_pct / 100.0
    new_home = min(max(p_home + shift, 0.0001), 0.9999)
    remaining = 1 - new_home
    old_remaining = p_draw + p_away
    if old_remaining <= 0:
        new_draw = new_away = remaining / 2
    else:
        new_draw = remaining * (p_draw / old_remaining)
        new_away = remaining * (p_away / old_remaining)
    return new_home, new_draw, new_away


def meets_accuracy_floor(accuracy_pct: float, floor_pct: float) -> bool:
    if accuracy_pct is None or (isinstance(accuracy_pct, float) and math.isnan(accuracy_pct)):
        return False
    return accuracy_pct >= floor_pct


def passes_market_gate(confidence_pct: float, ev_pct: float, confidence_floor_pct: float, min_ev_pct: float) -> bool:
    """A market on the 22-row valuation sheet is only 'allowed' if BOTH:
      - its own confidence (the higher of its Dixon-Coles and Monte Carlo
        probability for THAT SPECIFIC market, not the match's outright
        1X2 confidence) clears the floor, AND
      - its EV edge clears the minimum EV gate.
    This is an AND, not an OR - a high-EV pick the model isn't actually
    confident about is exactly the risky case this feature exists to
    catch, so either condition failing blocks the market."""
    return confidence_pct >= confidence_floor_pct and ev_pct >= min_ev_pct


def market_gate_status(confidence_pct: float, ev_pct: float, confidence_floor_pct: float, min_ev_pct: float) -> str:
    """A short, human-readable reason a market is or isn't currently
    allowed - used directly as a displayed column, not just for styling."""
    below_confidence = confidence_pct < confidence_floor_pct
    below_ev = ev_pct < min_ev_pct
    if not below_confidence and not below_ev:
        return "✅ Allowed"
    if below_confidence and below_ev:
        return "⛔ Below Confidence & EV"
    if below_confidence:
        return "⛔ Below Confidence Floor"
    return "⛔ Below EV Gate"


# ---------------------------------------------------------------------------
# Core Parameter B: volatility auto-calibrator
# ---------------------------------------------------------------------------

@dataclass
class VolatilityProfile:
    dispersion_ratio: float
    squad_turnover_index: float
    vol_dampener: float
    adjusted: bool


def compute_volatility_profile(settled_df: pd.DataFrame) -> VolatilityProfile:
    if settled_df.empty:
        return VolatilityProfile(1.0, 0.0, 1.0, False)
    total_goals = (settled_df["home_goals"].fillna(0) + settled_df["away_goals"].fillna(0)).to_numpy(dtype=float)
    mean_goals = float(np.mean(total_goals)) if len(total_goals) else 1.0
    variance_goals = float(np.var(total_goals)) if len(total_goals) else 0.0
    std_goals = float(np.std(total_goals)) if len(total_goals) else 0.0

    dispersion_ratio = variance_goals / mean_goals if mean_goals > 0 else 1.0
    squad_turnover_index = std_goals / max(0.1, mean_goals)

    vol_dampener = 1.0
    adjusted = False
    if squad_turnover_index > DISPERSION_ADJUST_TRIGGER:
        vol_dampener = dispersion_ratio * DISPERSION_ADJUST_FACTOR
        adjusted = True
    else:
        vol_dampener = dispersion_ratio

    return VolatilityProfile(dispersion_ratio, squad_turnover_index, vol_dampener, adjusted)


# ---------------------------------------------------------------------------
# Squad Streak Momentum Tracker
# ---------------------------------------------------------------------------

OPPONENT_QUALITY_MIN_RATIO = 0.5   # floor - even a very weak opponent still earns SOME credit toward a win streak, never zero
OPPONENT_QUALITY_MAX_RATIO = 1.8   # ceiling - guards a single extreme PPG outlier (thin sample) from dominating the whole streak's effect
OPPONENT_QUALITY_MIN_MATCHES = 3   # an opponent's own sample below this is too thin to trust - treated as league-average (ratio 1.0) instead


def _league_average_ppg(all_matches_df: pd.DataFrame) -> float:
    """Points-per-game averaged across every team with a settled match in
    all_matches_df - the baseline every individual opponent's own PPG is
    compared against, to turn 'beat a team' into 'beat a team that was
    X% stronger/weaker than this league's average side'."""
    settled = all_matches_df.dropna(subset=["home_goals", "away_goals"])
    if settled.empty:
        return 1.3  # generic football fallback - close to a typical league-wide mean PPG
    standings = compute_standings_table(settled)
    if standings.empty or standings["Played"].sum() == 0:
        return 1.3
    ppg_per_team = standings["Points"] / standings["Played"].replace(0, np.nan)
    avg = ppg_per_team.mean(skipna=True)
    return float(avg) if pd.notna(avg) and avg > 0 else 1.3


def _team_ppg_asof(all_matches_df: pd.DataFrame, team: str, as_of_date) -> tuple[float | None, int]:
    """This team's points-per-game using only matches strictly BEFORE
    as_of_date - so when this is used to grade a STREAK opponent's
    quality, it reflects what that opponent had actually shown up to
    that point, never hindsight from results that happened afterward."""
    rows = all_matches_df[
        (all_matches_df["home_team"] == team) | (all_matches_df["away_team"] == team)
    ].dropna(subset=["date", "home_goals", "away_goals"])
    if as_of_date is not None:
        rows = rows[rows["date"] < as_of_date]
    if rows.empty:
        return None, 0
    pts = 0
    for _, r in rows.iterrows():
        gf, ga = (r["home_goals"], r["away_goals"]) if r["home_team"] == team else (r["away_goals"], r["home_goals"])
        if gf > ga:
            pts += 3
        elif gf == ga:
            pts += 1
    return pts / len(rows), len(rows)


GD_XGD_DEFAULT_WINDOW = 5


@dataclass
class GDvsXGDInsight:
    team: str
    window_matches: int
    goal_difference: float
    xg_difference: "float | None"
    gap: "float | None"   # GD - xGD: positive = results running AHEAD of process, negative = process running ahead of results
    quadrant: str


def team_gd_vs_xgd(
    all_matches_df: pd.DataFrame, team: str, as_of_date=None, window: int = GD_XGD_DEFAULT_WINDOW,
) -> "GDvsXGDInsight | None":
    """Real Goal Difference vs. expected Goal Difference (xGF - xGA) over
    this team's last `window` matches (any venue, chronological, strictly
    before as_of_date if given - no lookahead). Classifies into the 2x2
    "process vs. results" read analysts commonly use:

      - GD positive, xGD positive  -> genuinely dominant, deserved, sustainable
      - GD positive, xGD negative  -> overperforming - results ahead of the
        underlying numbers, a classic regression-DOWN candidate
      - GD negative, xGD positive  -> unlucky - the underlying numbers are
        better than the results show, a regression-UP candidate
      - GD negative, xGD negative  -> genuinely struggling, no hidden upside

    Returns None only if the team has no matches at all. When xG data
    isn't in this dataset (or this team doesn't have enough of it in the
    window), xg_difference/gap come back None and quadrant reads
    'Insufficient xG data' - GD alone is still returned in that case,
    just without the xG half of the comparison; callers must treat None
    as 'not computable', never as zero."""
    rows = all_matches_df[
        (all_matches_df["home_team"] == team) | (all_matches_df["away_team"] == team)
    ].dropna(subset=["date", "home_goals", "away_goals"]).sort_values("date")
    if as_of_date is not None:
        rows = rows[rows["date"] < as_of_date]
    if rows.empty:
        return None

    recent = rows.tail(window)
    n = len(recent)
    has_xg = "home_xg" in recent.columns and "away_xg" in recent.columns

    gd_total = 0.0
    xgd_total = 0.0
    xg_rows_counted = 0
    for _, r in recent.iterrows():
        if r["home_team"] == team:
            gd_total += r["home_goals"] - r["away_goals"]
            if has_xg and pd.notna(r.get("home_xg")) and pd.notna(r.get("away_xg")):
                xgd_total += r["home_xg"] - r["away_xg"]
                xg_rows_counted += 1
        else:
            gd_total += r["away_goals"] - r["home_goals"]
            if has_xg and pd.notna(r.get("home_xg")) and pd.notna(r.get("away_xg")):
                xgd_total += r["away_xg"] - r["home_xg"]
                xg_rows_counted += 1

    if xg_rows_counted < max(2, window // 2):
        # Too little xG coverage in this window to trust a comparison - same
        # "don't guess on a thin sample" principle used everywhere else here.
        return GDvsXGDInsight(
            team=team, window_matches=n, goal_difference=round(gd_total, 2),
            xg_difference=None, gap=None, quadrant="Insufficient xG data",
        )

    gap = gd_total - xgd_total
    gd_sign = "positive" if gd_total >= 0 else "negative"
    xgd_sign = "positive" if xgd_total >= 0 else "negative"
    quadrant = {
        ("positive", "positive"): "🔵 Dominant — results match strong underlying performance",
        ("positive", "negative"): "🔥 Overperforming — results ahead of process (regression risk DOWN)",
        ("negative", "positive"): "🍀 Unlucky — process better than results (regression risk UP)",
        ("negative", "negative"): "⚠️ Struggling — deserved, no hidden upside",
    }[(gd_sign, xgd_sign)]

    return GDvsXGDInsight(
        team=team, window_matches=n, goal_difference=round(gd_total, 2),
        xg_difference=round(xgd_total, 2), gap=round(gap, 2), quadrant=quadrant,
    )


XG_TREND_DEFAULT_WINDOW = 8
XG_TREND_FLAT_THRESHOLD = 0.05  # slope magnitude below this counts as "stable" - a heuristic
                                 # starting point (xG values typically run 0-3 per match, so this
                                 # is a modest per-match shift), not yet empirically calibrated.


@dataclass
class XGTrendInsight:
    team: str
    window_matches: int
    slope_per_match: float   # positive = underlying xG performance improving match-to-match; negative = declining
    recent_xgd_avg: float     # average xGD over the window, for context alongside the slope
    trend_label: str


def team_xg_trend(
    all_matches_df: pd.DataFrame, team: str, as_of_date=None, window: int = XG_TREND_DEFAULT_WINDOW,
) -> "XGTrendInsight | None":
    """Is this team's underlying xG performance (xGF-xGA per match)
    IMPROVING or DECLINING over their last `window` matches - a genuine
    TREND (linear slope across match index), distinct from
    team_gd_vs_xgd() above, which compares GD to xGD in ONE fixed
    window and says nothing about direction of travel within it. A team
    can have a solidly positive xGD snapshot while still be on a
    declining trend (started the window very strong, fading recently) -
    these two insights answer different questions and are meant to be
    read together, not as duplicates.

    Fits a simple linear slope (np.polyfit) of per-match xGD against
    match order over the window - needs at least 4 matches with xG data
    to fit anything meaningful; returns None below that, or if xg
    columns aren't in this dataset at all, or the team has no matches."""
    if "home_xg" not in all_matches_df.columns or "away_xg" not in all_matches_df.columns:
        return None
    rows = all_matches_df[
        (all_matches_df["home_team"] == team) | (all_matches_df["away_team"] == team)
    ].dropna(subset=["date"]).sort_values("date")
    if as_of_date is not None:
        rows = rows[rows["date"] < as_of_date]
    if rows.empty:
        return None

    recent = rows.tail(window)
    xgd_values = []
    for _, r in recent.iterrows():
        if pd.isna(r.get("home_xg")) or pd.isna(r.get("away_xg")):
            continue
        if r["home_team"] == team:
            xgd_values.append(r["home_xg"] - r["away_xg"])
        else:
            xgd_values.append(r["away_xg"] - r["home_xg"])

    if len(xgd_values) < 4:
        return None

    match_index = np.arange(len(xgd_values))
    slope = float(np.polyfit(match_index, xgd_values, 1)[0])
    recent_avg = float(np.mean(xgd_values))

    if slope > XG_TREND_FLAT_THRESHOLD:
        label = "📈 Improving — underlying xG performance trending up"
    elif slope < -XG_TREND_FLAT_THRESHOLD:
        label = "📉 Declining — underlying xG performance trending down"
    else:
        label = "➖ Stable — no clear trend"

    return XGTrendInsight(
        team=team, window_matches=len(xgd_values), slope_per_match=round(slope, 3),
        recent_xgd_avg=round(recent_avg, 2), trend_label=label,
    )


def _regression_dampener(gd_total: float, xgd_total: float, is_win: bool) -> float:
    """Returns a multiplier in [0.3, 1.2], applied ON TOP of the opponent-
    quality-weighted streak effect in team_streak_multiplier below -
    dampens a WIN streak whose results are running well ahead of the
    underlying xG (regression risk down), and dampens a LOSING streak
    whose underlying xG is better than the results show (regression risk
    up). A streak fully backed by matching xG gets no dampening at all
    (ratio 1.0); the mild 1.2x ceiling means this can only ever nudge a
    streak's effect up slightly when the process is even stronger than
    the results already suggest - it's a caution signal first, not an
    independent confidence booster."""
    if is_win:
        if gd_total <= 0:
            return 1.0
        ratio = xgd_total / gd_total
    else:
        if gd_total >= 0:
            return 1.0
        ratio = xgd_total / gd_total
    return float(np.clip(ratio, 0.3, 1.2))


def team_streak_multiplier(all_matches_df: pd.DataFrame, team: str):
    """Looks at ALL of a team's matches (home and away combined, sorted
    chronologically) to find their CURRENT streak, and returns
    (multiplier, description).

    Each match in the streak is weighted by the OPPONENT'S OWN STRENGTH
    at the time (points-per-game up to that date, relative to the
    league-average PPG), instead of every win/loss counting as the same
    flat +2%/-3% regardless of who it came against. A mid-table team's
    win streak built entirely against bottom-table sides now produces a
    smaller boost than the same length streak that included wins over
    top-table opposition - the model no longer treats "5 wins" as
    interchangeable regardless of opposition quality, which is exactly
    the gap flat per-match increments had: a streak's real predictive
    value depends on WHO it came against, not just how long it is.

    Opponents with too thin a sample of their own (< OPPONENT_QUALITY_MIN_MATCHES)
    are treated as league-average (ratio 1.0) rather than trusted on a
    noisy PPG - same shrink-to-baseline instinct used everywhere else in
    this engine for small samples. The quality ratio is also clamped to
    [OPPONENT_QUALITY_MIN_RATIO, OPPONENT_QUALITY_MAX_RATIO] so one
    extreme opponent can't single-handedly dominate a multi-match streak's
    effect.

    Base per-match rates (2% for a win, 3% for a loss, at league-average
    opponent quality) and the overall 0.88-1.12 cap are unchanged from
    the previous flat version - so a streak built entirely against
    average opposition produces almost exactly the same multiplier as
    before; what changes is everything AWAY from average opposition."""
    rows = all_matches_df[
        (all_matches_df["home_team"] == team) | (all_matches_df["away_team"] == team)
    ].dropna(subset=["date"]).sort_values("date")
    if rows.empty:
        return 1.0, "no data"

    results = []
    for _, r in rows.iterrows():
        if r["home_team"] == team:
            gf, ga, opponent = r["home_goals"], r["away_goals"], r["away_team"]
        else:
            gf, ga, opponent = r["away_goals"], r["home_goals"], r["home_team"]
        if pd.isna(gf) or pd.isna(ga):
            continue
        outcome = "W" if gf > ga else ("L" if gf < ga else "D")
        results.append({"result": outcome, "opponent": opponent, "date": r["date"]})

    if not results:
        return 1.0, "no settled matches"

    last = results[-1]["result"]
    if last not in ("W", "L"):
        return 1.0, "last match was a draw"

    streak_matches = []
    for r in reversed(results):
        if r["result"] == last:
            streak_matches.append(r)
        else:
            break

    streak = len(streak_matches)
    if streak < 2:
        return 1.0, f"streak of {streak} - below the 2-match trigger"

    league_avg_ppg = _league_average_ppg(all_matches_df)
    quality_ratios = []
    for match in streak_matches:
        opp_ppg, opp_n = _team_ppg_asof(all_matches_df, match["opponent"], match["date"])
        if opp_ppg is None or opp_n < OPPONENT_QUALITY_MIN_MATCHES or league_avg_ppg <= 0:
            ratio = 1.0
        else:
            ratio = np.clip(opp_ppg / league_avg_ppg, OPPONENT_QUALITY_MIN_RATIO, OPPONENT_QUALITY_MAX_RATIO)
        quality_ratios.append(float(ratio))
    avg_quality_ratio = float(np.mean(quality_ratios))

    # Regression dampener: is this streak's GD actually backed by similar
    # xGD, or is it running ahead/behind the underlying process? Uses the
    # SAME team_gd_vs_xgd() window (default last 5) as the standalone
    # insight - not necessarily the exact streak length, since this is a
    # broader "does recent form hold up" check, not scoped tightly to the
    # streak itself. Gracefully no-ops (dampener=1.0) when xG data isn't
    # in this dataset - existing behaviour is fully preserved for anyone
    # without xG columns.
    gd_xgd = team_gd_vs_xgd(all_matches_df, team)
    dampener = 1.0
    dampener_note = ""
    if gd_xgd is not None and gd_xgd.xg_difference is not None:
        dampener = _regression_dampener(gd_xgd.goal_difference, gd_xgd.xg_difference, is_win=(last == "W"))
        if dampener < 0.95:
            dampener_note = f" [dampened {dampener:.2f}x - {gd_xgd.quadrant.split(' — ')[0].strip()}, last {gd_xgd.window_matches}: GD {gd_xgd.goal_difference:+.1f} vs xGD {gd_xgd.xg_difference:+.1f}]"
        elif dampener > 1.05:
            dampener_note = f" [boosted {dampener:.2f}x - {gd_xgd.quadrant.split(' — ')[0].strip()}, last {gd_xgd.window_matches}: GD {gd_xgd.goal_difference:+.1f} vs xGD {gd_xgd.xg_difference:+.1f}]"

    if last == "W":
        # Beating stronger-than-average sides scales the boost UP;
        # beating weaker-than-average sides scales it DOWN.
        total_effect = sum(0.02 * ratio for ratio in quality_ratios) * dampener
        mult = min(1.0 + total_effect, 1.12)
        quality_note = "above-average" if avg_quality_ratio > 1.05 else ("below-average" if avg_quality_ratio < 0.95 else "average")
        return mult, (
            f"{streak}-match win streak vs. {quality_note} opposition "
            f"(avg opponent strength {avg_quality_ratio:.2f}x league PPG, +{(mult - 1) * 100:.1f}%)" + dampener_note
        )
    else:
        # Losing to stronger-than-average sides is less damning (smaller
        # penalty); losing to weaker-than-average sides is scaled UP
        # (bigger penalty) - a losing streak against bottom-table teams
        # should hurt more than the same streak against the league's best.
        total_effect = sum(0.03 * (2.0 - ratio) for ratio in quality_ratios) * dampener
        mult = max(1.0 - total_effect, 0.88)
        quality_note = "above-average" if avg_quality_ratio > 1.05 else ("below-average" if avg_quality_ratio < 0.95 else "average")
        return mult, (
            f"{streak}-match losing streak vs. {quality_note} opposition "
            f"(avg opponent strength {avg_quality_ratio:.2f}x league PPG, {(mult - 1) * 100:.1f}%)" + dampener_note
        )


# ---------------------------------------------------------------------------
# Section 6: tactical & environmental multipliers
# ---------------------------------------------------------------------------

@dataclass
class TeamAdjustment:
    attack: float = 1.0
    defense: float = 1.0  # multiplicative on the "allowed" ratio - >1 makes defense WORSE


@dataclass
class TacticalInputs:
    home_newly_relegated: bool = False
    away_newly_relegated: bool = False
    home_relegation_threat: bool = False
    away_relegation_threat: bool = False
    # Injuries are now split by role - see the realism note above
    # apply_tactical_multipliers for why a striker injury and a defender
    # injury are no longer the same checkbox.
    home_striker_injury: bool = False
    away_striker_injury: bool = False
    home_defender_injury: bool = False
    away_defender_injury: bool = False
    home_bogey: bool = False
    away_bogey: bool = False
    home_new_manager: bool = False
    away_new_manager: bool = False
    home_boardroom_crisis: bool = False
    away_boardroom_crisis: bool = False
    home_dead_rubber: bool = False
    away_dead_rubber: bool = False
    home_travel_fatigue_units: int = 0  # 0-3, applies to the AWAY team traveling to home team's ground
    host_travel_fatigue_units: int = 0  # 0-3, the HOST's own midweek travel fatigue (e.g. a midweek away European leg) carried into THIS home fixture
    coastal_shock: bool = False  # applies to the traveling (away) team
    home_cup_distraction: bool = False
    away_cup_distraction: bool = False
    # New selectors
    home_tactical_setup: str = "Standard Open Play"   # or "Deep Ultra-Defensive Low-Block" / "High-Intensity Counter-Pressing Style"
    away_tactical_setup: str = "Standard Open Play"
    pitch_surface: str = "Standard Optimized Turf"     # or "Waterlogged Mud" / "Dry Uneven Grass, short and narrow"
    weather: str = "Clear Sky (no rain/wind)"          # or "Torrential Rain Storm" / "Gale-Force Wind Interference"
    referee_strictness: str = "Standard Average"        # or "Lenient (Flow Enforcer)" / "Hyper-Strict (Card Trigger)"
    pre_season_fixture: bool = False
    # Split departure/signing impact - two independent sliders per team
    # (see realism note below for why these replaced the old single
    # combined "transfer_impact_pct" slider).
    home_departure_impact_pct: float = 0.0   # 0 to -20 typically: a key player LEFT
    away_departure_impact_pct: float = 0.0
    home_new_signing_impact_pct: float = 0.0  # 0 to +20 typically: a quality player ARRIVED
    away_new_signing_impact_pct: float = 0.0
    # Derby / rivalry match - a property of the FIXTURE, not one team.
    derby_match: bool = False
    # Fixture congestion - COMPUTED AUTOMATICALLY from real match dates
    # (see compute_fixture_congestion below), not manually selected like
    # the toggles above. This is a count of matches played in the recent
    # window before this fixture, one per team.
    home_congestion_matches: int = 0
    away_congestion_matches: int = 0
    # Heat stress - a genuinely different mechanism from `weather` above
    # (ball physics: rain/wind) and `coastal_shock` (humidity, away-only):
    # this is cardiovascular fatigue accumulation from playing above ~25C,
    # affecting BOTH teams but asymmetrically depending on acclimatization
    # (see apply_tactical_multipliers' heat block for the reasoning). 0 =
    # below 25C / not applicable; 1=25-30C, 2=30-35C, 3=35C+.
    heat_stress_level: int = 0
    home_heat_acclimatized: bool = False  # this team's normal home climate is characteristically hot
    away_heat_acclimatized: bool = False


FIXTURE_CONGESTION_WINDOW_DAYS = 10  # matches within this many days before kickoff count toward congestion


def compute_fixture_congestion(all_matches_df: pd.DataFrame, team: str, as_of_date, window_days: float = FIXTURE_CONGESTION_WINDOW_DAYS) -> tuple[int, str]:
    """Counts how many matches `team` played in the WINDOW_DAYS immediately
    before as_of_date (never including as_of_date itself or anything
    after it - no lookahead). Returns (match_count, description) - feed
    match_count straight into TacticalInputs.home_congestion_matches /
    away_congestion_matches; apply_tactical_multipliers only starts
    applying a fatigue penalty at 2+ matches in the window, so a normal
    once-a-week schedule produces 0 or 1 and no penalty at all."""
    if as_of_date is None or pd.isna(as_of_date):
        return 0, "no reference date available"
    window_start = as_of_date - pd.Timedelta(days=window_days)
    rows = all_matches_df[
        ((all_matches_df["home_team"] == team) | (all_matches_df["away_team"] == team))
        & (all_matches_df["date"] >= window_start) & (all_matches_df["date"] < as_of_date)
    ]
    count = len(rows)
    if count >= 2:
        return count, f"{count} matches in the {int(window_days)} days before this fixture - squad rotation fatigue likely"
    return count, f"{count} match(es) in the {int(window_days)} days before this fixture - normal schedule"


def apply_tactical_multipliers(
    home_attack: float, home_defense: float, away_attack: float, away_defense: float,
    base_volatility: float, tactics: TacticalInputs,
):
    """Returns (home_adjustment, away_adjustment, adjusted_volatility, log).
    Every multiplier here only scales the INPUT rates - see design note
    #3 at the top of the file; nothing here sets a probability directly.

    REALISM NOTE (checked against actual research, not assumed):
    - New manager bounce: Premier League data (2021/22-2025/26, 35
      mid-season appointments) shows clubs jumping from ~0.90 to ~1.27
      points per game in the first 5 games under a new manager - a ~41%
      relative swing. That number is intentionally NOT applied at full
      strength here: most of it is regression to the mean (clubs sack
      managers exactly when results are at their worst, so ANY manager
      would see some bounce-back), plus small-sample noise that fades by
      games 11-20. A conservative 10% attack/defense bump is kept as a
      defensible middle estimate of the sustained part of the effect,
      not the full raw PPG swing.
    - Striker vs defender injuries are now DIFFERENT, not the same
      checkbox: losing a primary goal-scorer is a fairly predictable,
      attack-specific quality loss. Losing a first-choice center-back
      tends to show up more as increased defensive VARIANCE (makeshift
      back-lines make more individual errors) on top of a quality loss -
      this matches the general direction of the injury-performance
      literature (Hägglund et al. 2013, BJSM; and follow-up Bundesliga
      cost studies) even though no single published number cleanly
      separates "striker" vs "defender" effect size - these remain
      reasoned estimates, not a proven precise coefficient.
    - Key player transfer impact is now TWO independent sliders per team
      (departure, new signing) instead of one combined "net" slider - a
      team can lose a starter AND sign a replacement in the same window,
      and those are genuinely different events with different timing/
      certainty (a departure's absence is immediate and known; a new
      signing's actual impact often takes weeks to show up in the team's
      real output). Collapsing them into one net number hid that. Like
      the old combined slider, there's no universal research figure for
      a SPECIFIC transfer's impact the way there is for e.g. a manager
      change, so both remain manual sliders, left to your own judgement.
    - Derby/rivalry matches: research on rivalry games consistently finds
      MORE cards/fouls and tighter, more cagey scorelines than the same
      two teams' underlying quality would suggest in a normal fixture -
      heightened stakes and motivation tend to suppress the stronger
      side's usual scoring edge more than they boost the weaker side's.
      Modelled here as a modest tightening of BOTH teams' defense (goals
      get harder to come by for either side) plus a volatility bump
      (individual moments - a red card, a set piece - matter more than
      the run of play), rather than favoring either side specifically.
    - Fixture congestion is COMPUTED AUTOMATICALLY (see
      compute_fixture_congestion below) from each team's actual match
      dates, not manually toggled - 3+ matches in a 10-day window is
      treated as squad-rotation fatigue territory, scaling smoothly
      rather than a flat on/off switch.
    """
    home = TeamAdjustment(home_attack, home_defense)
    away = TeamAdjustment(away_attack, away_defense)
    vol = base_volatility
    log = []

    def general_decline_or_boost(adj: TeamAdjustment, factor: float, label: str, side: str):
        adj.attack *= factor
        adj.defense /= factor  # factor<1 (decline) -> defense worsens; factor>1 (boost) -> defense improves
        log.append(f"{side}: {label} -> attack x{factor:.2f}, defense x{1/factor:.2f}")

    if tactics.home_newly_relegated:
        general_decline_or_boost(home, 0.90, "Newly relegated", "Home")
    if tactics.away_newly_relegated:
        general_decline_or_boost(away, 0.90, "Newly relegated", "Away")

    if tactics.home_relegation_threat:
        home.defense /= 1.08
        log.append("Home: Live relegation threat -> defense x0.926 (+8% grit)")
    if tactics.away_relegation_threat:
        away.defense /= 1.08
        log.append("Away: Live relegation threat -> defense x0.926 (+8% grit)")

    # --- Injuries, now split by role (see realism note above) ---
    if tactics.home_striker_injury:
        home.attack *= 0.88
        vol *= 1.03
        log.append("Home: Key striker/attacker out -> attack x0.88, volatility x1.03")
    if tactics.away_striker_injury:
        away.attack *= 0.88
        vol *= 1.03
        log.append("Away: Key striker/attacker out -> attack x0.88, volatility x1.03")
    if tactics.home_defender_injury:
        home.defense /= 0.90  # defense WORSENS ~11% - a makeshift back-line concedes more
        vol *= 1.08
        log.append("Home: Key defender out -> defense x1.11 (worse), volatility x1.08")
    if tactics.away_defender_injury:
        away.defense /= 0.90
        vol *= 1.08
        log.append("Away: Key defender out -> defense x1.11 (worse), volatility x1.08")

    # --- Key player departure / new signing (split, manual sliders) ---
    if tactics.home_departure_impact_pct != 0:
        factor = max(0.01, 1 + tactics.home_departure_impact_pct / 100)
        general_decline_or_boost(home, factor, f"Key departure ({tactics.home_departure_impact_pct:+.0f}%, manually set)", "Home")
    if tactics.away_departure_impact_pct != 0:
        factor = max(0.01, 1 + tactics.away_departure_impact_pct / 100)
        general_decline_or_boost(away, factor, f"Key departure ({tactics.away_departure_impact_pct:+.0f}%, manually set)", "Away")
    if tactics.home_new_signing_impact_pct != 0:
        factor = max(0.01, 1 + tactics.home_new_signing_impact_pct / 100)
        general_decline_or_boost(home, factor, f"New signing ({tactics.home_new_signing_impact_pct:+.0f}%, manually set)", "Home")
    if tactics.away_new_signing_impact_pct != 0:
        factor = max(0.01, 1 + tactics.away_new_signing_impact_pct / 100)
        general_decline_or_boost(away, factor, f"New signing ({tactics.away_new_signing_impact_pct:+.0f}%, manually set)", "Away")

    # --- Derby / rivalry match (affects both teams together, not per-side) ---
    if tactics.derby_match:
        home.defense /= 1.06  # tighter defense for BOTH sides - goals harder to come by either way
        away.defense /= 1.06
        vol *= 1.12
        log.append("Derby/rivalry match: both defenses tighten x0.943, volatility x1.12 (tighter, feistier affair)")

    # --- Fixture congestion (computed automatically, not manually toggled) ---
    if tactics.home_congestion_matches >= 2:
        fatigue_factor = max(0.80, 1 - 0.05 * (tactics.home_congestion_matches - 1))
        home.attack *= fatigue_factor
        vol *= 1 + 0.02 * (tactics.home_congestion_matches - 1)
        log.append(
            f"Home: fixture congestion ({tactics.home_congestion_matches} matches in the recent window) "
            f"-> attack x{fatigue_factor:.2f}, volatility x{1 + 0.02 * (tactics.home_congestion_matches - 1):.2f}"
        )
    if tactics.away_congestion_matches >= 2:
        fatigue_factor = max(0.80, 1 - 0.05 * (tactics.away_congestion_matches - 1))
        away.attack *= fatigue_factor
        vol *= 1 + 0.02 * (tactics.away_congestion_matches - 1)
        log.append(
            f"Away: fixture congestion ({tactics.away_congestion_matches} matches in the recent window) "
            f"-> attack x{fatigue_factor:.2f}, volatility x{1 + 0.02 * (tactics.away_congestion_matches - 1):.2f}"
        )

    if tactics.home_bogey:
        general_decline_or_boost(home, 0.95, "Historical bogey hex", "Home")
    if tactics.away_bogey:
        general_decline_or_boost(away, 0.95, "Historical bogey hex", "Away")

    if tactics.home_new_manager:
        general_decline_or_boost(home, 1.10, "New manager bounce", "Home")
    if tactics.away_new_manager:
        general_decline_or_boost(away, 1.10, "New manager bounce", "Away")

    if tactics.home_boardroom_crisis:
        general_decline_or_boost(home, 0.85, "Boardroom crisis", "Home")
    if tactics.away_boardroom_crisis:
        general_decline_or_boost(away, 0.85, "Boardroom crisis", "Away")

    if tactics.home_dead_rubber:
        general_decline_or_boost(home, 0.90, "Dead rubber / beach mode", "Home")
        vol *= 0.90
    if tactics.away_dead_rubber:
        general_decline_or_boost(away, 0.90, "Dead rubber / beach mode", "Away")
        vol *= 0.90

    if tactics.home_travel_fatigue_units:
        factor = max(0.01, 1 - 0.04 * tactics.home_travel_fatigue_units)
        away.attack *= factor  # the AWAY team is the one traveling to the home team's ground
        log.append(f"Away: Travel fatigue x{tactics.home_travel_fatigue_units} unit(s) -> attack x{factor:.2f}")

    if tactics.host_travel_fatigue_units:
        factor = max(0.01, 1 - 0.04 * tactics.host_travel_fatigue_units)
        home.attack *= factor  # the HOST's own midweek travel, independent of this match's venue
        log.append(f"Home: Travel fatigue x{tactics.host_travel_fatigue_units} unit(s) -> attack x{factor:.2f}")

    if tactics.coastal_shock:
        away.attack *= 0.95
        vol *= 0.92
        log.append("Away: Coastal humidity shock -> attack x0.95, volatility x0.92")

    # --- Tactical setup selector (replaces the old low-block-only checkbox) ---
    if tactics.home_tactical_setup == "Deep Ultra-Defensive Low-Block":
        home.attack *= 0.85
        vol *= 0.82
        log.append("Home: Deep low-block -> attack x0.85, volatility x0.82")
    elif tactics.home_tactical_setup == "High-Intensity Counter-Pressing Style":
        # Not specified in the original spec - a reasoned estimate: winning
        # the ball back higher up creates more transition chances (attack
        # up) but also more end-to-end chaos (volatility up).
        home.attack *= 1.06
        vol *= 1.05
        log.append("Home: High-intensity counter-press -> attack x1.06, volatility x1.05")
    if tactics.away_tactical_setup == "Deep Ultra-Defensive Low-Block":
        away.attack *= 0.85
        vol *= 0.82
        log.append("Away: Deep low-block -> attack x0.85, volatility x0.82")
    elif tactics.away_tactical_setup == "High-Intensity Counter-Pressing Style":
        away.attack *= 1.06
        vol *= 1.05
        log.append("Away: High-intensity counter-press -> attack x1.06, volatility x1.05")

    # --- Pitch surface (affects BOTH teams equally - it's the same pitch) ---
    if tactics.pitch_surface == "Waterlogged Mud":
        home.attack *= 0.90
        away.attack *= 0.90
        vol *= 1.10
        log.append("Both: Waterlogged mud pitch -> attack x0.90 each, volatility x1.10")
    elif tactics.pitch_surface == "Dry Uneven Grass, short and narrow":
        home.attack *= 0.95
        away.attack *= 0.95
        vol *= 1.05
        log.append("Both: Dry uneven/narrow pitch -> attack x0.95 each, volatility x1.05")

    # --- Weather (affects BOTH teams equally) ---
    if tactics.weather == "Torrential Rain Storm":
        home.attack *= 0.92
        away.attack *= 0.92
        vol *= 1.12
        log.append("Both: Torrential rain -> attack x0.92 each, volatility x1.12")
    elif tactics.weather == "Gale-Force Wind Interference":
        home.attack *= 0.88
        away.attack *= 0.88
        vol *= 1.15
        log.append("Both: Gale-force wind -> attack x0.88 each, volatility x1.15")

    # --- Heat stress (affects BOTH teams, asymmetrically by acclimatization) ---
    # Distinct from `weather` above (rain/wind affect the BALL; heat affects
    # the PLAYERS - cardiovascular fatigue accumulation, not skill execution)
    # and from `coastal_shock` (humidity, hard-coded to the away team only).
    # An acclimatized team - one whose normal home climate is characteristically
    # hot - suffers roughly half the attack penalty and half the extra
    # volatility of a non-acclimatized team at the same temperature level,
    # reflecting real physiological adaptation to heat, not just travel.
    if tactics.heat_stress_level:
        level = tactics.heat_stress_level
        home_heat_factor = max(0.01, 1 - (0.0125 if tactics.home_heat_acclimatized else 0.025) * level)
        away_heat_factor = max(0.01, 1 - (0.0125 if tactics.away_heat_acclimatized else 0.025) * level)
        home.attack *= home_heat_factor
        away.attack *= away_heat_factor
        heat_vol_factor = 1 + (0.01 if (tactics.home_heat_acclimatized and tactics.away_heat_acclimatized) else 0.02) * level
        vol *= heat_vol_factor
        log.append(
            f"Both: Heat stress level {level} (>25C) -> "
            f"Home attack x{home_heat_factor:.3f}{' (acclimatized)' if tactics.home_heat_acclimatized else ''}, "
            f"Away attack x{away_heat_factor:.3f}{' (acclimatized)' if tactics.away_heat_acclimatized else ''}, "
            f"volatility x{heat_vol_factor:.2f}"
        )

    # --- Referee strictness (affects match chaos/variance, not attack directly) ---
    if tactics.referee_strictness == "Hyper-Strict (Card Trigger)":
        vol *= 1.15
        log.append("Match: Hyper-strict referee -> volatility x1.15 (per spec)")
    elif tactics.referee_strictness == "Lenient (Flow Enforcer)":
        vol *= 0.90
        log.append("Match: Lenient referee -> volatility x0.90 (reasoned estimate - not specified in original spec)")

    # --- Pre-season fixture (universal, both teams) ---
    if tactics.pre_season_fixture:
        home.attack *= 0.90
        away.attack *= 0.90
        log.append("Both: Pre-season fixture -> attack x0.90 each (stamina/sharpness penalty)")

    if tactics.home_cup_distraction:
        general_decline_or_boost(home, 0.88, "Look-ahead cup penalty", "Home")
    if tactics.away_cup_distraction:
        general_decline_or_boost(away, 0.88, "Look-ahead cup penalty", "Away")

    # --- Stacking realism clamp ---
    # With this many independent toggles, an extreme (if unlikely) combo -
    # e.g. relegated + striker injury + bogey + boardroom crisis + dead
    # rubber + low-block + cup distraction + a bad pitch + bad weather +
    # pre-season, all at once - multiplies out to roughly a 0.31x combined
    # attack factor (verified directly: 0.90*0.88*0.95*0.85*0.90*0.85*
    # 0.88*0.90*0.88*0.90 ≈ 0.307). No real single match swings a team's
    # underlying quality by ~70%, no matter how many bad things coincide -
    # that's several standard deviations beyond anything in the injury/
    # situational-factors literature this file's realism notes are based
    # on. This clamps the COMBINED RATIO applied by tactical multipliers
    # (relative to the real, data-derived baseline that came in), not the
    # baseline itself - so a genuinely strong team can still clearly
    # outperform a genuinely weak one; this only stops the SITUATIONAL
    # layer from compounding into an implausible extreme.
    COMBINED_RATIO_FLOOR = 0.55
    COMBINED_RATIO_CEILING = 1.55

    def clamp_ratio(adjusted_value, original_value, side_label, metric_label):
        if original_value <= 0:
            return adjusted_value
        ratio = adjusted_value / original_value
        clamped_ratio = float(np.clip(ratio, COMBINED_RATIO_FLOOR, COMBINED_RATIO_CEILING))
        if abs(clamped_ratio - ratio) > 1e-9:
            log.append(
                f"{side_label}: combined {metric_label} multiplier of x{ratio:.2f} "
                f"clamped to x{clamped_ratio:.2f} (realism cap - see stacking note)"
            )
            return original_value * clamped_ratio
        return adjusted_value

    home.attack = clamp_ratio(home.attack, home_attack, "Home", "attack")
    home.defense = clamp_ratio(home.defense, home_defense, "Home", "defense")
    away.attack = clamp_ratio(away.attack, away_attack, "Away", "attack")
    away.defense = clamp_ratio(away.defense, away_defense, "Away", "defense")
    vol = float(np.clip(vol, base_volatility * 0.5, base_volatility * 1.8))

    return home, away, vol, log


# ---------------------------------------------------------------------------
# Section 5, Engine A: Dixon-Coles Poisson model with data-fitted rho
# ---------------------------------------------------------------------------

def fit_rho(settled_df: pd.DataFrame) -> float:
    """Fits rho by grid search: pick the rho that makes the tau-adjusted
    Poisson-Poisson grid's predicted frequency of the four low-score
    cells (0-0, 1-0, 0-1, 1-1) match what's ACTUALLY observed in this
    league's settled matches, rather than assuming a fixed constant."""
    if settled_df.empty or len(settled_df) < MIN_SAMPLE_ROWS:
        return 0.0

    lam_h = float(settled_df["home_goals"].mean())
    lam_a = float(settled_df["away_goals"].mean())
    if lam_h <= 0 or lam_a <= 0:
        return 0.0

    total = len(settled_df)
    observed = {
        (0, 0): float(((settled_df["home_goals"] == 0) & (settled_df["away_goals"] == 0)).sum()) / total,
        (1, 0): float(((settled_df["home_goals"] == 1) & (settled_df["away_goals"] == 0)).sum()) / total,
        (0, 1): float(((settled_df["home_goals"] == 0) & (settled_df["away_goals"] == 1)).sum()) / total,
        (1, 1): float(((settled_df["home_goals"] == 1) & (settled_df["away_goals"] == 1)).sum()) / total,
    }

    best_rho, best_error = 0.0, float("inf")
    for rho_candidate in np.arange(-0.25, 0.251, 0.01):
        error = 0.0
        for (x, y), obs_p in observed.items():
            base = scipy_poisson.pmf(x, lam_h) * scipy_poisson.pmf(y, lam_a)
            pred_p = base * dixon_coles_tau(x, y, lam_h, lam_a, rho_candidate)
            error += (pred_p - obs_p) ** 2
        if error < best_error:
            best_error = error
            best_rho = float(rho_candidate)
    return best_rho


def dixon_coles_tau(x: int, y: int, lam_h: float, lam_a: float, rho: float) -> float:
    if x == 0 and y == 0:
        return 1 - lam_h * lam_a * rho
    if x == 0 and y == 1:
        return 1 + lam_h * rho
    if x == 1 and y == 0:
        return 1 + lam_a * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def build_score_matrix(lam_h: float, lam_a: float, rho: float, goal_cap: int = GOAL_CAP) -> np.ndarray:
    """Builds the Dixon-Coles-adjusted score matrix and normalizes it so
    the whole grid sums to exactly 1.0 (100%)."""
    matrix = np.zeros((goal_cap + 1, goal_cap + 1))
    for x in range(goal_cap + 1):
        px = scipy_poisson.pmf(x, lam_h)
        for y in range(goal_cap + 1):
            py = scipy_poisson.pmf(y, lam_a)
            matrix[x, y] = px * py * dixon_coles_tau(x, y, lam_h, lam_a, rho)
    matrix = np.clip(matrix, 0, None)
    total = matrix.sum()
    if total > 0:
        matrix = matrix / total
    return matrix


# ---------------------------------------------------------------------------
# Corners engine - a genuinely separate signal from goals (a team can
# dominate corners without scoring, or vice versa), so this gets its OWN
# baseline/lambda estimation rather than reusing the goals Dixon-Coles
# machinery. No rho-style low-count correlation adjustment is applied here
# (that correction exists specifically for football's well-documented
# 0-0/1-0/1-1 clustering, which has no established equivalent for corner
# counts) - home and away corners are modelled as independent Poisson
# variables, which is the standard, defensible baseline assumption for a
# count stat like this without a specific reason to complicate it further.
# ---------------------------------------------------------------------------

CORNER_GOAL_CAP = 20  # corners run higher than goals, so the grid needs more headroom than GOAL_CAP


def corners_league_baseline(settled_df: pd.DataFrame) -> "dict | None":
    """Thin, corners-specific wrapper around compute_league_metric_baseline
    - kept separate from the blended set-piece engine's baseline (which
    mixes corner_kicks with fouled_in_final_third/accurate_crosses/aerial
    duels for the ATTACK-STRENGTH blend) because the corners MARKET itself
    should be predicted from corner history specifically, not a blended
    attacking-style composite."""
    return compute_league_metric_baseline(settled_df, ["corner_kicks"])


def corners_team_profile(league_df, team, venue, half_life_days, reference_date, use_match_index=False, half_life_matches=FROZEN_HALF_LIFE_MATCHES):
    return team_metric_profile(league_df, team, venue, half_life_days, reference_date, ["corner_kicks"], use_match_index, half_life_matches)


def expected_corners(
    home_profile: "MetricProfile | None", away_profile: "MetricProfile | None",
    league_baseline: "dict | None",
) -> "tuple[float, float] | None":
    """Same attack*defense*baseline structure as expected_goals(), just
    for corner counts instead of goals - home_corner_strength (this
    team's own corner-winning rate vs. league average) combined with the
    opponent's corners-conceded rate, scaled by the league's average
    home/away corner count. Returns None if corner_kicks isn't in this
    dataset, or either team's sample is too thin to trust (same
    MIN_SAMPLE_ROWS floor used everywhere else)."""
    if league_baseline is None or home_profile is None or away_profile is None:
        return None
    if home_profile.n_matches < MIN_SAMPLE_ROWS or away_profile.n_matches < MIN_SAMPLE_ROWS:
        return None

    home_attack = metric_strength(home_profile, league_baseline, "home", ["corner_kicks"], (1.0,), direction="for")
    away_defense = metric_strength(away_profile, league_baseline, "away", ["corner_kicks"], (1.0,), direction="against")
    away_attack = metric_strength(away_profile, league_baseline, "away", ["corner_kicks"], (1.0,), direction="for")
    home_defense = metric_strength(home_profile, league_baseline, "home", ["corner_kicks"], (1.0,), direction="against")
    if None in (home_attack, away_defense, away_attack, home_defense):
        return None

    league_avg_home_corners = league_baseline["corner_kicks"]["home_for"]
    league_avg_away_corners = league_baseline["corner_kicks"]["away_for"]

    lambda_home_corners = league_avg_home_corners * home_attack * away_defense
    lambda_away_corners = league_avg_away_corners * away_attack * home_defense
    return max(lambda_home_corners, 0.5), max(lambda_away_corners, 0.5)


def corner_probs_analytical(lambda_home_corners: float, lambda_away_corners: float, cap: int = CORNER_GOAL_CAP) -> dict:
    """Exact (not sampled) corners market probabilities from two
    independent Poisson variables. Over/Under totals use the fact that
    the SUM of two independent Poissons is itself Poisson with the
    combined rate (an exact identity, not an approximation) - so those
    come straight from scipy_poisson.cdf. 'Most corners' (home/away/
    equal) needs the actual joint grid, since it depends on the two
    counts separately, not just their sum - built the same way
    build_score_matrix() builds the goals grid, minus any rho-style
    correlation term (see module-level note above)."""
    combined_lambda = lambda_home_corners + lambda_away_corners
    over = {line: float(1 - scipy_poisson.cdf(math.floor(line), combined_lambda)) for line in CORNERS_LINES}
    under = {line: float(scipy_poisson.cdf(math.floor(line), combined_lambda)) for line in CORNERS_LINES}

    idx = np.arange(cap + 1)
    home_pmf = scipy_poisson.pmf(idx, lambda_home_corners)
    away_pmf = scipy_poisson.pmf(idx, lambda_away_corners)
    joint = np.outer(home_pmf, away_pmf)
    joint = joint / joint.sum()  # renormalise for the truncation at `cap`
    home_grid, away_grid = np.meshgrid(idx, idx, indexing="ij")
    most_home = float(joint[home_grid > away_grid].sum())
    most_away = float(joint[home_grid < away_grid].sum())
    most_equal = float(joint[home_grid == away_grid].sum())

    result = {}
    for line in CORNERS_LINES:
        result[f"Corners Over {line}"] = over[line]
        result[f"Corners Under {line}"] = under[line]
    result["Most Corners Home"] = most_home
    result["Most Corners Away"] = most_away
    result["Most Corners Equal"] = most_equal
    return result


def corner_probs_simulated(lambda_home_corners: float, lambda_away_corners: float, iterations: int = MC_ITERATIONS, rng=None) -> dict:
    """A genuinely independent stochastic cross-check for corners,
    mirroring how monte_carlo_simulate() draws goals - real sampling
    noise, not a copy of the analytical numbers, so a corners
    convergence score means the same thing it does for every other
    market (two different computations agreeing, not one echoing itself)."""
    rng = rng or np.random.default_rng()
    home_corners = rng.poisson(lambda_home_corners, size=iterations)
    away_corners = rng.poisson(lambda_away_corners, size=iterations)
    total = home_corners + away_corners

    result = {}
    for line in CORNERS_LINES:
        result[f"Corners Over {line}"] = float(np.mean(total > line))
        result[f"Corners Under {line}"] = float(np.mean(total < line))
    result["Most Corners Home"] = float(np.mean(home_corners > away_corners))
    result["Most Corners Away"] = float(np.mean(home_corners < away_corners))
    result["Most Corners Equal"] = float(np.mean(home_corners == away_corners))
    return result


# ---------------------------------------------------------------------------
# Section 5, Engine B: 10,000-iteration Monte Carlo simulator
# ---------------------------------------------------------------------------

def monte_carlo_simulate(
    lam_h: float, lam_a: float, volatility_dampener: float = 1.0, iterations: int = MC_ITERATIONS,
    rng=None,
):
    """Draws directly from the raw expected-goal rates via
    np.random.poisson for `iterations` mock matches - deliberately
    bypassing the Dixon-Coles matrix entirely (per spec: "bypasses the
    post-processed matrix entirely to prevent flattening errors"). A
    volatility_dampener != 1.0 jitters the lambda per-simulation (rather
    than the goal draw itself) via a clipped normal multiplier, which is
    how the Section 6 volatility-affecting toggles and Core Parameter B's
    auto-calibrated dampener actually widen or narrow the simulated
    spread of results."""
    rng = rng or np.random.default_rng()
    noise_std = max(0.0, (volatility_dampener - 1.0)) + 0.10  # always some baseline match-to-match noise
    home_jitter = np.clip(rng.normal(1.0, noise_std, size=iterations), 0.05, None)
    away_jitter = np.clip(rng.normal(1.0, noise_std, size=iterations), 0.05, None)
    home_goals = rng.poisson(lam_h * home_jitter)
    away_goals = rng.poisson(lam_a * away_jitter)
    return home_goals, away_goals


# ---------------------------------------------------------------------------
# Section 8: 22-market probability extraction
# ---------------------------------------------------------------------------

MARKET_LIST = [
    "Home Win", "Draw", "Away Win",
    "Double Chance 1X", "Double Chance 12", "Double Chance X2",
    "Draw No Bet Home", "Draw No Bet Away",
    "Over 1.5 Goals", "Under 1.5 Goals",
    "Over 2.5 Goals", "Under 2.5 Goals",
    "Over 3.5 Goals", "Under 3.5 Goals",
    "BTTS - Yes", "BTTS - No",
    "Home Clean Sheet", "Away Clean Sheet",
    "Home Win to Nil", "Away Win to Nil",
    "Asian Handicap Home -1.5", "Asian Handicap Away +1.5",
    "Asian Handicap Home +1.5", "Asian Handicap Away -1.5",
] + MULTI_SCORE_MARKETS + CORNERS_MARKETS + DOUBLE_CHANCE_BTTS_MARKETS + DOUBLE_CHANCE_TOTAL_GOALS_MARKETS + MATCH_RESULT_TOTAL_GOALS_MARKETS + TOTAL_GOALS_BTTS_MARKETS + GOAL_RANGE_MARKETS + MATCH_RESULT_BTTS_MARKETS + EXTRA_LINE_COMBO_MARKETS + GOAL_STRUCTURE_MARKETS
assert len(MARKET_LIST) == 103, len(MARKET_LIST)  # 103 distinct markets after removing 11 duplicates and adding 11 new ones
assert not set(REDUNDANT_WRAPPERS) & set(MARKET_LIST), 'a removed duplicate is still in the market list'

# ---------------------------------------------------------------------------
# Display groups for the odds-input screen (each group becomes one drop-down). Every market belongs to exactly one.
# ---------------------------------------------------------------------------
MARKET_GROUP_ORDER = [
    "Match result", "Handicaps", "Total goals", "Both teams to score & shutouts", "Team goals & winning margin",
    "Multi Score", "Corners", "Result + BTTS", "Total goals + BTTS",
    "Result + total goals (1.5 line)", "Result + total goals (2.5 line)", "Result + total goals (3.5 line)",
]


def market_group(name: str) -> str:
    """Which odds-screen group a market belongs to (rule-based, so a market added later lands in the right place)."""
    if name.startswith(("Corners", "Most Corners")):
        return "Corners"
    if name.startswith("Multi Score"):
        return "Multi Score"
    if name.startswith("Asian Handicap"):
        return "Handicaps"
    if name in ("Home Win", "Draw", "Away Win") or name.startswith(("Double Chance", "Draw No Bet")):
        return "Match result"
    if name.startswith(("Home Team", "Away Team", "Home Wins by", "Away Wins by")):
        return "Team goals & winning margin"
    if name.startswith(("BTTS", "Home Clean", "Away Clean", "Home Win to Nil", "Away Win to Nil")):
        return "Both teams to score & shutouts"
    if re.match(r"^(Over|Under) \d\.\d Goals$", name) or name.startswith(("Goal Range", "Total Goals", "Exactly")):
        return "Total goals"
    if " & " in name:
        left, right = name.split(" & ", 1)
        if left.startswith(("Over", "Under")):
            return "Total goals + BTTS"
        line = re.search(r"(Over|Under) (\d\.\d)$", right)
        if line:
            return f"Result + total goals ({line.group(2)} line)"
        return "Result + BTTS"
    return "Other"


def grouped_markets() -> dict:
    """{group title: [markets in MARKET_LIST order]} in screen order; empty groups are dropped."""
    out = {title: [] for title in MARKET_GROUP_ORDER}
    for m in MARKET_LIST:
        out.setdefault(market_group(m), []).append(m)
    return {k: v for k, v in out.items() if v}
assert len(set(MARKET_LIST)) == len(MARKET_LIST)
# original 22 + Draw No Bet (2) + Multi Score (6 groups + Other) + Corners (3 lines x2 + 3 most-corners)
# + DC+BTTS (6) + DC+TotalGoals (6) + 1X2+TotalGoals (6) + TotalGoals+BTTS (4) + Goal Range (4) + 1X2+BTTS (6) = 72, plus the 1.5/3.5-line total-goals combos (see EXTRA_LINE_COMBO_MARKETS)


def market_probs_from_matrix(matrix: np.ndarray, corner_lambdas: "tuple | None" = None) -> dict:
    """Every market probability computed by literally summing the
    correct cells of the real Dixon-Coles grid - no shortcuts."""
    size = matrix.shape[0]
    idx = np.arange(size)
    home_grid, away_grid = np.meshgrid(idx, idx, indexing="ij")

    home_win = matrix[home_grid > away_grid].sum()
    draw = matrix[home_grid == away_grid].sum()
    away_win = matrix[home_grid < away_grid].sum()

    total_goals = home_grid + away_grid
    over = {t: matrix[total_goals > t].sum() for t in (1.5, 2.5, 3.5)}
    under = {t: matrix[total_goals < t].sum() for t in (1.5, 2.5, 3.5)}

    btts_yes = matrix[(home_grid >= 1) & (away_grid >= 1)].sum()
    btts_no = 1 - btts_yes

    home_clean_sheet = matrix[away_grid == 0].sum()
    away_clean_sheet = matrix[home_grid == 0].sum()

    home_win_to_nil = matrix[(home_grid > away_grid) & (away_grid == 0)].sum()
    away_win_to_nil = matrix[(away_grid > home_grid) & (home_grid == 0)].sum()

    ah_home_minus_1_5 = matrix[(home_grid - away_grid) > 1.5].sum()
    ah_away_plus_1_5 = 1 - ah_home_minus_1_5
    ah_away_minus_1_5 = matrix[(away_grid - home_grid) > 1.5].sum()
    ah_home_plus_1_5 = 1 - ah_away_minus_1_5

    # Draw No Bet - conditional on the match NOT being a draw (the stake
    # is void, not lost, on an actual draw), so these are home_win/away_win
    # renormalised over just the non-draw probability mass.
    decisive_mass = home_win + away_win
    dnb_home = home_win / decisive_mass if decisive_mass > 0 else 0.5
    dnb_away = away_win / decisive_mass if decisive_mass > 0 else 0.5

    # Multi Score - each market is the SUM of several literal cells of the real grid (one bet, several
    # scorelines), plus "Other" catching every scoreline outside all groups (the whole set sums to exactly 1).
    multi_score_probs = {}
    named_total = 0.0
    for name, cells in MULTI_SCORE_GROUPS:
        p = float(sum(matrix[h, a] for h, a in cells if h < size and a < size))
        multi_score_probs[name] = p
        named_total += p
    multi_score_probs[MULTI_SCORE_OTHER] = max(0.0, 1.0 - named_total)

    # Combo markets - boolean masks reused across all four combo groups below
    # (Double Chance x BTTS/Total Goals doesn't sum to 1 across its own 6
    # markets, since 1X/12/X2 overlap each other - only Goal Range,
    # 1X2+Total Goals, and Total Goals+BTTS are genuine mutually-exclusive-
    # and-exhaustive partitions; see CALIBRATION_COMPLEMENTARY_GROUPS below
    # for exactly which combo groups get the sum-to-1 renormalisation).
    btts_yes_mask = (home_grid >= 1) & (away_grid >= 1)
    over_25_mask = total_goals > 2.5
    dc_1x_mask = (home_grid > away_grid) | (home_grid == away_grid)
    dc_12_mask = (home_grid > away_grid) | (home_grid < away_grid)
    dc_x2_mask = (home_grid == away_grid) | (home_grid < away_grid)
    home_win_mask, draw_mask, away_win_mask = home_grid > away_grid, home_grid == away_grid, home_grid < away_grid

    combo_probs = {
        "1X & BTTS Yes": matrix[dc_1x_mask & btts_yes_mask].sum(),
        "12 & BTTS Yes": matrix[dc_12_mask & btts_yes_mask].sum(), "12 & BTTS No": matrix[dc_12_mask & ~btts_yes_mask].sum(),
        "X2 & BTTS Yes": matrix[dc_x2_mask & btts_yes_mask].sum(),
        "1X & Over 2.5": matrix[dc_1x_mask & over_25_mask].sum(), "1X & Under 2.5": matrix[dc_1x_mask & ~over_25_mask].sum(),
        "12 & Over 2.5": matrix[dc_12_mask & over_25_mask].sum(), "12 & Under 2.5": matrix[dc_12_mask & ~over_25_mask].sum(),
        "X2 & Over 2.5": matrix[dc_x2_mask & over_25_mask].sum(), "X2 & Under 2.5": matrix[dc_x2_mask & ~over_25_mask].sum(),
        "Home & Over 2.5": matrix[home_win_mask & over_25_mask].sum(), "Home & Under 2.5": matrix[home_win_mask & ~over_25_mask].sum(),
        "Draw & Over 2.5": matrix[draw_mask & over_25_mask].sum(),
        "Away & Over 2.5": matrix[away_win_mask & over_25_mask].sum(), "Away & Under 2.5": matrix[away_win_mask & ~over_25_mask].sum(),
        "Over 2.5 & BTTS Yes": matrix[over_25_mask & btts_yes_mask].sum(), "Over 2.5 & BTTS No": matrix[over_25_mask & ~btts_yes_mask].sum(),
        "Under 2.5 & BTTS Yes": matrix[~over_25_mask & btts_yes_mask].sum(), "Under 2.5 & BTTS No": matrix[~over_25_mask & ~btts_yes_mask].sum(),
        "Home & BTTS Yes": matrix[home_win_mask & btts_yes_mask].sum(),
        "Draw & BTTS Yes": matrix[draw_mask & btts_yes_mask].sum(), "Draw & BTTS No": matrix[draw_mask & ~btts_yes_mask].sum(),
        "Away & BTTS Yes": matrix[away_win_mask & btts_yes_mask].sum(),
        "Goal Range 0-1": matrix[(total_goals >= 0) & (total_goals <= 1)].sum(),
        "Goal Range 2-3": matrix[(total_goals >= 2) & (total_goals <= 3)].sum(),
        "Goal Range 4-5": matrix[(total_goals >= 4) & (total_goals <= 5)].sum(),
        "Goal Range 6+": matrix[total_goals >= 6].sum(),
    }
    combo_probs.update(_extra_line_combo_values(
        {"home": home_win_mask, "draw": draw_mask, "away": away_win_mask, "dc_1x": dc_1x_mask, "dc_12": dc_12_mask,
         "dc_x2": dc_x2_mask, "btts": btts_yes_mask},
        total_goals, lambda m: matrix[m].sum()))
    combo_probs.update(_goal_structure_values(home_grid, away_grid, lambda m: matrix[m].sum()))

    result = {
        "Home Win": home_win, "Draw": draw, "Away Win": away_win,
        "Double Chance 1X": home_win + draw,
        "Double Chance 12": home_win + away_win,
        "Double Chance X2": draw + away_win,
        "Draw No Bet Home": dnb_home, "Draw No Bet Away": dnb_away,
        "Over 1.5 Goals": over[1.5], "Under 1.5 Goals": under[1.5],
        "Over 2.5 Goals": over[2.5], "Under 2.5 Goals": under[2.5],
        "Over 3.5 Goals": over[3.5], "Under 3.5 Goals": under[3.5],
        "BTTS - Yes": btts_yes, "BTTS - No": btts_no,
        "Home Clean Sheet": home_clean_sheet, "Away Clean Sheet": away_clean_sheet,
        "Home Win to Nil": home_win_to_nil, "Away Win to Nil": away_win_to_nil,
        "Asian Handicap Home -1.5": ah_home_minus_1_5, "Asian Handicap Away +1.5": ah_away_plus_1_5,
        "Asian Handicap Home +1.5": ah_home_plus_1_5, "Asian Handicap Away -1.5": ah_away_minus_1_5,
        **multi_score_probs,
        **combo_probs,
    }
    # Corners keys are ONLY added when corner_lambdas is actually available
    # (real corner_kicks data, enough sample to trust) - never a fabricated
    # neutral guess. Every consumer of MARKET_LIST elsewhere already treats
    # an absent key as "not applicable" (dict.get() with a skip/0.0
    # fallback), so omitting these keys is the correct way to signal
    # "not computable here", not a bug.
    if corner_lambdas is not None:
        result.update(corner_probs_analytical(corner_lambdas[0], corner_lambdas[1]))
    return result


def market_probs_from_simulation(home_goals: np.ndarray, away_goals: np.ndarray, corner_lambdas: "tuple | None" = None, rng=None) -> dict:
    """Same markets as market_probs_from_matrix (goals/1X2/BTTS/O-U/
    clean sheets/AH/multi score/combo markets/goal range - everything
    in MARKET_LIST except corners, which is added separately by the
    caller when corner_lambdas is supplied), computed as raw frequencies
    across the Monte Carlo simulation array instead of matrix cell sums -
    an entirely independent probability path from the matrix above,
    which is what makes the convergence score a genuine cross-check
    rather than comparing a method to itself. Every key here must match
    market_probs_from_matrix's keys, or build_valuation_sheet silently
    reads a missing key as mc_prob=0.0 (see the combo_probs block below
    for why that's a real, previously-hit bug, not just a theoretical risk)."""
    home_win = np.mean(home_goals > away_goals)
    draw = np.mean(home_goals == away_goals)
    away_win = np.mean(home_goals < away_goals)
    total_goals = home_goals + away_goals

    over = {t: np.mean(total_goals > t) for t in (1.5, 2.5, 3.5)}
    under = {t: np.mean(total_goals < t) for t in (1.5, 2.5, 3.5)}

    btts_yes = np.mean((home_goals >= 1) & (away_goals >= 1))
    btts_no = 1 - btts_yes

    home_clean_sheet = np.mean(away_goals == 0)
    away_clean_sheet = np.mean(home_goals == 0)

    home_win_to_nil = np.mean((home_goals > away_goals) & (away_goals == 0))
    away_win_to_nil = np.mean((away_goals > home_goals) & (home_goals == 0))

    ah_home_minus_1_5 = np.mean((home_goals - away_goals) > 1.5)
    ah_away_plus_1_5 = 1 - ah_home_minus_1_5
    ah_away_minus_1_5 = np.mean((away_goals - home_goals) > 1.5)
    ah_home_plus_1_5 = 1 - ah_away_minus_1_5

    decisive_mask = home_goals != away_goals
    decisive_count = int(np.sum(decisive_mask))
    dnb_home = float(np.mean(home_goals[decisive_mask] > away_goals[decisive_mask])) if decisive_count > 0 else 0.5
    dnb_away = 1 - dnb_home if decisive_count > 0 else 0.5

    multi_score_probs = {}
    named_total = 0.0
    for name, cells in MULTI_SCORE_GROUPS:
        in_group = np.zeros(len(home_goals), dtype=bool)
        for h, a in cells:
            in_group |= (home_goals == h) & (away_goals == a)
        p = float(np.mean(in_group))
        multi_score_probs[name] = p
        named_total += p
    multi_score_probs[MULTI_SCORE_OTHER] = max(0.0, 1.0 - named_total)

    # Combo markets - SAME boolean-mask logic as market_probs_from_matrix's
    # combo_probs block, just evaluated against the simulation array via
    # np.mean() instead of summing matrix cells. This MUST stay in sync with
    # the matrix version - previously it didn't, and every one of these 26
    # keys silently fell back to mc_prob=0.0 in build_valuation_sheet
    # (dict.get(market, 0.0)), which doesn't crash but tanks convergence_score
    # to near-zero for any real dc_prob, silently keeping every one of these
    # markets below the confidence gate regardless of actual model confidence.
    btts_yes_mask = (home_goals >= 1) & (away_goals >= 1)
    over_25_mask = total_goals > 2.5
    dc_1x_mask = (home_goals > away_goals) | (home_goals == away_goals)
    dc_12_mask = (home_goals > away_goals) | (home_goals < away_goals)
    dc_x2_mask = (home_goals == away_goals) | (home_goals < away_goals)
    home_win_mask, draw_mask, away_win_mask = home_goals > away_goals, home_goals == away_goals, home_goals < away_goals

    combo_probs = {
        "1X & BTTS Yes": np.mean(dc_1x_mask & btts_yes_mask),
        "12 & BTTS Yes": np.mean(dc_12_mask & btts_yes_mask), "12 & BTTS No": np.mean(dc_12_mask & ~btts_yes_mask),
        "X2 & BTTS Yes": np.mean(dc_x2_mask & btts_yes_mask),
        "1X & Over 2.5": np.mean(dc_1x_mask & over_25_mask), "1X & Under 2.5": np.mean(dc_1x_mask & ~over_25_mask),
        "12 & Over 2.5": np.mean(dc_12_mask & over_25_mask), "12 & Under 2.5": np.mean(dc_12_mask & ~over_25_mask),
        "X2 & Over 2.5": np.mean(dc_x2_mask & over_25_mask), "X2 & Under 2.5": np.mean(dc_x2_mask & ~over_25_mask),
        "Home & Over 2.5": np.mean(home_win_mask & over_25_mask), "Home & Under 2.5": np.mean(home_win_mask & ~over_25_mask),
        "Draw & Over 2.5": np.mean(draw_mask & over_25_mask),
        "Away & Over 2.5": np.mean(away_win_mask & over_25_mask), "Away & Under 2.5": np.mean(away_win_mask & ~over_25_mask),
        "Over 2.5 & BTTS Yes": np.mean(over_25_mask & btts_yes_mask), "Over 2.5 & BTTS No": np.mean(over_25_mask & ~btts_yes_mask),
        "Under 2.5 & BTTS Yes": np.mean(~over_25_mask & btts_yes_mask), "Under 2.5 & BTTS No": np.mean(~over_25_mask & ~btts_yes_mask),
        "Home & BTTS Yes": np.mean(home_win_mask & btts_yes_mask),
        "Draw & BTTS Yes": np.mean(draw_mask & btts_yes_mask), "Draw & BTTS No": np.mean(draw_mask & ~btts_yes_mask),
        "Away & BTTS Yes": np.mean(away_win_mask & btts_yes_mask),
        "Goal Range 0-1": np.mean((total_goals >= 0) & (total_goals <= 1)),
        "Goal Range 2-3": np.mean((total_goals >= 2) & (total_goals <= 3)),
        "Goal Range 4-5": np.mean((total_goals >= 4) & (total_goals <= 5)),
        "Goal Range 6+": np.mean(total_goals >= 6),
    }
    combo_probs.update(_extra_line_combo_values(
        {"home": home_win_mask, "draw": draw_mask, "away": away_win_mask, "dc_1x": dc_1x_mask, "dc_12": dc_12_mask,
         "dc_x2": dc_x2_mask, "btts": btts_yes_mask},
        total_goals, lambda m: np.mean(m)))
    combo_probs.update(_goal_structure_values(home_goals, away_goals, lambda m: np.mean(m)))
    combo_probs = {k: float(v) for k, v in combo_probs.items()}

    result = {
        "Home Win": home_win, "Draw": draw, "Away Win": away_win,
        "Double Chance 1X": home_win + draw,
        "Double Chance 12": home_win + away_win,
        "Double Chance X2": draw + away_win,
        "Draw No Bet Home": dnb_home, "Draw No Bet Away": dnb_away,
        "Over 1.5 Goals": over[1.5], "Under 1.5 Goals": under[1.5],
        "Over 2.5 Goals": over[2.5], "Under 2.5 Goals": under[2.5],
        "Over 3.5 Goals": over[3.5], "Under 3.5 Goals": under[3.5],
        "BTTS - Yes": btts_yes, "BTTS - No": btts_no,
        "Home Clean Sheet": home_clean_sheet, "Away Clean Sheet": away_clean_sheet,
        "Home Win to Nil": home_win_to_nil, "Away Win to Nil": away_win_to_nil,
        "Asian Handicap Home -1.5": ah_home_minus_1_5, "Asian Handicap Away +1.5": ah_away_plus_1_5,
        "Asian Handicap Home +1.5": ah_home_plus_1_5, "Asian Handicap Away -1.5": ah_away_minus_1_5,
        **multi_score_probs,
        **combo_probs,
    }
    if corner_lambdas is not None:
        # A genuine independent sample (see corner_probs_simulated), not a
        # copy of the analytical numbers - keeps the corners convergence
        # score meaningful, same as every other market on the sheet.
        result.update(corner_probs_simulated(corner_lambdas[0], corner_lambdas[1], iterations=len(home_goals), rng=rng))
    return result


# ---------------------------------------------------------------------------
# Section 8: EV, convergence, fair odds, verdicts, recommended action
# ---------------------------------------------------------------------------

VOLATILITY_LOW, VOLATILITY_HIGH = 0.85, 1.15
EV_ELITE_THRESHOLD = 0.03  # +3.0%


def convergence_score(p_dc: float, p_mc: float) -> float:
    """1.0 = the two engines agree perfectly, 0.0 = maximally apart."""
    return max(0.0, 1.0 - abs(p_dc - p_mc) * 2)


def consensus_prob(p_dc: float, p_mc: float, mode: "str | None" = None) -> float:
    """The ONE probability used for EV, fair odds, confidence gates, Kelly and parlays.

    'min'  (default) - the probability BOTH engines support. EV computed from it is the edge
                       both engines agree on, so a market only shows value when Dixon-Coles AND
                       Monte Carlo each see it. (Taking the HIGHER of the two reports an edge
                       that only one engine sees - a different statement.)
    'mean'           - average of the two; keeps every market group summing to 100% when each
                       engine's own set does.
    'max'            - most generous engine (old behaviour; builds in an upward bias)."""
    mode = (mode or EV_ENGINE_MODE).lower()
    if mode == "max":
        return max(p_dc, p_mc)
    if mode == "mean":
        return (p_dc + p_mc) / 2.0
    return min(p_dc, p_mc)


def fair_odds(p_dc: float, p_mc: float, mode: "str | None" = None) -> float:
    p = consensus_prob(p_dc, p_mc, mode)
    if p <= 0:
        return float("inf")
    return 1.0 / p


def expected_value(model_prob: float, bookmaker_odds: float) -> float:
    return model_prob * bookmaker_odds - 1.0


def volatility_tier(vol_dampener: float) -> str:
    if vol_dampener < VOLATILITY_LOW:
        return "Low Chaos"
    if vol_dampener > VOLATILITY_HIGH:
        return "High Chaos"
    return "Balanced"


def value_verdict(ev: float) -> str:
    return "🔥 ELITE VALUE" if ev > EV_ELITE_THRESHOLD else "⚠️ HIGH-JUICE TRAP"


def recommended_action(ev: float, convergence: float, tier: str) -> str:
    """A rule-based synthesis of edge size, engine agreement, and
    volatility - never a market probability itself, just a plain-English
    action label built on top of numbers that were already computed."""
    if ev <= 0:
        return "🚫 AVOID - NO EDGE"
    if convergence < 0.5:
        return "❓ LOW CONFIDENCE - ENGINES DISAGREE"
    if ev > EV_ELITE_THRESHOLD and convergence >= 0.75:
        if tier == "High Chaos":
            return "🔥 STRONG BET (small stake - high chaos)"
        return "🔥 STRONG BET"
    if ev > EV_ELITE_THRESHOLD:
        return "✅ VALUE BET"
    return "🟡 MARGINAL - SMALL STAKE ONLY"


@dataclass
class MarketRow:
    market: str
    bookmaker_odds: float
    dc_prob: float
    mc_prob: float
    convergence: float
    fair_odds: float
    ev: float
    volatility_tier: str
    verdict: str
    recommended_action: str
    prob_used: float = 0.0          # the consensus probability EV/fair odds/confidence were computed from
    ev_dc: float = float("-inf")    # EV if only Dixon-Coles were believed
    ev_mc: float = float("-inf")    # EV if only Monte Carlo were believed


def build_valuation_sheet(
    dc_probs: dict, mc_probs: dict, bookmaker_odds: dict, vol_dampener: float, ev_mode: "str | None" = None,
):
    tier = volatility_tier(vol_dampener)
    rows = []
    for market in MARKET_LIST:
        p_dc = dc_probs.get(market, 0.0)
        p_mc = mc_probs.get(market, 0.0)
        odds = bookmaker_odds.get(market, 0.0)
        conv = convergence_score(p_dc, p_mc)
        p_used = consensus_prob(p_dc, p_mc, ev_mode)
        f_odds = fair_odds(p_dc, p_mc, ev_mode)
        ev = expected_value(p_used, odds) if odds > 0 else float("-inf")
        ev_dc = expected_value(p_dc, odds) if odds > 0 else float("-inf")
        ev_mc = expected_value(p_mc, odds) if odds > 0 else float("-inf")
        rows.append(MarketRow(
            market=market, bookmaker_odds=odds, dc_prob=p_dc, mc_prob=p_mc,
            convergence=conv, fair_odds=f_odds, ev=ev, volatility_tier=tier,
            verdict=value_verdict(ev) if odds > 0 else "-",
            recommended_action=recommended_action(ev, conv, tier) if odds > 0 else "ENTER ODDS",
            prob_used=p_used, ev_dc=ev_dc, ev_mc=ev_mc,
        ))
    return rows


# ---------------------------------------------------------------------------
# Kelly Criterion & parlay combination
# ---------------------------------------------------------------------------

def kelly_stake_fraction(model_prob: float, bookmaker_odds: float, kelly_multiplier: float) -> float:
    b = bookmaker_odds - 1.0
    if b <= 0:
        return 0.0
    edge = model_prob * bookmaker_odds - 1.0
    full_kelly = edge / b
    return max(0.0, full_kelly) * kelly_multiplier


def round_to_nearest(amount: float, denomination: float = 10.0) -> float:
    return round(amount / denomination) * denomination


def _leg_value(leg, name):
    """Supports either a MarketRow object (attribute access) or a plain
    dict with the same field names - the cross-league bet slip stores
    legs as dicts (since they need to persist after you've moved on to a
    different fixture/league, long after the original MarketRow objects
    for that match are gone), while the single-fixture parlay builder
    still uses MarketRow objects directly."""
    return getattr(leg, name) if hasattr(leg, name) else leg[name]


def combine_parlay_legs(legs, ev_mode: "str | None" = None):
    """Returns (combined_odds, combined_model_probability). Works with a
    mix of MarketRow objects and/or plain dicts, in any combination.
    Each leg uses the same consensus probability as the valuation sheet. NOTE: legs are
    multiplied as if independent - legs from the SAME match are correlated (e.g. Home Win +
    Over 2.5), so treat same-match parlay probabilities as approximate."""
    combined_odds = 1.0
    combined_prob = 1.0
    for leg in legs:
        combined_odds *= _leg_value(leg, "bookmaker_odds")
        combined_prob *= consensus_prob(_leg_value(leg, "dc_prob"), _leg_value(leg, "mc_prob"), ev_mode)
    return combined_odds, combined_prob


# ---------------------------------------------------------------------------
# Section 10: Deserved Points (xPts) and season Monte Carlo forecast
# ---------------------------------------------------------------------------

BOX_TOUCH_WEIGHT = 0.015


def compute_standings_table(settled_df: pd.DataFrame) -> pd.DataFrame:
    """The classic, literal league table - Played/Won/Drawn/Lost/GF/GA/
    GD/Points, computed directly from real match results. NOT the same
    thing as compute_xpts_table() below, which is model-based (deserved
    points from territory dominance) - this is the actual, real
    standings, with the standard tiebreak order (Points, then GD, then
    GF). This is the table that gets archived at end of season - see
    the app's season-archive functions, which snapshot exactly this
    table's output alongside a season label and date range, so a
    league's final table stays retrievable even after new-season
    matches start accumulating in the same underlying dataset."""
    columns = ["Position", "Team", "Played", "Won", "Drawn", "Lost", "GF", "GA", "GD", "Points"]
    if settled_df.empty:
        return pd.DataFrame(columns=columns)

    records: dict[str, dict] = {}

    def ensure(team: str) -> dict:
        if team not in records:
            records[team] = {"Team": team, "Played": 0, "Won": 0, "Drawn": 0, "Lost": 0, "GF": 0, "GA": 0}
        return records[team]

    for _, row in settled_df.iterrows():
        home, away = row.get("home_team"), row.get("away_team")
        hg, ag = row.get("home_goals"), row.get("away_goals")
        if pd.isna(hg) or pd.isna(ag) or not isinstance(home, str) or not isinstance(away, str):
            continue
        hg, ag = int(hg), int(ag)
        h, a = ensure(home), ensure(away)
        h["Played"] += 1
        a["Played"] += 1
        h["GF"] += hg
        h["GA"] += ag
        a["GF"] += ag
        a["GA"] += hg
        if hg > ag:
            h["Won"] += 1
            a["Lost"] += 1
        elif hg < ag:
            a["Won"] += 1
            h["Lost"] += 1
        else:
            h["Drawn"] += 1
            a["Drawn"] += 1

    if not records:
        return pd.DataFrame(columns=columns)

    rows = []
    for r in records.values():
        r["GD"] = r["GF"] - r["GA"]
        r["Points"] = r["Won"] * 3 + r["Drawn"]
        rows.append(r)

    table = pd.DataFrame(rows).sort_values(
        ["Points", "GD", "GF"], ascending=[False, False, False]
    ).reset_index(drop=True)
    table.insert(0, "Position", range(1, len(table) + 1))
    return table[columns]


def compute_xpts_table(settled_df: pd.DataFrame) -> pd.DataFrame:
    """Loops through every finished fixture and works out a "deserved"
    result from box-touch territory dominance (using a fixed 0.015 box
    touch weight per the spec), rather than the real final score. Also
    tracks the real GP/W/D/L/GD alongside it for context."""
    teams = pd.unique(settled_df[["home_team", "away_team"]].values.ravel("K"))
    records = {
        t: {"played": 0, "wins": 0, "draws": 0, "losses": 0, "goals_for": 0.0,
            "goals_against": 0.0, "actual_pts": 0.0, "xpts": 0.0}
        for t in teams
    }

    for _, row in settled_df.iterrows():
        home, away = row["home_team"], row["away_team"]
        hbt = float(row.get("home_box_touches", 0) or 0)
        abt = float(row.get("away_box_touches", 0) or 0)
        dominance_diff = (hbt - abt) * BOX_TOUCH_WEIGHT

        if dominance_diff > 0.05:
            home_xpts, away_xpts = 3.0, 0.0
        elif dominance_diff < -0.05:
            home_xpts, away_xpts = 0.0, 3.0
        else:
            home_xpts, away_xpts = 1.0, 1.0

        hg, ag = row.get("home_goals", np.nan), row.get("away_goals", np.nan)
        if pd.notna(hg) and pd.notna(ag):
            home_actual = 3.0 if hg > ag else (1.0 if hg == ag else 0.0)
            away_actual = 3.0 if ag > hg else (1.0 if hg == ag else 0.0)
        else:
            home_actual = away_actual = 0.0
            hg = ag = 0.0

        records[home]["played"] += 1
        records[home]["actual_pts"] += home_actual
        records[home]["xpts"] += home_xpts
        records[home]["goals_for"] += hg
        records[home]["goals_against"] += ag
        records[away]["played"] += 1
        records[away]["actual_pts"] += away_actual
        records[away]["xpts"] += away_xpts
        records[away]["goals_for"] += ag
        records[away]["goals_against"] += hg

        if home_actual == 3.0:
            records[home]["wins"] += 1
            records[away]["losses"] += 1
        elif away_actual == 3.0:
            records[away]["wins"] += 1
            records[home]["losses"] += 1
        else:
            records[home]["draws"] += 1
            records[away]["draws"] += 1

    out = pd.DataFrame([
        {
            "team": t, "played": r["played"], "wins": r["wins"], "draws": r["draws"],
            "losses": r["losses"], "goal_difference": round(r["goals_for"] - r["goals_against"], 1),
            "actual_points": r["actual_pts"], "expected_points": round(r["xpts"], 2),
            "points_difference": round(r["actual_pts"] - r["xpts"], 2),
        }
        for t, r in records.items()
    ])
    # Sorted by ACTUAL points first (real league position), highest on top -
    # per the request to sort by current/actual points rather than xPts.
    return out.sort_values(
        ["actual_points", "expected_points"], ascending=[False, False]
    ).reset_index(drop=True)


def current_season_slice(settled_df: pd.DataFrame, upcoming_df: "pd.DataFrame | None" = None):
    """Tables, deserved points and the season forecast describe ONE season. When the data holds several
    (explicit season column, or inferred from dates) this returns just the current one, so points are
    never added up across seasons. Returns (settled_current, upcoming_current, label, n_seasons).
    Current season = the season of the latest played match - unless that season has no fixtures left
    and the upcoming fixtures belong to a newer season that has not started yet, in which case the
    newer one is used (and its table is empty). Single-season data is returned unchanged."""
    if settled_df is None or settled_df.empty:
        return settled_df, upcoming_df, None, 0
    seasons, method = derive_season_series(settled_df)
    if method == "fallback":
        return settled_df, upcoming_df, None, 1
    seasons = seasons.reset_index(drop=True)
    dates = pd.to_datetime(settled_df["date"], errors="coerce").reset_index(drop=True) if "date" in settled_df.columns else pd.Series(dtype="datetime64[ns]")
    latest_label = seasons.loc[dates.idxmax()] if len(dates) and dates.notna().any() else seasons.iloc[-1]
    n_seasons = int(seasons.nunique())
    up_labels = None
    if upcoming_df is not None and not upcoming_df.empty:
        up_labels, _ = derive_season_series(upcoming_df)
        up_labels = up_labels.reset_index(drop=True)
    current_label = latest_label
    if up_labels is not None and latest_label not in set(up_labels):
        current_label = up_labels.mode().iloc[0]  # a newer season whose fixtures exist but has not started
    if n_seasons <= 1 and (up_labels is None or current_label == latest_label):
        return settled_df, upcoming_df, latest_label, n_seasons
    settled_cur = settled_df[(seasons == current_label).to_numpy()]
    upcoming_cur = upcoming_df
    if up_labels is not None:
        upcoming_cur = upcoming_df[(up_labels == current_label).to_numpy()]
    return settled_cur, upcoming_cur, current_label, max(n_seasons, 2 if current_label != latest_label else 1)


def compute_team_strength_table(
    history_df: pd.DataFrame, teams: list, half_life_days: float, reference_date: pd.Timestamp,
    weights: tuple = TERRITORY_WEIGHTS_DEFAULT, engine_weights: "dict | None" = None, duel_strength: float = 1.0,
    reliabilities: "dict | None" = None, baseline: "LeagueBaseline | None" = None,
) -> dict:
    """Every team's home AND away attack/defence ratings from the SAME pipeline match predictions use
    (engine blend with the league's engine weights, duel multiplier at its tuned strength, momentum) plus
    its tactical style at home and away. One compute_match_strengths call per team (a team as both host and
    visitor yields both venues). A team that cannot be rated gets neutral 1.0 ratings, never NaN."""
    if baseline is None:
        baseline = compute_league_baseline(history_df)
    if reliabilities is None:
        reliabilities = compute_engine_reliabilities(history_df, weights)
    out = {}
    for team in teams:
        try:
            ms = compute_match_strengths(history_df, team, team, half_life_days, reference_date, weights,
                                         reliabilities=reliabilities, baseline=baseline,
                                         engine_weights=engine_weights, duel_strength=duel_strength)
            vals = dict(ha=ms.home_attack, hd=ms.home_defense, aa=ms.away_attack, ad=ms.away_defense,
                        hs=ms.home_style, as_=ms.away_style)
            for k in ("ha", "hd", "aa", "ad"):
                if vals[k] is None or not np.isfinite(vals[k]):
                    vals[k] = 1.0
            out[team] = vals
        except Exception:
            out[team] = dict(ha=1.0, hd=1.0, aa=1.0, ad=1.0, hs=STYLE_BALANCED, as_=STYLE_BALANCED)
    return out


def simulate_season(
    settled_df: pd.DataFrame, upcoming_df: pd.DataFrame, iterations: int = MC_ITERATIONS,
    rng=None, relegation_spots: int = 3, title_odds: dict | None = None,
    weights: tuple = TERRITORY_WEIGHTS_DEFAULT, points_df: "pd.DataFrame | None" = None,
    half_life_days: float = FROZEN_HALF_LIFE_DAYS, engine_weights: "dict | None" = None, duel_strength: float = 1.0,
    mismatch_table: "dict | None" = None, mismatch_strength: float = 1.0,
) -> pd.DataFrame:
    """Team ratings now come from the SAME pipeline as the match predictions (engine blend, duel, momentum, plus
    the learned tactical-mismatch multiplier per fixture). `settled_df` is the full history used for the ratings;
    `points_df` (default: settled_df) is only the CURRENT season's results used for the points already won -
    pass current_season_slice()'s output so seasons are never added together.

    Runs the full remaining-season Monte Carlo forecast. The
    crash-proof safety shield: any team with no historical matches (a
    brand-new call-up, or a data gap) gets its capability vector clamped
    to a safe floor of 0.01 rather than letting a NaN reach
    np.random.poisson.

    Now also tracks the FULL final-position distribution (not just
    "won it" / "finished last"), so a real "chance of finishing exactly
    Nth" percentage is available for every position, not just the two
    extremes. relegation_spots controls how many bottom places count as
    "relegated" (default 3, the most common real-world convention) -
    this replaces the old "only counts literally finishing dead last"
    behavior, which understated relegation risk for teams that are
    likely-but-not-certain to finish bottom.

    title_odds, if supplied (dict of team -> bookmaker odds to win the
    league), adds an "edge" column: (title_win_pct/100 * odds) - 1, the
    same EV formula used everywhere else in this app.
    """
    rng = rng or np.random.default_rng()
    points_df = settled_df if points_df is None else points_df
    baseline = compute_league_baseline(settled_df)
    all_teams = pd.unique(
        pd.concat([points_df[["home_team", "away_team"]], upcoming_df[["home_team", "away_team"]]])
        .values.ravel("K")
    )
    all_teams = [t for t in all_teams if isinstance(t, str)]

    reference_date = settled_df["date"].max() if not settled_df.empty and settled_df["date"].notna().any() else pd.Timestamp.now()
    strengths = compute_team_strength_table(settled_df, all_teams, half_life_days, reference_date, weights,
                                            engine_weights, duel_strength, baseline=baseline)

    current_points = {t: 0 for t in all_teams}
    for _, row in points_df.iterrows():
        hg, ag = row.get("home_goals"), row.get("away_goals")
        if pd.isna(hg) or pd.isna(ag):
            continue
        if hg > ag:
            current_points[row["home_team"]] = current_points.get(row["home_team"], 0) + 3
        elif hg < ag:
            current_points[row["away_team"]] = current_points.get(row["away_team"], 0) + 3
        else:
            current_points[row["home_team"]] = current_points.get(row["home_team"], 0) + 1
            current_points[row["away_team"]] = current_points.get(row["away_team"], 0) + 1

    n_teams = len(all_teams)
    title_wins = {t: 0 for t in all_teams}
    relegation_finishes = {t: 0 for t in all_teams}
    position_counts = {t: {p: 0 for p in range(1, n_teams + 1)} for t in all_teams}
    fixtures = list(upcoming_df[["home_team", "away_team"]].itertuples(index=False, name=None))
    relegation_spots = max(1, min(relegation_spots, n_teams))

    # Expected goals for each remaining fixture never change between simulated seasons, so compute them once.
    fixture_lams = []
    for home, away in fixtures:
        if home not in strengths or away not in strengths:
            continue
        h, a = strengths[home], strengths[away]
        lam_h, lam_a = expected_goals(h["ha"], a["ad"], a["aa"], h["hd"], baseline)
        mm_h, mm_a = mismatch_multipliers(mismatch_table or {}, h["hs"], a["as_"], mismatch_strength)
        fixture_lams.append((home, away, max(lam_h * mm_h, 0.01), max(lam_a * mm_a, 0.01)))

    for _ in range(iterations):
        sim_points = dict(current_points)
        for home, away, lam_h, lam_a in fixture_lams:
            hg = rng.poisson(lam_h)
            ag = rng.poisson(lam_a)
            if hg > ag:
                sim_points[home] = sim_points.get(home, 0) + 3
            elif hg < ag:
                sim_points[away] = sim_points.get(away, 0) + 3
            else:
                sim_points[home] = sim_points.get(home, 0) + 1
                sim_points[away] = sim_points.get(away, 0) + 1

        if not sim_points:
            continue
        # Full final ranking this iteration, ties broken by team name (a
        # neutral, deterministic tiebreak - real tables use goal
        # difference, but that's a whole extra simulated dimension not
        # tracked per-iteration here, so this stays a reasonable
        # approximation rather than pretending precision it doesn't have).
        ranked = sorted(sim_points.items(), key=lambda kv: (-kv[1], kv[0]))
        for position, (team, _pts) in enumerate(ranked, start=1):
            position_counts[team][position] += 1
        champion = ranked[0][0]
        title_wins[champion] += 1
        for team, _pts in ranked[-relegation_spots:]:
            relegation_finishes[team] += 1

    def risk_flag(pct: float) -> str:
        if pct >= 40:
            return "🚨"
        if pct >= 15:
            return "⚠️"
        return "🟢"

    records = []
    for t in all_teams:
        title_pct = round(100 * title_wins[t] / iterations, 2)
        row = {
            "team": t,
            "current_points": current_points.get(t, 0),
            "title_win_pct": title_pct,
            "relegation_risk_pct": round(100 * relegation_finishes[t] / iterations, 2),
            "relegation_flag": risk_flag(100 * relegation_finishes[t] / iterations),
        }
        # Chance of finishing in each exact position - only worth showing
        # for a reasonably small league table, but computed for all sizes.
        for position in range(1, n_teams + 1):
            row[f"finish_pos_{position}_pct"] = round(100 * position_counts[t][position] / iterations, 2)
        if title_odds and t in title_odds and title_odds[t] and title_odds[t] > 0:
            row["title_odds"] = title_odds[t]
            row["title_edge_pct"] = round(expected_value(title_pct / 100, title_odds[t]) * 100, 2)
        records.append(row)

    out = pd.DataFrame(records)
    # Sorted by current (real) points first, highest on top - per the
    # request to sort by current standing rather than simulated title %.
    return out.sort_values(["current_points", "title_win_pct"], ascending=[False, False]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Section 7: Sisonke Gold Mine Strategy Panel
# ---------------------------------------------------------------------------
# NOTE ON HONESTY: these are general football-knowledge-based starting
# hints (e.g. leagues broadly known for open, high-scoring football vs
# ones known for being cagey/defensive), NOT the output of a rigorous
# statistical backtest of each specific league. Treat them as an
# editable starting point, not verified fact - swap any of these out
# once you've backtested your own data and have real numbers to trust
# instead of a general reputation.

GOLD_MINE_STRATEGY = {
    ("Premier Division", "South Africa"): "OVER 1.5 GOALS / DOUBLE CHANCE",
    ("Championship", "England"): "OVER 2.5 GOALS / BOTH TEAMS TO SCORE (YES)",
    ("League One", "England"): "OVER 2.5 GOALS / HOME DOUBLE CHANCE",
    ("Premier League", "England"): "BOTH TEAMS TO SCORE (YES) / OVER 2.5 GOALS",
    ("LaLiga", "Spain"): "UNDER 2.5 GOALS / HOME OR DRAW",
    ("LaLiga 2", "Spain"): "UNDER 2.5 GOALS",
    ("Bundesliga", "Germany"): "OVER 2.5 GOALS / BOTH TEAMS TO SCORE (YES)",
    ("2. Bundesliga", "Germany"): "OVER 2.5 GOALS / BOTH TEAMS TO SCORE (YES)",
    ("Serie A", "Italy"): "UNDER 2.5 GOALS",
    ("Serie B", "Italy"): "UNDER 2.5 GOALS / DOUBLE CHANCE",
    ("Ligue 1", "France"): "UNDER 2.5 GOALS",
    ("Ligue 2", "France"): "UNDER 2.5 GOALS",
    ("Bundesliga", "Austria"): "OVER 2.5 GOALS",
    ("Pro League", "Belgium"): "OVER 2.5 GOALS / BOTH TEAMS TO SCORE (YES)",
    ("Challenger Pro League", "Belgium"): "OVER 1.5 GOALS",
    ("Brasileirão Série A", "Brazil"): "UNDER 2.5 GOALS / DOUBLE CHANCE",
    ("Brasileirão Série B", "Brazil"): "UNDER 2.5 GOALS",
    ("Premier League", "Canada"): "OVER 2.5 GOALS",
    ("MLS", "USA"): "OVER 2.5 GOALS / BOTH TEAMS TO SCORE (YES)",
    ("USL Championship", "USA"): "OVER 2.5 GOALS",
    ("HNL", "Croatia"): "HOME DOUBLE CHANCE",
    ("Danish Superliga", "Denmark"): "OVER 2.5 GOALS",
    ("Premier Division", "Ireland"): "OVER 1.5 GOALS",
    ("J1 League", "Japan"): "BOTH TEAMS TO SCORE (YES)",
    ("Eredivisie", "Netherlands"): "OVER 2.5 GOALS / BOTH TEAMS TO SCORE (YES)",
    ("Eerste Divisie", "Netherlands"): "OVER 2.5 GOALS",
    ("Eliteserien", "Norway"): "OVER 2.5 GOALS",
    ("Ekstraklasa", "Poland"): "UNDER 2.5 GOALS",
    ("Premier League", "Russia"): "UNDER 2.5 GOALS",
    ("Liga Portugal 2", "Portugal"): "UNDER 2.5 GOALS",
    ("Allsvenskan", "Sweden"): "OVER 2.5 GOALS",
    ("Super League", "Switzerland"): "OVER 2.5 GOALS",
    ("Challenge League", "Switzerland"): "OVER 1.5 GOALS",
    ("Super League", "China"): "UNDER 2.5 GOALS",
    ("Chilean Primera División", "Chile"): "UNDER 2.5 GOALS",
    ("Serie A", "Ecuador"): "HOME DOUBLE CHANCE",
    ("Liga 1", "Peru"): "HOME DOUBLE CHANCE",
    ("Primera A", "Colombia"): "UNDER 2.5 GOALS",
    ("First League", "Czech Republic"): "OVER 2.5 GOALS",
    ("A-League Men", "Australia"): "OVER 2.5 GOALS / BOTH TEAMS TO SCORE (YES)",
    ("Botola Pro", "Morocco"): "UNDER 2.5 GOALS",
    ("Egyptian Premier League", "Egypt"): "UNDER 2.5 GOALS",
    ("Trendyol Süper Lig", "Turkey"): "OVER 2.5 GOALS / BOTH TEAMS TO SCORE (YES)",
    ("Besta deild karla", "Iceland"): "OVER 2.5 GOALS",
    ("División Profesional", "Bolivia"): "OVER 2.5 GOALS (altitude factor)",
    ("Premiership", "Scotland"): "HOME DOUBLE CHANCE",
    ("Indian Super League", "India"): "UNDER 2.5 GOALS",
    ("Liga MX Apertura", "Mexico"): "OVER 2.5 GOALS",
    ("Super Liga", "Romania"): "UNDER 2.5 GOALS",
}


def gold_mine_hint(division_text: str) -> str:
    """Direct lookup first, then a keyword-matching fallback loop so a
    slightly different bracket/formatting in the CSV (e.g. 'England
    Premier League' vs 'Premier League (England)') still locks onto the
    right entry.

    IMPORTANT: candidates are tried LONGEST-LEAGUE-NAME-FIRST. Several
    real league names are literal substrings of another real league name
    in this same dictionary (e.g. 'Bundesliga' is a substring of
    '2. Bundesliga'; 'LaLiga' is a substring of 'LaLiga 2') - matching in
    dictionary insertion order would let the shorter, WRONG division win
    just because its name happens to appear inside the correct one's
    text. Sorting by length descending means the more specific name is
    always tried before the shorter one it's contained in."""
    if not division_text:
        return "No Gold Mine data for this division yet."
    text_lower = division_text.lower()

    candidates_by_length = sorted(
        GOLD_MINE_STRATEGY.items(), key=lambda item: len(item[0][0]), reverse=True
    )

    for (league, country), hint in candidates_by_length:
        if league.lower() in text_lower and country.lower() in text_lower:
            return f"SISONKE GOLD MINE MARKET ({league}, {country}): Target {hint}"

    for (league, country), hint in candidates_by_length:
        if league.lower() in text_lower or country.lower() in text_lower:
            return f"SISONKE GOLD MINE MARKET (closest match: {league}, {country}): Target {hint}"

    return "No Gold Mine data matched this division - showing raw model output only."


# ---------------------------------------------------------------------------
# League playstyle profile banner - qualitative reputation tags per league,
# same honesty caveat as GOLD_MINE_STRATEGY above: general football
# knowledge, not a backtested statistical fingerprint of each league.
# ---------------------------------------------------------------------------

LEAGUE_PLAYSTYLE_PROFILE = {
    ("Premier Division", "South Africa"): "Physical duels, moderate tempo, set-piece reliant",
    ("Championship", "England"): "High box-touch intensity, direct/transition-heavy, congested fixture list",
    ("League One", "England"): "Direct play, high pressing, moderate technical quality",
    ("Premier League", "England"): "Fast transitions, high pressing, open end-to-end play",
    ("LaLiga", "Spain"): "Possession-based, patient build-up, low box-touch chaos",
    ("LaLiga 2", "Spain"): "Cagey, low-tempo, defensively organised",
    ("Bundesliga", "Germany"): "Fast transition attack, high pressing, open play",
    ("2. Bundesliga", "Germany"): "High-intensity transitions, aggressive pressing",
    ("Serie A", "Italy"): "Tactically disciplined, low box-touch intensity, defensively structured",
    ("Serie B", "Italy"): "Cagey, set-piece reliant, moderate tempo",
    ("Ligue 1", "France"): "Counter-attacking, uneven quality gaps, moderate tempo",
    ("Ligue 2", "France"): "Physical, direct, low technical intensity",
    ("Bundesliga", "Austria"): "Open play, fast transitions",
    ("Pro League", "Belgium"): "Technical, open play, high box-touch intensity",
    ("Challenger Pro League", "Belgium"): "Direct, moderate intensity",
    ("Brasileirão Série A", "Brazil"): "Technical, congested calendar fatigue, set-piece reliant",
    ("Brasileirão Série B", "Brazil"): "Physical, direct, low technical polish",
    ("Premier League", "Canada"): "Open play, moderate intensity",
    ("MLS", "USA"): "High tempo, open play, travel-fatigue heavy (large geography)",
    ("USL Championship", "USA"): "Direct, physical, moderate intensity",
    ("HNL", "Croatia"): "Home-dominant, technical, low away scoring",
    ("Danish Superliga", "Denmark"): "High pressing, fast transitions",
    ("Premier Division", "Ireland"): "Physical, direct, set-piece reliant",
    ("J1 League", "Japan"): "Technical, disciplined pressing, open play",
    ("Eredivisie", "Netherlands"): "Fast transition attack, high box-touch intensity, open play",
    ("Eerste Divisie", "Netherlands"): "Open, high-tempo, developmental squads",
    ("Eliteserien", "Norway"): "Direct, physical, weather-affected variance",
    ("Ekstraklasa", "Poland"): "Cagey, defensively structured",
    ("Premier League", "Russia"): "Low-tempo, defensively disciplined",
    ("Liga Portugal 2", "Portugal"): "Technical, low-scoring, patient build-up",
    ("Allsvenskan", "Sweden"): "Direct, physical, weather-affected variance",
    ("Super League", "Switzerland"): "High-tempo, open play",
    ("Challenge League", "Switzerland"): "Cagey, moderate intensity",
    ("Super League", "China"): "Cagey, defensively structured",
    ("Chilean Primera División", "Chile"): "Technical, low-scoring, patient build-up",
    ("Serie A", "Ecuador"): "Home-dominant (altitude factor in some venues), physical",
    ("Liga 1", "Peru"): "Home-dominant (altitude factor in some venues), physical",
    ("Primera A", "Colombia"): "Technical, low-scoring, patient build-up",
    ("First League", "Czech Republic"): "High-tempo, open play",
    ("A-League Men", "Australia"): "High tempo, open play, travel-fatigue heavy (large geography)",
    ("Botola Pro", "Morocco"): "Cagey, defensively structured",
    ("Egyptian Premier League", "Egypt"): "Cagey, low-scoring, physical",
    ("Trendyol Süper Lig", "Turkey"): "High-tempo, open play, high chaos/card variance",
    ("Besta deild karla", "Iceland"): "Weather-affected variance, direct play",
    ("División Profesional", "Bolivia"): "High-scoring (altitude factor), open play",
    ("Premiership", "Scotland"): "Physical duels, high box-touch intensity, direct play",
    ("Indian Super League", "India"): "Cagey, low-scoring, physical",
    ("Liga MX Apertura", "Mexico"): "Technical, high-altitude variance in some venues, open play",
    ("Super Liga", "Romania"): "Cagey, defensively structured",
}


def league_playstyle_profile(division_text: str) -> str:
    """Same longest-name-first matching logic as gold_mine_hint, kept as
    a separate lookup since the playstyle tag and the market hint are
    conceptually different things a user might want independently."""
    if not division_text:
        return "No playstyle profile for this division yet."
    text_lower = division_text.lower()
    candidates_by_length = sorted(
        LEAGUE_PLAYSTYLE_PROFILE.items(), key=lambda item: len(item[0][0]), reverse=True
    )
    for (league, country), tags in candidates_by_length:
        if league.lower() in text_lower and country.lower() in text_lower:
            return tags
    for (league, country), tags in candidates_by_length:
        if league.lower() in text_lower or country.lower() in text_lower:
            return tags
    return "No playstyle profile matched this division."


# ---------------------------------------------------------------------------
# Dynamic prediction explanation - a plain-language readout of everything
# that fed into one specific projection, generated fresh each time rather
# than a canned template string.
# ---------------------------------------------------------------------------

def generate_prediction_explanation(
    home_team: str, away_team: str,
    half_life_days: float, half_life_frozen: bool,
    home_attack_raw: float, away_attack_raw: float,
    home_momentum_mult: float, home_momentum_desc: str,
    away_momentum_mult: float, away_momentum_desc: str,
    tactic_log: list[str],
    rho: float,
    lam_home: float, lam_away: float,
    dc_probs: dict, mc_probs: dict,
    vol_dampener_adjusted: float,
) -> str:
    """Builds a fresh, specific explanation from the ACTUAL numbers that
    went into this one projection - not a fixed template that just fills
    in team names. Every section only appears if it was actually
    relevant (e.g. the momentum section is skipped entirely if neither
    team has a real streak), so the explanation reads differently for
    different fixtures rather than always listing the same boilerplate."""
    lines = [f"### Why the model predicts what it does: {home_team} vs {away_team}", ""]

    stronger = home_team if home_attack_raw > away_attack_raw else away_team
    lines.append(
        f"**Territory data**: based on venue-isolated big chances, shots on target, and box "
        f"touches, {stronger} shows the stronger underlying attacking territory profile "
        f"(host rating {home_attack_raw:.2f} vs visitor rating {away_attack_raw:.2f}). These "
        f"ratings are each team's own numbers divided by THIS LEAGUE'S real, separately-"
        f"computed average - 1.00 always means 'exactly average for this specific league' by "
        f"definition (a number divided by itself is always 1), so it looks the same across "
        f"every league on purpose. The actual underlying average - e.g. average goals, big "
        f"chances, shots on target, box touches for this exact league - genuinely differs "
        f"league to league and is shown in the 'This League's Real Baseline' panel above."
    )

    decay_note = (
        f"a frozen {half_life_days:.0f}-day window (manually locked)" if half_life_frozen
        else f"an auto-optimised {half_life_days:.0f}-day half-life, chosen by backtesting "
             f"candidate windows against this division's own real results"
    )
    lines.append(f"**Recency weighting**: recent matches are weighted more heavily using {decay_note}.")

    momentum_bits = []
    if abs(home_momentum_mult - 1.0) > 1e-6:
        momentum_bits.append(f"{home_team}: {home_momentum_desc} (x{home_momentum_mult:.2f} to attack)")
    if abs(away_momentum_mult - 1.0) > 1e-6:
        momentum_bits.append(f"{away_team}: {away_momentum_desc} (x{away_momentum_mult:.2f} to attack)")
    if momentum_bits:
        lines.append(f"**Streak/momentum**: {'; '.join(momentum_bits)}.")
    else:
        lines.append("**Streak/momentum**: neither team is currently on a qualifying win or loss streak, so no adjustment applied.")

    if tactic_log:
        lines.append("**Manual tactical/environmental adjustments applied**:")
        for entry in tactic_log:
            lines.append(f"  - {entry}")
    else:
        lines.append("**Manual tactical/environmental adjustments applied**: none - this is the model's baseline data-only projection.")

    lines.append(
        f"**Low-score correlation**: a Dixon-Coles rho of {rho:+.3f} was fitted from this "
        f"division's own historical 0-0/1-1/1-0/0-1 frequency (0 means no measurable "
        f"correlation was found in the data)."
    )
    lines.append(f"**Final expected goals**: {home_team} {lam_home:.2f} - {lam_away:.2f} {away_team}.")

    dc_pick = max([("Home", dc_probs.get("Home Win", 0)), ("Draw", dc_probs.get("Draw", 0)), ("Away", dc_probs.get("Away Win", 0))], key=lambda kv: kv[1])
    mc_pick = max([("Home", mc_probs.get("Home Win", 0)), ("Draw", mc_probs.get("Draw", 0)), ("Away", mc_probs.get("Away Win", 0))], key=lambda kv: kv[1])
    agree_note = "agree" if dc_pick[0] == mc_pick[0] else "DISAGREE"
    lines.append(
        f"**Engine cross-check**: Dixon-Coles favors **{dc_pick[0]}** "
        f"({dc_pick[1]*100:.1f}%), Monte Carlo favors **{mc_pick[0]}** ({mc_pick[1]*100:.1f}%) - "
        f"the two engines {agree_note}."
    )
    lines.append(f"**Match volatility dampener** (used by the Monte Carlo engine): {vol_dampener_adjusted:.3f}.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 13: Auxiliary engines - set-piece, transition/counter-attack, and
# defensive-resilience (duels + goalkeeping) - built to accommodate leagues
# where the territorial engine (big chances / SOT / box touches) doesn't
# correlate well with real results (see compute_league_reliability_score's
# xg_proxy_correlation - the original diagnostic that flagged this gap).
#
# ARCHITECTURE NOTE, worth reading before using any of this: rather than
# building three fully separate Poisson/Monte-Carlo pipelines that would
# each need their OWN convergence/EV comparison against Dixon-Coles (a much
# larger rewrite of how predictions flow through this app), these auxiliary
# engines plug into the SAME attack/defense-strength -> expected_goals()
# pipeline the territorial engine already uses. Each engine produces its own
# attack/defense strength ratio (1.0 = league average, same convention as
# attack_strength()/defense_strength() above), and blend_engine_strengths()
# combines all available engines' ratios into ONE final strength, weighted
# by each engine's own per-league reliability score - so a league where
# territorial correlation is weak automatically leans more on set-piece
# and/or transition signal instead, without anyone hand-flagging which
# leagues need which engine. This is the SAME reliability-score mechanism
# compute_league_reliability_score() already uses for the territorial
# engine, just generalised so it isn't hard-coded to one engine's columns.
#
# DATA AVAILABILITY: none of these columns exist in the original schema
# (goals/big-chances/SOT/box-touches) - every function below checks for
# column presence and returns None (never a fabricated neutral guess
# masquerading as real data) when the columns aren't there, so an older
# dataset without this data keeps working exactly as before, just without
# these engines contributing anything to the blend.
# ---------------------------------------------------------------------------

SETPIECE_COUNT_METRICS = ["corner_kicks", "fouled_in_final_third", "accurate_crosses", "xg_set_play"]
SETPIECE_RATE_METRICS = ["aerial_duels_percentage", "successful_crosses_percentage"]
# corners, fouled in final third, crosses (VOLUME), xG from set plays (DIRECT shot-quality
# measurement), aerial duels, crosses (QUALITY). xg_set_play gets the largest single weight
# below deliberately - it's a real per-shot quality measurement, not a reconstructed proxy
# the way corners/crosses/fouls are, so it should dominate this engine's signal whenever a
# league has it (same reasoning as xG vs. the territorial engine's own proxy - see
# compute_auxiliary_engine_reliability's xG docstring). ONLY total xG (`xg`, its own separate
# engine) was deliberately kept OUT of this list - xg_set_play + xg_open_play are a
# decomposition of total xg, so running total xg as a THIRD peer alongside its own two parts
# would be triple-counting the same underlying shot quality (see the multicollinearity
# discussion this was built from) - not yet empirically backtested (same starting-point
# status TERRITORY_WEIGHTS_DEFAULT had before optimize_territory_weights existed).
SETPIECE_WEIGHTS_DEFAULT = (0.15, 0.15, 0.15, 0.30, 0.10, 0.15)

TRANSITION_COUNT_METRICS = ["accurate_long_passes", "final_third_entries", "xg_open_play"]
TRANSITION_RATE_METRICS = ["dribbles_percentage", "successful_long_passes_percentage"]
# long passes VOLUME, final third entries, xG from open play (DIRECT shot-quality
# measurement - same "gets the largest weight" reasoning as xg_set_play above), dribbles,
# long passes ACCURACY. Both long-pass fields describe the SAME underlying action (volume
# vs. accuracy%) and now share one name: accurate_long_passes / successful_long_passes_percentage.
# Older files that call the volume column accurate_long_balls are renamed automatically
# on load (see COLUMN_RENAMES / standardise_columns).
TRANSITION_WEIGHTS_DEFAULT = (0.15, 0.15, 0.35, 0.20, 0.15)

# xG-by-situation (xg_set_play, xg_open_play) IS folded directly into
# SETPIECE_COUNT_METRICS/TRANSITION_COUNT_METRICS above (each given the
# largest single weight in its engine, since it's a direct shot-quality
# measurement rather than a reconstructed proxy) - deliberately NOT kept
# as separate peer entries in blend_engine_strengths() alongside total
# `xg`. xg_set_play + xg_open_play sum to approximately xg_total by
# construction, so running all three ("xg", "xg_set_play", "xg_open_play")
# as independent, separately-reliability-weighted peers in the SAME
# blend would inflate the same underlying shot-quality signal's influence
# 2-3x over relative to every other engine - the exact multicollinearity
# this design was built to avoid. Folding xg_set_play/xg_open_play into
# their respective proxy engines instead means each contributes ONCE.
# This dataset only includes leagues with full metric coverage (a
# deliberate choice made when this was built), so the all-or-nothing
# column-availability check on these lists is never actually a problem
# in practice here.


def setpiece_active_metrics(league_df: pd.DataFrame) -> tuple[list[str], tuple[float, ...]]:
    """Returns the fixed metrics/weights (corners, fouls, crosses,
    xg_set_play, aerial duels, crosses%) - xg_set_play is folded directly
    into this single list (see the module note above), not treated as a
    separate peer engine. Kept as a function (rather than inlining the
    fixed lists at every call site) purely so setpiece_profile/
    setpiece_strength always resolve the identical metric list, the
    consistency fix noted on setpiece_strength below."""
    return list(SETPIECE_COUNT_METRICS) + list(SETPIECE_RATE_METRICS), SETPIECE_WEIGHTS_DEFAULT


def transition_active_metrics(league_df: pd.DataFrame) -> tuple[list[str], tuple[float, ...]]:
    """Same reasoning as setpiece_active_metrics() above, for xg_open_play
    folded into TRANSITION_COUNT_METRICS."""
    return list(TRANSITION_COUNT_METRICS) + list(TRANSITION_RATE_METRICS), TRANSITION_WEIGHTS_DEFAULT

# Duel/defensive-resilience is framed purely as a DEFENSE multiplier (not
# attack) - it answers "can this team hold up physically/technically
# against territorial dominance", which is exactly the mechanism named in
# the original discussion (a mid-table side surviving a stronger side's
# pressure through duels, not through out-territorying them).
DUEL_RATE_METRICS = ["tackles_won_percentage", "ground_duels_percentage", "aerial_duels_percentage"]
DUEL_WEIGHTS_DEFAULT = (0.4, 0.3, 0.3)  # tackles won, ground duels, aerial duels


@dataclass
class MetricProfile:
    """Generic decay-weighted, venue-isolated profile for an arbitrary
    set of raw match columns - the same machinery team_territory_profile
    uses for big_chances/SOT/box_touches, generalised so the set-piece,
    transition, and duel engines can reuse it instead of three near-
    duplicate dataclasses. Works for both COUNT-type columns (corner_kicks,
    accurate_crosses - has a real "for vs. against" split) and RATE-type
    columns (tackles_won_percentage - "against" here means what the
    OPPONENT'S rate looked like in that same match, still a meaningful
    comparison, just not a summable count)."""
    n_matches: int
    effective_sample_size: float
    values_for: dict    # {metric_name: decay-weighted average}
    values_against: dict


def _columns_available(df: pd.DataFrame, metrics: list[str]) -> bool:
    """True only if EVERY home_/away_ pair for these metrics exists in df
    AND actually contains at least one real (non-null) value - partial
    coverage, or a column that's structurally present but entirely empty
    (every row blank), is treated as unavailable rather than silently
    computing a profile from whichever columns happen to exist. A column
    that exists but is all-NaN would otherwise pass a presence-only check,
    then get fillna(0.0)'d downstream into a profile that LOOKS like real
    data but is actually every value defaulting to zero - this is what
    stops that from reaching the blend at all, rather than relying on
    _safe_ratio's 0/0->1.0 guard and a NaN correlation to merely keep it
    harmless after the fact."""
    if not metrics:
        return True
    for m in metrics:
        home_col, away_col = f"home_{m}", f"away_{m}"
        if home_col not in df.columns or away_col not in df.columns:
            return False
        if df[home_col].notna().sum() == 0 or df[away_col].notna().sum() == 0:
            return False
    return True


def team_metric_profile(
    league_df: pd.DataFrame, team: str, venue: str, half_life_days: float, reference_date: pd.Timestamp,
    metrics: list[str], use_match_index: bool = False, half_life_matches: float = FROZEN_HALF_LIFE_MATCHES,
) -> "MetricProfile | None":
    """Same decay-weighting and venue-isolation as team_territory_profile,
    generalised to any list of metric names. Returns None (not a neutral
    profile) if the required columns aren't in league_df at all, or if
    this team has no rows at this venue - callers must handle None by
    excluding that engine from the blend, never by guessing a value."""
    if not _columns_available(league_df, metrics):
        return None
    if venue == "home":
        rows = league_df[league_df["home_team"] == team]
        for_prefix, against_prefix = "home_", "away_"
    else:
        rows = league_df[league_df["away_team"] == team]
        for_prefix, against_prefix = "away_", "home_"
    if rows.empty:
        return None

    if use_match_index:
        rows = rows.sort_values("date", ascending=False)
        w = decay_weights_by_index(len(rows), half_life_matches)
    else:
        w = decay_weights(rows["date"], reference_date, half_life_days)

    def wmean(col: str) -> float:
        vals = rows[col].to_numpy(dtype=float)
        return weighted_mean(vals, w)

    values_for = {m: wmean(f"{for_prefix}{m}") for m in metrics}
    values_against = {m: wmean(f"{against_prefix}{m}") for m in metrics}

    metric_cols = [f"{p}{m}" for p in (for_prefix, against_prefix) for m in metrics]
    usable = rows[metric_cols].notna().any(axis=1).to_numpy()
    if not usable.any():
        return None
    return MetricProfile(
        n_matches=int(usable.sum()), effective_sample_size=round(effective_sample_size(np.where(usable, w, 0.0)), 1),
        values_for=values_for, values_against=values_against,
    )


def compute_league_metric_baseline(settled_df: pd.DataFrame, metrics: list[str]) -> "dict | None":
    """League-wide home/away for/against averages per metric - the generic
    equivalent of compute_league_baseline(), needed to turn a team's raw
    metric average into a ratio-to-league-average (the same convention
    attack_strength()/defense_strength() use). Returns None if the
    columns aren't present."""
    if not _columns_available(settled_df, metrics) or settled_df.empty:
        return None
    # Season-aware and NaN-aware, same as compute_league_baseline().
    w = season_row_weights(settled_df)
    baseline = {}
    for m in metrics:
        home_mean = _wcol_mean(settled_df, f"home_{m}", w)
        away_mean = _wcol_mean(settled_df, f"away_{m}", w)
        baseline[m] = {"home_for": home_mean, "home_against": away_mean, "away_for": away_mean, "away_against": home_mean}
    return baseline


def metric_strength(
    profile: "MetricProfile | None", league_metric_baseline: "dict | None", venue: str,
    metrics: list[str], weights: tuple, direction: str = "for",
) -> "float | None":
    """Generic version of attack_strength()/defense_strength() - a
    weighted average of {profile ratio vs. league baseline} across the
    given metrics, for either the 'for' (attacking) or 'against'
    (defensive) direction. Returns None (not a neutral 1.0) when the
    profile or baseline is unavailable, or the sample is too thin to
    trust - the caller decides how to handle a missing engine, this
    function never manufactures a fake-neutral contribution."""
    if profile is None or league_metric_baseline is None or profile.n_matches < MIN_SAMPLE_ROWS:
        return None
    ratios = []
    values = profile.values_for if direction == "for" else profile.values_against
    key = f"{venue}_{direction}"
    for m in metrics:
        baseline_value = league_metric_baseline[m][key]
        ratios.append(_safe_ratio(values[m], baseline_value))
    arr = np.asarray(ratios, dtype=float)
    w_arr = np.asarray(weights, dtype=float)
    usable = np.isfinite(arr) & (w_arr > 0)
    if not usable.any():
        return None  # nothing measurable -> this engine sits out, never a fake neutral value
    return float(np.average(arr[usable], weights=w_arr[usable]))


# --- Engine-specific wrappers ------------------------------------------------

XG_METRICS = ["xg"]
XG_WEIGHTS_DEFAULT = (1.0,)  # single metric - no blend weighting needed, unlike the multi-metric engines above

# xGOT / xG-set-play / xG-open-play are each kept as their OWN independent
# single-metric engine, deliberately NOT folded into the existing setpiece/
# transition COUNT_METRICS lists above. Reason: _columns_available() requires
# EVERY metric in a list to be present before that whole engine activates -
# if xg_set_play were merged into SETPIECE_COUNT_METRICS, a league missing
# ONLY that one field would silently disable the entire set-piece engine,
# throwing away perfectly good corner_kicks/crosses/aerial-duel signal along
# with it. Provider coverage for these three is real but partial ("most
# leagues, not all"), so each gets its own availability check via
# compute_auxiliary_engine_reliability - a league missing xg_set_play simply
# doesn't get that ONE engine's contribution; the original set-piece engine
# (and everything else) keeps working exactly as before, completely
# unaffected. blend_engine_strengths() already treats missing engines as
# "skip and renormalize" for free, so this costs nothing structurally.
#
# WHY THESE THREE, SEPARATE FROM `xg` ABOVE:
#   xgot           - POST-shot quality (placement/power, conditioned on
#                    already being on target) vs. xg's PRE-shot quality
#                    (location/angle/situation) - a genuinely different
#                    question: xG asks "how good was the chance", xGOT asks
#                    "how well did they execute once they shot". This is the
#                    analyst-grade version of what SOT-conversion (goals/SOT,
#                    currently display-only) was approximating - unlike that
#                    metric, this DOES feed the live attack/defense blend.
#   xg_set_play    - xG specifically from set-piece situations - sharpens
#                    the set-piece engine's actual shot-quality signal
#                    rather than the corners/crosses volume proxy alone.
#   xg_open_play   - xG specifically from open play - same idea for the
#                    transition engine's actual shot-quality signal.
# All three correlate with total xG by construction (open+set play sum to
# roughly total xG), which is exactly why they're NOT added alongside `xg`
# itself as competing peers - each instead targets a DIFFERENT existing
# engine (xgot -> finishing quality, generically blended; xg_set_play ->
# set-piece; xg_open_play -> transition), avoiding double-counting the same
# underlying shots.
XGOT_METRICS = ["xgot"]
XGOT_WEIGHTS_DEFAULT = (1.0,)
XG_SET_PLAY_METRICS = ["xg_set_play"]
XG_SET_PLAY_WEIGHTS_DEFAULT = (1.0,)
XG_OPEN_PLAY_METRICS = ["xg_open_play"]
XG_OPEN_PLAY_WEIGHTS_DEFAULT = (1.0,)


def setpiece_profile(league_df, team, venue, half_life_days, reference_date, metrics=None, use_match_index=False, half_life_matches=FROZEN_HALF_LIFE_MATCHES):
    if metrics is None:
        metrics, _ = setpiece_active_metrics(league_df)
    return team_metric_profile(
        league_df, team, venue, half_life_days, reference_date,
        metrics, use_match_index, half_life_matches,
    )


def transition_profile(league_df, team, venue, half_life_days, reference_date, metrics=None, use_match_index=False, half_life_matches=FROZEN_HALF_LIFE_MATCHES):
    if metrics is None:
        metrics, _ = transition_active_metrics(league_df)
    return team_metric_profile(
        league_df, team, venue, half_life_days, reference_date,
        metrics, use_match_index, half_life_matches,
    )


def duel_profile(league_df, team, venue, half_life_days, reference_date, use_match_index=False, half_life_matches=FROZEN_HALF_LIFE_MATCHES):
    return team_metric_profile(
        league_df, team, venue, half_life_days, reference_date,
        DUEL_RATE_METRICS, use_match_index, half_life_matches,
    )


def xg_profile(league_df, team, venue, half_life_days, reference_date, use_match_index=False, half_life_matches=FROZEN_HALF_LIFE_MATCHES):
    """Decay-weighted, venue-isolated provider-xG profile - values_for is
    this team's own xG generated, values_against is xG conceded (the
    opponent's xG in the same matches). Same MetricProfile shape as
    every other engine here - see XG_METRICS module note above for how
    this differs from the territorial engine's own proxy."""
    return team_metric_profile(
        league_df, team, venue, half_life_days, reference_date,
        XG_METRICS, use_match_index, half_life_matches,
    )


def xgot_profile(league_df, team, venue, half_life_days, reference_date, use_match_index=False, half_life_matches=FROZEN_HALF_LIFE_MATCHES):
    """Post-shot expected goals (xGOT) - see the module note above XGOT_METRICS
    for how this differs from pre-shot xg. Returns None if this league's
    provider doesn't cover xgot (partial coverage across leagues is expected)."""
    return team_metric_profile(
        league_df, team, venue, half_life_days, reference_date,
        XGOT_METRICS, use_match_index, half_life_matches,
    )


def xg_set_play_profile(league_df, team, venue, half_life_days, reference_date, use_match_index=False, half_life_matches=FROZEN_HALF_LIFE_MATCHES):
    return team_metric_profile(
        league_df, team, venue, half_life_days, reference_date,
        XG_SET_PLAY_METRICS, use_match_index, half_life_matches,
    )


def xg_open_play_profile(league_df, team, venue, half_life_days, reference_date, use_match_index=False, half_life_matches=FROZEN_HALF_LIFE_MATCHES):
    return team_metric_profile(
        league_df, team, venue, half_life_days, reference_date,
        XG_OPEN_PLAY_METRICS, use_match_index, half_life_matches,
    )


def setpiece_strength(profile, baseline, venue, direction="for", metrics=None, weights=None, league_df=None):
    """BUG FIX: previously defaulted to the fixed proxy-only metrics/weights
    regardless of whether the profile passed in actually included
    xg_set_play - meaning the xG-enhanced signal could be computed into
    the profile by setpiece_profile() and then silently never used here.
    Now resolves the SAME active-metrics list (via setpiece_active_metrics)
    unless explicitly overridden, so profile and strength always agree on
    which metrics are actually in play. Pass league_df (the same settled_div
    used to build the profile) so this can run the same availability check -
    if omitted, falls back to the fixed proxy-only list (the old behaviour),
    which is always safe since it's a subset of what the profile might have."""
    if metrics is None or weights is None:
        if league_df is not None:
            metrics, weights = setpiece_active_metrics(league_df)
        else:
            metrics = SETPIECE_COUNT_METRICS + SETPIECE_RATE_METRICS
            weights = SETPIECE_WEIGHTS_DEFAULT
    return metric_strength(profile, baseline, venue, metrics, weights, direction)


def transition_strength(profile, baseline, venue, direction="for", metrics=None, weights=None, league_df=None):
    """Same fix and same reasoning as setpiece_strength() above, for
    xg_open_play feeding the transition engine."""
    if metrics is None or weights is None:
        if league_df is not None:
            metrics, weights = transition_active_metrics(league_df)
        else:
            metrics = TRANSITION_COUNT_METRICS + TRANSITION_RATE_METRICS
            weights = TRANSITION_WEIGHTS_DEFAULT
    return metric_strength(profile, baseline, venue, metrics, weights, direction)


def duel_defense_strength(profile, baseline, venue, weights=DUEL_WEIGHTS_DEFAULT):
    """Duels are a DEFENSIVE-RESILIENCE multiplier applied on top of the blended defence ratio.

    CONVENTION (this is where the old version was wrong): in this engine a defence value is an
    'ALLOWED ratio' - ABOVE 1.0 = concedes MORE than league average = WORSE defence. A team that
    wins MORE duels than the league average (duel-win ratio > 1) is MORE resilient, so its
    multiplier must be BELOW 1.0. The old code multiplied the raw duel-win ratio straight in,
    which punished the strong duel teams and rewarded the weak ones (65% duel win rate gave
    x1.29, i.e. a worse defence). The ratio is now INVERTED and clamped to
    [DUEL_MULT_FLOOR, DUEL_MULT_CEILING] so it stays a nudge rather than a dominant term.

    Returns None when there is not enough duel data (the engine then sits out)."""
    win_rate_ratio = metric_strength(profile, baseline, venue, DUEL_RATE_METRICS, weights, direction="for")
    if win_rate_ratio is None or not np.isfinite(win_rate_ratio) or win_rate_ratio <= 0:
        return None
    return float(np.clip(1.0 / win_rate_ratio, DUEL_MULT_FLOOR, DUEL_MULT_CEILING))


def xg_strength(profile, baseline, venue, direction="for", weights=XG_WEIGHTS_DEFAULT):
    return metric_strength(profile, baseline, venue, XG_METRICS, weights, direction)


def xgot_strength(profile, baseline, venue, direction="for", weights=XGOT_WEIGHTS_DEFAULT):
    return metric_strength(profile, baseline, venue, XGOT_METRICS, weights, direction)


def xg_set_play_strength(profile, baseline, venue, direction="for", weights=XG_SET_PLAY_WEIGHTS_DEFAULT):
    return metric_strength(profile, baseline, venue, XG_SET_PLAY_METRICS, weights, direction)


def xg_open_play_strength(profile, baseline, venue, direction="for", weights=XG_OPEN_PLAY_WEIGHTS_DEFAULT):
    return metric_strength(profile, baseline, venue, XG_OPEN_PLAY_METRICS, weights, direction)


# --- Goalkeeper save quality (extends the SOT-conversion-against work) -----

def team_save_efficiency_profile(
    league_df: pd.DataFrame, team: str, venue: str, half_life_days: float, reference_date: pd.Timestamp,
    league_avg_save_efficiency: float, use_match_index: bool = False, half_life_matches: float = FROZEN_HALF_LIFE_MATCHES,
    shrinkage_k: float = SOT_CONVERSION_SHRINKAGE_K,
) -> "SOTConversionProfile | None":
    """Save efficiency = this team's own goalkeeper_saves / shots on
    target FACED - the mirror image of team_sot_conversion_profile's
    'against' side, isolating shot-stopping quality specifically (a
    signal genuinely absent from every other engine in this codebase:
    a team can face heavy territorial/SOT pressure and still survive
    purely on goalkeeping, which box touches, big chances, and SOT
    volume alone can never see). Reuses SOTConversionProfile's shape
    and the same ESS-based shrinkage toward the league average - see
    team_sot_conversion_profile() for the shrinkage rationale. Returns
    None if goalkeeper_saves isn't in league_df."""
    if "home_goalkeeper_saves" not in league_df.columns or "away_goalkeeper_saves" not in league_df.columns:
        return None
    if venue == "home":
        rows = league_df[league_df["home_team"] == team]
        own_prefix, faced_prefix = "home_", "away_"
    else:
        rows = league_df[league_df["away_team"] == team]
        own_prefix, faced_prefix = "away_", "home_"
    if rows.empty:
        return None

    if use_match_index:
        rows = rows.sort_values("date", ascending=False)
        w = decay_weights_by_index(len(rows), half_life_matches)
    else:
        w = decay_weights(rows["date"], reference_date, half_life_days)

    ess = effective_sample_size(w)
    saves = rows[f"{own_prefix}goalkeeper_saves"].to_numpy(dtype=float)
    sot_faced = rows[f"{faced_prefix}shots_on_target"].to_numpy(dtype=float) if f"{faced_prefix}shots_on_target" in rows.columns else None

    if sot_faced is None:
        return None

    raw_efficiency = _ratio_sums(saves, sot_faced, w)
    shrink_weight = ess / (ess + shrinkage_k) if (ess + shrinkage_k) > 0 else 0.0
    shrunk_efficiency = _shrink(raw_efficiency, shrink_weight, league_avg_save_efficiency)

    return SOTConversionProfile(
        n_matches=len(rows), effective_sample_size=round(ess, 1),
        sot_conversion_for=round(shrunk_efficiency, 3), sot_conversion_against=round(shrunk_efficiency, 3),
        raw_sot_conversion_for=round(raw_efficiency, 3), raw_sot_conversion_against=round(raw_efficiency, 3),
        league_avg_conversion=round(league_avg_save_efficiency, 3), shrinkage_weight=round(shrink_weight, 3),
    )  # NOTE: reuses SOTConversionProfile's shape (save efficiency has no separate "for vs against" meaning of its own, so both fields carry the same value) - field names are inherited from that dataclass, read sot_conversion_for/against here as "save efficiency", not shot conversion.


def league_average_save_efficiency(settled_df: pd.DataFrame) -> "float | None":
    if "home_goalkeeper_saves" not in settled_df.columns or "away_goalkeeper_saves" not in settled_df.columns:
        return None
    if "home_shots_on_target" not in settled_df.columns or "away_shots_on_target" not in settled_df.columns:
        return None
    pooled = _pooled_ratio(settled_df, [("home_goalkeeper_saves", "away_shots_on_target"), ("away_goalkeeper_saves", "home_shots_on_target")])
    return pooled if np.isfinite(pooled) else None


# --- Field penetration efficiency (needs no new data - box_touches and
# final_third_entries are both already tracked) --------------------------
#
# final_third_entries (the transition engine) is a VOLUME metric - how
# often a team gets into the final third - the exact same limitation raw
# SOT had before SOT-conversion split shot quality out from shot count.
# Field penetration efficiency = box_touches / final_third_entries: of the
# times a team reaches the final third, how often do they actually push
# on into the box, rather than stalling out, losing the ball, or being
# forced backward. Same ratio-of-sums, decay-weighted, shrink-to-league-
# average treatment as SOT conversion above, one step earlier in the
# buildup chain.

def league_average_penetration_efficiency(settled_df: pd.DataFrame) -> "float | None":
    if "home_final_third_entries" not in settled_df.columns or "away_final_third_entries" not in settled_df.columns:
        return None
    if "home_box_touches" not in settled_df.columns or "away_box_touches" not in settled_df.columns:
        return None
    pooled = _pooled_ratio(settled_df, [("home_box_touches", "home_final_third_entries"), ("away_box_touches", "away_final_third_entries")])
    return pooled if np.isfinite(pooled) else None


def team_penetration_efficiency_profile(
    league_df: pd.DataFrame, team: str, venue: str, half_life_days: float, reference_date: pd.Timestamp,
    league_avg_efficiency: float, use_match_index: bool = False, half_life_matches: float = FROZEN_HALF_LIFE_MATCHES,
    shrinkage_k: float = SOT_CONVERSION_SHRINKAGE_K,
) -> "SOTConversionProfile | None":
    """Same shape and shrinkage as team_sot_conversion_profile - reused
    directly rather than a new dataclass, since 'a ratio of two decay-
    weighted counts, shrunk toward a league average by ESS' is exactly
    the same computation regardless of which two columns it's applied
    to. Read sot_conversion_for/against here as 'penetration efficiency
    FOR/faced', not shot conversion. Returns None if either required
    column pair is missing."""
    if "home_final_third_entries" not in league_df.columns or "away_final_third_entries" not in league_df.columns:
        return None
    if "home_box_touches" not in league_df.columns or "away_box_touches" not in league_df.columns:
        return None
    if venue == "home":
        rows = league_df[league_df["home_team"] == team]
        for_prefix, against_prefix = "home_", "away_"
    else:
        rows = league_df[league_df["away_team"] == team]
        for_prefix, against_prefix = "away_", "home_"
    if rows.empty:
        return None

    if use_match_index:
        rows = rows.sort_values("date", ascending=False)
        w = decay_weights_by_index(len(rows), half_life_matches)
    else:
        w = decay_weights(rows["date"], reference_date, half_life_days)

    ess = effective_sample_size(w)

    box_for = rows[f"{for_prefix}box_touches"].to_numpy(dtype=float)
    entries_for = rows[f"{for_prefix}final_third_entries"].to_numpy(dtype=float)
    box_against = rows[f"{against_prefix}box_touches"].to_numpy(dtype=float)
    entries_against = rows[f"{against_prefix}final_third_entries"].to_numpy(dtype=float)

    raw_for = _ratio_sums(box_for, entries_for, w)
    raw_against = _ratio_sums(box_against, entries_against, w)

    shrink_weight = ess / (ess + shrinkage_k) if (ess + shrinkage_k) > 0 else 0.0
    shrunk_for = _shrink(raw_for, shrink_weight, league_avg_efficiency)
    shrunk_against = _shrink(raw_against, shrink_weight, league_avg_efficiency)

    return SOTConversionProfile(
        n_matches=len(rows), effective_sample_size=round(ess, 1),
        sot_conversion_for=round(shrunk_for, 3), sot_conversion_against=round(shrunk_against, 3),
        raw_sot_conversion_for=round(raw_for, 3) if np.isfinite(raw_for) else float('nan'), raw_sot_conversion_against=round(raw_against, 3) if np.isfinite(raw_against) else float('nan'),
        league_avg_conversion=round(league_avg_efficiency, 3), shrinkage_weight=round(shrink_weight, 3),
    )


# --- Per-engine reliability scoring (generalised from xg_proxy_correlation) -

def metric_differential_correlation(
    settled_df: pd.DataFrame, count_metrics: list[str], rate_metrics: list[str], weights: tuple,
) -> "float | None":
    """Generalised xg_proxy_correlation(): Pearson correlation between a
    standardised, weighted differential of the given metrics (home minus
    away) and the actual goal difference. Returns None (not NaN) when the
    columns aren't in this dataset at all, so callers can tell "not
    computable here" apart from "computed and found weak" (that second
    case still returns NaN internally when there's genuinely no
    variance, same as xg_proxy_correlation)."""
    all_metrics = count_metrics + rate_metrics
    if not _columns_available(settled_df, all_metrics):
        return None
    needed = [f"{side}_{m}" for m in all_metrics for side in ("home", "away")] + ["home_goals", "away_goals"]
    df = settled_df.dropna(subset=[c for c in needed if c in settled_df.columns]).copy()
    if len(df) < MIN_SAMPLE_ROWS:
        return float("nan")

    def z(series):
        s = series.std(ddof=0)
        return (series - series.mean()) / s if s > 0 else series * 0.0

    diffs = [z(df[f"home_{m}"] - df[f"away_{m}"]) for m in all_metrics]
    combined = sum(w * d for w, d in zip(weights, diffs))
    goal_diff = df["home_goals"] - df["away_goals"]
    if combined.std(ddof=0) == 0 or goal_diff.std(ddof=0) == 0:
        return float("nan")
    return float(np.corrcoef(combined, goal_diff)[0, 1])


@dataclass
class AuxiliaryEngineReliability:
    """Same shape as LeagueReliabilityScore, one per auxiliary engine.
    available=False means the required columns simply aren't in this
    dataset - composite_score is meaningless in that case and MUST be
    excluded from any blend, not treated as a low score."""
    engine_name: str
    available: bool
    proxy_correlation: float
    composite_score: float


def compute_auxiliary_engine_reliability(settled_df: pd.DataFrame, engine_name: str) -> AuxiliaryEngineReliability:
    """engine_name is 'setpiece', 'transition', 'duel', 'xg', 'xgot',
    'xg_set_play', or 'xg_open_play'. Mirrors
    compute_league_reliability_score()'s normalisation approach but
    without a BSS term (these engines don't have their own walk-forward
    backtest infrastructure - just the direct proxy-correlation check),
    so the composite here is the correlation component alone, scaled
    the same way xg_corr_norm is in compute_league_reliability_score.

    'xg' is provider-tracked expected goals (NOT the territorial
    engine's box-touches/big-chances/SOT proxy) - when a data provider
    supplies real per-match xG, it's typically a substantially stronger
    predictor of results than any indirect proxy, since it's a direct
    shot-quality measurement rather than something reconstructed from
    volume metrics. It still goes through the exact same correlation
    check as every other engine here rather than being hard-coded to a
    high trust level - a provider's xG field could still be noisy or
    inconsistently tracked, and the whole point of this mechanism is
    letting the data demonstrate its own reliability rather than
    assuming it."""
    if engine_name == "setpiece":
        corr = metric_differential_correlation(settled_df, SETPIECE_COUNT_METRICS, SETPIECE_RATE_METRICS, SETPIECE_WEIGHTS_DEFAULT)
    elif engine_name == "transition":
        corr = metric_differential_correlation(settled_df, TRANSITION_COUNT_METRICS, TRANSITION_RATE_METRICS, TRANSITION_WEIGHTS_DEFAULT)
    elif engine_name == "duel":
        corr = metric_differential_correlation(settled_df, [], DUEL_RATE_METRICS, DUEL_WEIGHTS_DEFAULT)
    elif engine_name == "xg":
        corr = metric_differential_correlation(settled_df, XG_METRICS, [], XG_WEIGHTS_DEFAULT)
    elif engine_name == "xgot":
        corr = metric_differential_correlation(settled_df, XGOT_METRICS, [], XGOT_WEIGHTS_DEFAULT)
    elif engine_name == "xg_set_play":
        corr = metric_differential_correlation(settled_df, XG_SET_PLAY_METRICS, [], XG_SET_PLAY_WEIGHTS_DEFAULT)
    elif engine_name == "xg_open_play":
        corr = metric_differential_correlation(settled_df, XG_OPEN_PLAY_METRICS, [], XG_OPEN_PLAY_WEIGHTS_DEFAULT)
    else:
        raise ValueError(f"Unknown engine_name: {engine_name}")

    if corr is None:
        return AuxiliaryEngineReliability(engine_name=engine_name, available=False, proxy_correlation=float("nan"), composite_score=float("nan"))

    corr_norm = float(np.clip((abs(corr) - 0.1) / 0.4, 0, 1)) if not math.isnan(corr) else 0.0
    return AuxiliaryEngineReliability(
        engine_name=engine_name, available=True,
        proxy_correlation=round(corr, 3) if not math.isnan(corr) else float("nan"),
        composite_score=round(corr_norm, 3),
    )


# --- Blending all available engines into one final strength ----------------

def blend_engine_strengths(
    territorial_strength: float, territorial_reliability: float,
    auxiliary_engines: "list[tuple[str, float, AuxiliaryEngineReliability]]" = (),
    engine_weights: "dict | None" = None,
) -> tuple[float, dict]:
    """Combines every AVAILABLE engine's strength ratio into one final
    value, weighted by each engine's own per-league reliability
    composite_score (normalised so the weights actually used sum to 1).
    A league where territorial correlation is weak and set-piece (or xG)
    correlation is strong automatically ends up leaning on that signal
    instead - no manual per-league engine selection required, the same
    principle behind compute_league_reliability_score() extended across
    engines instead of applied to just one.

    auxiliary_engines is a list of (name, strength_value, reliability)
    tuples - pass as many or as few as are available for this fixture;
    entries with strength_value=None or an unavailable reliability are
    skipped automatically, never manufactured. This list-based interface
    (rather than one fixed parameter per engine) is what lets new engines
    - like xG here - get added without changing this function's signature
    again.

    Returns (blended_strength, weight_breakdown) - the breakdown dict is
    for display/audit (which engines contributed, and how much), not
    required by the caller."""
    contributions = [("territorial", territorial_strength, max(territorial_reliability, 0.05))]
    for name, strength_value, reliability in auxiliary_engines:
        if strength_value is not None and reliability is not None and reliability.available:
            contributions.append((name, strength_value, max(reliability.composite_score, 0.05)))

    # Optional per-league override chosen by optimize_engine_weights() (walk-forward Brier score).
    # Weights are used only for engines that are actually available for THIS fixture and are
    # re-normalised over them; an engine may be given weight 0 (territorial included), so a league
    # that is mostly decided by set pieces is no longer forced to lean on territorial data.
    # If none of the chosen engines is available here, fall back to the reliability weights above.
    if engine_weights:
        custom = [(name, val, float(engine_weights.get(name, 0.0))) for name, val, _ in contributions]
        custom = [c for c in custom if c[2] > 0]
        if custom:
            contributions = custom

    total_weight = sum(w for _, _, w in contributions)
    if total_weight <= 0:
        return territorial_strength, {"territorial": 1.0}

    blended = sum(val * w for _, val, w in contributions) / total_weight
    breakdown = {name: round(w / total_weight, 3) for name, _, w in contributions}
    return float(blended), breakdown


# ---------------------------------------------------------------------------
# Optional Telegram notification - the ONE piece of this app that needs
# real internet access (sending a message inherently requires it), unlike
# everything else which stays fully local/offline. Only ever called when
# the user explicitly clicks "Send" - never runs automatically.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ONE strength pipeline shared by the live prediction and the walk-forward backtest
# ---------------------------------------------------------------------------
# Previously the live app blended territorial + auxiliary engines + duel + momentum while
# the backtest (BSS badge, reliability, calibrators) only ran the territorial model, so the
# validation described a different model from the one making predictions. Everything that
# turns history into attack/defence ratings now lives here and both paths call it.

AUX_ENGINE_NAMES = ["setpiece", "transition", "xg", "xg_set_play", "xg_open_play"]


def compute_engine_reliabilities(history_df: pd.DataFrame, weights: tuple = TERRITORY_WEIGHTS_DEFAULT) -> dict:
    """Per-engine trust scores computed ONLY from `history_df` (so a backtest step never sees
    the match it is predicting). The territorial score uses no BSS term (neutral 0.5 there) -
    BSS comes from backtesting this very model, so feeding it back in would be circular."""
    out = {"territorial": compute_league_reliability_score(history_df, None, weights).composite_score}
    for name in AUX_ENGINE_NAMES:
        out[name] = compute_auxiliary_engine_reliability(history_df, name)
    return out


# ---------------------------------------------------------------------------
# Tactical STYLES and automatic MISMATCH multipliers
# ---------------------------------------------------------------------------
# Every team is labelled with a playing style from its own measured ratings (no manual input), and
# every home-style vs away-style pairing - ALL combinations - gets a multiplier on each side's
# expected goals. The multipliers are LEARNED from this league's own walk-forward history: for each
# pairing, "did the goals scored beat or fall short of what the model (without any mismatch term)
# expected?", pooled over the matches where that exact pairing occurred, shrunk towards 1.0 for thin
# samples and capped. So a pairing like low block vs counter-attacking only moves the numbers if this
# league's results actually show an effect - the app never assumes one.

STYLE_DOMINANT = "Dominant / high press"
STYLE_COUNTER = "Counter-attacking / direct"
STYLE_LOW_BLOCK = "Low block"
STYLE_SET_PIECE = "Set-piece reliant"
STYLE_OPEN = "Open / high-event"
STYLE_STRUGGLING = "Struggling (weak both ways)"
STYLE_BALANCED = "Balanced"
TACTICAL_STYLES = [STYLE_DOMINANT, STYLE_COUNTER, STYLE_LOW_BLOCK, STYLE_SET_PIECE, STYLE_OPEN, STYLE_STRUGGLING, STYLE_BALANCED]

MISMATCH_PRIOR_GOALS = 150.0      # shrinkage: a pairing needs ~100+ matches of evidence before it can move far from 1.0
MISMATCH_CAP = (0.90, 1.10)       # a mismatch can never move a side's expected goals by more than +-10%
MISMATCH_MIN_MATCHES_FOR_TUNING = 150


def classify_style(attack, defense, transition_attack=None, setpiece_attack=None) -> str:
    """All inputs are ratios vs the league average (1.0 = average). `defense` is the ALLOWED ratio
    (below 1.0 = concedes less). Priority order matters - first rule that fits wins:
      Dominant       creates a lot AND concedes no more than average (high press / possession control)
      Counter/direct strong transition output (long passes, final-third entries, dribbles) without
                     controlling the territory
      Low block      creates little AND concedes little (compact, deep, hard to break down)
      Set-piece      lives off set-piece output while not creating much in open territory
      Open           creates above average but also leaks (end-to-end, high-event)
      Struggling     creates little AND leaks
      Balanced       everything else"""
    att = 1.0 if attack is None or not np.isfinite(attack) else float(attack)
    dfn = 1.0 if defense is None or not np.isfinite(defense) else float(defense)
    trans = None if transition_attack is None or not np.isfinite(transition_attack) else float(transition_attack)
    sp = None if setpiece_attack is None or not np.isfinite(setpiece_attack) else float(setpiece_attack)
    if att >= 1.08 and dfn <= 1.00:
        return STYLE_DOMINANT
    if trans is not None and trans >= 1.08 and att < 1.08:
        return STYLE_COUNTER
    if att <= 0.95 and dfn <= 1.00:
        return STYLE_LOW_BLOCK
    if sp is not None and sp >= 1.10 and att < 1.05:
        return STYLE_SET_PIECE
    if att >= 1.00 and dfn >= 1.08:
        return STYLE_OPEN
    if att <= 0.95 and dfn >= 1.05:
        return STYLE_STRUGGLING
    return STYLE_BALANCED


def style_from_inputs(inp: dict, side: str) -> str:
    """Style of the home or away team from the raw (pre-blend, pre-multiplier) strength ratios."""
    k_att, k_def = ("ha", "hd") if side == "home" else ("aa", "ad")
    return classify_style(inp["terr"][k_att], inp["terr"][k_def], inp["eng"]["transition"][k_att], inp["eng"]["setpiece"][k_att])


class MismatchLedger:
    """Running per-pairing totals of (goals scored, goals the base model expected). Because it is
    filled strictly chronologically in a walk-forward pass, the multiplier read for a match only
    ever reflects matches played BEFORE it - no look-ahead."""

    def __init__(self, prior_goals: float = MISMATCH_PRIOR_GOALS):
        self.prior = prior_goals
        self.cells: dict = {}

    def add(self, home_style, away_style, goals_home, lam_home, goals_away, lam_away):
        c = self.cells.setdefault((home_style, away_style), [0.0, 0.0, 0.0, 0.0, 0])
        c[0] += float(goals_home); c[1] += float(lam_home); c[2] += float(goals_away); c[3] += float(lam_away); c[4] += 1

    def raw_multipliers(self, home_style, away_style) -> tuple:
        c = self.cells.get((home_style, away_style))
        if not c:
            return 1.0, 1.0
        lo, hi = MISMATCH_CAP
        return (float(np.clip((c[0] + self.prior) / (c[1] + self.prior), lo, hi)),
                float(np.clip((c[2] + self.prior) / (c[3] + self.prior), lo, hi)))

    def table(self) -> dict:
        out = {}
        for (hs, a_s), c in self.cells.items():
            mh, ma = self.raw_multipliers(hs, a_s)
            out[(hs, a_s)] = {"home": mh, "away": ma, "n": int(c[4])}
        return out


def compute_mismatch_table(backtest_df: pd.DataFrame) -> dict:
    """Learned multiplier for every home-style x away-style pairing seen in a walk-forward backtest
    (needs the lam_*_base / *_style columns walk_forward_backtest records)."""
    ledger = MismatchLedger()
    needed = ["home_style", "away_style", "lam_home_base", "lam_away_base", "home_goals", "away_goals"]
    if backtest_df is None or backtest_df.empty or any(c not in backtest_df.columns for c in needed):
        return {}
    for r in backtest_df.dropna(subset=needed).itertuples(index=False):
        ledger.add(r.home_style, r.away_style, r.home_goals, r.lam_home_base, r.away_goals, r.lam_away_base)
    return ledger.table()


def mismatch_multipliers(table: dict, home_style: str, away_style: str, strength: float = 1.0) -> tuple:
    """(home multiplier, away multiplier) for this pairing at the league's tuned `strength`
    (0 = off, 1 = full learned effect). Unknown pairings return (1.0, 1.0)."""
    cell = (table or {}).get((home_style, away_style))
    if not cell:
        return 1.0, 1.0
    return 1.0 + strength * (cell["home"] - 1.0), 1.0 + strength * (cell["away"] - 1.0)


def serialize_mismatch_table(table: dict) -> dict:
    return {f"{h}||{a}": v for (h, a), v in (table or {}).items()}


def deserialize_mismatch_table(blob: dict) -> dict:
    out = {}
    for k, v in (blob or {}).items():
        h, a = k.split("||", 1)
        out[(h, a)] = v
    return out


@dataclass
class MatchStrengths:
    home_attack: float            # final ratings used for the Poisson means (blend x duel x momentum)
    home_defense: float
    away_attack: float
    away_defense: float
    home_attack_raw: float        # blended attack BEFORE momentum (for the explanation panel)
    away_attack_raw: float
    home_profile: object
    away_profile: object
    home_attack_territorial: float
    home_defense_territorial: float
    away_attack_territorial: float
    away_defense_territorial: float
    home_attack_breakdown: dict
    home_defense_breakdown: dict
    away_attack_breakdown: dict
    away_defense_breakdown: dict
    home_du_defense: "float | None"
    away_du_defense: "float | None"
    home_mom_mult: float
    home_mom_desc: str
    away_mom_mult: float
    away_mom_desc: str
    baseline: LeagueBaseline
    reliabilities: dict
    home_style: str = STYLE_BALANCED
    away_style: str = STYLE_BALANCED


def _engine_specs():
    return [
        ("setpiece", setpiece_profile, setpiece_strength, SETPIECE_COUNT_METRICS + SETPIECE_RATE_METRICS),
        ("transition", transition_profile, transition_strength, TRANSITION_COUNT_METRICS + TRANSITION_RATE_METRICS),
        ("xg", xg_profile, xg_strength, XG_METRICS),
        ("xg_set_play", xg_set_play_profile, xg_set_play_strength, XG_SET_PLAY_METRICS),
        ("xg_open_play", xg_open_play_profile, xg_open_play_strength, XG_OPEN_PLAY_METRICS),
    ]


def gather_engine_inputs(
    history_df: pd.DataFrame, home_team: str, away_team: str, half_life_days: float,
    reference_date: pd.Timestamp, weights: tuple = TERRITORY_WEIGHTS_DEFAULT,
    use_match_index: bool = False, include_momentum: bool = True, baseline: "LeagueBaseline | None" = None,
) -> dict:
    """Step 1 of the strength pipeline: every engine's raw strength ratios for this fixture, BEFORE
    they are blended. Split out so the engine-weight optimiser can compute these once per historical
    match and then try many different blends cheaply."""
    if baseline is None:
        baseline = compute_league_baseline(history_df)
    kw = dict(use_match_index=use_match_index)
    hp = team_territory_profile(history_df, home_team, "home", half_life_days, reference_date, **kw)
    ap = team_territory_profile(history_df, away_team, "away", half_life_days, reference_date, **kw)
    terr = {
        "ha": attack_strength(hp, baseline, "home", weights), "hd": defense_strength(hp, baseline, "home", weights),
        "aa": attack_strength(ap, baseline, "away", weights), "ad": defense_strength(ap, baseline, "away", weights),
    }
    eng = {}
    for name, profile_fn, strength_fn, metrics in _engine_specs():
        eng_baseline = compute_league_metric_baseline(history_df, metrics)
        h_prof = profile_fn(history_df, home_team, "home", half_life_days, reference_date, **kw)
        a_prof = profile_fn(history_df, away_team, "away", half_life_days, reference_date, **kw)
        eng[name] = {
            "ha": strength_fn(h_prof, eng_baseline, "home", direction="for"),
            "hd": strength_fn(h_prof, eng_baseline, "home", direction="against"),
            "aa": strength_fn(a_prof, eng_baseline, "away", direction="for"),
            "ad": strength_fn(a_prof, eng_baseline, "away", direction="against"),
        }
    du_baseline = compute_league_metric_baseline(history_df, DUEL_RATE_METRICS)
    home_du = duel_defense_strength(duel_profile(history_df, home_team, "home", half_life_days, reference_date, **kw), du_baseline, "home")
    away_du = duel_defense_strength(duel_profile(history_df, away_team, "away", half_life_days, reference_date, **kw), du_baseline, "away")
    h_mom, h_mom_desc, a_mom, a_mom_desc = 1.0, "", 1.0, ""
    if include_momentum:
        h_mom, h_mom_desc = team_streak_multiplier(history_df, home_team)
        a_mom, a_mom_desc = team_streak_multiplier(history_df, away_team)
    return dict(baseline=baseline, hp=hp, ap=ap, terr=terr, eng=eng, home_du=home_du, away_du=away_du,
                h_mom=h_mom, h_mom_desc=h_mom_desc, a_mom=a_mom, a_mom_desc=a_mom_desc)


def blend_match_strengths(
    inp: dict, reliabilities: dict, engine_weights: "dict | None" = None, duel_strength: float = 1.0,
) -> MatchStrengths:
    """Step 2: blend the engines (reliability-weighted by default, or with the league's optimised
    engine_weights), then apply the duel multiplier on defence (scaled by the league's tuned
    `duel_strength`: 0 = ignore duels, 1 = full effect) and momentum on attack."""
    terr_rel = reliabilities["territorial"]

    def blend(key, terr_value):
        aux = [(name, inp["eng"][name][key], reliabilities[name]) for name in AUX_ENGINE_NAMES]
        return blend_engine_strengths(terr_value, terr_rel, aux, engine_weights)

    home_attack, home_attack_bd = blend("ha", inp["terr"]["ha"])
    home_defense, home_defense_bd = blend("hd", inp["terr"]["hd"])
    away_attack, away_attack_bd = blend("aa", inp["terr"]["aa"])
    away_defense, away_defense_bd = blend("ad", inp["terr"]["ad"])
    if inp["home_du"] is not None:
        home_defense *= 1.0 + duel_strength * (inp["home_du"] - 1.0)
    if inp["away_du"] is not None:
        away_defense *= 1.0 + duel_strength * (inp["away_du"] - 1.0)
    home_attack_raw, away_attack_raw = home_attack, away_attack
    home_attack *= inp["h_mom"]
    away_attack *= inp["a_mom"]
    return MatchStrengths(
        home_attack, home_defense, away_attack, away_defense, home_attack_raw, away_attack_raw,
        inp["hp"], inp["ap"], inp["terr"]["ha"], inp["terr"]["hd"], inp["terr"]["aa"], inp["terr"]["ad"],
        home_attack_bd, home_defense_bd, away_attack_bd, away_defense_bd,
        inp["home_du"], inp["away_du"], inp["h_mom"], inp["h_mom_desc"], inp["a_mom"], inp["a_mom_desc"],
        inp["baseline"], reliabilities,
        home_style=style_from_inputs(inp, "home"), away_style=style_from_inputs(inp, "away"),
    )


def compute_match_strengths(
    history_df: pd.DataFrame, home_team: str, away_team: str, half_life_days: float,
    reference_date: pd.Timestamp, weights: tuple = TERRITORY_WEIGHTS_DEFAULT,
    use_match_index: bool = False, reliabilities: "dict | None" = None,
    include_momentum: bool = True, baseline: "LeagueBaseline | None" = None,
    engine_weights: "dict | None" = None, duel_strength: float = 1.0,
) -> MatchStrengths:
    inp = gather_engine_inputs(history_df, home_team, away_team, half_life_days, reference_date, weights,
                               use_match_index, include_momentum, baseline)
    if reliabilities is None:
        reliabilities = compute_engine_reliabilities(history_df, weights)
    return blend_match_strengths(inp, reliabilities, engine_weights, duel_strength)


# ---------------------------------------------------------------------------
# Per-league ENGINE WEIGHT optimisation (walk-forward Brier score, holdout-validated)
# ---------------------------------------------------------------------------
# By default engines are blended by their correlation-based reliability, and the territorial engine
# is always present. optimize_engine_weights() lets the DATA decide instead: it searches blends of
# territorial / set-piece / transition / xG / xG-set-play / xG-open-play (any engine, including
# territorial, can end up with weight ~0), scoring each by the Brier score of walk-forward 1X2
# predictions. The blend is chosen on the earlier part of the evaluation window and only ADOPTED if
# it also beats the automatic weighting on the later, untouched part - so a lucky fit to noise is
# not silently trusted.

def _outcome_probs_vec(lam_h: np.ndarray, lam_a: np.ndarray):
    k = np.arange(GOAL_CAP + 1)[:, None]
    ph = scipy_poisson.pmf(k, lam_h[None, :])
    pa = scipy_poisson.pmf(k, lam_a[None, :])
    joint = np.einsum("in,jn->nij", ph, pa)
    idx = np.arange(GOAL_CAP + 1)
    home_mask = idx[:, None] > idx[None, :]
    draw_mask = idx[:, None] == idx[None, :]
    away_mask = ~(home_mask | draw_mask)
    p_h, p_d, p_a = [(joint * m).sum(axis=(1, 2)) for m in (home_mask, draw_mask, away_mask)]
    total = p_h + p_d + p_a
    return p_h / total, p_d / total, p_a / total


def _brier_per_match(lam_h: np.ndarray, lam_a: np.ndarray, actual: np.ndarray) -> np.ndarray:
    p_h, p_d, p_a = _outcome_probs_vec(np.asarray(lam_h, float), np.asarray(lam_a, float))
    return (p_h - (actual == "H")) ** 2 + (p_d - (actual == "D")) ** 2 + (p_a - (actual == "A")) ** 2


def _records_brier(records: list, engine_weights: "dict | None", per_match: bool = False, duel_strength: float = 1.0):
    lam_h, lam_a, actual = [], [], []
    for rec in records:
        ms = blend_match_strengths(rec["inp"], rec["rel"], engine_weights, duel_strength)
        lh, la = expected_goals(ms.home_attack, ms.away_defense, ms.away_attack, ms.home_defense, rec["inp"]["baseline"])
        lam_h.append(lh); lam_a.append(la); actual.append(rec["actual"])
    per = _brier_per_match(np.array(lam_h), np.array(lam_a), np.array(actual))
    return per if per_match else float(np.mean(per))


def _paired_z(candidate_per_match: np.ndarray, reference_per_match: np.ndarray) -> tuple:
    """(mean improvement of candidate over reference, z-score). z <= -1 means 'better by >= 1 standard error'."""
    diff = np.asarray(candidate_per_match) - np.asarray(reference_per_match)
    if len(diff) < 3 or np.std(diff, ddof=1) <= 0:
        return float(-np.mean(diff)), 0.0
    return float(-np.mean(diff)), float(np.mean(diff) / (np.std(diff, ddof=1) / np.sqrt(len(diff))))


def _collect_records(settled_df, half_life_days, weights, max_matches_evaluated, reliability_refresh_every):
    df = settled_df.dropna(subset=["date", "home_goals", "away_goals"]).sort_values("date").reset_index(drop=True)
    if len(df) < MIN_SAMPLE_ROWS * 8:
        return None, f"only {len(df)} settled matches - not enough to optimise reliably (need about {MIN_SAMPLE_ROWS * 8}+); staying on automatic settings."
    eval_start = max(MIN_SAMPLE_ROWS * 3, len(df) - max_matches_evaluated)
    records, rel, rel_at = [], None, -10 ** 9
    for i in range(eval_start, len(df)):
        history, row = df.iloc[:i], df.iloc[i]
        if rel is None or i - rel_at >= reliability_refresh_every:
            rel, rel_at = compute_engine_reliabilities(history, weights), i  # history only: no look-ahead
        inp = gather_engine_inputs(history, row["home_team"], row["away_team"], half_life_days, row["date"], weights)
        hg, ag = row["home_goals"], row["away_goals"]
        records.append({"inp": inp, "rel": rel, "actual": "H" if hg > ag else ("D" if hg == ag else "A")})
    if len(records) < 60:
        return None, "fewer than 60 evaluable matches - staying on automatic settings."
    return records, None


def _search_engine_weights(records, holdout_fraction=0.30, step=0.05, max_iterations=15):
    n_val = max(20, int(len(records) * holdout_fraction))
    train, valid = records[:-n_val], records[-n_val:]
    available = ["territorial"] + [
        n for n in AUX_ENGINE_NAMES
        if any(r["inp"]["eng"][n]["ha"] is not None and r["rel"][n].available for r in train)
    ]
    last = records[-1]
    auto_raw = {"territorial": max(last["rel"]["territorial"], 0.05)}
    for n in available[1:]:
        auto_raw[n] = max(last["rel"][n].composite_score, 0.05)
    auto_snapshot = {n: w / sum(auto_raw.values()) for n, w in auto_raw.items()}

    def norm(wd):
        tot = sum(wd.values())
        return {k: v / tot for k, v in wd.items() if v > 1e-9}

    candidates = {"automatic (reliability-weighted)": None, "equal across engines": {n: 1.0 for n in available}}
    for e in available:
        candidates[f"only {e}"] = {e: 1.0}
        if len(available) > 1:
            candidates[f"lean on {e} (60%)"] = {n: (0.6 if n == e else 0.4 / (len(available) - 1)) for n in available}
    train_scores = {name: _records_brier(train, None if w is None else norm(w)) for name, w in candidates.items()}
    best_name = min(train_scores, key=train_scores.get)
    best_w = norm(candidates[best_name]) if candidates[best_name] is not None else dict(auto_snapshot)
    best_score = train_scores[best_name]
    for _ in range(max_iterations):  # hill-climb: move 5% of weight between engines while Brier improves
        improved = None
        for src_e in list(best_w):
            for dst_e in available:
                if dst_e == src_e:
                    continue
                move = min(step, best_w[src_e])
                trial = dict(best_w); trial[src_e] -= move; trial[dst_e] = trial.get(dst_e, 0.0) + move
                trial = norm(trial)
                score = _records_brier(train, trial)
                if score < best_score - 1e-6 and (improved is None or score < improved[0]):
                    improved = (score, trial)
        if improved is None:
            break
        best_score, best_w = improved
    chosen = {k: round(v, 3) for k, v in sorted(best_w.items(), key=lambda kv: -kv[1])}

    val = {
        "automatic": _records_brier(valid, None),
        "chosen": _records_brier(valid, chosen),
        "equal": _records_brier(valid, norm({n: 1.0 for n in available})),
        "territorial only": _records_brier(valid, {"territorial": 1.0}),
    }
    improvement, z = _paired_z(_records_brier(valid, chosen, per_match=True), _records_brier(valid, None, per_match=True))
    adopted = bool(improvement > 0 and z <= -1.0)
    val["improvement_vs_automatic"], val["improvement_z_score"] = improvement, z
    info = {
        "chosen_weights": chosen, "adopted": adopted, "n_train": len(train), "n_validation": len(valid),
        "train_brier": {"chosen (after fine-tuning)": best_score, **train_scores},
        "validation_brier": val, "engines_available": available,
        "reason": ("Chosen blend beat the automatic weighting on the held-out matches - now in use for this league."
                   if adopted else
                   "The searched blend did not clearly beat the automatic weighting on held-out matches (difference within noise, or overfit to the training window) - keeping automatic weights."),
    }
    return (chosen if adopted else None), info, train, valid


def optimize_engine_weights(
    settled_df: pd.DataFrame, half_life_days: float, weights: tuple = TERRITORY_WEIGHTS_DEFAULT,
    max_matches_evaluated: int = 300, holdout_fraction: float = 0.30, step: float = 0.05,
    reliability_refresh_every: int = 25, max_iterations: int = 15,
):
    """Returns (engine_weights | None, info). engine_weights is a dict {engine: weight} summing to 1
    (engines not listed get 0), or None when the data does not support beating the automatic
    reliability weighting (in which case the app keeps using the automatic weights)."""
    records, reason = _collect_records(settled_df, half_life_days, weights, max_matches_evaluated, reliability_refresh_every)
    if records is None:
        return None, {"reason": reason}
    chosen, info, _, _ = _search_engine_weights(records, holdout_fraction, step, max_iterations)
    return chosen, info


DUEL_STRENGTH_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
MISMATCH_STRENGTH_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def optimize_mismatch_strength(backtest_df: pd.DataFrame, holdout_fraction: float = 0.30):
    """Tunes how much of the learned tactical-mismatch effect to apply (0 = none, 1 = full), by Brier
    score on a walk-forward backtest that recorded the model's mismatch-free expected goals and each
    match's style pairing. Default stays at 1.0 (automatic, full) unless another strength is clearly
    better (paired z <= -1) on the held-out matches."""
    need = ["home_style", "away_style", "lam_home_base", "lam_away_base", "home_goals", "away_goals", "actual"]
    d = backtest_df.dropna(subset=["home_style", "away_style", "lam_home_base", "lam_away_base"]) if backtest_df is not None and not backtest_df.empty and all(c in backtest_df.columns for c in need) else None
    if d is None or len(d) < MISMATCH_MIN_MATCHES_FOR_TUNING:
        return 1.0, {"reason": f"fewer than {MISMATCH_MIN_MATCHES_FOR_TUNING} matches with style data - mismatch multipliers stay at the default full strength.", "adopted": False}
    d = d.reset_index(drop=True)
    ledger = MismatchLedger()
    raw_h, raw_a = np.ones(len(d)), np.ones(len(d))
    for i, r in enumerate(d.itertuples(index=False)):
        raw_h[i], raw_a[i] = ledger.raw_multipliers(r.home_style, r.away_style)  # BEFORE this match is added
        ledger.add(r.home_style, r.away_style, r.home_goals, r.lam_home_base, r.away_goals, r.lam_away_base)
    base_h, base_a, actual = d["lam_home_base"].to_numpy(float), d["lam_away_base"].to_numpy(float), d["actual"].to_numpy()
    per = {sv: _brier_per_match(np.maximum(0.05, base_h * (1 + sv * (raw_h - 1))), np.maximum(0.05, base_a * (1 + sv * (raw_a - 1))), actual)
           for sv in MISMATCH_STRENGTH_GRID}
    n_val = max(30, int(len(d) * holdout_fraction))
    train_scores = {sv: float(np.mean(per[sv][:-n_val])) for sv in per}
    best = min(train_scores, key=train_scores.get)
    improvement, z = _paired_z(per[best][-n_val:], per[1.0][-n_val:])
    adopted = bool(best != 1.0 and improvement > 0 and z <= -1.0)
    chosen = best if adopted else 1.0
    return chosen, {
        "adopted": adopted, "chosen_strength": chosen, "best_on_training": best, "n_matches": len(d), "n_validation": n_val,
        "validation_brier": {sv: float(np.mean(per[sv][-n_val:])) for sv in per},
        "improvement_vs_default": improvement, "improvement_z_score": z,
        "reason": (f"Mismatch strength tuned to {chosen:.2f} (clearly better than the default 1.00 on held-out matches)." if adopted else
                   "Default full-strength mismatch multipliers kept (no other strength was clearly better on held-out matches)."),
    }


def optimize_league_tactics(
    settled_df: pd.DataFrame, half_life_days: float, weights: tuple = TERRITORY_WEIGHTS_DEFAULT,
    max_matches_evaluated: int = 300, mismatch_history: int = 600, reliability_refresh_every: int = 25,
):
    """One pass that tunes everything league-specific about HOW the model reads tactics, always
    scored by walk-forward Brier score and adopted only when it beats the default on held-out matches:
      1. engine weights   - territorial / set-piece / transition / xG / xG set play / xG open play
      2. duel strength    - how much the duel-resilience defence multiplier counts (0 = off ... 1 = full)
      3. mismatch strength- how much of the learned style-vs-style multiplier to apply
    Returns (config, info); config = {"weights": dict|None, "duel_strength": float, "mismatch_strength": float}.
    Call again whenever the territory weights change - every step here is scored with them."""
    default_cfg = {"weights": None, "duel_strength": 1.0, "mismatch_strength": 1.0}
    records, reason = _collect_records(settled_df, half_life_days, weights, max_matches_evaluated, reliability_refresh_every)
    if records is None:
        return default_cfg, {"reason": reason, "engine": {"reason": reason}, "duel": {}, "mismatch": {}}
    chosen_w, eng_info, train, valid = _search_engine_weights(records)

    train_scores = {sv: _records_brier(train, chosen_w, duel_strength=sv) for sv in DUEL_STRENGTH_GRID}
    best_duel = min(train_scores, key=train_scores.get)
    d_improve, d_z = _paired_z(_records_brier(valid, chosen_w, per_match=True, duel_strength=best_duel),
                               _records_brier(valid, chosen_w, per_match=True, duel_strength=1.0))
    duel_adopted = bool(best_duel != 1.0 and d_improve > 0 and d_z <= -1.0)
    duel_strength = best_duel if duel_adopted else 1.0
    duel_info = {
        "adopted": duel_adopted, "chosen_strength": duel_strength, "best_on_training": best_duel,
        "validation_brier": {sv: _records_brier(valid, chosen_w, duel_strength=sv) for sv in DUEL_STRENGTH_GRID},
        "improvement_vs_default": d_improve, "improvement_z_score": d_z,
        "reason": (f"Duel strength tuned to {duel_strength:.2f} (clearly better than full strength on held-out matches)." if duel_adopted
                   else "Duel multiplier kept at full strength (no other strength was clearly better on held-out matches)."),
    }

    tail = settled_df.dropna(subset=["date", "home_goals", "away_goals"]).sort_values("date").tail(mismatch_history)
    bt = walk_forward_backtest(tail, half_life_days=half_life_days, weights=weights, include_all_markets=False,
                               full_model=True, engine_weights=chosen_w, duel_strength=duel_strength, mismatch_strength=0.0,
                               reliability_refresh_every=reliability_refresh_every)
    mismatch_strength, mm_info = optimize_mismatch_strength(bt)
    cfg = {"weights": chosen_w, "duel_strength": duel_strength, "mismatch_strength": mismatch_strength}
    return cfg, {"engine": eng_info, "duel": duel_info, "mismatch": mm_info, "mismatch_table": compute_mismatch_table(bt),
                 "reason": eng_info.get("reason", "")}


# ---------------------------------------------------------------------------
# Upload schema check (runs BEFORE anything is saved)
# ---------------------------------------------------------------------------

@dataclass
class SchemaReport:
    errors: list = field(default_factory=list)    # blocks the upload
    warnings: list = field(default_factory=list)  # saved, but you should know
    info: list = field(default_factory=list)
    fixes: list = field(default_factory=list)     # things that were corrected automatically

    @property
    def ok(self) -> bool:
        return not self.errors


_METRIC_ALIASES = {"big_chances_created": "big_chances", **COLUMN_RENAMES}


def _apply_column_aliases(df: pd.DataFrame, report: SchemaReport) -> pd.DataFrame:
    """Accepts the 'metric_home / metric_away' naming (e.g. goals_home, team_home,
    big_chances_created_home) by converting it to this engine's 'home_metric / away_metric'
    naming. Only runs when the engine's own naming is not already present, and never
    overwrites an existing column."""
    renamed = {}
    existing = set(df.columns)
    if not {"home_team", "away_team"} <= existing:
        for c in list(df.columns):
            for suffix, prefix in (("_home", "home_"), ("_away", "away_")):
                if c.endswith(suffix):
                    base = _METRIC_ALIASES.get(c[: -len(suffix)], c[: -len(suffix)])
                    target = f"{prefix}{base}"
                    if target not in existing and target not in renamed.values():
                        renamed[c] = target
    for prefix in ("home_", "away_"):
        for old, new in _METRIC_ALIASES.items():
            c = f"{prefix}{old}"
            if c in df.columns and f"{prefix}{new}" not in existing and f"{prefix}{new}" not in renamed.values():
                renamed[c] = f"{prefix}{new}"
    if renamed:
        df = df.rename(columns=renamed)
        report.fixes.append(f"Renamed {len(renamed)} columns to the engine's naming (e.g. " + ", ".join(f"{a} -> {b}" for a, b in list(renamed.items())[:4]) + ").")
    return df


def prepare_upload(raw_df: pd.DataFrame) -> "tuple[pd.DataFrame, SchemaReport]":
    """Cleans and validates a freshly uploaded match file. Returns (clean_df, report); if
    report.errors is non-empty the file must NOT be saved."""
    report = SchemaReport()
    df = standardise_columns(raw_df)
    df = _apply_column_aliases(df, report)

    # 1. required columns -----------------------------------------------------------
    date_col = find_date_column(df)
    required = [c for c in REQUIRED_BASE_COLUMNS if c != "date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        report.errors.append(
            "Missing required column(s): " + ", ".join(missing) + ". The engine needs home_team, away_team, home_goals, "
            "away_goals plus home_/away_ shots_on_target, big_chances and box_touches (the 'metric_home' naming, e.g. "
            "goals_home, is converted automatically). Without them predictions cannot be built correctly."
        )
        return df, report
    if date_col is None:
        report.warnings.append("No date column found - time-decay weighting and season detection fall back to match order.")

    # 2. whitespace hygiene on text columns ----------------------------------------
    text_cols = ["home_team", "away_team"] + [c for c in (find_division_column(df), find_season_column(df)) if c]
    changed = 0
    for c in text_cols:
        cleaned = df[c].apply(lambda v: re.sub(r"\s+", " ", str(v)).strip() if pd.notna(v) else v)
        changed += int((cleaned.fillna("") != df[c].fillna("")).sum())
        df[c] = cleaned
    if changed:
        report.fixes.append(f"Trimmed stray spaces in {changed} team/league/season cells (e.g. 'Chelsea ' vs 'Chelsea' would otherwise be two teams).")

    blank_team = df["home_team"].isna() | df["away_team"].isna() | (df["home_team"] == "") | (df["away_team"] == "")
    if blank_team.any():
        report.fixes.append(f"Removed {int(blank_team.sum())} row(s) with a blank home or away team name.")
        df = df.loc[~blank_team].copy()
    same_team = df["home_team"].str.lower() == df["away_team"].str.lower()
    if same_team.any():
        report.warnings.append(f"{int(same_team.sum())} row(s) have the same team as home and away - check them.")

    # 3. duplicates ------------------------------------------------------------------
    key_parts = [df["home_team"].str.lower(), df["away_team"].str.lower()]
    div_col = find_division_column(df)
    if div_col:
        key_parts.insert(0, df[div_col].astype(str).str.lower())
    if date_col:
        parsed_for_key = pd.to_datetime(df[date_col], errors="coerce", format="mixed").dt.strftime("%Y-%m-%d").fillna(df[date_col].astype(str))
        key_parts.insert(0, parsed_for_key)
    key = key_parts[0].astype(str)
    for part in key_parts[1:]:
        key = key + "|" + part.astype(str)
    completeness = df.notna().sum(axis=1).to_numpy()
    order = np.argsort(-completeness, kind="stable")
    dup_sorted = key.iloc[order].duplicated().to_numpy()
    keep_positions = np.sort(order[~dup_sorted])
    n_dupes = len(df) - len(keep_positions)
    if n_dupes:
        df = df.iloc[keep_positions].copy()
        report.fixes.append(f"Removed {n_dupes} duplicate match row(s) (same league/date/home/away) - the most complete copy was kept.")

    # 4. played matches + stat coverage ---------------------------------------------
    played = ~df.apply(lambda r: is_unplayed(r["home_goals"], r["away_goals"]), axis=1)
    n_played = int(played.sum())
    report.info.append(f"{len(df)} rows: {n_played} played, {len(df) - n_played} unplayed/upcoming.")
    if n_played == 0:
        report.warnings.append("No played matches found - the file contains fixtures only, so nothing can be modelled from it alone.")
    else:
        played_df = df.loc[played]
        for col in required:
            if col in ("home_team", "away_team"):
                continue
            raw = played_df[col].replace("", np.nan)
            nums = pd.to_numeric(raw, errors="coerce")
            non_numeric = int((raw.notna() & nums.isna()).sum())
            coverage = float(nums.notna().mean())
            if non_numeric:
                report.warnings.append(f"'{col}' has {non_numeric} non-numeric value(s) among played matches (treated as missing).")
            if coverage == 0:
                report.errors.append(f"'{col}' is empty for EVERY played match - predictions that depend on it would be meaningless.")
            elif coverage < 0.90:
                by_team = nums.isna().groupby(played_df["home_team" if col.startswith("home_") else "away_team"]).mean().sort_values(ascending=False)
                worst = ", ".join(f"{t} ({v:.0%} missing)" for t, v in by_team.head(3).items() if v > 0)
                report.warnings.append(f"'{col}' is only {coverage:.0%} filled among played matches. Missing values are EXCLUDED (not counted as 0), but thin coverage weakens the ratings. Worst: {worst}.")
            if (nums < 0).any():
                report.warnings.append(f"'{col}' contains negative values.")
        goals = pd.to_numeric(played_df["home_goals"], errors="coerce")
        if ((goals % 1 != 0) | (goals > 20)).any():
            report.warnings.append("home_goals contains non-integer or implausibly large values (>20).")

    # 5. dates -------------------------------------------------------------------------
    if date_col:
        raw_dates = df[date_col]
        parsed = pd.to_datetime(raw_dates, errors="coerce", format="mixed")
        unreadable = raw_dates.notna() & (raw_dates.astype(str).str.strip() != "") & parsed.isna()
        blank_played = (raw_dates.isna() | (raw_dates.astype(str).str.strip() == "")) & played
        if unreadable.any():
            examples = ", ".join(map(str, raw_dates[unreadable].head(3).tolist()))
            report.warnings.append(f"{int(unreadable.sum())} row(s) have an unreadable date (e.g. {examples}). Those rows are excluded from time-weighted calculations.")
        if blank_played.any():
            report.warnings.append(f"{int(blank_played.sum())} played match(es) have no date and will be excluded from time-weighted calculations.")

    # 6. auxiliary engines + season detection (information) -----------------------------
    probe = df.copy()
    for c in probe.columns:
        if c.startswith(("home_", "away_")) and c not in ("home_team", "away_team"):
            probe[c] = pd.to_numeric(probe[c], errors="coerce")
    engines = {
        "set-piece": SETPIECE_COUNT_METRICS + SETPIECE_RATE_METRICS, "transition": TRANSITION_COUNT_METRICS + TRANSITION_RATE_METRICS,
        "duel": DUEL_RATE_METRICS, "xG": XG_METRICS, "xG set play": XG_SET_PLAY_METRICS, "xG open play": XG_OPEN_PLAY_METRICS,
    }
    have = [name for name, metrics in engines.items() if _columns_available(probe, metrics)]
    lack = [name for name in engines if name not in have]
    report.info.append("Auxiliary engines usable from this file: " + (", ".join(have) if have else "none") + (f". Not available (columns missing/empty): {', '.join(lack)}." if lack else "."))
    try:
        probe_dates = probe.copy()
        if date_col:
            probe_dates["date"] = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
        _, method = derive_season_series(probe_dates)
        if method == "fallback":
            report.warnings.append("No season column and no usable dates - the season-aware league baseline cannot tell seasons apart and will pool all matches.")
        else:
            report.info.append(f"Seasons identified via: {method}.")
    except Exception:
        pass
    return df, report


def send_telegram_message(bot_token: str, chat_id: str, text: str) -> tuple[bool, str]:
    if not bot_token or not chat_id:
        return False, "Bot token and chat ID are both required."
    if _requests is None:
        return False, "The 'requests' package isn't installed - run: pip install requests"
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        resp = _requests.post(url, data={"chat_id": chat_id, "text": text}, timeout=10)
        if resp.status_code == 200:
            return True, "Sent."
        return False, f"Telegram API returned HTTP {resp.status_code}: {resp.text[:200]}"
    except Exception as exc:  # noqa: BLE001
        return False, f"Request failed: {exc}"


