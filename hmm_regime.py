"""
hmm_regime.py
-------------
A 3-state Hidden Markov Model for gold's macro regime, built on the same
daily data your briefing already caches (price_history.csv).

WHY 3 STATES, NOT 5: tested head-to-head against your actual ~1 year of
data. 5 states catastrophically overfit (train/test log-likelihood gap of
~248 nats/observation -- the model was memorizing noise, not finding real
structure), and even 4 states added nothing: the model left one of its 4
slots completely unused and reproduced the exact same 3 clusters found at
n=3. The data currently supports 3 real, distinguishable states:

    - Range        : quiet, near-zero average daily return
    - Rising        : positive drift, moderate volatility
    - Shock/Selloff : rare, large-magnitude down days (a handful of extreme
                      events, not a regime the market spends real time in --
                      treat this as an anomaly flag more than a "state" you
                      expect to sit in for days at a stretch)

As more daily data accumulates in price_history.csv (this file grows every
time you run main.py), re-run the 3 vs 4 vs 5 comparison periodically --
more history may eventually support more states. Don't just bump
N_STATES below without re-validating; that's exactly the overfitting trap
we hit before.
"""

import warnings

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

N_STATES = 3
FEATURE_COLS = ["gold_ret", "dxy_ret", "real_yield_chg", "gold_vol_5d"]
STATE_LABELS_ASCENDING = ["Declining", "Range", "Rising"]  # sorted by mean gold_ret; NOT
# hardcoded as "rare" anymore -- earlier testing showed this state can be
# 20%+ of days depending on the sample, which isn't rare at all. It
# represents the cluster with the lowest average daily return, which in a
# grinding multi-month decline can mean "sustained decline days," not
# isolated shock events. Actual frequency is reported alongside the label
# so you can judge for yourself rather than trust a static name.

warnings.filterwarnings("ignore", category=UserWarning, module="hmmlearn")


ROLL_ARTIFACT_THRESHOLD = 0.06  # same threshold used in xau_currency_score.py,
# kept consistent so both modules agree on what counts as implausible


def flag_roll_artifacts(feat_df: pd.DataFrame, threshold=ROLL_ARTIFACT_THRESHOLD):
    """
    Flags days where gold's single-day return exceeds `threshold` --
    real gold essentially never moves this much in one day outside a
    genuine historic crisis, so this is almost always a GC=F futures
    contract-roll artifact (we found exactly this pattern before: an
    "11% single-day drop" that turned out to be a data splicing glitch,
    not a real market event).

    This was previously only applied in xau_currency_score.py; extending
    it here so the main regime pipeline (and --extended specifically,
    which covers ~23 years and therefore many more roll events) gets the
    same protection.
    """
    df = feat_df.copy()
    df["likely_roll_artifact"] = df["gold_ret"].abs() > threshold
    return df


def build_features(price_df: pd.DataFrame) -> pd.DataFrame:
    """Engineer the 4 features the HMM trains on from raw cached price data.
    Returns an empty DataFrame (not a raised error) if required columns are
    missing -- e.g. FRED_API_KEY wasn't set this run, so real yield data
    wasn't fetched. That's a normal, recoverable situation, not a crash."""
    df = price_df.copy()
    required = {"gold_spot", "dxy", "dfii10"}
    missing = required - set(df.columns)
    if missing:
        print(f"[warn] Regime model needs {missing}, not present this run "
              f"(likely FRED_API_KEY wasn't set, or market data fetch failed). "
              f"Skipping regime model for today.")
        return pd.DataFrame()

    df = df.dropna(subset=["gold_spot", "dxy", "dfii10"])
    df["gold_ret"] = df["gold_spot"].pct_change()
    df["dxy_ret"] = df["dxy"].pct_change()
    df["real_yield_chg"] = df["dfii10"].diff()
    df["gold_vol_5d"] = df["gold_ret"].rolling(5).std()

    return df[FEATURE_COLS].dropna()


def _standardize(X: np.ndarray, mean=None, std=None):
    """Manual z-score standardization (avoids adding scikit-learn as a
    dependency just for this one function)."""
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
        std[std == 0] = 1.0  # guard against a constant column
    return (X - mean) / std, mean, std


def diagnostic_check(feat_df: pd.DataFrame, n_states=N_STATES, test_frac=0.2):
    """
    Chronological train/test split to check for overfitting BEFORE trusting
    the production model. Prints a warning if the gap looks unhealthy.
    Returns True if the model looks reasonably well-behaved.
    """
    split_idx = int(len(feat_df) * (1 - test_frac))
    train, test = feat_df.iloc[:split_idx], feat_df.iloc[split_idx:]

    if len(train) < 50 or len(test) < 10:
        print("[warn] Not enough data yet for a reliable train/test overfitting check.")
        return True  # don't block on this, just skip the check

    X_train, mean, std = _standardize(train.values)
    X_test, _, _ = _standardize(test.values, mean, std)

    model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000, random_state=42)
    model.fit(X_train)

    train_ll = model.score(X_train) / len(X_train)
    test_ll = model.score(X_test) / len(X_test)
    gap = train_ll - test_ll

    print(f"[info] Regime model health check -- train LL/obs: {train_ll:.2f}, "
          f"test LL/obs: {test_ll:.2f}, gap: {gap:.2f}")

    if gap > 10:
        print(
            "[warn] Large train/test gap detected -- the model may be overfitting "
            "to recent data. Treat today's regime read with extra skepticism."
        )
        return False

    return True


def compare_state_counts(feat_df: pd.DataFrame, candidates=(2, 3, 4, 5, 6), test_frac=0.2):
    """
    Formalizes the informal 3-vs-4-vs-5 state investigation done early in
    this project's development (which is WHY N_STATES=3 was chosen in the
    first place) into reusable code, so it can be re-run on new/larger
    datasets rather than re-derived by hand each time.

    Fits each candidate state count on a chronological train split, scores
    train and test log-likelihood, and reports the gap -- a large gap means
    that state count is overfitting for this data size. Also reports how
    many of the requested states were actually used (a model can be given
    N slots and use fewer, exactly as happened when 4 states collapsed to 3
    in the original investigation).
    """
    split_idx = int(len(feat_df) * (1 - test_frac))
    train, test = feat_df.iloc[:split_idx], feat_df.iloc[split_idx:]

    if len(train) < 50 or len(test) < 10:
        print("[warn] Not enough data for a reliable state-count comparison.")
        return {}

    X_train, mean, std = _standardize(train.values)
    X_test, _, _ = _standardize(test.values, mean, std)

    results = {}
    for n in candidates:
        try:
            model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000, random_state=42)
            model.fit(X_train)
            train_ll = model.score(X_train) / len(X_train)
            test_ll = model.score(X_test) / len(X_test)
            train_states = model.predict(X_train)
            state_sizes = pd.Series(train_states).value_counts().sort_index().tolist()
            results[n] = {
                "train_ll": train_ll,
                "test_ll": test_ll,
                "gap": train_ll - test_ll,
                "n_states_used": len(set(train_states)),
                "state_sizes": state_sizes,
            }
        except Exception as e:
            results[n] = {"error": str(e)}
    return results


def recommend_state_count(comparison: dict, max_acceptable_gap=10):
    """
    Picks the largest state count whose train/test gap stays under a
    healthy threshold AND that actually uses all its states (not
    collapsing states the way n=4 did in the original investigation).
    Prefers more states only when the data actually supports them --
    defaults to the smallest candidate if nothing looks clean.
    """
    healthy = [
        n for n, r in comparison.items()
        if "error" not in r and r["gap"] < max_acceptable_gap and r["n_states_used"] == n
    ]
    if not healthy:
        valid = [n for n, r in comparison.items() if "error" not in r]
        return min(valid) if valid else None
    return max(healthy)


def fit_production_model(feat_df: pd.DataFrame, n_states=N_STATES):
    """
    Fits the HMM on ALL available data (not just the train split) for the
    actual state labels / transition matrix / current-day inference --
    the diagnostic_check() above already validated this is a reasonable
    number of states, so we use the full history here for the best estimate.
    """
    X, mean, std = _standardize(feat_df.values)
    model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000, random_state=42)
    model.fit(X)
    return model, mean, std


def label_states(feat_df: pd.DataFrame, model, mean, std):
    """Assigns a human-readable label to each hidden state, ordered by mean
    gold return (ascending), and returns (labeled_df, label_map, small_sample_states)."""
    X, _, _ = _standardize(feat_df.values, mean, std)
    raw_states = model.predict(X)

    df = feat_df.copy()
    df["raw_state"] = raw_states

    order = df.groupby("raw_state")["gold_ret"].mean().sort_values().index.tolist()
    n = len(order)
    labels = STATE_LABELS_ASCENDING if n == 3 else [f"State {i}" for i in range(n)]
    label_map = {order[i]: labels[i] for i in range(n)}

    df["label"] = df["raw_state"].map(label_map)

    counts = df["label"].value_counts()
    small_sample_states = counts[counts < 15].index.tolist()

    return df, label_map, small_sample_states


def build_transition_matrix(labeled_df: pd.DataFrame):
    """Empirical day-to-day transition matrix from the labeled state sequence."""
    labels_seq = labeled_df["label"].tolist()
    unique_labels = sorted(labeled_df["label"].unique(), key=lambda l: STATE_LABELS_ASCENDING.index(l)
                            if l in STATE_LABELS_ASCENDING else 999)

    matrix = pd.DataFrame(0, index=unique_labels, columns=unique_labels, dtype=float)
    for i in range(len(labels_seq) - 1):
        matrix.loc[labels_seq[i], labels_seq[i + 1]] += 1

    row_sums = matrix.sum(axis=1)
    matrix = matrix.div(row_sums.replace(0, np.nan), axis=0).fillna(0)
    return matrix


def build_recent_transition_matrix(labeled_df: pd.DataFrame, window_days=63):
    """
    Same as build_transition_matrix, but computed ONLY on the most recent
    `window_days` trading days (default ~63 = roughly 3 calendar months).

    This exists because a full-history matrix is a single average over
    however much data you have, and can look nothing like current
    behavior if the earlier and later portions of history had genuinely
    different dynamics (e.g. a long clean trending stretch early on,
    versus choppy regime-switching more recently). Always compare this
    against the full-history matrix rather than trusting either alone.
    """
    recent = labeled_df.tail(window_days)
    return build_transition_matrix(recent), recent.index.min(), recent.index.max()


def state_frequency(labeled_df: pd.DataFrame):
    """What fraction of days actually fall in each state -- reported
    alongside labels so a name like 'Declining' isn't silently assumed to
    mean 'rare' when it might be a large fraction of the sample."""
    counts = labeled_df["label"].value_counts()
    pct = (counts / len(labeled_df) * 100).round(1)
    return pct.to_dict()


def state_characteristics(labeled_df: pd.DataFrame):
    """
    Describes what each state ACTUALLY looks like -- mean/std of every
    feature, plus sample count. Essential when state count != 3, since
    those states only get generic names ("State 0", "State 1", ...) with
    no description of what they represent. This is the table that answers
    "what is State 3 actually characterized by."
    """
    cols = [c for c in FEATURE_COLS if c in labeled_df.columns]
    grouped = labeled_df.groupby("label")[cols].agg(["mean", "std"])
    counts = labeled_df["label"].value_counts()

    rows = []
    for label in grouped.index:
        row = {"label": label, "n": int(counts.get(label, 0))}
        for col in cols:
            row[f"{col}_mean"] = grouped.loc[label, (col, "mean")]
            row[f"{col}_std"] = grouped.loc[label, (col, "std")]
        rows.append(row)

    # Sort by mean gold_ret ascending, matching the same convention used
    # for the 3-state descriptive labels (lowest return first)
    rows.sort(key=lambda r: r.get("gold_ret_mean", 0))
    return rows


def current_state_probabilities(feat_df: pd.DataFrame, model, mean, std, label_map):
    """Probability distribution over states for the MOST RECENT observation."""
    X, _, _ = _standardize(feat_df.values, mean, std)
    posteriors = model.predict_proba(X)
    latest = posteriors[-1]

    result = {}
    for raw_state, prob in enumerate(latest):
        label = label_map.get(raw_state, f"State {raw_state}")
        result[label] = result.get(label, 0) + prob  # in case of label collisions
    return dict(sorted(result.items(), key=lambda kv: -kv[1]))


def analyze_regime(price_df: pd.DataFrame, recent_window_days=63, n_states=N_STATES, precomputed_feat_df=None):
    """
    Full pipeline: features -> diagnostic check -> production fit -> labels
    -> BOTH full-history and recent-window transition matrices -> current
    state probabilities. Returns a dict with everything main.py needs to
    print, or None if there wasn't enough data to run at all.

    n_states defaults to the validated N_STATES=3 for normal use. The
    extended-history command can pass a different value here if
    compare_state_counts() finds a larger dataset supports more states --
    never hardcode a different number without running that comparison first.

    precomputed_feat_df: if provided, SKIPS rebuilding features from
    price_df and uses this directly instead. Needed so artifact-filtered
    feature sets (e.g. roll-artifact days removed) actually flow into the
    final fitted model, rather than silently being rebuilt from raw,
    unfiltered price data and reintroducing the artifacts.
    """
    feat_df = precomputed_feat_df if precomputed_feat_df is not None else build_features(price_df)

    if len(feat_df) < 60:
        print(f"[warn] Only {len(feat_df)} usable observations -- too few to "
              f"fit a meaningful regime model yet. Need more history.")
        return None

    model_healthy = diagnostic_check(feat_df, n_states=n_states)

    model, mean, std = fit_production_model(feat_df, n_states=n_states)
    labeled_df, label_map, small_sample_states = label_states(feat_df, model, mean, std)
    transition_matrix = build_transition_matrix(labeled_df)
    current_probs = current_state_probabilities(feat_df, model, mean, std, label_map)
    freq = state_frequency(labeled_df)
    characteristics = state_characteristics(labeled_df)

    recent_matrix, recent_start, recent_end = None, None, None
    if len(labeled_df) >= recent_window_days + 10:
        recent_matrix, recent_start, recent_end = build_recent_transition_matrix(
            labeled_df, window_days=recent_window_days
        )

    return {
        "labeled_df": labeled_df,
        "transition_matrix": transition_matrix,
        "recent_matrix": recent_matrix,
        "recent_window_start": recent_start,
        "recent_window_end": recent_end,
        "recent_window_days": recent_window_days,
        "current_probs": current_probs,
        "model_healthy": model_healthy,
        "small_sample_states": small_sample_states,
        "state_frequency": freq,
        "state_characteristics": characteristics,
        "n_observations": len(feat_df),
        # Raw fitted model + standardization params, so OTHER feature
        # vectors (e.g. today's actual conditions from a different,
        # shorter-window model) can be scored against THIS fitted model --
        # needed for the "how do today's conditions compare to the
        # extended taxonomy" cross-scoring.
        "model": model,
        "mean": mean,
        "std": std,
        "label_map": label_map,
        "feat_df": feat_df,
    }


def stationary_distribution(transmat, max_iter=10000, tol=1e-12):
    """
    Computes the long-run equilibrium (stationary) distribution of the
    fitted transition matrix via power iteration -- the genuine "typical
    day, no other information" base rate for each state.

    This exists because hmmlearn's own predict_proba(), when given a
    single-observation sequence, falls back to using the model's
    startprob_ as the prior -- and startprob_ is NOT a general-purpose
    prior. It represents the model's belief about the FIRST observation
    of the TRAINING sequence specifically (an arbitrary artifact of
    whatever regime happened to be active when the historical data window
    starts), and empirically is often close to a degenerate one-hot
    vector (fit against a single real "day one" during EM). Using it to
    score an unrelated, recent, isolated day silently biases the result
    toward whichever state got that historical accident of a starting
    point, regardless of the actual data being scored.
    """
    n = transmat.shape[0]
    dist = np.ones(n) / n
    for _ in range(max_iter):
        new_dist = dist @ transmat
        if np.allclose(new_dist, dist, atol=tol):
            return new_dist
        dist = new_dist
    return dist


def _diag_gaussian_likelihood(x, mean_vec, cov_matrix):
    """Likelihood of x under a diagonal-covariance multivariate Gaussian
    (product of independent per-feature normal densities)."""
    variances = np.diag(cov_matrix)
    likelihood = 1.0
    for i in range(len(x)):
        var = max(variances[i], 1e-12)  # guard against a collapsed-to-zero variance
        diff = x[i] - mean_vec[i]
        likelihood *= (1.0 / np.sqrt(2 * np.pi * var)) * np.exp(-0.5 * diff ** 2 / var)
    return likelihood


def score_feature_vector_isolated(feat_row, model, mean, std, label_map):
    """
    Scores a SINGLE feature vector IN ISOLATION -- no history, no
    yesterday, no transition context. Uses the model's STATIONARY
    distribution as the prior (see stationary_distribution() above for
    why this replaces hmmlearn's default startprob_-based fallback, which
    is not a defensible "no information" prior).

    IMPORTANT: even with this fix, this still answers a genuinely cruder
    question than score_feature_vector_sequence_aware() below -- "which
    state's historical distribution does today's raw number resemble
    most, using the long-run base rate as a prior" -- not "what state are
    we likely in today given yesterday." Prefer the sequence-aware
    version whenever you have historical data to append to (which this
    system always does); use this only as a genuine last-resort fallback.
    """
    row_values = np.array([feat_row[c] for c in FEATURE_COLS])
    X, _, _ = _standardize(row_values.reshape(1, -1), mean, std)
    x = X[0]

    stat_dist = stationary_distribution(model.transmat_)
    likelihoods = np.array([
        _diag_gaussian_likelihood(x, model.means_[k], model.covars_[k])
        for k in range(model.n_components)
    ])
    posterior_unnorm = stat_dist * likelihoods
    total = posterior_unnorm.sum()
    posterior = posterior_unnorm / total if total > 0 else stat_dist

    result = {}
    for raw_state, prob in enumerate(posterior):
        label = label_map.get(raw_state, f"State {raw_state}")
        result[label] = result.get(label, 0) + prob
    return dict(sorted(result.items(), key=lambda kv: -kv[1]))


def score_feature_vector_sequence_aware(new_row, historical_feat_df, model, mean, std, label_map):
    """
    The methodologically correct way to ask "what state does today's data
    put us in, according to THIS model" -- appends the new row onto the
    END of the model's own actual fitted sequence, runs predict_proba on
    the WHOLE combined sequence, and reads off the LAST row's posterior.

    This correctly incorporates the fitted transition matrix and the
    sequence's recent trajectory (effectively "what state were we likely
    in yesterday, and how does today's data update that") -- unlike
    score_feature_vector_isolated(), which has no memory of anything.
    """
    import numpy as np
    import pandas as pd

    new_row_df = pd.DataFrame([{c: new_row[c] for c in FEATURE_COLS}])
    combined = pd.concat([historical_feat_df[FEATURE_COLS], new_row_df], ignore_index=True)

    X, _, _ = _standardize(combined.values, mean, std)
    posteriors = model.predict_proba(X)
    last_posterior = posteriors[-1]

    result = {}
    for raw_state, prob in enumerate(last_posterior):
        label = label_map.get(raw_state, f"State {raw_state}")
        result[label] = result.get(label, 0) + prob
    return dict(sorted(result.items(), key=lambda kv: -kv[1]))


def name_state_archetype(row, all_rows):
    """
    Assigns an economic, human-readable archetype name based on the
    state's mean return and volatility RELATIVE to the other states in
    THIS fit -- not hardcoded thresholds (exact numbers vary run to run
    and with how much data you have), but the same underlying return x
    volatility taxonomy logic a human analyst would apply by eye.

    Uses a clean 3x3 grid (return tier x volatility tier, each Low/Mid/
    High by rank) so every combination is covered -- no boundary gaps.
    Generalizes to however many states your data actually supports.
    """
    returns = sorted(r.get("gold_ret_mean", 0) for r in all_rows)
    vols = sorted(r.get("gold_vol_5d_mean", 0) for r in all_rows)

    def _rank(value, sorted_list):
        if len(sorted_list) <= 1:
            return 0.5
        return sorted_list.index(value) / (len(sorted_list) - 1)

    def _tier(rank):
        if rank <= 1 / 3:
            return "Low"
        elif rank <= 2 / 3:
            return "Mid"
        return "High"

    ret_tier = _tier(_rank(row.get("gold_ret_mean", 0), returns))
    vol_tier = _tier(_rank(row.get("gold_vol_5d_mean", 0), vols))

    grid = {
        ("Low", "High"): ("Shock Liquidation", "Sharp selloff with high volatility -- panic/deleveraging character."),
        ("Low", "Mid"): ("Yield Pressure", "Steady decline without panic -- a grinding, not violent, down-move."),
        ("Low", "Low"): ("Yield Pressure", "Orderly, low-volatility decline -- consistent downward pressure."),
        ("Mid", "Low"): ("Quiet Consolidation", "Little net movement, low volatility -- sideways/range character."),
        ("Mid", "Mid"): ("Quiet Consolidation", "Near-flat on average, moderate noise -- still range-like."),
        ("Mid", "High"): ("Choppy Range", "Flat on average but with large swings -- volatile without a clear trend."),
        ("High", "Low"): ("Macro Tailwind", "Highest returns with contained volatility -- the most favorable environment in this sample."),
        ("High", "Mid"): ("Steady Bull", "Orderly uptrend with moderate, contained volatility."),
        ("High", "High"): ("Momentum Bull", "Strong uptrend but with elevated volatility -- large, fast moves within an up-trend."),
    }
    return grid[(ret_tier, vol_tier)]
