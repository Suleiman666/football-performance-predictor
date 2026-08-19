"""
model.py
Trains and evaluates a match-outcome prediction model.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, classification_report


FEATURE_COLS = [
    "home_form_points_avg", "home_form_goals_for_avg", "home_form_goals_against_avg",
    "home_form_goal_diff_avg", "home_form_win_rate",
    "away_form_points_avg", "away_form_goals_for_avg", "away_form_goals_against_avg",
    "away_form_goal_diff_avg", "away_form_win_rate",
]
TARGET_COL = "result"


def load_dataset(path: str = "data/processed/matches.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["match_date"])

    # Drop matches where either team has no form history yet — no signal to learn from
    df = df[(df["home_form_sample_size"] > 0) & (df["away_form_sample_size"] > 0)]
    df = df.sort_values("match_date").reset_index(drop=True)
    return df


def chronological_split(df: pd.DataFrame, train_frac: float = 0.8):
    """Split by date order, not randomly — train on earlier matches, test on later ones."""
    split_idx = int(len(df) * train_frac)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test


def train_logistic_regression(train: pd.DataFrame):
    X_train = train[FEATURE_COLS]
    y_train = train[TARGET_COL]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_random_forest(train: pd.DataFrame):
    X_train = train[FEATURE_COLS]
    y_train = train[TARGET_COL]

    model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate(model, test: pd.DataFrame, label: str):
    X_test = test[FEATURE_COLS]
    y_test = test[TARGET_COL]

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    acc = accuracy_score(y_test, preds)
    ll = log_loss(y_test, probs, labels=model.classes_)

    print(f"\n--- {label} ---")
    print(f"Accuracy: {acc:.3f}")
    print(f"Log-loss: {ll:.3f}")
    print(classification_report(y_test, preds))

    return {"accuracy": acc, "log_loss": ll}


def baseline_majority_class(train: pd.DataFrame, test: pd.DataFrame):
    """A naive baseline: always predict the most common outcome in training data.
    Any real model should beat this — if it doesn't, something's wrong."""
    most_common = train[TARGET_COL].mode()[0]
    preds = [most_common] * len(test)
    acc = accuracy_score(test[TARGET_COL], preds)
    print(f"\n--- Majority-class baseline (always predict '{most_common}') ---")
    print(f"Accuracy: {acc:.3f}")
    return acc


if __name__ == "__main__":
    df = load_dataset()
    print(f"Loaded {len(df)} usable matches (after dropping early-season no-history rows)")

    train, test = chronological_split(df)
    print(f"Train: {len(train)} matches | Test: {len(test)} matches")

    baseline_majority_class(train, test)

    log_reg = train_logistic_regression(train)
    evaluate(log_reg, test, "Logistic Regression")

    rf = train_random_forest(train)
    evaluate(rf, test, "Random Forest")