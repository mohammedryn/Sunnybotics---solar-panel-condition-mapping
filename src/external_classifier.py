"""
External mode's supervised clean/damaged benchmark - a SEPARATE, explicitly
labeled analysis, not part of the label-free inference path.

The zero-shot rule classifier in condition_analysis.py (5 categories,
thresholds calibrated on synthetic renders) is never touched by this
module and never sees a folder label. This module answers a narrower,
explicitly supervised question instead: "if we actually calibrate against
Sunnybotics' clean/damaged folder labels via cross-validation, do these
same 10 numeric features (feature_extraction.py's output) predict
clean/damaged on real photos better than chance?"

Where the label is and isn't used:
  - ingest.py, feature_extraction.py, condition_analysis.py,
    priority_score.py, annotate.py, export.py: NEVER read a folder label.
  - This module: reads the folder label ONLY here, ONLY to train/evaluate
    this explicitly supervised benchmark. Every reported prediction in
    external_binary_eval.csv is made by a model that did not see that
    image's label during its own fold's training (standard k-fold
    cross-validation - see cross_validate()).
  - The "final" model (fit_final_model) is trained on all external images
    and used only to produce a deployed-style prediction per image
    (external_predicted_label/external_confidence in results.csv) - it is
    NOT evaluated for accuracy anywhere, since it has seen every label.
    Only the held-out cross-validation predictions (external_cv_predicted_
    label/external_cv_correct) are ever used for accuracy claims.

This is a real-photo sample benchmark on ~100 images, not a production
accuracy claim - see the "note" fields in the saved summary JSON.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats as _stats

from . import config

try:
    from sklearn.linear_model import LogisticRegression
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

FEATURE_COLUMNS = [
    "brightness_median",
    "shadow_area_ratio",
    "shadow_directionality_score",
    "dirt_area_ratio",
    "dirt_hue_shift_score",
    "glare_area_ratio",
    "glare_clipped_pixel_ratio",
    "damage_line_density",
    "damage_irregular_line_score",
    "laplacian_variance",
]
LABELS = ("clean", "damaged")
N_FOLDS = 5
RANDOM_SEED = 42
MIN_SAMPLES_PER_CLASS = 2  # below this, cross-validation is not meaningful


# --------------------------------------------------------------------------
# Data loading - the ONLY place the folder label meets the feature table.
# --------------------------------------------------------------------------

def load_feature_table(condition_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> pd.DataFrame:
    """Joins condition_analysis's saved per-image feature JSON (already
    written by feature_extraction.py during the normal, label-free
    inference pass) with the external ground-truth labels. Only images
    that reached feature extraction (visual_analysis_status == "ok") and
    have a folder label are included."""
    gt_lookup = dict(zip(ground_truth_df["image_id"], ground_truth_df["true_label"]))
    usable = condition_df[condition_df["visual_analysis_status"] == "ok"]

    rows = []
    for _, row in usable.iterrows():
        image_id = row["image_id"]
        true_label = gt_lookup.get(image_id)
        feat_path = row.get("feature_summary_json")
        if true_label not in LABELS or not feat_path or not os.path.exists(feat_path):
            continue
        with open(feat_path) as f:
            features = json.load(f)
        rows.append({"image_id": image_id, "true_label": true_label, **{c: features[c] for c in FEATURE_COLUMNS}})

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Standardization - always fit on a train set only, applied to train+test.
# --------------------------------------------------------------------------

def _standardize_fit(X: np.ndarray) -> tuple:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0  # constant feature in this fold - avoid div by zero, leaves it at its (centered) value
    return mean, std


def _standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


# --------------------------------------------------------------------------
# Classifier backends - sklearn LogisticRegression preferred, numpy
# nearest-centroid fallback with an analogous confidence measure.
# --------------------------------------------------------------------------

class _NearestCentroidClassifier:
    """Fallback when sklearn isn't available: classify by nearest
    (Euclidean) standardized class centroid. Confidence is a softmax over
    negative distances to each centroid - the closer centroid dominates,
    same shape/interpretation as a 2-class probability."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_NearestCentroidClassifier":
        self.classes_ = np.array(sorted(set(y)))
        self.centroids_ = {c: X[y == c].mean(axis=0) for c in self.classes_}
        return self

    def predict_with_confidence(self, X: np.ndarray) -> tuple:
        dist_matrix = np.column_stack([
            np.linalg.norm(X - self.centroids_[c], axis=1) for c in self.classes_
        ])
        neg = -dist_matrix
        neg -= neg.max(axis=1, keepdims=True)
        exp = np.exp(neg)
        proba = exp / exp.sum(axis=1, keepdims=True)
        pred_idx = np.argmin(dist_matrix, axis=1)
        preds = self.classes_[pred_idx]
        confidences = proba[np.arange(len(X)), pred_idx]
        return preds, confidences


def _fit_backend(X: np.ndarray, y: np.ndarray):
    if _HAS_SKLEARN:
        model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED)
        model.fit(X, y)
        return model
    return _NearestCentroidClassifier().fit(X, y)


def _predict_with_confidence(model, X: np.ndarray) -> tuple:
    if _HAS_SKLEARN and isinstance(model, LogisticRegression):
        proba = model.predict_proba(X)
        idx_max = np.argmax(proba, axis=1)
        preds = model.classes_[idx_max]
        confidences = proba[np.arange(len(X)), idx_max]
        return preds, confidences
    return model.predict_with_confidence(X)


def backend_name() -> str:
    return "logistic_regression" if _HAS_SKLEARN else "nearest_centroid"


# --------------------------------------------------------------------------
# Stratified k-fold split - implemented directly (not via sklearn) so CV
# behavior is identical regardless of which classifier backend is used.
# --------------------------------------------------------------------------

def _stratified_kfold_indices(y: np.ndarray, n_folds: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    fold_indices = [[] for _ in range(n_folds)]
    for c in sorted(set(y)):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        for i, split in enumerate(np.array_split(idx, n_folds)):
            fold_indices[i].extend(split.tolist())

    all_idx = set(range(len(y)))
    folds = []
    for i in range(n_folds):
        test_idx = np.array(sorted(fold_indices[i]), dtype=int)
        train_idx = np.array(sorted(all_idx - set(fold_indices[i])), dtype=int)
        folds.append((train_idx, test_idx))
    return folds


def _effective_n_folds(y: np.ndarray, requested: int) -> int:
    """Shrinks the fold count on small datasets (e.g. tests) so every
    class has at least one sample per fold, never crashing or silently
    producing empty test folds for a class."""
    counts = pd.Series(y).value_counts()
    return max(2, min(requested, int(counts.min())))


# --------------------------------------------------------------------------
# Cross-validation: every reported prediction comes from a fold that did
# NOT see that image during training.
# --------------------------------------------------------------------------

def cross_validate(feature_table: pd.DataFrame, n_folds: int = N_FOLDS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    X = feature_table[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = feature_table["true_label"].to_numpy()

    effective_folds = _effective_n_folds(y, n_folds)
    folds = _stratified_kfold_indices(y, effective_folds, seed)

    pred_labels = np.empty(len(y), dtype=object)
    pred_conf = np.full(len(y), np.nan)

    for train_idx, test_idx in folds:
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        mean, std = _standardize_fit(X[train_idx])
        X_train = _standardize_apply(X[train_idx], mean, std)
        X_test = _standardize_apply(X[test_idx], mean, std)

        model = _fit_backend(X_train, y[train_idx])
        preds, confs = _predict_with_confidence(model, X_test)
        pred_labels[test_idx] = preds
        pred_conf[test_idx] = confs

    out = feature_table[["image_id", "true_label"]].copy()
    out["cv_predicted_label"] = pred_labels
    out["cv_confidence"] = pred_conf
    out["cv_correct"] = out["cv_predicted_label"] == out["true_label"]
    return out


# --------------------------------------------------------------------------
# Final model: trained on ALL external images. Used only to produce a
# per-image "what would the deployed model say" prediction - never scored
# for accuracy (it has seen every label), unlike the CV predictions above.
# --------------------------------------------------------------------------

def fit_final_model(feature_table: pd.DataFrame) -> dict:
    X = feature_table[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = feature_table["true_label"].to_numpy()
    mean, std = _standardize_fit(X)
    X_std = _standardize_apply(X, mean, std)
    model = _fit_backend(X_std, y)
    return {"backend": backend_name(), "model": model, "mean": mean, "std": std}


def predict_with_final_model(final_model_info: dict, feature_table: pd.DataFrame) -> pd.DataFrame:
    X = feature_table[FEATURE_COLUMNS].to_numpy(dtype=float)
    X_std = _standardize_apply(X, final_model_info["mean"], final_model_info["std"])
    preds, confs = _predict_with_confidence(final_model_info["model"], X_std)
    out = feature_table[["image_id"]].copy()
    out["external_predicted_label"] = preds
    out["external_confidence"] = confs
    return out


# --------------------------------------------------------------------------
# Feature importance / coefficient inspection (Part 2).
# --------------------------------------------------------------------------

def compute_feature_importance(final_model_info: dict) -> pd.DataFrame:
    """Positive coefficient/diff -> pushes toward "damaged" (classes_[1]
    for logistic regression since LABELS sorts alphabetically to
    ["clean", "damaged"]; damaged-minus-clean centroid diff for the
    nearest-centroid fallback, same sign convention). Diagnostic only -
    which real-photo signals this particular model leaned on, not a
    causal claim about what damage physically looks like."""
    if final_model_info["backend"] == "logistic_regression":
        coefs = final_model_info["model"].coef_.ravel()
    else:
        centroids = final_model_info["model"].centroids_
        classes = sorted(centroids.keys())  # ["clean", "damaged"]
        coefs = centroids[classes[1]] - centroids[classes[0]]

    df = pd.DataFrame({"feature": FEATURE_COLUMNS, "coefficient": coefs})
    df["abs_coefficient"] = df["coefficient"].abs()
    df["direction"] = np.where(df["coefficient"] > 0, "toward_damaged", "toward_clean")
    total_abs = df["abs_coefficient"].sum()
    df["normalized_importance"] = df["abs_coefficient"] / total_abs if total_abs > 0 else 0.0
    return df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Confidence-stratified accuracy (Part 3) - held-out CV predictions only.
# --------------------------------------------------------------------------

def _partition_by_confidence(cv_eval_df: pd.DataFrame) -> dict:
    """Splits rows (sorted by descending cv_confidence) into top 20% /
    middle 60% / bottom 20%, covering every row exactly once. On very
    small inputs where 20%+20% would overlap or exceed the row count,
    shrinks top/bottom to fit and the middle stratum may end up empty -
    still a valid, non-overlapping partition."""
    df = cv_eval_df.sort_values("cv_confidence", ascending=False).reset_index(drop=True)
    n = len(df)
    if n == 0:
        return {"top20": df, "middle60": df, "bottom20": df}

    n_top = max(1, int(round(n * 0.2)))
    n_bottom = max(1, int(round(n * 0.2)))
    if n_top + n_bottom >= n:
        n_top = (n + 1) // 2
        n_bottom = n - n_top
    n_middle = n - n_top - n_bottom

    return {
        "top20": df.iloc[:n_top],
        "middle60": df.iloc[n_top:n_top + n_middle],
        "bottom20": df.iloc[n_top + n_middle:],
    }


def _stratum_stats(name: str, sub_df: pd.DataFrame) -> dict:
    n = len(sub_df)
    n_correct = int(sub_df["cv_correct"].sum()) if n else 0
    return {
        "stratum": name,
        "n": n,
        "mean_confidence": float(sub_df["cv_confidence"].mean()) if n else None,
        "accuracy": (n_correct / n) if n else None,
        "n_correct": n_correct,
        "n_incorrect": n - n_correct,
    }


def confidence_strata(cv_eval_df: pd.DataFrame) -> pd.DataFrame:
    parts = _partition_by_confidence(cv_eval_df)
    return pd.DataFrame([_stratum_stats(name, sub) for name, sub in parts.items()])


# --------------------------------------------------------------------------
# Summary metrics (Part 1): accuracy, confusion matrix, precision/recall/
# F1, naive baselines.
# --------------------------------------------------------------------------

def _confusion_matrix(true_labels: np.ndarray, pred_labels: np.ndarray) -> dict:
    return {
        t: {p: int(((true_labels == t) & (pred_labels == p)).sum()) for p in LABELS}
        for t in LABELS
    }


def _precision_recall_f1(true_labels: np.ndarray, pred_labels: np.ndarray) -> dict:
    metrics = {}
    for pos in LABELS:
        tp = int(((pred_labels == pos) & (true_labels == pos)).sum())
        fp = int(((pred_labels == pos) & (true_labels != pos)).sum())
        fn = int(((pred_labels != pos) & (true_labels == pos)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (2 * precision * recall / (precision + recall)) if precision and recall and (precision + recall) > 0 else None
        metrics[pos] = {"precision": precision, "recall": recall, "f1": f1}
    return metrics


def _wilson_ci(n_correct: int, n_total: int, confidence: float = 0.95) -> tuple:
    """Wilson score interval for a binomial proportion - preferred over
    the naive Wald interval (p +/- z*sqrt(p(1-p)/n)) because Wald is
    known to be poorly calibrated exactly where this benchmark sits:
    a proportion well above 0.5 with n in the low hundreds. Treats each
    held-out CV prediction as an independent Bernoulli trial, which is a
    simplification (predictions within a fold share a fitted model), but
    is the standard back-of-envelope treatment for this size of sample
    and is far more informative than reporting the point estimate alone."""
    if n_total == 0:
        return (None, None)
    p = n_correct / n_total
    z = _stats.norm.ppf(1 - (1 - confidence) / 2)
    denom = 1 + z ** 2 / n_total
    center = (p + z ** 2 / (2 * n_total)) / denom
    margin = (z / denom) * np.sqrt(p * (1 - p) / n_total + z ** 2 / (4 * n_total ** 2))
    return (max(0.0, center - margin), min(1.0, center + margin))


def _naive_baselines(true_labels: np.ndarray) -> dict:
    n = len(true_labels)
    if n == 0:
        return {"always_predict_clean_accuracy": None, "always_predict_damaged_accuracy": None,
                "majority_class_baseline_accuracy": None}
    n_clean = int((true_labels == "clean").sum())
    n_damaged = int((true_labels == "damaged").sum())
    return {
        "always_predict_clean_accuracy": n_clean / n,
        "always_predict_damaged_accuracy": n_damaged / n,
        "majority_class_baseline_accuracy": max(n_clean, n_damaged) / n,
    }


def build_summary(cv_eval_df: pd.DataFrame, importance_df: pd.DataFrame, strata_df: pd.DataFrame) -> dict:
    true_labels = cv_eval_df["true_label"].to_numpy()
    pred_labels = cv_eval_df["cv_predicted_label"].to_numpy()
    n = len(cv_eval_df)
    n_correct = int((pred_labels == true_labels).sum())
    accuracy = float(n_correct / n) if n else None
    ci_low, ci_high = _wilson_ci(n_correct, n)

    return {
        "n_total": n,
        "n_clean_true": int((true_labels == "clean").sum()),
        "n_damaged_true": int((true_labels == "damaged").sum()),
        "accuracy": accuracy,
        "accuracy_95ci_wilson": [ci_low, ci_high],
        "confusion_matrix": _confusion_matrix(true_labels, pred_labels),
        "precision_recall_f1": _precision_recall_f1(true_labels, pred_labels),
        "naive_baselines": _naive_baselines(true_labels),
        "classifier_backend": backend_name(),
        "cv_folds": _effective_n_folds(true_labels, N_FOLDS) if n else None,
        "top_features": importance_df.head(5)[["feature", "coefficient", "direction"]].to_dict("records"),
        "confidence_strata": strata_df.to_dict("records"),
        "note": (
            "Accuracy/precision/recall/F1 above are from stratified k-fold "
            "cross-validation on Sunnybotics' real external sample only - "
            "every prediction was made by a model that did not see that "
            "image's label during its own fold's training. This is a small-"
            "sample benchmark (n reflects the real dataset size), not a "
            "production accuracy claim. accuracy_95ci_wilson is a Wilson "
            "score 95% confidence interval treating each held-out prediction "
            "as an independent trial - a rough but standard way to show that "
            "a point estimate from ~100 images carries real sampling "
            "uncertainty, not a precise measurement."
        ),
        "feature_importance_note": (
            "Coefficients are computed on STANDARDIZED features (mean 0, "
            "std 1), so magnitudes are comparable across features. A "
            "positive coefficient pushes the prediction toward 'damaged'; "
            "negative pushes toward 'clean'. This is diagnostic - it shows "
            "what this particular model leaned on - not a causal claim "
            "about what real damage physically looks like."
        ),
    }


# --------------------------------------------------------------------------
# Orchestration + persistence.
# --------------------------------------------------------------------------

def run_supervised_benchmark(condition_df: pd.DataFrame, ground_truth_df: pd.DataFrame):
    """Returns None if there isn't enough labeled, feature-bearing data to
    run a meaningful benchmark (e.g. too few images of one class) -
    caller should skip saving outputs and say so, rather than fabricate
    a result from too little data."""
    feature_table = load_feature_table(condition_df, ground_truth_df)
    class_counts = feature_table["true_label"].value_counts() if len(feature_table) else pd.Series(dtype=int)
    if len(class_counts) < 2 or class_counts.min() < MIN_SAMPLES_PER_CLASS:
        return None

    cv_eval_df = cross_validate(feature_table)
    final_model_info = fit_final_model(feature_table)
    importance_df = compute_feature_importance(final_model_info)
    strata_df = confidence_strata(cv_eval_df)
    summary = build_summary(cv_eval_df, importance_df, strata_df)
    final_predictions_df = predict_with_final_model(final_model_info, feature_table)

    return {
        "feature_table": feature_table,
        "cv_eval_df": cv_eval_df,
        "importance_df": importance_df,
        "strata_df": strata_df,
        "summary": summary,
        "final_predictions_df": final_predictions_df,
    }


def save_binary_eval(cv_eval_df: pd.DataFrame, path: str = None):
    path = path or os.path.join(config.OUTPUTS_DIR, "external_binary_eval.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv_eval_df[["image_id", "true_label", "cv_predicted_label", "cv_confidence", "cv_correct"]].to_csv(path, index=False)


def save_binary_eval_summary(summary: dict, path: str = None):
    path = path or os.path.join(config.OUTPUTS_DIR, "external_binary_eval_summary.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)


def save_feature_importance(importance_df: pd.DataFrame, path: str = None):
    path = path or os.path.join(config.OUTPUTS_DIR, "external_feature_importance.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    importance_df.to_csv(path, index=False)


def save_confidence_strata(strata_df: pd.DataFrame, path: str = None):
    path = path or os.path.join(config.OUTPUTS_DIR, "external_confidence_strata.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    strata_df.to_csv(path, index=False)


def merge_into_results(export_df: pd.DataFrame, cv_eval_df: pd.DataFrame, final_predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Adds external_predicted_label/external_confidence (final, all-data
    model) and external_cv_predicted_label/external_cv_correct (held-out
    CV result) onto the main results export, on image_id. Images without
    a resolvable folder label (shouldn't happen for this dataset, but
    defensively handled) simply get NaN in these columns."""
    merged = export_df.merge(final_predictions_df, on="image_id", how="left")
    cv_cols = cv_eval_df[["image_id", "cv_predicted_label", "cv_correct"]].rename(
        columns={"cv_predicted_label": "external_cv_predicted_label", "cv_correct": "external_cv_correct"}
    )
    merged = merged.merge(cv_cols, on="image_id", how="left")
    return merged
