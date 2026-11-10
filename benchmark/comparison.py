"""Per-case comparison against historical result CSVs with explicit numerical tolerance."""
import numpy as np
import pandas as pd


def compare_with_historical(
    new_df: pd.DataFrame,
    hist_df: pd.DataFrame,
    metrics: list[str],
    tolerance: float = 1e-4,
) -> tuple[pd.DataFrame, dict]:
    """
    Compare new per-case metrics against a historical CSV by matching exact unique case_ids.
    Does not compare unmatched rows; reports missing/extra case IDs explicitly.
    """
    if tolerance < 0:
        raise ValueError("Tolerance must be non-negative")
    for name, df in (("new", new_df), ("historical", hist_df)):
        if "case_id" not in df.columns or df["case_id"].isna().any() or df["case_id"].duplicated().any():
            raise ValueError(f"Missing or duplicate case_id in {name} table")

    new_ids = set(new_df["case_id"].astype(int))
    hist_ids = set(hist_df["case_id"].astype(int))
    matched_ids = sorted(new_ids & hist_ids)
    only_new = sorted(new_ids - hist_ids)
    only_hist = sorted(hist_ids - new_ids)

    merged = (
        new_df[new_df["case_id"].isin(matched_ids)]
        .merge(
            hist_df[hist_df["case_id"].isin(matched_ids)],
            on="case_id",
            validate="one_to_one",
            suffixes=("_new", "_hist"),
        )
        .sort_values("case_id")
        .reset_index(drop=True)
    )

    out_cols = {"case_id": merged["case_id"].astype(int)}
    metric_Diff_counts = {}
    for m in metrics:
        if f"{m}_new" not in merged.columns or f"{m}_hist" not in merged.columns:
            raise ValueError(f"Metric {m} missing from comparison tables")
        v_new = merged[f"{m}_new"].to_numpy(dtype=float)
        v_hist = merged[f"{m}_hist"].to_numpy(dtype=float)
        diff = v_new - v_hist
        abs_diff = np.abs(diff)
        both_nan = np.isnan(v_new) & np.isnan(v_hist)
        one_nan = np.isnan(v_new) ^ np.isnan(v_hist)
        exceeds = (~both_nan) & (one_nan | (abs_diff > tolerance))

        out_cols[f"{m}_new"] = v_new
        out_cols[f"{m}_hist"] = v_hist
        out_cols[f"{m}_diff"] = diff
        out_cols[f"{m}_abs_diff"] = abs_diff
        out_cols[f"{m}_exceeds_tol"] = exceeds
        metric_Diff_counts[m] = {
            "exceeds_tolerance_count": int(exceeds.sum()),
            "max_abs_diff": float(np.nanmax(abs_diff)) if not np.all(np.isnan(abs_diff)) else 0.0,
            "mean_diff": float(np.nanmean(diff)) if not np.all(np.isnan(diff)) else 0.0,
        }

    diff_df = pd.DataFrame(out_cols)
    summary = {
        "tolerance": float(tolerance),
        "matched_cases": int(len(matched_ids)),
        "only_in_new": only_new,
        "only_in_historical": only_hist,
        "metrics": metric_Diff_counts,
    }
    return diff_df, summary

