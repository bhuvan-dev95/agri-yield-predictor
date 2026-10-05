"""Flask dashboard for data-driven crop-yield analysis."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from flask import Flask, jsonify, render_template, request
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "crop_yield_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

FEATURES = [
    "rainfall_mm",
    "soil_quality_index",
    "farm_size_hectares",
    "sunlight_hours",
    "fertilizer_kg",
]
TARGET = "crop_yield"
REQUIRED_COLUMNS = FEATURES + [TARGET]
FIGURE_NAMES = {
    "distributions": "feature_distributions.png",
    "relationships": "feature_vs_crop_yield.png",
    "correlation": "correlation_heatmap.png",
}


def _make_empty_state(error: str | None = None) -> dict[str, Any]:
    return {
        "available": False,
        "error": error,
        "raw_rows": 0,
        "raw_columns": 0,
        "missing_before": {},
        "invalid_values_handled": 0,
        "duplicate_rows_removed": 0,
        "cleaned_rows": 0,
        "cleaned_columns": len(REQUIRED_COLUMNS),
        "features": FEATURES,
        "target": TARGET,
        "statistics": {},
        "models": [],
        "best_model": None,
        "model_rmse": None,
        "figure_urls": {},
        "eda": {},
        "model": None,
        "cleaned_data": None,
    }


def _save_eda_figures(data: pd.DataFrame) -> dict[str, str]:
    """Create and persist distribution, relationship, and correlation plots."""
    sns.set_theme(style="whitegrid", palette="YlGn")
    urls: dict[str, str] = {}

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for index, column in enumerate(REQUIRED_COLUMNS):
        axis = axes.flat[index]
        sns.histplot(data=data, x=column, kde=len(data) > 1, ax=axis, color="#2f855a")
        axis.set_title(column.replace("_", " ").title())
    axes.flat[-1].axis("off")
    fig.suptitle("Dataset feature distributions", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / FIGURE_NAMES["distributions"], dpi=140, bbox_inches="tight")
    plt.close(fig)
    urls["distributions"] = f"/outputs/figures/{FIGURE_NAMES['distributions']}"

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for index, column in enumerate(FEATURES):
        axis = axes.flat[index]
        sns.scatterplot(data=data, x=column, y=TARGET, ax=axis, color="#2f855a", alpha=0.7)
        axis.set_title(f"{column.replace('_', ' ').title()} vs. crop yield")
    axes.flat[-1].axis("off")
    fig.suptitle("Feature relationships with crop yield", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / FIGURE_NAMES["relationships"], dpi=140, bbox_inches="tight")
    plt.close(fig)
    urls["relationships"] = f"/outputs/figures/{FIGURE_NAMES['relationships']}"

    fig, axis = plt.subplots(figsize=(9, 7))
    sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="YlGn", fmt=".2f", ax=axis)
    axis.set_title("Correlation heatmap")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / FIGURE_NAMES["correlation"], dpi=140, bbox_inches="tight")
    plt.close(fig)
    urls["correlation"] = f"/outputs/figures/{FIGURE_NAMES['correlation']}"
    return urls


def _prepare_dataset(path: Path) -> dict[str, Any]:
    """Load, validate, clean, summarize, and model the CSV dataset."""
    if not path.is_file():
        return _make_empty_state(
            f"Dataset not found: {path.name}. Place crop_yield_data.csv beside main.py."
        )

    try:
        raw_data = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as error:
        return _make_empty_state(f"Could not read {path.name}: {error}")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in raw_data.columns]
    if missing_columns:
        state = _make_empty_state("Dataset is missing required columns: " + ", ".join(missing_columns))
        state.update(
            {
                "raw_rows": int(len(raw_data)),
                "raw_columns": int(len(raw_data.columns)),
                "missing_before": raw_data.isna().sum().astype(int).to_dict(),
            }
        )
        return state
    if raw_data.empty:
        return _make_empty_state("The dataset has column headers but contains no rows.")

    selected = raw_data[REQUIRED_COLUMNS].copy()
    missing_before = selected.isna().sum().astype(int).to_dict()
    invalid_values = 0
    for column in REQUIRED_COLUMNS:
        converted = pd.to_numeric(selected[column], errors="coerce")
        invalid_values += int((selected[column].notna() & converted.isna()).sum())
        selected[column] = converted.replace([np.inf, -np.inf], np.nan)

    # These physical quantities cannot be negative; no bounds are assumed for the index.
    nonnegative_columns = ["rainfall_mm", "sunlight_hours", "fertilizer_kg", TARGET]
    for column in nonnegative_columns:
        invalid_mask = selected[column] < 0
        invalid_values += int(invalid_mask.sum())
        selected.loc[invalid_mask, column] = np.nan
    invalid_farm_size = selected["farm_size_hectares"] <= 0
    invalid_values += int(invalid_farm_size.sum())
    selected.loc[invalid_farm_size, "farm_size_hectares"] = np.nan
    invalid_values += int(
        sum(
            np.isinf(pd.to_numeric(raw_data[column], errors="coerce").to_numpy(dtype=float, na_value=np.nan)).sum()
            for column in REQUIRED_COLUMNS
        )
    )

    duplicate_rows = int(selected.duplicated().sum())
    selected = selected.drop_duplicates()
    # Rows without the target cannot train or evaluate a supervised model.
    selected = selected.dropna(subset=[TARGET])
    for column in FEATURES:
        if selected[column].isna().any():
            median = selected[column].median()
            selected[column] = selected[column].fillna(median)
    selected = selected.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    if selected.empty:
        return _make_empty_state("No usable rows remain after cleaning the dataset.")

    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    selected.to_csv(RESULTS_DIR / "cleaned_dataset.csv", index=False)
    figure_urls = _save_eda_figures(selected)
    statistics = selected[REQUIRED_COLUMNS].describe().round(4).to_dict()

    state = _make_empty_state()
    state.update(
        {
            "available": True,
            "error": None,
            "raw_rows": int(len(raw_data)),
            "raw_columns": int(len(raw_data.columns)),
            "missing_before": missing_before,
            "invalid_values_handled": invalid_values,
            "duplicate_rows_removed": duplicate_rows,
            "cleaned_rows": int(len(selected)),
            "cleaned_columns": int(len(selected.columns)),
            "statistics": statistics,
            "figure_urls": figure_urls,
            "cleaned_data": selected,
            "eda": {
                "features": FEATURES,
                "target": TARGET,
                "correlation": selected.corr(numeric_only=True).round(4).to_dict(),
                "statistics": statistics,
            },
        }
    )

    if len(selected) < 6:
        state["error"] = (
            "The cleaned dataset needs at least 6 usable rows to create a reliable "
            "train/test comparison. Dataset summary and EDA are available."
        )
        return state

    x_train, x_test, y_train, y_test = train_test_split(
        selected[FEATURES], selected[TARGET], test_size=0.2, random_state=42
    )
    candidates = [
        ("Linear Regression", LinearRegression()),
        ("Decision Tree Regressor", DecisionTreeRegressor(random_state=42)),
        ("Random Forest Regressor", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
    ]
    evaluation_rows: list[dict[str, Any]] = []
    for name, candidate in candidates:
        candidate.fit(x_train, y_train)
        predictions = candidate.predict(x_test)
        mse = float(mean_squared_error(y_test, predictions))
        evaluation_rows.append(
            {
                "Model": name,
                "MAE": float(mean_absolute_error(y_test, predictions)),
                "MSE": mse,
                "RMSE": math.sqrt(mse),
                "R2": float(r2_score(y_test, predictions)) if len(y_test) > 1 else None,
            }
        )
    evaluation_rows.sort(key=lambda row: row["RMSE"])
    evaluation_frame = pd.DataFrame(evaluation_rows)
    evaluation_frame.to_csv(RESULTS_DIR / "model_evaluation.csv", index=False)
    best_name = evaluation_rows[0]["Model"]
    best_model = candidates[[name for name, _ in candidates].index(best_name)][1]
    best_model.fit(selected[FEATURES], selected[TARGET])
    state.update(
        {
            "models": evaluation_rows,
            "best_model": best_name,
            "model_rmse": evaluation_rows[0]["RMSE"],
            "model": best_model,
        }
    )
    return state


def _public_summary(state: dict[str, Any]) -> dict[str, Any]:
    """Return dataset and model metadata without internal model/data objects."""
    missing_total = int(sum(state["missing_before"].values()))
    return {
        "available": state["available"],
        "error": state["error"],
        "dataset": {
            "filename": DATASET_PATH.name,
            "rows_before_cleaning": state["raw_rows"],
            "columns": state["raw_columns"],
            "missing_values_before_cleaning": state["missing_before"],
            "missing_values_count": missing_total,
            "features": state["features"],
            "target": state["target"],
        },
        "cleaning": {
            "invalid_values_handled": state["invalid_values_handled"],
            "duplicate_rows_removed": state["duplicate_rows_removed"],
            "final_rows": state["cleaned_rows"],
            "final_columns": state["cleaned_columns"],
            "output_file": "/outputs/results/cleaned_dataset.csv" if state["cleaned_data"] is not None else None,
        },
        "statistics": state["statistics"],
        "models": state["models"],
        "best_model": state["best_model"],
        "model_rmse": state["model_rmse"],
        "figure_urls": state["figure_urls"],
    }


def create_app() -> Flask:
    app = Flask(__name__)
    for folder in (OUTPUT_DIR, FIGURES_DIR, RESULTS_DIR, PREDICTIONS_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    state = _prepare_dataset(DATASET_PATH)
    app.config["PROJECT_STATE"] = state

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/outputs/<path:filename>")
    def output_file(filename: str):
        from flask import send_from_directory

        return send_from_directory(OUTPUT_DIR, filename)

    @app.get("/api/summary")
    def api_summary():
        return jsonify(_public_summary(app.config["PROJECT_STATE"]))

    @app.get("/api/eda")
    def api_eda():
        current = app.config["PROJECT_STATE"]
        if current["cleaned_data"] is None:
            return jsonify({"available": False, "error": current["error"], "features": FEATURES}), 503
        return jsonify(
            {
                "available": True,
                "features": FEATURES,
                "target": TARGET,
                "statistics": current["statistics"],
                "correlation": current["eda"]["correlation"],
                "figure_urls": current["figure_urls"],
            }
        )

    @app.post("/api/predict")
    def api_predict():
        current = app.config["PROJECT_STATE"]
        if not current["available"] or current["model"] is None:
            return jsonify({"error": current["error"] or "A trained model is not available."}), 503
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Request body must be a JSON object containing all five input values."}), 400

        values: dict[str, float] = {}
        for feature in FEATURES:
            value = payload.get(feature)
            if value is None or isinstance(value, bool):
                return jsonify({"error": f"{feature} is required and must be a finite number."}), 400
            try:
                number = float(value)
            except (TypeError, ValueError):
                return jsonify({"error": f"{feature} must be a numeric value."}), 400
            if not math.isfinite(number):
                return jsonify({"error": f"{feature} must be a finite number."}), 400
            if feature in ("rainfall_mm", "farm_size_hectares", "sunlight_hours", "fertilizer_kg") and number < 0:
                return jsonify({"error": f"{feature} cannot be negative."}), 400
            if feature == "farm_size_hectares" and number <= 0:
                return jsonify({"error": "farm_size_hectares must be greater than zero."}), 400
            values[feature] = number

        input_frame = pd.DataFrame([values], columns=FEATURES)
        predicted_yield = float(current["model"].predict(input_frame)[0])
        expected_production = values["farm_size_hectares"] * predicted_yield
        dataset_medians = current["cleaned_data"][FEATURES].median()
        insights = []
        insight_text = {
            "rainfall_mm": ("Rainfall", "Compare rainfall with the dataset median; lower input may warrant checking local water availability, while higher input may warrant drainage planning."),
            "soil_quality_index": ("Soil quality index", "This index is interpreted relative to the dataset median only; confirm soil conditions with local testing."),
            "sunlight_hours": ("Sunlight", "Compare sunlight hours with the dataset median and consider local crop requirements and seasonal variation."),
            "fertilizer_kg": ("Fertilizer", "Compare fertilizer input with the dataset median; base any application changes on soil testing and crop-specific guidance."),
            "farm_size_hectares": ("Farm size", "Expected production scales the predicted yield by the entered farm size; plan labor, storage, and logistics for that area."),
        }
        for feature in FEATURES:
            title, message = insight_text[feature]
            baseline = float(dataset_medians[feature])
            direction = "above" if values[feature] > baseline else "below" if values[feature] < baseline else "equal to"
            insights.append(
                {
                    "parameter": title,
                    "value": values[feature],
                    "dataset_median": baseline,
                    "comparison": direction,
                    "insight": message,
                }
            )

        result = {
            "inputs": values,
            "predicted_crop_yield": predicted_yield,
            "expected_production": expected_production,
            "production_formula": "farm size × predicted crop yield",
            "production_unit_note": "Production unit is not specified because the dataset does not document the crop_yield unit.",
            "best_model": current["best_model"],
            "model_rmse": current["model_rmse"],
            "insights": insights,
        }
        prediction_record = {
            **values,
            "predicted_crop_yield": predicted_yield,
            "expected_production": expected_production,
            "best_model": current["best_model"],
            "model_rmse": current["model_rmse"],
        }
        pd.DataFrame([prediction_record]).to_csv(
            PREDICTIONS_DIR / "latest_prediction.csv", index=False
        )
        return jsonify(result)

    @app.errorhandler(500)
    def internal_server_error(error):
        app.logger.exception("Unhandled server error: %s", error)
        return jsonify({"error": "An unexpected server error occurred. Check the server log for details."}), 500

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
