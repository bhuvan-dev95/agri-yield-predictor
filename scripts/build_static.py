"""Build the GitHub Pages version of the dashboard from the real CSV."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import FEATURES, app


SITE = ROOT / "site"


def serialize_tree(estimator: DecisionTreeRegressor) -> dict[str, list[float] | list[int]]:
    tree = estimator.tree_
    return {
        "children_left": tree.children_left.tolist(),
        "children_right": tree.children_right.tolist(),
        "feature": tree.feature.tolist(),
        "threshold": tree.threshold.tolist(),
        "value": tree.value[:, 0, 0].tolist(),
    }


def serialize_model(model: object) -> dict[str, object]:
    if isinstance(model, LinearRegression):
        return {
            "kind": "linear",
            "intercept": float(model.intercept_),
            "coefficients": np.asarray(model.coef_, dtype=float).tolist(),
        }
    if isinstance(model, DecisionTreeRegressor):
        return {"kind": "tree", "tree": serialize_tree(model)}
    if isinstance(model, RandomForestRegressor):
        return {"kind": "forest", "trees": [serialize_tree(tree) for tree in model.estimators_]}
    raise TypeError(f"Cannot publish unsupported model type: {type(model).__name__}")


def main() -> None:
    state = app.config["PROJECT_STATE"]
    if not state["available"] or state["model"] is None:
        raise SystemExit(f"Cannot build GitHub Pages site: {state['error'] or 'model unavailable'}")

    SITE.mkdir(parents=True, exist_ok=True)
    (SITE / "static").mkdir(exist_ok=True)
    (SITE / "data").mkdir(exist_ok=True)
    for folder in ("figures", "results"):
        (SITE / "outputs" / folder).mkdir(parents=True, exist_ok=True)

    summary = {
        "available": True,
        "error": None,
        "dataset": {
            "filename": "crop_yield_data.csv",
            "rows_before_cleaning": state["raw_rows"],
            "columns": state["raw_columns"],
            "missing_values_before_cleaning": state["missing_before"],
            "missing_values_count": int(sum(state["missing_before"].values())),
            "features": FEATURES,
            "target": "crop_yield",
        },
        "cleaning": {
            "invalid_values_handled": state["invalid_values_handled"],
            "duplicate_rows_removed": state["duplicate_rows_removed"],
            "final_rows": state["cleaned_rows"],
            "final_columns": state["cleaned_columns"],
            "output_file": "./outputs/results/cleaned_dataset.csv",
        },
        "statistics": state["statistics"],
        "models": state["models"],
        "best_model": state["best_model"],
        "model_rmse": state["model_rmse"],
        "figure_urls": {
            name: url.replace("/outputs/", "./outputs/")
            for name, url in state["figure_urls"].items()
        },
    }
    model_data = {
        "features": FEATURES,
        "medians": {name: float(state["cleaned_data"][name].median()) for name in FEATURES},
        "best_model": state["best_model"],
        "model_rmse": state["model_rmse"],
        "model": serialize_model(state["model"]),
    }
    (SITE / "data" / "summary.json").write_text(json.dumps(summary, allow_nan=False), encoding="utf-8")
    (SITE / "data" / "model.json").write_text(json.dumps(model_data, allow_nan=False), encoding="utf-8")

    with app.test_request_context("/"):
        from flask import render_template

        html = render_template("index.html")
    html = html.replace("<body>", '<body data-static-mode="true">')
    html = html.replace("/static/", "./static/")
    (SITE / "index.html").write_text(html, encoding="utf-8")

    for filename in ("style.css", "script.js"):
        shutil.copy2(ROOT / "static" / filename, SITE / "static" / filename)
    for filename in state["figure_urls"].values():
        name = Path(filename).name
        shutil.copy2(ROOT / "outputs" / "figures" / name, SITE / "outputs" / "figures" / name)
    for filename in ("cleaned_dataset.csv", "model_evaluation.csv"):
        shutil.copy2(ROOT / "outputs" / "results" / filename, SITE / "outputs" / "results" / filename)
    shutil.copy2(ROOT / "crop_yield_data.csv", SITE / "crop_yield_data.csv")
    shutil.copy2(ROOT / "README.md", SITE / "README.md")
    print(f"Built static dashboard in {SITE}")
    print(f"Dataset rows: {state['cleaned_rows']}; selected model: {state['best_model']}")


if __name__ == "__main__":
    main()
