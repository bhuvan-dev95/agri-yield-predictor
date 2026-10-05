# Agricultural Earth Resource Management – Crop Yield Prediction

A college PBL mini-project that analyzes a supplied crop-yield CSV, compares regression models, and serves a responsive dashboard for yield estimation and dataset-relative resource insights.

## Problem statement

Agricultural outcomes are influenced by environmental conditions and farm inputs. This project uses the provided observations to explore how rainfall, soil quality, farm size, sunlight, and fertilizer relate to crop yield. Its predictions are data-driven estimates, not guaranteed agricultural advice.

## Objectives

- Inspect and clean the supplied dataset without changing its required column names or inventing observations.
- Explore feature distributions, feature-to-yield relationships, correlations, and descriptive statistics.
- Train and compare three regression algorithms using a train/test split.
- Select the model with the lowest test-set RMSE and expose a prediction API.
- Present an expected-production calculation and cautious, dataset-relative resource insights in a responsive dashboard.

## Technologies

Python, Flask, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, HTML5, CSS3, JavaScript, and Chart.js.

## Dataset description

Place `crop_yield_data.csv` in the same directory as `main.py`. The CSV must contain these exact columns:

| Column | Use |
| --- | --- |
| `rainfall_mm` | Model input |
| `soil_quality_index` | Model input |
| `farm_size_hectares` | Model input |
| `sunlight_hours` | Model input |
| `fertilizer_kg` | Model input |
| `crop_yield` | Prediction target |

The application reads the source CSV as provided and selects only these columns for analysis. It does not generate substitute data. The unit for `crop_yield` is not inferred; expected production is shown without claiming a unit.

## Methodology

1. **Problem understanding:** estimate crop yield from five agricultural/environmental inputs.
2. **Data collection:** read `crop_yield_data.csv` at application startup and verify all required columns.
3. **Cleaning and preparation:** coerce feature/target values to numeric, count invalid entries, treat non-finite values as missing, remove duplicate rows, discard rows without a target, and median-impute missing feature values. Negative physical quantities and negative yield values are treated as invalid; no range is assumed for the soil-quality index.
4. **Exploratory analysis:** generate feature-distribution plots, feature-versus-yield plots, a correlation heatmap, and descriptive statistics.
5. **Model building:** split the cleaned dataset into training and test sets (80/20, fixed random seed), compare the models below, and refit the lowest-RMSE model on all cleaned rows for use in predictions.
6. **Evaluation:** report MAE, MSE, RMSE, and R² on the held-out test set. Datasets with fewer than six usable rows are summarized and plotted but not modeled.
7. **Prediction:** validate inputs and predict with the selected model. Expected production is calculated as `farm size × predicted crop yield`.
8. **Resource analysis:** compare inputs to the corresponding cleaned-dataset medians and provide contextual prompts. These comparisons are not agronomic prescriptions.

## Machine-learning models

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

The best model is selected primarily by the **lowest test-set RMSE**.

## Evaluation metrics

- **MAE:** mean absolute error
- **MSE:** mean squared error
- **RMSE:** square root of mean squared error
- **R²:** coefficient of determination

## Project structure

```text
.
├── main.py
├── crop_yield_data.csv             # Supply the original dataset here
├── requirements.txt
├── README.md
├── templates/
│   └── index.html
├── static/
│   ├── script.js
│   └── style.css
└── outputs/
    ├── figures/
    ├── predictions/
    └── results/
```

Generated files are written under `outputs/`:

- `outputs/results/cleaned_dataset.csv`
- `outputs/results/model_evaluation.csv`
- `outputs/figures/feature_distributions.png`
- `outputs/figures/feature_vs_crop_yield.png`
- `outputs/figures/correlation_heatmap.png`
- `outputs/predictions/latest_prediction.csv` (after a successful prediction)

## Installation

Use Python 3.10 or newer. From the project directory:

```bash
pip install -r requirements.txt
```

Add the source `crop_yield_data.csv` beside `main.py`. The application validates it at startup. If the CSV is absent or invalid, the dashboard still starts and reports the issue; model comparison and predictions remain unavailable until a valid dataset is supplied.

## How to run

```bash
python main.py
```

Open <http://127.0.0.1:5000>.

## API

- `GET /api/summary` — dataset dimensions, cleaning counts, descriptive statistics, evaluation results, and figure paths.
- `GET /api/eda` — descriptive statistics, correlations, and EDA figure paths.
- `POST /api/predict` — validate inputs, predict yield, calculate expected production, and return dataset-relative insights.

The request body is a JSON object with numeric values for the five exact feature names listed in the dataset table. Use values from a real scenario; the project does not provide substitute dataset observations or hard-coded predictions.

## Sample prediction workflow

1. Start the app with a valid CSV in the project directory.
2. Review the cleaned dataset summary, EDA plots, and model comparison.
3. Enter values for all five model inputs in the scenario form.
4. Select **Predict crop yield** to view the model estimate, expected-production calculation, selected model and RMSE, and dataset-relative insights.
5. The latest successful scenario is saved to `outputs/predictions/latest_prediction.csv`.

## Future scope

- Add time-aware or location-aware validation if those fields are provided by the dataset.
- Track multiple prediction scenarios and export a report.
- Add crop-specific, locally validated agronomic recommendations.
- Monitor model performance on new observations and compare additional algorithms.

## Error handling and limitations

The application reports missing CSV files, unreadable CSVs, missing required columns, empty/cleaned-empty data, insufficient rows for modeling, invalid numeric inputs, and API errors. Model quality and estimates depend on the quality and coverage of the supplied data; this educational project is not a substitute for local agricultural expertise.
