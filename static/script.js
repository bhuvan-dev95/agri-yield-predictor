"use strict";

const numberFormat = new Intl.NumberFormat(undefined, { maximumFractionDigits: 4 });
let modelChart;
const staticMode = document.body.dataset.staticMode === "true";
let staticModelBundle;

function formatNumber(value) {
  return typeof value === "number" && Number.isFinite(value) ? numberFormat.format(value) : "—";
}

function showAlert(message) {
  const alert = document.getElementById("global-alert");
  alert.textContent = message;
  alert.classList.remove("hidden");
}

function setFigure(imageId, emptyId, url) {
  const image = document.getElementById(imageId);
  const empty = document.getElementById(emptyId);
  if (!url) return;
  image.src = url;
  image.classList.remove("hidden");
  empty.classList.add("hidden");
}

function renderSummary(summary) {
  const dataset = summary.dataset;
  const cleaning = summary.cleaning;
  document.getElementById("row-count").textContent = formatNumber(dataset.rows_before_cleaning);
  document.getElementById("column-count").textContent = formatNumber(dataset.columns);
  document.getElementById("missing-count").textContent = formatNumber(dataset.missing_values_count);
  document.getElementById("clean-row-count").textContent = formatNumber(cleaning.final_rows);
  document.getElementById("missing-detail").textContent = formatNumber(dataset.missing_values_count);
  document.getElementById("invalid-count").textContent = formatNumber(cleaning.invalid_values_handled);
  document.getElementById("duplicate-count").textContent = formatNumber(cleaning.duplicate_rows_removed);
  document.getElementById("final-size").textContent =
    `${formatNumber(cleaning.final_rows)} rows × ${formatNumber(cleaning.final_columns)} columns`;
  document.getElementById("sidebar-status").textContent = summary.best_model
    ? "Dataset & model ready"
    : summary.available
      ? "Dataset loaded · model unavailable"
      : "Dataset unavailable";

  const featureList = document.getElementById("feature-list");
  featureList.replaceChildren();
  (dataset.features || []).forEach((feature) => {
    const chip = document.createElement("span");
    chip.className = "feature-chip";
    chip.textContent = feature;
    const tag = document.createElement("small");
    tag.textContent = "input";
    chip.append(" ", tag);
    featureList.append(chip);
  });
  const targetChip = document.createElement("span");
  targetChip.className = "feature-chip target";
  targetChip.textContent = dataset.target;
  const targetTag = document.createElement("small");
  targetTag.textContent = "target";
  targetChip.append(" ", targetTag);
  featureList.append(targetChip);

  const urls = summary.figure_urls || {};
  setFigure("distribution-figure", "distribution-empty", urls.distributions);
  setFigure("relationship-figure", "relationship-empty", urls.relationships);
  setFigure("correlation-figure", "correlation-empty", urls.correlation);
  renderStatistics(summary.statistics);
  renderModels(summary.models || [], summary.best_model);

  const predictButton = document.getElementById("predict-button");
  predictButton.disabled = !summary.best_model;
  if (summary.error) showAlert(summary.error);
  else if (!summary.available) showAlert("Dataset unavailable. Add crop_yield_data.csv beside main.py using the required column names to enable analysis.");
}

function renderStatistics(statistics) {
  const table = document.getElementById("statistics-table");
  const measures = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"];
  const features = Object.keys(statistics || {});
  table.replaceChildren();
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  ["Measure", ...features].forEach((label) => {
    const cell = document.createElement("th");
    cell.textContent = label.replaceAll("_", " ");
    headRow.append(cell);
  });
  head.append(headRow);
  table.append(head);
  const body = document.createElement("tbody");
  measures.forEach((measure) => {
    const row = document.createElement("tr");
    const name = document.createElement("td");
    name.textContent = measure;
    row.append(name);
    features.forEach((feature) => {
      const cell = document.createElement("td");
      cell.textContent = formatNumber(statistics[feature]?.[measure]);
      row.append(cell);
    });
    body.append(row);
  });
  table.append(body);
}

function renderModels(models, bestModel) {
  const body = document.querySelector("#model-table tbody");
  body.replaceChildren();
  if (!models.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 5;
    cell.className = "muted";
    cell.textContent = "Model results will appear when enough valid data is available.";
    row.append(cell);
    body.append(row);
    return;
  }
  models.forEach((model) => {
    const row = document.createElement("tr");
    if (model.Model === bestModel) row.className = "best-row";
    [model.Model, formatNumber(model.MAE), formatNumber(model.MSE), formatNumber(model.RMSE), formatNumber(model.R2)]
      .forEach((value) => {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      });
    body.append(row);
  });

  document.getElementById("best-chip").classList.remove("hidden");
  if (typeof Chart !== "undefined") {
    if (modelChart) modelChart.destroy();
    modelChart = new Chart(document.getElementById("model-chart"), {
      type: "bar",
      data: {
        labels: models.map((model) => model.Model.replace(" Regressor", "")),
        datasets: [{
          label: "RMSE",
          data: models.map((model) => model.RMSE),
          backgroundColor: models.map((model) => model.Model === bestModel ? "#34845b" : "#c7d8bd"),
          borderRadius: 5,
          maxBarThickness: 38,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { display: false }, ticks: { font: { family: "DM Sans", size: 9 } } },
          y: { beginAtZero: true, grid: { color: "#eef1ed" }, ticks: { font: { family: "DM Sans", size: 9 } } },
        },
      },
    });
  }
}

async function loadDashboard() {
  try {
    const response = await fetch(staticMode ? "./data/summary.json" : "/api/summary");
    if (!response.ok) throw new Error(`Dashboard request failed (${response.status}).`);
    renderSummary(await response.json());
    if (staticMode) {
      const modelResponse = await fetch("./data/model.json");
      if (!modelResponse.ok) throw new Error(`Model request failed (${modelResponse.status}).`);
      staticModelBundle = await modelResponse.json();
    }
  } catch (error) {
    document.getElementById("sidebar-status").textContent = "Dashboard error";
    showAlert(error.message || "Could not load the project summary. Please refresh the page.");
  }
}

document.getElementById("prediction-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  const errorElement = document.getElementById("form-error");
  const button = document.getElementById("predict-button");
  errorElement.classList.add("hidden");

  const values = {};
  for (const feature of ["rainfall_mm", "soil_quality_index", "farm_size_hectares", "sunlight_hours", "fertilizer_kg"]) {
    const input = form.elements.namedItem(feature);
    const value = input.value.trim();
    const number = Number(value);
    if (value === "" || !Number.isFinite(number)) {
      errorElement.textContent = `${input.labels[0].textContent.trim()} must be a valid finite number.`;
      errorElement.classList.remove("hidden");
      input.focus();
      return;
    }
    if (["rainfall_mm", "farm_size_hectares", "sunlight_hours", "fertilizer_kg"].includes(feature) && number < 0) {
      errorElement.textContent = `${input.labels[0].textContent.trim()} cannot be negative.`;
      errorElement.classList.remove("hidden");
      input.focus();
      return;
    }
    if (feature === "farm_size_hectares" && number === 0) {
      errorElement.textContent = "Farm size must be greater than zero.";
      errorElement.classList.remove("hidden");
      input.focus();
      return;
    }
    values[feature] = number;
  }

  button.disabled = true;
  button.textContent = "Calculating…";
  try {
    if (staticMode) {
      if (!staticModelBundle) throw new Error("The prediction model has not loaded yet. Refresh and try again.");
      renderPrediction(createLocalPrediction(values, staticModelBundle));
      return;
    }
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(values),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `Prediction request failed (${response.status}).`);
    renderPrediction(result);
  } catch (error) {
    errorElement.textContent = error.message || "Could not calculate the prediction. Please try again.";
    errorElement.classList.remove("hidden");
  } finally {
    button.disabled = false;
    button.innerHTML = 'Predict crop yield <span aria-hidden="true">→</span>';
  }
});

function predictWithTree(tree, values) {
  let node = 0;
  while (tree.children_left[node] !== -1) {
    const featureValue = values[tree.feature[node]];
    node = featureValue <= tree.threshold[node] ? tree.children_left[node] : tree.children_right[node];
  }
  return tree.value[node];
}

function predictWithModel(values, model) {
  if (model.kind === "linear") {
    return model.intercept + model.coefficients.reduce((sum, coefficient, index) => sum + coefficient * values[index], 0);
  }
  if (model.kind === "tree") return predictWithTree(model.tree, values);
  if (model.kind === "forest") {
    return model.trees.reduce((sum, tree) => sum + predictWithTree(tree, values), 0) / model.trees.length;
  }
  throw new Error("The static site contains an unsupported prediction model.");
}

function createLocalPrediction(inputs, bundle) {
  const orderedValues = bundle.features.map((feature) => inputs[feature]);
  const predictedYield = predictWithModel(orderedValues, bundle.model);
  const medians = bundle.medians;
  const recommendations = {
    rainfall_mm: ["Rainfall", "Compare rainfall with the dataset median; lower input may warrant checking local water availability, while higher input may warrant drainage planning."],
    soil_quality_index: ["Soil quality index", "This index is interpreted relative to the dataset median only; confirm soil conditions with local testing."],
    sunlight_hours: ["Sunlight", "Compare sunlight hours with the dataset median and consider local crop requirements and seasonal variation."],
    fertilizer_kg: ["Fertilizer", "Compare fertilizer input with the dataset median; base any application changes on soil testing and crop-specific guidance."],
    farm_size_hectares: ["Farm size", "Expected production scales the predicted yield by the entered farm size; plan labor, storage, and logistics for that area."],
  };
  const insights = bundle.features.map((feature) => {
    const [parameter, insight] = recommendations[feature];
    const comparison = inputs[feature] > medians[feature] ? "above" : inputs[feature] < medians[feature] ? "below" : "equal to";
    return { parameter, value: inputs[feature], dataset_median: medians[feature], comparison, insight };
  });
  return {
    inputs,
    predicted_crop_yield: predictedYield,
    expected_production: inputs.farm_size_hectares * predictedYield,
    production_formula: "farm size × predicted crop yield",
    production_unit_note: "Production unit is not specified because the dataset does not document the crop_yield unit.",
    best_model: bundle.best_model,
    model_rmse: bundle.model_rmse,
    insights,
  };
}

function renderPrediction(result) {
  document.getElementById("result-empty").classList.add("hidden");
  document.getElementById("result-content").classList.remove("hidden");
  document.getElementById("predicted-yield").textContent = formatNumber(result.predicted_crop_yield);
  document.getElementById("expected-production").textContent = formatNumber(result.expected_production);
  document.getElementById("production-note").textContent = result.production_unit_note;
  document.getElementById("result-model").textContent = result.best_model;
  document.getElementById("result-rmse").textContent = formatNumber(result.model_rmse);

  const list = document.getElementById("insight-list");
  list.replaceChildren();
  result.insights.forEach((insight) => {
    const item = document.createElement("li");
    const title = document.createElement("strong");
    title.textContent = `${insight.parameter} · ${insight.comparison} dataset median`;
    item.append(title, document.createElement("br"), document.createTextNode(insight.insight));
    list.append(item);
  });
  const download = document.getElementById("prediction-download");
  if (staticMode) {
    const columns = [...Object.keys(result.inputs), "predicted_crop_yield", "expected_production", "best_model", "model_rmse"];
    const values = [...Object.values(result.inputs), result.predicted_crop_yield, result.expected_production, result.best_model, result.model_rmse];
    const csv = [columns.join(","), values.map((value) => `"${String(value).replaceAll('"', '""')}"`).join(",")].join("\n");
    download.href = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
    download.classList.remove("hidden");
  } else {
    download.classList.add("hidden");
  }
  document.getElementById("prediction-result").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

document.querySelectorAll(".nav-link").forEach((link) => {
  link.addEventListener("click", () => {
    document.querySelectorAll(".nav-link").forEach((item) => item.classList.remove("active"));
    link.classList.add("active");
  });
});

loadDashboard();
