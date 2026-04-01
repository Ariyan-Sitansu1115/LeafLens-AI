// LeafLens Insight client-side script
// Provides functions to load and render IoT sensor + 3-day forecast

const REFRESH_INTERVAL_MS = 7000;
const t = (key, fallback) => window.LeafLensI18n?.t?.(key, fallback) ?? fallback;
const carbonTrendHistory = [];
let carbonTrendChart = null;

function qs(id) {
  return document.getElementById(id);
}

async function loadInsightData() {
  showError(null);
  showLoading(true);
  try {
    const endpoint = "/api/insight-data";
    const resp = await fetch(endpoint, { method: "GET", cache: "no-store" });

    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`Server error: ${resp.status} ${text}`);
    }

    const data = await resp.json();

    const sensor = data?.sensor || null;
    const rainPrediction = data?.rain_prediction || null;
    const carbonFootprint = data?.carbon_footprint || null;

    renderSensorData(sensor);
    renderRainPrediction(rainPrediction);
    renderForecast(data.forecast || []);
    renderCarbonFootprint(carbonFootprint);
    renderCarbonAndIrrigation(data);
  } catch (err) {
    console.error("loadInsightData error", err);
    showError(err.message || t("insight_load_failed", "Failed to load data"));
    renderSensorData(null);
    renderRainPrediction(null);
    renderForecast([]);
    renderCarbonFootprint(null);
    renderCarbonAndIrrigation(null);
  } finally {
    showLoading(false);
  }
}

function translateCondition(value) {
  const raw = typeof value === "string" ? value.trim() : "";
  if (!raw) return t("condition_map.Dry", "Dry");

  const direct = t(`condition_map.${raw}`, raw);
  if (direct !== raw) return direct;

  const lowered = raw.toLowerCase();
  const aliasMap = {
    high: "High",
    moderate: "Moderate",
    low: "Low",
    dry: "Dry",
    "rain likely": "Rain Likely",
    humid: "Humid",
    wet: "Wet",
  };
  const canonical = aliasMap[lowered];
  return canonical ? t(`condition_map.${canonical}`, canonical) : raw;
}

async function refreshDashboard() {
  await loadInsightData();
}

function renderSensorData(data) {
  const temperature = data?.temperature;
  const humidity = data?.humidity;
  const soilMoisture = data?.soil_moisture;
  const stressIndex = data?.stress_index;

  const tempEl = qs("sensor-temp");
  const humidityEl = qs("sensor-humidity");
  const soilEl = qs("sensor-soil");
  const stressEl = qs("sensor-stress");

  if (tempEl) {
    tempEl.textContent = Number.isFinite(Number(temperature)) ? `${Number(temperature).toFixed(1)} °C` : "0.0 °C";
  }
  if (humidityEl) {
    humidityEl.textContent = Number.isFinite(Number(humidity)) ? `${Number(humidity).toFixed(1)} %` : "0.0 %";
  }
  if (soilEl) {
    soilEl.textContent = Number.isFinite(Number(soilMoisture)) ? `${Number(soilMoisture).toFixed(1)} %` : "0.0 %";
  }
  if (stressEl) {
    stressEl.textContent = Number.isFinite(Number(stressIndex)) ? `${Number(stressIndex).toFixed(1)}` : "0.0";
  }
}

function renderRainPrediction(rain) {
  const chanceRaw = `${rain?.chance || "Low"}`;
  const chance = ["High", "Moderate", "Low"].includes(chanceRaw) ? chanceRaw : "Low";
  const probabilityRaw = Number(rain?.probability);
  const probability = Number.isFinite(probabilityRaw) ? Math.max(0, Math.min(100, probabilityRaw)) : 0;

  const badge = qs("rain-badge");
  const chanceEl = qs("rain-chance");
  const probabilityEl = qs("rain-probability");

  const chanceLabelMap = {
    High: t("rain_chance_high", "High"),
    Moderate: t("rain_chance_moderate", "Moderate"),
    Low: t("rain_chance_low", "Low"),
  };
  const localizedChance = translateCondition(chanceLabelMap[chance] || chance);

  if (chanceEl) chanceEl.textContent = localizedChance;
  if (probabilityEl) probabilityEl.textContent = `${probability.toFixed(1)}%`;

  if (badge) {
    badge.textContent = localizedChance;
    badge.classList.remove("rain-high", "rain-moderate", "rain-low");
    if (chance === "High") {
      badge.classList.add("rain-high");
    } else if (chance === "Moderate") {
      badge.classList.add("rain-moderate");
    } else {
      badge.classList.add("rain-low");
    }
  }
}

function renderForecast(forecast) {
  const container = qs("forecast-list");
  if (!container) return;
  container.innerHTML = "";

  if (!Array.isArray(forecast) || forecast.length === 0) {
    container.textContent = t("no_data_available", "No data available");
    return;
  }

  const wrapper = document.createElement("div");
  wrapper.className = "forecast-card-wrapper";

  forecast.forEach((day, index) => {
    const card = document.createElement("div");
    card.className = "forecast-card";

    const dayLabel = typeof day.day === "string" && day.day.trim()
      ? day.day
      : `${t("day", "Day")} ${index + 1}`;

    const temperature = day.temperature;
    const humidity = day.humidity;
    const condition = translateCondition(
      typeof day.condition === "string" && day.condition.trim() ? day.condition : "Dry",
    );

    card.innerHTML = `
      <div class="forecast-date">${dayLabel}</div>
      <div class="forecast-item">
        <span>🌡 ${t("weather_display.temperature", "Temperature")}</span>
        <strong>${Number.isFinite(Number(temperature)) ? Number(temperature).toFixed(1) : "0.0"} °C</strong>
      </div>
      <div class="forecast-item">
        <span>💧 ${t("weather_display.humidity", "Humidity")}</span>
        <strong>${Number.isFinite(Number(humidity)) ? Number(humidity).toFixed(1) : "0.0"} %</strong>
      </div>
      <div class="forecast-item">
        <span>☁ ${t("weather_display.condition", "Condition")}</span>
        <strong>${condition}</strong>
      </div>
    `;

    wrapper.appendChild(card);
  });

  container.appendChild(wrapper);
}

function _safeNumber(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function renderCarbonFootprint(carbon) {
  const card = qs("carbon-card");
  if (!card) return;

  try {
    if (!carbon || typeof carbon !== "object") {
      card.classList.add("hidden");
      return;
    }

    const co2Kg = Math.max(0, _safeNumber(carbon.co2_kg, 0));
    const ecoScore = Math.max(0, Math.min(100, _safeNumber(carbon.eco_score, 0)));
    const statusRaw = typeof carbon.status === "string" ? carbon.status : "Low";
    const status = ["Low", "Moderate", "High"].includes(statusRaw) ? statusRaw : "Low";
    const recommendation =
      typeof carbon.recommendation === "string" && carbon.recommendation.trim()
        ? carbon.recommendation.trim()
        : "Good sustainable farming practice";

    const co2El = qs("carbon-co2");
    const scoreEl = qs("carbon-eco-score");
    const statusEl = qs("carbon-status-legacy");
    const tipEl = qs("carbon-tip-legacy");

    if (co2El) co2El.textContent = `${co2Kg.toFixed(3)} kg CO₂`;
    if (scoreEl) scoreEl.textContent = `${ecoScore.toFixed(0)}/100`;
    if (tipEl) tipEl.textContent = recommendation;

    if (statusEl) {
      statusEl.textContent = status;
      statusEl.classList.remove("carbon-status-low", "carbon-status-moderate", "carbon-status-high");
      if (status === "High") {
        statusEl.classList.add("carbon-status-high");
      } else if (status === "Moderate") {
        statusEl.classList.add("carbon-status-moderate");
      } else {
        statusEl.classList.add("carbon-status-low");
      }
    }

    card.classList.remove("hidden");
    trackCarbonTrend(co2Kg);
    renderCarbonTrendChart();
  } catch (error) {
    console.error("renderCarbonFootprint error", error);
    card.classList.add("hidden");
  }
}

function renderCarbonAndIrrigation(data) {
  const section = qs("carbon-section");
  if (!section) return;

  try {
    const carbon = data?.carbon_footprint;
    const pumpStatus = data?.pump_status;
    console.log("Pump Status:", data?.pump_status);

    if (!carbon || typeof carbon !== "object" || Array.isArray(carbon)) {
      section.style.display = "none";
      return;
    }

    section.style.display = "block";

    const irrigationEl = qs("irrigation-status");
    if (irrigationEl) {
      let irrigationText = "🛑 Not Irrigating (Motor OFF)";
      if (pumpStatus === "ON") {
        irrigationText = "🚰 Irrigating (Motor ON)";
      }
      irrigationEl.innerText = irrigationText;
    }

    const co2El = qs("co2-value");
    const statusEl = qs("carbon-status");
    const ecoScoreEl = qs("eco-score");
    const tipEl = qs("carbon-tip");

    const co2Raw = Number(carbon?.co2_kg);
    const co2Value = Number.isFinite(co2Raw) ? co2Raw : 0;
    const statusRaw = typeof carbon?.status === "string" ? carbon.status.trim() : "Low";
    const status = statusRaw || "Low";
    const ecoRaw = Number(carbon?.eco_score);
    const ecoScore = Number.isFinite(ecoRaw) ? ecoRaw : 0;
    const recommendation = typeof carbon?.recommendation === "string" && carbon.recommendation.trim()
      ? carbon.recommendation.trim()
      : "No recommendation available";

    if (co2El) {
      co2El.textContent = `${co2Value.toFixed(3)} kg CO₂`;
    }

    if (statusEl) {
      statusEl.textContent = status;
      statusEl.classList.remove("carbon-status-low", "carbon-status-moderate", "carbon-status-high");
      const normalized = status.toLowerCase();
      if (normalized === "high") {
        statusEl.classList.add("carbon-status-high");
      } else if (normalized === "moderate") {
        statusEl.classList.add("carbon-status-moderate");
      } else {
        statusEl.classList.add("carbon-status-low");
      }
    }

    if (ecoScoreEl) {
      ecoScoreEl.textContent = String(ecoScore);
    }

    if (tipEl) {
      tipEl.textContent = recommendation;
    }
  } catch (error) {
    console.error("renderCarbonAndIrrigation error", error);
    section.style.display = "none";
  }
}

function trackCarbonTrend(co2Kg) {
  carbonTrendHistory.push({
    label: new Date().toLocaleTimeString(),
    value: co2Kg,
  });

  if (carbonTrendHistory.length > 20) {
    carbonTrendHistory.shift();
  }
}

function renderCarbonTrendChart() {
  const chartWrap = qs("carbon-chart-wrap");
  const canvas = qs("carbon-chart");
  if (!chartWrap || !canvas) return;

  if (!window.Chart || carbonTrendHistory.length < 2) {
    chartWrap.classList.add("hidden");
    return;
  }

  chartWrap.classList.remove("hidden");

  const labels = carbonTrendHistory.map((item) => item.label);
  const values = carbonTrendHistory.map((item) => item.value);
  const rootStyles = window.getComputedStyle(document.documentElement);
  const lineColor = rootStyles.getPropertyValue("--success").trim() || "#22c55e";

  if (carbonTrendChart) {
    carbonTrendChart.destroy();
  }

  const ctx = canvas.getContext("2d");
  if (!ctx) {
    chartWrap.classList.add("hidden");
    return;
  }

  carbonTrendChart = new window.Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "CO₂ (kg)",
          data: values,
          borderColor: lineColor,
          backgroundColor: lineColor,
          tension: 0.35,
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
      },
      scales: {
        y: {
          beginAtZero: true,
        },
      },
    },
  });
}

function showLoading(show) {
  const spinner = qs("insight-spinner");
  const btn = qs("insight-refresh");
  if (spinner) spinner.classList.toggle("hidden", !show);
  if (btn) btn.disabled = !!show;
}

function showError(message) {
  const alertEl = qs("insight-alert");
  if (!alertEl) return;
  if (!message) {
    alertEl.classList.add("hidden");
    alertEl.textContent = "";
    return;
  }
  alertEl.classList.remove("hidden");
  alertEl.classList.add("alert-error");
  alertEl.textContent = message;
}

// Wiring UI events on DOM ready
document.addEventListener("DOMContentLoaded", () => {
  const yearEl = qs("year");
  if (yearEl) yearEl.textContent = String(new Date().getFullYear());
  window.LeafLensI18n?.applyTranslations?.();

  const refresh = qs("insight-refresh");
  let refreshTimerId = null;

  if (refresh) {
    refresh.addEventListener("click", () => {
      refreshDashboard().catch((e) => console.error(e));
    });
  }

  window.addEventListener("leaflens:language-changed", () => {
    window.LeafLensI18n?.applyTranslations?.();
    refreshDashboard().catch((e) => console.error(e));
  });

  // Initial load with default
  refreshDashboard().catch((e) => console.error(e));

  refreshTimerId = window.setInterval(() => {
    refreshDashboard().catch((e) => console.error(e));
  }, REFRESH_INTERVAL_MS);

  window.addEventListener("beforeunload", () => {
    if (refreshTimerId !== null) {
      window.clearInterval(refreshTimerId);
    }
  });
});
