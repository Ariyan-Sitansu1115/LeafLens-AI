/* LeafLens Weather Page – modular, no inline JS */

(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const t = (key, fallback) => window.LeafLensI18n?.t?.(key, fallback) ?? fallback;

  const els = {
    form: $("#weather-form"),
    cityInput: $("#weather-city"),
    languageSelect: $("#weather-language"),
    submitBtn: $("#weather-btn"),
    spinner: $("#weather-spinner"),
    alert: $("#weather-alert"),
    resultCard: $("#weather-result-card"),
    resultContent: $("#weather-result-content"),
    year: $("#year"),
  };

  function setAlert(message) {
    if (!els.alert) return;
    if (!message) {
      els.alert.classList.add("hidden");
      els.alert.textContent = "";
      return;
    }
    els.alert.textContent = message;
    els.alert.classList.remove("hidden");
  }

  function setLoading(isLoading) {
    if (!els.submitBtn) return;
    els.submitBtn.disabled = isLoading;
    if (els.spinner) {
      els.spinner.classList.toggle("hidden", !isLoading);
    }
    const label = els.submitBtn.querySelector(".btn-label");
    if (label) {
      label.textContent = isLoading
      ? t("loading", "Loading…")
      : t("btn_get_weather", "Get Weather");
    }
  }

  function translateCondition(value) {
    const raw = typeof value === "string" ? value.trim() : "";
    if (!raw) return "--";
    const direct = t(`condition_map.${raw}`, raw);
    if (direct !== raw) return direct;

    const aliasMap = {
      high: "High",
      moderate: "Moderate",
      low: "Low",
      dry: "Dry",
      "rain likely": "Rain Likely",
      humid: "Humid",
      wet: "Wet",
    };
    const canonical = aliasMap[raw.toLowerCase()];
    return canonical ? t(`condition_map.${canonical}`, canonical) : raw;
  }

  function applyWeatherUiTranslations() {
    const title = $(".card-hero .h1");
    const subtitle = $(".card-hero .muted");
    const cityLabel = document.querySelector("label[for='weather-city']");
    const languageLabel = document.querySelector("label[for='weather-language']");
    const hint = document.querySelector("#weather-form .row .hint");

    if (title) title.textContent = t("weather_title", "Weather");
    if (subtitle) {
      subtitle.textContent = t(
        "weather_subtitle",
        "Get current weather data for your location. Optionally enter a city name.",
      );
    }
    if (cityLabel) cityLabel.textContent = t("location_label", "City (optional)");
    if (languageLabel) languageLabel.textContent = t("label_language", "Language");
    if (hint) hint.textContent = t("weather_city_hint", "Leave blank to use auto-detected location.");
    if (els.cityInput) els.cityInput.placeholder = t("city_placeholder", "e.g. Delhi, Mumbai");

    const labels = {
      tempValue: t("weather_display.temperature", "Temperature"),
      humidityValue: t("weather_display.humidity", "Humidity"),
      rainValue: t("weather_display.rainfall", "Rainfall"),
      windValue: t("weather_display.wind_speed", "Wind Speed"),
      cloudValue: t("weather_display.clouds", "Cloud Cover"),
      conditionValue: t("weather_display.condition", "Condition"),
    };

    Object.entries(labels).forEach(([id, text]) => {
      const labelEl = document.getElementById(id)?.closest(".weather-card")?.querySelector(".weather-label");
      if (labelEl) labelEl.textContent = text;
    });
  }

  function renderWeather(data) {
    if (!data || !data.success) {
      els.resultContent.innerHTML = `<div class="muted">${t("weather_no_data", "No weather data available.")}</div>`;
      els.resultContent.classList.remove("hidden");
      document.getElementById("weather-report")?.classList.add("hidden");
      els.resultCard.classList.remove("hidden");
      return;
    }

    document.getElementById("weather-report")?.classList.remove("hidden");
    els.resultContent.classList.add("hidden");
    els.resultContent.innerHTML = "";

    const weather = data.weather || {};
    const loc = data.location || {};

    const tempEl = document.getElementById("tempValue");
    const humidityEl = document.getElementById("humidityValue");
    const rainEl = document.getElementById("rainValue");
    const windEl = document.getElementById("windValue");
    const cloudEl = document.getElementById("cloudValue");
    const conditionEl = document.getElementById("conditionValue");
    const weatherTitle = document.getElementById("weatherTitle");

    if (tempEl) tempEl.innerText = (weather.temperature ?? weather.temp ?? "--") + "°C";
    if (humidityEl) humidityEl.innerText = (weather.humidity ?? "--") + "%";
    if (rainEl) rainEl.innerText = (weather.rainfall ?? weather.rain ?? "--") + " mm";
    if (windEl) windEl.innerText = (weather.wind_speed ?? weather.wind ?? "--") + " m/s";
    if (cloudEl) cloudEl.innerText = (weather.clouds ?? "--") + "%";
    if (conditionEl) conditionEl.innerText = translateCondition(weather.condition ?? "--");

    if (weatherTitle) {
      weatherTitle.innerText =
        (loc.city ?? loc.region ?? t("weather_title", "Weather")) + " - " + t("weather_report", "Weather Report");
    }

    els.resultCard.classList.remove("hidden");
  }

  function renderError(message) {
    els.resultContent.innerHTML = `<div class="muted">${message || t("weather_fetch_failed", "Failed to fetch weather data.")}</div>`;
    els.resultContent.classList.remove("hidden");
    document.getElementById("weather-report")?.classList.add("hidden");
    els.resultCard.classList.remove("hidden");
  }

  async function fetchWeather() {
    setAlert("");
    const location = (els.cityInput?.value || "").trim();
    const languageCode = window.LeafLensI18n?.getLanguageCode?.() || els.languageSelect?.value || "en";

    const params = new URLSearchParams();
    if (location) params.set("location", location);
    params.set("language_code", languageCode);

    setLoading(true);
    els.resultCard.classList.add("hidden");

    try {
      const res = await fetch(`/api/weather?${params.toString()}`, { method: "GET" });
      const data = await res.json().catch(() => null);

      if (!res.ok) {
        const errMsg = data?.error || `Request failed (${res.status})`;
        setAlert(errMsg);
        renderError(errMsg);
        return;
      }

      renderWeather(data);
    } catch {
      const msg = t("weather_service_unreachable", "Could not reach weather service. Please try again.");
      setAlert(msg);
      renderError(msg);
    } finally {
      setLoading(false);
    }
  }

  function wireEvents() {
    if (els.form) {
      els.form.addEventListener("submit", (e) => {
        e.preventDefault();
        fetchWeather();
      });
    }
  }

  function boot() {
    if (els.year) els.year.textContent = String(new Date().getFullYear());
    window.LeafLensI18n?.applyTranslations?.();
    applyWeatherUiTranslations();
    wireEvents();

    window.addEventListener("leaflens:language-changed", () => {
      window.LeafLensI18n?.applyTranslations?.();
      applyWeatherUiTranslations();
      if (els.resultCard && !els.resultCard.classList.contains("hidden")) {
        fetchWeather();
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
