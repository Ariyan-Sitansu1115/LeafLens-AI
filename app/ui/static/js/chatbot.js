/* LeafLens Insight Chatbot – disease-aware multilingual text chat */

(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const t = (key, fallback) => window.LeafLensI18n?.t?.(key, fallback) ?? fallback;

  const els = {
    year: $("#year"),
    alert: $("#chatbot-alert"),
    contextText: $("#chatbot-context-text"),
    chat: $("#chatbot-chat"),
    input: $("#chatbot-input"),
    sendBtn: $("#chatbot-send"),
    spinner: $("#chatbot-spinner"),
    globalLanguageSelect: $("#global-language-select-chatbot"),
  };

  const state = {
    disease: "",
    crop: "",
    predictionId: null,
    languageCode: "en",
    introRendered: false,
  };

  function setAlert(message) {
    if (!els.alert) return;
    if (!message) {
      els.alert.textContent = "";
      els.alert.classList.add("hidden");
      return;
    }
    els.alert.textContent = message;
    els.alert.classList.remove("hidden");
  }

  function setSending(isSending) {
    if (els.sendBtn) els.sendBtn.disabled = !!isSending;
    if (els.spinner) els.spinner.classList.toggle("hidden", !isSending);
  }

  function escapeHtml(s) {
    return String(s)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function appendMessage(role, text) {
    if (!els.chat) return;

    const safeRole = role === "user" ? "user" : "ai";
    const roleLabel = safeRole === "user" ? t("chatbot_user", "You") : t("chatbot_bot", "LeafLens Bot");

    const node = document.createElement("div");
    node.className = `chat-bubble ${safeRole}`;
    node.innerHTML = `
      <div class="chat-role">${escapeHtml(roleLabel)}</div>
      <div class="chat-body"><p>${escapeHtml(text || "")}</p></div>
    `;

    els.chat.appendChild(node);
    els.chat.scrollTop = els.chat.scrollHeight;
  }

  function addTypingIndicator() {
    if (!els.chat) return null;
    const node = document.createElement("div");
    node.className = "typing-indicator";
    node.innerHTML = "<span></span><span></span><span></span>";
    els.chat.appendChild(node);
    els.chat.scrollTop = els.chat.scrollHeight;
    return node;
  }

  function resolveLanguageCode() {
    const selectedFromDom = String(els.globalLanguageSelect?.value || "").trim().toLowerCase();
    if (selectedFromDom) {
      return selectedFromDom;
    }
    return window.LeafLensI18n?.getLanguageCode?.() || state.languageCode || "en";
  }

  function getUrlContext() {
    const params = new URLSearchParams(window.location.search || "");
    const predictionIdRaw = params.get("prediction_id");
    const parsedPredictionId = predictionIdRaw ? Number(predictionIdRaw) : null;
    return {
      disease: (params.get("disease") || "").trim(),
      crop: (params.get("crop") || "").trim(),
      predictionId: Number.isFinite(parsedPredictionId) ? parsedPredictionId : null,
      languageCode: (params.get("language_code") || "").trim(),
    };
  }

  function getStorageContext() {
    try {
      const raw = window.localStorage.getItem("leaflens.chat.context");
      if (!raw) return {};
      const parsed = JSON.parse(raw);
      return {
        disease: String(parsed?.disease || "").trim(),
        crop: String(parsed?.crop || "").trim(),
        predictionId: typeof parsed?.predictionId === "number" ? parsed.predictionId : null,
      };
    } catch {
      return {};
    }
  }

  function hydrateContext() {
    const fromUrl = getUrlContext();
    const fromStorage = getStorageContext();
    const fromWindow = window.LeafLensPredictionContext || {};

    state.disease = fromUrl.disease || fromStorage.disease || String(fromWindow?.disease || "").trim();
    state.crop = fromUrl.crop || fromStorage.crop || String(fromWindow?.crop || "").trim();
    state.predictionId =
      fromUrl.predictionId ??
      fromStorage.predictionId ??
      (typeof fromWindow?.predictionId === "number" ? fromWindow.predictionId : null);

    state.languageCode = fromUrl.languageCode || resolveLanguageCode() || "en";
  }

  function renderContext() {
    if (!els.contextText) return;

    if (!state.disease) {
      els.contextText.textContent = t(
        "chatbot_waiting_prediction",
        "Run a prediction first. Then open this chatbot from the result card.",
      );
      return;
    }

    const cropPart = state.crop
      ? `${t("crop", "Crop")}: ${state.crop} • `
      : "";
    const diseasePart = `${t("predicted_label", "Predicted label")}: ${state.disease}`;
    els.contextText.textContent = `${cropPart}${diseasePart}`;
  }

  function renderIntroIfNeeded() {
    if (!state.disease || state.introRendered) return;
    const intro = `${t("chatbot_intro_prefix", "Hi! I’m your LeafLens assistant. Let’s discuss")} ${state.disease}.`;
    appendMessage("ai", intro);
    state.introRendered = true;
  }

  async function sendMessage() {
    setAlert("");

    const question = String(els.input?.value || "").trim();
    if (!question) {
      setAlert(t("chatbot_empty_question", "Please type a question for the chatbot."));
      return;
    }

    if (!state.disease) {
      setAlert(t("chatbot_no_disease", "No predicted disease context found. Please run prediction first."));
      return;
    }

    appendMessage("user", question);
    if (els.input) els.input.value = "";

    setSending(true);
    const typingNode = addTypingIndicator();

    try {
      const response = await fetch("/chatbot/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          disease: state.disease,
          crop: state.crop || null,
          language_code: resolveLanguageCode(),
        }),
      });

      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        const msg =
          (typeof data?.detail === "string" && data.detail) ||
          t("chatbot_error_response", "Could not get response. Please try again.");
        throw new Error(msg);
      }

      const answer = String(data?.response || "").trim();
      appendMessage("ai", answer || t("chatbot_empty_response", "I could not generate a response right now."));
    } catch (error) {
      appendMessage(
        "ai",
        `${t("chatbot_error_prefix", "Sorry, I could not process that.")} ${error?.message || ""}`.trim(),
      );
    } finally {
      if (typingNode && typingNode.parentElement) {
        typingNode.parentElement.removeChild(typingNode);
      }
      setSending(false);
    }
  }

  function wireEvents() {
    if (els.sendBtn) {
      els.sendBtn.addEventListener("click", sendMessage);
    }

    if (els.input) {
      els.input.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
          event.preventDefault();
          sendMessage();
        }
      });
    }

    window.addEventListener("leaflens:prediction-ready", (event) => {
      const disease = String(event?.detail?.disease || "").trim();
      const crop = String(event?.detail?.crop || "").trim();

      if (disease) {
        state.disease = disease;
      }
      if (crop) {
        state.crop = crop;
      }
      if (typeof event?.detail?.predictionId === "number") {
        state.predictionId = event.detail.predictionId;
      }

      renderContext();
      renderIntroIfNeeded();
    });

    window.addEventListener("leaflens:language-changed", (event) => {
      state.languageCode = event?.detail?.languageCode || "en";
      window.LeafLensI18n?.applyTranslations?.();
      renderContext();
    });
  }

  async function boot() {
    if (els.year) els.year.textContent = String(new Date().getFullYear());
    window.LeafLensI18n?.applyTranslations?.();

    if (window.LeafLensI18n?.ready) {
      try {
        await window.LeafLensI18n.ready;
      } catch {
        // ignore i18n initialization errors
      }
    }

    hydrateContext();
    renderContext();
    renderIntroIfNeeded();

    if (!state.disease) {
      setAlert(t("chatbot_no_disease", "No predicted disease context found. Please run prediction first."));
    }

    wireEvents();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
