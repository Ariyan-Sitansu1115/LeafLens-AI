/* LeafLens Voice Assistant – modular, post-prediction voice Q&A */

(() => {
  "use strict";

  const SpeechRecognitionCtor = window.SpeechRecognition || window.webkitSpeechRecognition;

  const els = {
    card: document.querySelector("#voice-assistant-card"),
    languageSelect: document.querySelector("#voice-language-select"),
    micBtn: document.querySelector("#voice-mic-btn"),
    resetBtn: document.querySelector("#voice-reset-btn"),
    status: document.querySelector("#voice-status"),
    listening: document.querySelector("#voice-listening"),
    transcript: document.querySelector("#voice-transcript"),
    responseText: document.querySelector("#voice-response-text"),
    responseIndicator: document.querySelector("#voice-response-indicator"),
    spinner: document.querySelector("#voice-processing-spinner"),
    quickTreatment: document.querySelector("#voice-quick-treatment"),
    quickCause: document.querySelector("#voice-quick-cause"),
    quickPrevention: document.querySelector("#voice-quick-prevention"),
  };

  if (!els.card) {
    return;
  }

  const LANG_MAP = {
    en: {
      speech: "en-IN",
      api: "en",
      recognitionCandidates: ["en-IN", "en-US", "en-GB"],
      ttsCandidates: ["en-IN", "en-US", "en-GB"],
    },
    hi: {
      speech: "hi-IN",
      api: "hi",
      recognitionCandidates: ["hi-IN", "hi", "en-IN"],
      ttsCandidates: ["hi-IN", "hi", "en-IN"],
    },
    or: {
      speech: "or-IN",
      api: "or",
      recognitionCandidates: ["or-IN", "or", "hi-IN"],
      ttsCandidates: ["or-IN", "or", "hi-IN", "en-IN"],
    },
    od: {
      speech: "or-IN",
      api: "or",
      recognitionCandidates: ["or-IN", "or", "hi-IN"],
      ttsCandidates: ["or-IN", "or", "hi-IN", "en-IN"],
    },
  };

  // Map dashboard language codes to voice assistant language codes
  const DASHBOARD_TO_VOICE_LANG = {
    en: "en",
    hi: "hi",
    od: "or",    // Odia
    or: "or",    // Odia (alternative code)
    // Languages not directly supported fall back to English
    bn: "en",    // Bengali → English
    gu: "en",    // Gujarati → English
    kn: "en",    // Kannada → English
    ml: "en",    // Malayalam → English
    mr: "en",    // Marathi → English
    ta: "en",    // Tamil → English
    te: "en",    // Telugu → English
  };

  const state = {
    disease: "",
    selectedLanguage: "en",
    transcriptText: "",
    isListening: false,
    isProcessing: false,
    recognition: null,
    dashboardLanguageSync: true, // Whether to sync with dashboard language
    fallbackRecognitionAttempted: false,
  };

  function getDiseaseFromDom() {
    const labelNode = document.querySelector("#predicted-label");
    const resultCard = document.querySelector("#result-card");
    const labelText = String(labelNode?.textContent || "").trim();
    const resultVisible = !!resultCard && !resultCard.classList.contains("hidden");
    if (!resultVisible || !labelText || labelText === "—") {
      return "";
    }
    return labelText;
  }

  function syncDiseaseContext() {
    const eventDisease = String(state.disease || "").trim();
    const contextDisease = String(window.LeafLensPredictionContext?.disease || "").trim();
    const domDisease = getDiseaseFromDom();
    const nextDisease = eventDisease || contextDisease || domDisease;

    if (!nextDisease) {
      return "";
    }

    if (state.disease !== nextDisease) {
      state.disease = nextDisease;
    }

    if (els.card) {
      els.card.classList.remove("hidden");
    }

    return state.disease;
  }

  function setStatus(text) {
    if (els.status) els.status.textContent = text;
  }

  function setResponseIndicator(text, kind) {
    if (!els.responseIndicator) return;
    els.responseIndicator.textContent = text;
    els.responseIndicator.classList.remove("voice-indicator-idle", "voice-indicator-processing", "voice-indicator-ready", "voice-indicator-error");
    if (kind) {
      els.responseIndicator.classList.add(kind);
    }
  }

  function showListening(show) {
    if (!els.listening) return;
    els.listening.classList.toggle("hidden", !show);
  }

  function setProcessing(on) {
    state.isProcessing = !!on;
    if (els.spinner) {
      els.spinner.classList.toggle("hidden", !on);
    }
    if (els.micBtn) {
      els.micBtn.disabled = on;
    }
    if (els.quickTreatment) els.quickTreatment.disabled = on;
    if (els.quickCause) els.quickCause.disabled = on;
    if (els.quickPrevention) els.quickPrevention.disabled = on;
  }

  function resolveLanguageConfig() {
    if (state.selectedLanguage === "od") {
      return LANG_MAP.or;
    }
    return LANG_MAP[state.selectedLanguage] || LANG_MAP.en;
  }

  function chooseBestVoice(candidates) {
    if (!window.speechSynthesis) return null;
    const voices = window.speechSynthesis.getVoices() || [];
    if (!voices.length) return null;

    const normalizedCandidates = (candidates || []).map((item) => String(item || "").toLowerCase());

    for (const candidate of normalizedCandidates) {
      const exact = voices.find((voice) => String(voice.lang || "").toLowerCase() === candidate);
      if (exact) return exact;
    }

    for (const candidate of normalizedCandidates) {
      const prefix = candidate.split("-")[0];
      const partial = voices.find((voice) => String(voice.lang || "").toLowerCase().startsWith(prefix));
      if (partial) return partial;
    }

    return null;
  }

  function resolveRecognitionLang(useFallback = false) {
    const config = resolveLanguageConfig();
    const candidates = Array.isArray(config.recognitionCandidates) && config.recognitionCandidates.length
      ? config.recognitionCandidates
      : [config.speech || "en-IN"];

    if (!useFallback) {
      return candidates[0];
    }

    return candidates[1] || candidates[0] || "en-IN";
  }

  function syncLanguageFromDashboard(dashboardLanguageCode) {
    const normalizedCode = String(dashboardLanguageCode || "en").toLowerCase();
    const voiceLangCode = DASHBOARD_TO_VOICE_LANG[normalizedCode] || "en";
    if (state.selectedLanguage !== voiceLangCode) {
      state.selectedLanguage = voiceLangCode;
      // Update the voice language selector if it exists
      if (els.languageSelect) {
        els.languageSelect.value = voiceLangCode;
      }
    }
  }

  function resetAssistantUI(keepCardVisible = true) {
    state.transcriptText = "";
    if (els.transcript) {
      els.transcript.textContent = "Waiting for your question…";
      els.transcript.classList.add("muted");
    }
    if (els.responseText) {
      els.responseText.textContent = "Response will appear here.";
      els.responseText.classList.add("muted");
    }
    setStatus(state.disease ? "Click mic" : "Run prediction to enable voice assistant");
    setResponseIndicator("Idle", "voice-indicator-idle");
    showListening(false);
    setProcessing(false);
    if (!keepCardVisible && els.card) {
      els.card.classList.add("hidden");
    }
  }

  function speakResponse(text) {
    if (!text || !window.speechSynthesis) {
      return;
    }

    try {
      const config = resolveLanguageConfig();
      const utter = new SpeechSynthesisUtterance(text);
      utter.lang = config.speech;

      const selectedVoice = chooseBestVoice(config.ttsCandidates || [config.speech]);
      if (selectedVoice) {
        utter.voice = selectedVoice;
        utter.lang = selectedVoice.lang || utter.lang;
      } else if (state.selectedLanguage === "or") {
        // Odia voices are often unavailable in browsers; force audible fallback instead of silence.
        utter.lang = "hi-IN";
      }

      utter.rate = 1;
      utter.pitch = 1;
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(utter);
    } catch {
      // Voice output is optional; fail silently without breaking text UX.
    }
  }

  function handleRecognitionError(event) {
    const code = event?.error || "unknown";

    if (code === "language-not-supported") {
      if (
        ["or", "od", "hi"].includes(state.selectedLanguage)
        && state.recognition
        && !state.fallbackRecognitionAttempted
      ) {
        state.fallbackRecognitionAttempted = true;
        const languageLabel = state.selectedLanguage === "hi" ? "Hindi" : "Odia";
        setStatus(`${languageLabel} speech input is limited in this browser. Switching to fallback input...`);
        setResponseIndicator("Fallback", "voice-indicator-processing");
        try {
          state.recognition.lang = resolveRecognitionLang(true);
          window.setTimeout(() => {
            try {
              state.recognition.start();
            } catch {
              setStatus("Speech recognition failed");
              setResponseIndicator("Error", "voice-indicator-error");
            }
          }, 150);
          return;
        } catch {
          setStatus("Speech recognition failed");
          setResponseIndicator("Error", "voice-indicator-error");
          return;
        }
      }

      setStatus("Selected language is not supported for voice input in this browser");
      setResponseIndicator("Unsupported", "voice-indicator-error");
      return;
    }

    if (code === "not-allowed" || code === "service-not-allowed") {
      setStatus("Microphone permission denied");
      setResponseIndicator("Mic Blocked", "voice-indicator-error");
      if (els.responseText) {
        els.responseText.textContent = "Microphone permission was denied. Please allow mic access in Chrome site settings.";
        els.responseText.classList.remove("muted");
      }
      return;
    }

    if (code === "no-speech") {
      setStatus("No speech detected");
      setResponseIndicator("No Speech", "voice-indicator-error");
      return;
    }

    if (code === "audio-capture") {
      setStatus("No microphone detected");
      setResponseIndicator("Mic Error", "voice-indicator-error");
      return;
    }

    setStatus("Speech recognition failed");
    setResponseIndicator("Error", "voice-indicator-error");
  }

  async function sendVoiceQuery(question) {
    const q = (question || "").trim();
    if (!q) {
      setStatus("Empty question. Please speak again.");
      setResponseIndicator("Empty", "voice-indicator-error");
      return;
    }

    const activeDisease = syncDiseaseContext();
    if (!activeDisease) {
      setStatus("Run prediction first");
      setResponseIndicator("No Disease", "voice-indicator-error");
      return;
    }

    setStatus("Processing...");
    setResponseIndicator("Processing", "voice-indicator-processing");
    setProcessing(true);

    try {
      const lang = resolveLanguageConfig();
      const response = await fetch("/voice-query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: q,
          disease: activeDisease,
          language: lang.api,
        }),
      });

      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        const msg = (typeof data?.detail === "string" && data.detail) || `Backend error (${response.status})`;
        throw new Error(msg);
      }

      const answer = typeof data?.response === "string" ? data.response.trim() : "";
      if (!answer) {
        throw new Error("No response generated");
      }

      if (els.responseText) {
        els.responseText.textContent = answer;
        els.responseText.classList.remove("muted");
      }

      setStatus("Answer Ready");
      setResponseIndicator("Ready", "voice-indicator-ready");
      speakResponse(answer);
    } catch (error) {
      if (els.responseText) {
        els.responseText.textContent = `Could not fetch answer: ${error?.message || "Please try again."}`;
        els.responseText.classList.remove("muted");
      }
      setStatus("Request failed");
      setResponseIndicator("Error", "voice-indicator-error");
    } finally {
      setProcessing(false);
    }
  }

  function startListening() {
    if (state.isProcessing) return;

    if (!syncDiseaseContext()) {
      setStatus("Run prediction first");
      setResponseIndicator("No Disease", "voice-indicator-error");
      return;
    }

    if (!state.recognition) {
      setStatus("Speech recognition not supported in this browser");
      setResponseIndicator("Unsupported", "voice-indicator-error");
      return;
    }

    state.fallbackRecognitionAttempted = false;
    state.transcriptText = "";
    if (els.transcript) {
      els.transcript.textContent = "";
      els.transcript.classList.add("muted");
    }

    try {
      state.recognition.lang = resolveRecognitionLang(false);
      state.recognition.start();
    } catch {
      setStatus("Could not start microphone");
      setResponseIndicator("Mic Error", "voice-indicator-error");
    }
  }

  function setupRecognition() {
    if (!SpeechRecognitionCtor) {
      setStatus("Speech recognition not supported in this browser");
      if (els.micBtn) {
        els.micBtn.disabled = true;
      }
      return;
    }

    const recognition = new SpeechRecognitionCtor();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
      state.isListening = true;
      setStatus("Listening...");
      setResponseIndicator("Listening", "voice-indicator-processing");
      showListening(true);
    };

    recognition.onresult = (event) => {
      let fullTranscript = "";
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const part = event.results[i]?.[0]?.transcript || "";
        fullTranscript += part;
      }
      const cleanText = fullTranscript.trim();
      state.transcriptText = cleanText;
      if (els.transcript) {
        els.transcript.textContent = cleanText || "Listening…";
        els.transcript.classList.toggle("muted", !cleanText);
      }
    };

    recognition.onerror = (event) => {
      handleRecognitionError(event);
    };

    recognition.onend = () => {
      state.isListening = false;
      showListening(false);

      const asked = (state.transcriptText || "").trim();
      if (!asked) {
        if (!state.isProcessing) {
          setStatus("Click mic");
          setResponseIndicator("Idle", "voice-indicator-idle");
        }
        return;
      }

      sendVoiceQuery(asked);
    };

    state.recognition = recognition;
  }

  function onQuickAction(question) {
    if (els.transcript) {
      els.transcript.textContent = question;
      els.transcript.classList.remove("muted");
    }
    state.transcriptText = question;
    sendVoiceQuery(question);
  }

  function wireEvents() {
    // Voice language selector is now controlled by the global dashboard language
    // Sync whenever the voice selector changes (manual override)
    if (els.languageSelect) {
      els.languageSelect.addEventListener("change", () => {
        const nextValue = (els.languageSelect.value || "en").toLowerCase();
        state.selectedLanguage = LANG_MAP[nextValue] ? nextValue : "en";
      });
    }

    // Listen to global dashboard language changes and sync voice language automatically
    window.addEventListener("leaflens:language-changed", (event) => {
      const dashboardLangCode = event?.detail?.languageCode || event?.detail?.currentLang || "en";
      syncLanguageFromDashboard(dashboardLangCode);
    });

    if (els.micBtn) {
      els.micBtn.addEventListener("click", startListening);
    }

    if (els.resetBtn) {
      els.resetBtn.addEventListener("click", () => {
        if (window.speechSynthesis) {
          window.speechSynthesis.cancel();
        }
        resetAssistantUI(true);
      });
    }

    if (els.quickTreatment) {
      els.quickTreatment.addEventListener("click", () => {
        onQuickAction("What is the treatment for this disease?");
      });
    }

    if (els.quickCause) {
      els.quickCause.addEventListener("click", () => {
        onQuickAction("What is the main cause of this disease?");
      });
    }

    if (els.quickPrevention) {
      els.quickPrevention.addEventListener("click", () => {
        onQuickAction("How can I prevent this disease in my field?");
      });
    }

    window.addEventListener("leaflens:prediction-ready", (event) => {
      const disease = String(event?.detail?.disease || event?.detail?.label || "").trim();
      if (!disease) return;

      state.disease = disease;
      els.card.classList.remove("hidden");
      resetAssistantUI(true);
    });

    window.addEventListener("leaflens:prediction-cleared", () => {
      state.disease = "";
      resetAssistantUI(false);
    });
  }

  function syncFromPredictionContext() {
    const context = window.LeafLensPredictionContext;
    const disease = String(context?.disease || "").trim();
    if (!disease) {
      return;
    }

    state.disease = disease;
    els.card.classList.remove("hidden");
    resetAssistantUI(true);
  }

  function boot() {
    // Sync voice language with global dashboard language
    // Try to get current dashboard language from i18n system
    const dashboardLang = window.LeafLensI18n?.getLanguageCode?.() || "en";
    syncLanguageFromDashboard(dashboardLang);

    // Update the voice language selector UI to reflect the synced language
    if (els.languageSelect && LANG_MAP[state.selectedLanguage]) {
      els.languageSelect.value = state.selectedLanguage;
    }

    if (window.speechSynthesis) {
      // Warm up voice list for browsers that load voices asynchronously.
      window.speechSynthesis.getVoices();
      window.speechSynthesis.onvoiceschanged = () => {
        window.speechSynthesis.getVoices();
      };
    }

    setupRecognition();
    wireEvents();
    resetAssistantUI(false);
    syncFromPredictionContext();

    // Safety check: if the prediction UI is already visible but event timing was missed,
    // recover disease from the rendered label.
    window.setTimeout(() => {
      if (syncDiseaseContext() && els.card.classList.contains("hidden")) {
        els.card.classList.remove("hidden");
      }
      if (syncDiseaseContext()) {
        resetAssistantUI(true);
      }
    }, 250);

    const observedLabel = document.querySelector("#predicted-label");
    if (observedLabel && window.MutationObserver) {
      const observer = new MutationObserver(() => {
        const previousDisease = state.disease;
        const currentDisease = syncDiseaseContext();
        if (currentDisease && currentDisease !== previousDisease) {
          resetAssistantUI(true);
        }
      });
      observer.observe(observedLabel, { childList: true, characterData: true, subtree: true });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
