/* LeafLens global i18n controller */

(() => {
  "use strict";

  const DEFAULT_LANGUAGE = "en";
  const STORAGE_KEY = "lang";

  const fallbackLanguages = {
    en: { display_name: "English", llm_name: "English" },
    hi: { display_name: "हिंदी (Hindi)", llm_name: "Hindi" },
    od: { display_name: "ଓଡ଼ିଆ (Odia)", llm_name: "Odia" },
    ta: { display_name: "தமிழ் (Tamil)", llm_name: "Tamil" },
  };

  const translations = {
    en: {
      nav_predict: "Predict",
      nav_weather: "Weather",
      nav_insight: "LeafLens Insight",
      nav_about: "About",
      nav_docs: "Documentation",
      label_language: "Language",
      brand_name: "LeafLens 🌿",
      weather_title: "Weather",
      weather_subtitle: "Get current weather data for your location. Optionally enter a city name.",
      location_label: "City (optional)",
      weather_city_hint: "Leave blank to use auto-detected location.",
      city_placeholder: "e.g. Delhi, Mumbai",
      btn_get_weather: "Get Weather",
      weather_report: "Weather Report",
      weather_no_data: "No weather data available.",
      weather_fetch_failed: "Failed to fetch weather data.",
      weather_service_unreachable: "Could not reach weather service. Please try again.",
      insight_title: "LeafLens Insight",
      insight_subtitle: "Real-time IoT readings and a 3-day weather forecast.",
      insight_load_failed: "Failed to load data",
      refresh: "Refresh",
      current_field_data: "Current Field Data",
      rain_prediction: "Rain Prediction",
      forecast_title: "Forecast",
      forecast_next_3_days: "Forecast (Next 3 Days)",
      forecast_loading: "Loading forecast…",
      no_data_available: "No data available",
      day: "Day",
      rain_chance: "Chance",
      rain_likely: "Rain Likely",
      soil_moisture: "Soil Moisture",
      stress_index: "Stress Index",
      weather_display: {
        temperature: "Temperature",
        humidity: "Humidity",
        rainfall: "Rainfall",
        wind_speed: "Wind Speed",
        clouds: "Cloud Cover",
        condition: "Condition",
      },
      condition_map: {
        High: "High",
        Moderate: "Moderate",
        Low: "Low",
        Dry: "Dry",
        "Rain Likely": "Rain Likely",
        Humid: "Humid",
        Wet: "Wet",
      },
      rain_chance_high: "High",
      rain_chance_moderate: "Moderate",
      rain_chance_low: "Low",
      loading: "Loading…",
      about_title: "About LeafLens",
      footer_tagline: "Built for fast field decisions.",
    },
    hi: {
      nav_predict: "पूर्वानुमान",
      nav_weather: "मौसम",
      nav_insight: "लीफलेंस इनसाइट",
      nav_about: "परिचय",
      nav_docs: "दस्तावेज़",
      label_language: "भाषा",
      brand_name: "LeafLens 🌿",
      weather_title: "मौसम",
      weather_subtitle: "अपने स्थान का वर्तमान मौसम देखें। चाहें तो शहर का नाम दर्ज करें।",
      location_label: "शहर (वैकल्पिक)",
      weather_city_hint: "खाली छोड़ें ताकि स्थान स्वतः पता चले।",
      city_placeholder: "जैसे: दिल्ली, मुंबई",
      btn_get_weather: "मौसम प्राप्त करें",
      weather_report: "मौसम रिपोर्ट",
      weather_no_data: "कोई मौसम डेटा उपलब्ध नहीं है।",
      weather_fetch_failed: "मौसम डेटा प्राप्त नहीं हुआ।",
      weather_service_unreachable: "मौसम सेवा तक पहुँचना संभव नहीं। कृपया पुनः प्रयास करें।",
      insight_title: "लीफलेंस इनसाइट",
      insight_subtitle: "रीयल-टाइम IoT रीडिंग और नियम-आधारित 3-दिवसीय पूर्वानुमान।",
      insight_load_failed: "डेटा लोड नहीं हुआ",
      refresh: "रीफ्रेश",
      current_field_data: "वर्तमान खेत डेटा",
      rain_prediction: "वर्षा पूर्वानुमान",
      forecast_title: "पूर्वानुमान",
      forecast_next_3_days: "पूर्वानुमान (अगले 3 दिन)",
      forecast_loading: "पूर्वानुमान लोड हो रहा है…",
      no_data_available: "कोई डेटा उपलब्ध नहीं",
      day: "दिन",
      rain_chance: "संभावना",
      rain_likely: "वर्षा संभावित",
      soil_moisture: "मिट्टी की नमी",
      stress_index: "तनाव सूचकांक",
      weather_display: {
        temperature: "तापमान",
        humidity: "आर्द्रता",
        rainfall: "वर्षा",
        wind_speed: "हवा की गति",
        clouds: "बादल आवरण",
        condition: "स्थिति",
      },
      condition_map: {
        High: "उच्च",
        Moderate: "मध्यम",
        Low: "कम",
        Dry: "शुष्क",
        "Rain Likely": "वर्षा संभावित",
        Humid: "आर्द्र",
        Wet: "गीला",
      },
      rain_chance_high: "उच्च",
      rain_chance_moderate: "मध्यम",
      rain_chance_low: "कम",
      loading: "लोड हो रहा है…",
      about_title: "LeafLens के बारे में",
      footer_tagline: "तेज़ खेत निर्णयों के लिए बनाया गया।",
    },
    od: {
      nav_predict: "ପୂର୍ବାନୁମାନ",
      nav_weather: "ଆବହାଓା",
      nav_insight: "ଲିଫଲେନ୍ସ ଇନସାଇଟ୍",
      nav_about: "ବିଷୟରେ",
      nav_docs: "ଡକ୍ୟୁମେଣ୍ଟେସନ୍",
      label_language: "ଭାଷା",
      brand_name: "LeafLens 🌿",
      weather_title: "ଆବହାଓା",
      weather_subtitle: "ଆପଣଙ୍କ ସ୍ଥାନର ବର୍ତ୍ତମାନ ଆବହାଓା ଦେଖନ୍ତୁ। ଇଚ୍ଛା କଲେ ସହର ନାମ ଦିଅନ୍ତୁ।",
      location_label: "ସହର (ଇଚ୍ଛାଧୀନ)",
      weather_city_hint: "ଖାଲି ରଖିଲେ ସ୍ଥାନ ସ୍ୱୟଂଚାଳିତ ଭାବେ ଚିହ୍ନଟ ହେବ।",
      city_placeholder: "ଉଦାହରଣ: ଦିଲ୍ଲୀ, ମୁମ୍ବାଇ",
      btn_get_weather: "ଆବହାଓା ନିଅନ୍ତୁ",
      weather_report: "ଆବହାଓା ରିପୋର୍ଟ",
      weather_no_data: "କୌଣସି ଆବହାଓା ତଥ୍ୟ ମିଳିଲା ନାହିଁ।",
      weather_fetch_failed: "ଆବହାଓା ତଥ୍ୟ ଆଣିବାରେ ବିଫଳ।",
      weather_service_unreachable: "ଆବହାଓା ସେବାକୁ ପହଞ୍ଚି ପାରିଲା ନାହିଁ। ପୁନଃଚେଷ୍ଟା କରନ୍ତୁ।",
      insight_title: "ଲିଫଲେନ୍ସ ଇନସାଇଟ୍",
      insight_subtitle: "ରିଅଲ୍-ଟାଇମ୍ IoT ପଢ଼ା ଏବଂ ନିୟମାଧାରିତ 3 ଦିନର ପୂର୍ବାନୁମାନ।",
      insight_load_failed: "ତଥ୍ୟ ଲୋଡ୍ ହୋଇନି",
      refresh: "ରିଫ୍ରେଶ",
      current_field_data: "ବର୍ତ୍ତମାନ କ୍ଷେତ୍ର ତଥ୍ୟ",
      rain_prediction: "ବର୍ଷା ପୂର୍ବାନୁମାନ",
      forecast_title: "ପୂର୍ବାନୁମାନ",
      forecast_next_3_days: "ପୂର୍ବାନୁମାନ (ଆଗାମୀ 3 ଦିନ)",
      forecast_loading: "ପୂର୍ବାନୁମାନ ଲୋଡ୍ ହେଉଛି…",
      no_data_available: "ତଥ୍ୟ ଉପଲବ୍ଧ ନାହିଁ",
      day: "ଦିନ",
      rain_chance: "ସମ୍ଭାବନା",
      rain_likely: "ବର୍ଷାର ସମ୍ଭାବନା",
      soil_moisture: "ମାଟି ଆର୍ଦ୍ରତା",
      stress_index: "ଚାପ ସୂଚକ",
      weather_display: {
        temperature: "ତାପମାତ୍ରା",
        humidity: "ଆର୍ଦ୍ରତା",
        rainfall: "ବର୍ଷାପାତ",
        wind_speed: "ପବନ ବେଗ",
        clouds: "ମେଘ ଆବରଣ",
        condition: "ଅବସ୍ଥା",
      },
      condition_map: {
        High: "ଉଚ୍ଚ",
        Moderate: "ମଧ୍ୟମ",
        Low: "ନିମ୍ନ",
        Dry: "ଶୁଷ୍କ",
        "Rain Likely": "ବର୍ଷାର ସମ୍ଭାବନା",
        Humid: "ଆର୍ଦ୍ର",
        Wet: "ଭିଜା",
      },
      rain_chance_high: "ଉଚ୍ଚ",
      rain_chance_moderate: "ମଧ୍ୟମ",
      rain_chance_low: "ନିମ୍ନ",
      loading: "ଲୋଡ୍ ହେଉଛି…",
      about_title: "LeafLens ବିଷୟରେ",
      footer_tagline: "ଦ୍ରୁତ କ୍ଷେତ୍ର ସିଦ୍ଧାନ୍ତ ପାଇଁ ନିର୍ମିତ।",
    },
    ta: {
      nav_predict: "கணிப்பு",
      nav_weather: "வானிலை",
      nav_insight: "லீஃப்லென்ஸ் இன்சைட்",
      nav_about: "பற்றி",
      nav_docs: "ஆவணங்கள்",
      label_language: "மொழி",
      brand_name: "LeafLens 🌿",
      weather_title: "வானிலை",
      weather_subtitle: "உங்கள் இடத்தின் தற்போதைய வானிலைத் தகவலைப் பெறுங்கள். விரும்பினால் நகரப் பெயரை உள்ளிடுங்கள்.",
      location_label: "நகரம் (விருப்பம்)",
      weather_city_hint: "காலியாக விட்டால் இடம் தானாக கண்டறியப்படும்.",
      city_placeholder: "எ.கா., டெல்லி, மும்பை",
      btn_get_weather: "வானிலை பெறுக",
      weather_report: "வானிலை அறிக்கை",
      weather_no_data: "வானிலைத் தரவு கிடைக்கவில்லை.",
      weather_fetch_failed: "வானிலைத் தரவை பெற முடியவில்லை.",
      weather_service_unreachable: "வானிலை சேவையை அணுக முடியவில்லை. மீண்டும் முயற்சிக்கவும்.",
      insight_title: "லீஃப்லென்ஸ் இன்சைட்",
      insight_subtitle: "நேரடி IoT வாசிப்புகள் மற்றும் விதி அடிப்படையிலான 3 நாள் முன்னறிவிப்பு.",
      insight_load_failed: "தரவை ஏற்ற முடியவில்லை",
      refresh: "புதுப்பி",
      current_field_data: "தற்போதைய புலத் தரவு",
      rain_prediction: "மழை முன்னறிவிப்பு",
      forecast_title: "முன்னறிவிப்பு",
      forecast_next_3_days: "முன்னறிவிப்பு (அடுத்த 3 நாட்கள்)",
      forecast_loading: "முன்னறிவிப்பு ஏற்றப்படுகிறது…",
      no_data_available: "தரவு இல்லை",
      day: "நாள்",
      rain_chance: "சாத்தியம்",
      rain_likely: "மழை வாய்ப்பு",
      soil_moisture: "மண் ஈரப்பதம்",
      stress_index: "அழுத்த குறியீடு",
      weather_display: {
        temperature: "வெப்பநிலை",
        humidity: "ஈரப்பதம்",
        rainfall: "மழைப்பொழிவு",
        wind_speed: "காற்றின் வேகம்",
        clouds: "மேக மூடல்",
        condition: "நிலை",
      },
      condition_map: {
        High: "அதிகம்",
        Moderate: "மிதமான",
        Low: "குறைவு",
        Dry: "உலர்",
        "Rain Likely": "மழை வாய்ப்பு",
        Humid: "ஈரமான",
        Wet: "நனைந்த",
      },
      rain_chance_high: "அதிகம்",
      rain_chance_moderate: "மிதமான",
      rain_chance_low: "குறைவு",
      loading: "ஏற்றப்படுகிறது…",
      about_title: "LeafLens பற்றி",
      footer_tagline: "விரைவான புல முடிவுகளுக்காக உருவாக்கப்பட்டது.",
    },
  };

  const localeOverrides = {
    en: {
      predict_title: "Predict crop disease",
      predict_subtitle: "Upload a leaf image, choose the crop, and get predictions, confidence, and explanations.",
      crop_hint: "Auto-populated from the model registry if available.",
      upload_image: "Upload image",
      dropzone_title: "Drag & drop your image here",
      dropzone_subtitle: "or browse to upload",
      prediction_result: "Prediction result",
      cached: "Cached",
      predicted_label: "Predicted label",
      prediction_analytics: "Prediction analytics",
      model_version: "Model version",
      inference_time: "Inference time",
      device: "Device",
      image_hash: "Image hash",
      class_probabilities: "Class probabilities",
      gradcam_visualization: "Grad-CAM visualization",
      feedback_question: "Was this prediction correct?",
      feedback_correct: "👍 Correct",
      feedback_incorrect: "👎 Incorrect",
      explanation: "Explanation",
      ai_mode: "AI Mode",
      powered_by_gemini: "Powered by Gemini",
      dashboard_language: "Dashboard Language",
      simple_explanation: "Simple Explanation",
      explain_in_detail: "Explain in Detail (AI)",
      run_prediction_for_explanation: "Run a prediction to unlock explanations.",
      about_description: "LeafLens combines multi-crop disease detection with Grad-CAM visualizations and a hybrid explanation engine: deterministic knowledge base insights plus optional LLM expansion with caching.",
      about_fast_reliable_title: "Fast & reliable",
      about_fast_reliable_text: "Production-grade API with defensive error handling.",
      about_transparent_title: "Transparent",
      about_transparent_text: "Grad-CAM highlights what the model focused on.",
      about_actionable_title: "Actionable",
      about_actionable_text: "Knowledge-base guidance: cause, symptoms, spread, treatment, prevention.",
      file_too_large_title: "File too large",
      file_too_large_body: "Please upload an image under 5MB.",
      no_probability_distribution: "No probability distribution available.",
      no_explanation_available: "No explanation available.",
      loading_crops: "Loading crops…",
      using_default_crop_title: "Using default crop",
      using_default_crop_body: "Model registry crops could not be loaded yet.",
      offline_mode_title: "Offline mode",
      offline_mode_body: "Could not load crop list. Using a default option.",
      model_loaded: "Model Loaded",
      model_unavailable: "Model Unavailable",
      kb_active: "Knowledge Base Active",
      kb_unavailable: "Knowledge Base Unavailable",
      llm_enabled: "LLM Enabled",
      llm_optional: "LLM Optional",
      unsupported_crop_title: "Unsupported crop",
      available_crops: "Available crops",
      prediction_failed_title: "Prediction failed",
      confidence_medium: "Medium",
      prediction_id: "Prediction",
      not_logged: "Not logged",
      choose_explanation_mode: "Choose an explanation mode above.",
      prediction_complete_title: "Prediction complete",
      inference_finished_in: "Inference finished in",
      no_prediction_yet_title: "No prediction yet",
      no_prediction_yet_body: "Run a prediction before submitting feedback.",
      feedback_failed_title: "Feedback failed",
      feedback_recorded_status: "Thanks! Your feedback has been recorded.",
      feedback_recorded_title: "Feedback recorded",
      feedback_recorded_body: "Thank you for helping improve LeafLens.",
      feedback_unexpected_error: "Unexpected error while submitting feedback.",
      feedback_error_title: "Feedback error",
      reset_title: "Reset",
      reset_body: "Ready for a new prediction.",
      image_selected_title: "Image selected",
      image_selected_body: "Ready to run prediction.",
      hide_probability_distribution: "Hide probability distribution",
      hide_gradcam: "Hide Grad-CAM",
    },
    hi: {
      predict_title: "फसल रोग का पूर्वानुमान",
      predict_subtitle: "पत्ती की छवि अपलोड करें, फसल चुनें और पूर्वानुमान, कॉन्फिडेंस व व्याख्या प्राप्त करें।",
      crop_hint: "यदि उपलब्ध हो तो मॉडल रजिस्ट्री से स्वतः भरा जाता है।",
      upload_image: "छवि अपलोड करें",
      dropzone_title: "अपनी छवि यहाँ खींचें और छोड़ें",
      dropzone_subtitle: "या ब्राउज़ करके अपलोड करें",
      prediction_result: "पूर्वानुमान परिणाम",
      cached: "कैश्ड",
      predicted_label: "पूर्वानुमानित लेबल",
      prediction_analytics: "पूर्वानुमान विश्लेषण",
      model_version: "मॉडल संस्करण",
      inference_time: "इन्फरेंस समय",
      device: "डिवाइस",
      image_hash: "इमेज हैश",
      class_probabilities: "क्लास संभावनाएँ",
      gradcam_visualization: "Grad-CAM विज़ुअलाइज़ेशन",
      feedback_question: "क्या यह पूर्वानुमान सही था?",
      feedback_correct: "👍 सही",
      feedback_incorrect: "👎 गलत",
      explanation: "व्याख्या",
      ai_mode: "AI मोड",
      powered_by_gemini: "Gemini द्वारा संचालित",
      dashboard_language: "डैशबोर्ड भाषा",
      simple_explanation: "सरल व्याख्या",
      explain_in_detail: "विस्तृत व्याख्या (AI)",
      run_prediction_for_explanation: "व्याख्या देखने के लिए पहले पूर्वानुमान चलाएँ।",
      about_description: "LeafLens मल्टी-क्रॉप रोग पहचान, Grad-CAM विज़ुअलाइज़ेशन और हाइब्रिड व्याख्या इंजन को जोड़ता है।",
      about_fast_reliable_title: "तेज़ और विश्वसनीय",
      about_fast_reliable_text: "डिफेंसिव एरर हैंडलिंग के साथ प्रोडक्शन-ग्रेड API।",
      about_transparent_title: "पारदर्शी",
      about_transparent_text: "Grad-CAM दिखाता है कि मॉडल ने कहाँ ध्यान दिया।",
      about_actionable_title: "कारगर",
      about_actionable_text: "नॉलेज-बेस मार्गदर्शन: कारण, लक्षण, फैलाव, उपचार, रोकथाम।",
      file_too_large_title: "फ़ाइल बहुत बड़ी है",
      file_too_large_body: "कृपया 5MB से छोटी छवि अपलोड करें।",
      no_probability_distribution: "कोई प्रायिकता वितरण उपलब्ध नहीं है।",
      no_explanation_available: "कोई व्याख्या उपलब्ध नहीं है।",
      loading_crops: "फसलें लोड हो रही हैं…",
      using_default_crop_title: "डिफ़ॉल्ट फसल उपयोग में",
      using_default_crop_body: "मॉडल रजिस्ट्री फसलें अभी लोड नहीं हो सकीं।",
      offline_mode_title: "ऑफ़लाइन मोड",
      offline_mode_body: "फसल सूची लोड नहीं हुई। डिफ़ॉल्ट विकल्प उपयोग किया गया।",
      model_loaded: "मॉडल लोडेड",
      model_unavailable: "मॉडल उपलब्ध नहीं",
      kb_active: "नॉलेज बेस सक्रिय",
      kb_unavailable: "नॉलेज बेस उपलब्ध नहीं",
      llm_enabled: "LLM सक्षम",
      llm_optional: "LLM वैकल्पिक",
      unsupported_crop_title: "असमर्थित फसल",
      available_crops: "उपलब्ध फसलें",
      prediction_failed_title: "पूर्वानुमान विफल",
      confidence_medium: "मध्यम",
      prediction_id: "पूर्वानुमान",
      not_logged: "लॉग नहीं हुआ",
      choose_explanation_mode: "ऊपर से व्याख्या मोड चुनें।",
      prediction_complete_title: "पूर्वानुमान पूर्ण",
      inference_finished_in: "इन्फरेंस पूरा हुआ",
      no_prediction_yet_title: "अभी कोई पूर्वानुमान नहीं",
      no_prediction_yet_body: "फीडबैक देने से पहले पूर्वानुमान चलाएँ।",
      feedback_failed_title: "फीडबैक विफल",
      feedback_recorded_status: "धन्यवाद! आपका फीडबैक दर्ज हो गया है।",
      feedback_recorded_title: "फीडबैक दर्ज हुआ",
      feedback_recorded_body: "LeafLens को बेहतर बनाने में मदद के लिए धन्यवाद।",
      feedback_unexpected_error: "फीडबैक भेजते समय अप्रत्याशित त्रुटि हुई।",
      feedback_error_title: "फीडबैक त्रुटि",
      reset_title: "रीसेट",
      reset_body: "नई भविष्यवाणी के लिए तैयार।",
      image_selected_title: "छवि चुनी गई",
      image_selected_body: "पूर्वानुमान चलाने के लिए तैयार।",
      hide_probability_distribution: "प्रायिकता वितरण छुपाएँ",
      hide_gradcam: "Grad-CAM छुपाएँ",
    },
    od: {
      predict_title: "ଫସଲ ରୋଗ ପୂର୍ବାନୁମାନ",
      predict_subtitle: "ପତ୍ରର ଛବି ଅପଲୋଡ୍ କରନ୍ତୁ, ଫସଲ ବାଛନ୍ତୁ ଏବଂ ପୂର୍ବାନୁମାନ, ବିଶ୍ୱସନୀୟତା ଓ ବ୍ୟାଖ୍ୟା ପାଆନ୍ତୁ।",
      crop_hint: "ଉପଲବ୍ଧ ଥିଲେ ମଡେଲ୍ ରେଜିଷ୍ଟ୍ରିରୁ ସ୍ୱୟଂଚାଳିତ ଭାବେ ଭରାଯାଏ।",
      upload_image: "ଛବି ଅପଲୋଡ୍ କରନ୍ତୁ",
      dropzone_title: "ଆପଣଙ୍କ ଛବିକୁ ଏଠାରେ ଡ୍ରାଗ୍ ଏବଂ ଡ୍ରପ୍ କରନ୍ତୁ",
      dropzone_subtitle: "କିମ୍ବା ବ୍ରାଉଜ୍ କରି ଅପଲୋଡ୍ କରନ୍ତୁ",
      prediction_result: "ପୂର୍ବାନୁମାନ ଫଳାଫଳ",
      cached: "କ୍ୟାଶ୍ଡ",
      predicted_label: "ପୂର୍ବାନୁମାନିତ ଲେବେଲ୍",
      prediction_analytics: "ପୂର୍ବାନୁମାନ ବିଶ୍ଳେଷଣ",
      model_version: "ମଡେଲ୍ ସଂସ୍କରଣ",
      inference_time: "ଇନଫରେନ୍ସ ସମୟ",
      device: "ଡିଭାଇସ୍",
      image_hash: "ଛବି ହ୍ୟାଶ୍",
      class_probabilities: "ଶ୍ରେଣୀ ସମ୍ଭାବନା",
      gradcam_visualization: "Grad-CAM ଭିଜୁଆଲାଇଜେସନ୍",
      feedback_question: "ଏହି ପୂର୍ବାନୁମାନ ଠିକ୍ ଥିଲା କି?",
      feedback_correct: "👍 ଠିକ୍",
      feedback_incorrect: "👎 ଭୁଲ",
      explanation: "ବ୍ୟାଖ୍ୟା",
      ai_mode: "AI ମୋଡ୍",
      powered_by_gemini: "Gemini ଦ୍ୱାରା ସଚଳିତ",
      dashboard_language: "ଡ୍ୟାସବୋର୍ଡ ଭାଷା",
      simple_explanation: "ସରଳ ବ୍ୟାଖ୍ୟା",
      explain_in_detail: "ବିସ୍ତୃତ ବ୍ୟାଖ୍ୟା (AI)",
      run_prediction_for_explanation: "ବ୍ୟାଖ୍ୟା ପାଇଁ ପ୍ରଥମେ ପୂର୍ବାନୁମାନ ଚଳାନ୍ତୁ।",
      about_description: "LeafLens ମଲ୍ଟି-କ୍ରପ୍ ରୋଗ ଚିହ୍ନଟ, Grad-CAM ଭିଜୁଆଲାଇଜେସନ୍ ଏବଂ ହାଇବ୍ରିଡ୍ ବ୍ୟାଖ୍ୟା ଇଞ୍ଜିନ୍କୁ ଯୋଡ଼େ।",
      about_fast_reliable_title: "ଦ୍ରୁତ ଏବଂ ନିର୍ଭରଯୋଗ୍ୟ",
      about_fast_reliable_text: "ରକ୍ଷାତ୍ମକ ଏରର୍ ହ୍ୟାଣ୍ଡଲିଂ ସହିତ ପ୍ରୋଡକ୍ସନ୍-ଗ୍ରେଡ୍ API।",
      about_transparent_title: "ପାରଦର୍ଶୀ",
      about_transparent_text: "ମଡେଲ୍ କେଉଁଠାରେ ଧ୍ୟାନ ଦେଇଛି, Grad-CAM ଦେଖାଏ।",
      about_actionable_title: "କାର୍ଯ୍ୟକାରୀ",
      about_actionable_text: "ଜ୍ଞାନଭଣ୍ଡାର ଗାଇଡ୍: କାରଣ, ଲକ୍ଷଣ, ପ୍ରସାର, ଚିକିତ୍ସା, ପ୍ରତିରୋଧ।",
      file_too_large_title: "ଫାଇଲ୍ ବହୁତ ବଡ଼",
      file_too_large_body: "ଦୟାକରି 5MB ରୁ କମ୍ ଛବି ଅପଲୋଡ୍ କରନ୍ତୁ।",
      no_probability_distribution: "କୌଣସି ସମ୍ଭାବନା ବିତରଣ ଉପଲବ୍ଧ ନାହିଁ।",
      no_explanation_available: "କୌଣସି ବ୍ୟାଖ୍ୟା ଉପଲବ୍ଧ ନାହିଁ।",
      loading_crops: "ଫସଲ ଲୋଡ୍ ହେଉଛି…",
      using_default_crop_title: "ଡିଫଲ୍ଟ ଫସଲ ବ୍ୟବହାର ହେଉଛି",
      using_default_crop_body: "ମଡେଲ୍ ରେଜିଷ୍ଟ୍ରି ଫସଲଗୁଡ଼ିକ ଏବେ ଲୋଡ୍ ହେବାକୁ ବଞ୍ଚିତ।",
      offline_mode_title: "ଅଫଲାଇନ୍ ମୋଡ୍",
      offline_mode_body: "ଫସଲ ତାଲିକା ଲୋଡ୍ ହୋଇନି। ଡିଫଲ୍ଟ ବିକଳ୍ପ ବ୍ୟବହୃତ।",
      model_loaded: "ମଡେଲ୍ ଲୋଡ୍ ହୋଇଛି",
      model_unavailable: "ମଡେଲ୍ ଉପଲବ୍ଧ ନୁହେଁ",
      kb_active: "ଜ୍ଞାନଭଣ୍ଡାର ସକ୍ରିୟ",
      kb_unavailable: "ଜ୍ଞାନଭଣ୍ଡାର ଉପଲବ୍ଧ ନୁହେଁ",
      llm_enabled: "LLM ସକ୍ରିୟ",
      llm_optional: "LLM ବିକଳ୍ପୀୟ",
      unsupported_crop_title: "ଅସମର୍ଥିତ ଫସଲ",
      available_crops: "ଉପଲବ୍ଧ ଫସଲ",
      prediction_failed_title: "ପୂର୍ବାନୁମାନ ବିଫଳ",
      confidence_medium: "ମଧ୍ୟମ",
      prediction_id: "ପୂର୍ବାନୁମାନ",
      not_logged: "ଲଗ୍ ହୋଇନି",
      choose_explanation_mode: "ଉପରୁ ବ୍ୟାଖ୍ୟା ମୋଡ୍ ଚୟନ କରନ୍ତୁ।",
      prediction_complete_title: "ପୂର୍ବାନୁମାନ ସମ୍ପୂର୍ଣ୍ଣ",
      inference_finished_in: "ଇନଫରେନ୍ସ ସମାପ୍ତ",
      no_prediction_yet_title: "ଏପର୍ଯ୍ୟନ୍ତ ପୂର୍ବାନୁମାନ ନାହିଁ",
      no_prediction_yet_body: "ଫିଡବ୍ୟାକ୍ ପୂର୍ବରୁ ପୂର୍ବାନୁମାନ ଚଳାନ୍ତୁ।",
      feedback_failed_title: "ଫିଡବ୍ୟାକ୍ ବିଫଳ",
      feedback_recorded_status: "ଧନ୍ୟବାଦ! ଆପଣଙ୍କ ଫିଡବ୍ୟାକ୍ ରେକର୍ଡ୍ ହେଲା।",
      feedback_recorded_title: "ଫିଡବ୍ୟାକ୍ ରେକର୍ଡ୍ ହେଲା",
      feedback_recorded_body: "LeafLens କୁ ଉନ୍ନତ କରିବାରେ ସହଯୋଗ ପାଇଁ ଧନ୍ୟବାଦ।",
      feedback_unexpected_error: "ଫିଡବ୍ୟାକ୍ ପଠାଇବା ସମୟରେ ଅପେକ୍ଷିତ ତ୍ରୁଟି ହେଲା।",
      feedback_error_title: "ଫିଡବ୍ୟାକ୍ ତ୍ରୁଟି",
      reset_title: "ରିସେଟ୍",
      reset_body: "ନୂଆ ପୂର୍ବାନୁମାନ ପାଇଁ ପ୍ରସ୍ତୁତ।",
      image_selected_title: "ଛବି ବାଛାଯାଇଛି",
      image_selected_body: "ପୂର୍ବାନୁମାନ ଚଳାଇବାକୁ ପ୍ରସ୍ତୁତ।",
      hide_probability_distribution: "ସମ୍ଭାବନା ବିତରଣ ଲୁଚାନ୍ତୁ",
      hide_gradcam: "Grad-CAM ଲୁଚାନ୍ତୁ",
    },
    ta: {
      predict_title: "பயிர் நோய் கணிப்பு",
      predict_subtitle: "இலை படத்தை பதிவேற்றி, பயிரை தேர்வு செய்து கணிப்பு, நம்பகத்தன்மை மற்றும் விளக்கத்தைப் பெறுங்கள்.",
      crop_hint: "இருந்தால் மாடல் பதிவேட்டில் இருந்து தானாக நிரப்பப்படும்.",
      upload_image: "படத்தை பதிவேற்று",
      dropzone_title: "உங்கள் படத்தை இங்கே இழுத்து விடுங்கள்",
      dropzone_subtitle: "அல்லது உலாவி பதிவேற்றுங்கள்",
      prediction_result: "கணிப்பு முடிவு",
      cached: "கேஷ்",
      predicted_label: "கணிக்கப்பட்ட லேபல்",
      prediction_analytics: "கணிப்பு பகுப்பாய்வு",
      model_version: "மாடல் பதிப்பு",
      inference_time: "இன்ஃபரன்ஸ் நேரம்",
      device: "சாதனம்",
      image_hash: "பட ஹாஷ்",
      class_probabilities: "வகுப்பு சாத்தியங்கள்",
      gradcam_visualization: "Grad-CAM காட்சி",
      feedback_question: "இந்த கணிப்பு சரியா?",
      feedback_correct: "👍 சரி",
      feedback_incorrect: "👎 தவறு",
      explanation: "விளக்கம்",
      ai_mode: "AI முறை",
      powered_by_gemini: "Gemini மூலம் இயக்கப்படுகிறது",
      dashboard_language: "டாஷ்போர்டு மொழி",
      simple_explanation: "எளிய விளக்கம்",
      explain_in_detail: "விரிவாக விளக்கு (AI)",
      run_prediction_for_explanation: "விளக்கத்தைப் பெற முதலில் கணிப்பை இயக்குங்கள்.",
      about_description: "LeafLens பல-பயிர் நோய் கண்டறிதல், Grad-CAM காட்சி மற்றும் கலப்பு விளக்க இயந்திரத்தை இணைக்கிறது.",
      about_fast_reliable_title: "வேகமும் நம்பகத்தன்மையும்",
      about_fast_reliable_text: "பாதுகாப்பான பிழை கையாளுதலுடன் தயாரிப்பு தர API.",
      about_transparent_title: "தெளிவு",
      about_transparent_text: "மாடல் எதைக் கவனித்தது என்பதை Grad-CAM காட்டுகிறது.",
      about_actionable_title: "செயல்படக்கூடியது",
      about_actionable_text: "அறிவு அடிப்படை வழிகாட்டுதல்: காரணம், அறிகுறிகள், பரவல், சிகிச்சை, தடுப்பு.",
      file_too_large_title: "கோப்பு மிகவும் பெரியது",
      file_too_large_body: "5MB க்கும் குறைவான படத்தை பதிவேற்றவும்.",
      no_probability_distribution: "சாத்திய விநியோகம் இல்லை.",
      no_explanation_available: "விளக்கம் இல்லை.",
      loading_crops: "பயிர்கள் ஏற்றப்படுகின்றன…",
      using_default_crop_title: "இயல்புநிலை பயிர் பயன்படுத்தப்படுகிறது",
      using_default_crop_body: "மாடல் பதிவேட்டு பயிர்களை இப்போது ஏற்ற முடியவில்லை.",
      offline_mode_title: "ஆஃப்லைன் முறை",
      offline_mode_body: "பயிர் பட்டியலை ஏற்ற முடியவில்லை. இயல்புநிலை விருப்பம் பயன்படுத்தப்பட்டது.",
      model_loaded: "மாடல் ஏற்றப்பட்டது",
      model_unavailable: "மாடல் கிடைக்கவில்லை",
      kb_active: "அறிவு தளம் செயல்பாட்டில்",
      kb_unavailable: "அறிவு தளம் கிடைக்கவில்லை",
      llm_enabled: "LLM இயங்குகிறது",
      llm_optional: "LLM விருப்பமானது",
      unsupported_crop_title: "ஆதரிக்காத பயிர்",
      available_crops: "கிடைக்கும் பயிர்கள்",
      prediction_failed_title: "கணிப்பு தோல்வி",
      confidence_medium: "நடுத்தரம்",
      prediction_id: "கணிப்பு",
      not_logged: "பதியப்படவில்லை",
      choose_explanation_mode: "மேலே உள்ள விளக்கம் முறையைத் தேர்வுசெய்க.",
      prediction_complete_title: "கணிப்பு முடிந்தது",
      inference_finished_in: "இன்ஃபரன்ஸ் முடிந்த நேரம்",
      no_prediction_yet_title: "இன்னும் கணிப்பு இல்லை",
      no_prediction_yet_body: "பின்பற்றியை சமர்ப்பிக்கும் முன் கணிப்பை இயக்குங்கள்.",
      feedback_failed_title: "பின்பற்றி தோல்வி",
      feedback_recorded_status: "நன்றி! உங்கள் பின்பற்றி பதிவு செய்யப்பட்டது.",
      feedback_recorded_title: "பின்பற்றி பதிவு செய்யப்பட்டது",
      feedback_recorded_body: "LeafLens மேம்பட உதவியதற்கு நன்றி.",
      feedback_unexpected_error: "பின்பற்றி சமர்ப்பிக்கும் போது எதிர்பாராத பிழை.",
      feedback_error_title: "பின்பற்றி பிழை",
      reset_title: "மீட்டமை",
      reset_body: "புதிய கணிப்பிற்கு தயார்.",
      image_selected_title: "படம் தேர்ந்தெடுக்கப்பட்டது",
      image_selected_body: "கணிப்பை இயக்க தயாராக உள்ளது.",
      hide_probability_distribution: "சாத்திய விநியோகத்தை மறை",
      hide_gradcam: "Grad-CAM மறை",
    },
  };

  const state = {
    languages: { ...fallbackLanguages },
    translations: {},
    languageCode: DEFAULT_LANGUAGE,
  };

  let currentLang = DEFAULT_LANGUAGE;
  try {
    currentLang = window.localStorage.getItem("lang") || "en";
    if (!currentLang) {
      currentLang = window.localStorage.getItem("leaflens.language") || DEFAULT_LANGUAGE;
    }
  } catch {
    currentLang = DEFAULT_LANGUAGE;
  }
  if (!(currentLang in state.languages)) {
    currentLang = DEFAULT_LANGUAGE;
  }
  window.currentLang = currentLang;

  function getNested(obj, path) {
    if (!obj || !path) return undefined;
    let cursor = obj;
    const parts = String(path).split(".");
    for (const part of parts) {
      if (cursor && typeof cursor === "object" && part in cursor) {
        cursor = cursor[part];
      } else {
        return undefined;
      }
    }
    return cursor;
  }

  function deepMerge(baseObj, overrideObj) {
    const output = { ...(baseObj || {}) };
    for (const [key, value] of Object.entries(overrideObj || {})) {
      if (
        value
        && typeof value === "object"
        && !Array.isArray(value)
        && output[key]
        && typeof output[key] === "object"
        && !Array.isArray(output[key])
      ) {
        output[key] = deepMerge(output[key], value);
      } else {
        output[key] = value;
      }
    }
    return output;
  }

  Object.keys(localeOverrides).forEach((langCode) => {
    translations[langCode] = deepMerge(translations[langCode] || {}, localeOverrides[langCode]);
  });

  function t(key, fallback = "") {
    const found = getNested(state.translations, key);
    return typeof found === "string" && found.trim() ? found : fallback || key;
  }

  function setStoredLang(langCode) {
    try {
      window.localStorage.setItem(STORAGE_KEY, langCode);
      window.localStorage.setItem("leaflens.language", langCode);
    } catch {
      // localStorage not available; ignore.
    }
  }

  async function fetchJson(url) {
    const response = await fetch(url, { method: "GET", cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Request failed: ${response.status}`);
    }
    return response.json();
  }

  async function loadLanguages() {
    try {
      const payload = await fetchJson("/i18n/languages");
      const serverLanguages = payload?.languages;
      if (serverLanguages && typeof serverLanguages === "object") {
        state.languages = { ...state.languages, ...serverLanguages };
      }
    } catch {
      state.languages = { ...fallbackLanguages, ...state.languages };
    }
  }

  async function loadTranslations(languageCode) {
    const requestedLang = languageCode || currentLang || DEFAULT_LANGUAGE;
    let serverTranslations = {};
    let resolvedCode = requestedLang;

    try {
      const params = new URLSearchParams({ language_code: requestedLang });
      const payload = await fetchJson(`/i18n/translations?${params.toString()}`);
      serverTranslations = payload?.translations && typeof payload.translations === "object"
        ? payload.translations
        : {};
      resolvedCode = payload?.language_code || requestedLang;
    } catch {
      resolvedCode = requestedLang;
      serverTranslations = {};
    }

    if (!(resolvedCode in state.languages)) {
      resolvedCode = DEFAULT_LANGUAGE;
    }

    const base = deepMerge(translations.en || {}, serverTranslations || {});
    const localeFallback = translations[resolvedCode] || {};
    state.translations = deepMerge(base, localeFallback);
    state.languageCode = resolvedCode;
    currentLang = resolvedCode;
    window.currentLang = resolvedCode;

    if (document?.documentElement) {
      document.documentElement.setAttribute("lang", state.languageCode || DEFAULT_LANGUAGE);
    }
  }

  function buildLanguageOptions(selectEl) {
    if (!selectEl) return;
    const activeCode = state.languageCode || currentLang || DEFAULT_LANGUAGE;
    const entries = Object.entries(state.languages || {});
    selectEl.innerHTML = "";

    entries.forEach(([code, meta]) => {
      const option = document.createElement("option");
      option.value = code;
      option.textContent = meta?.display_name || code;
      option.selected = code === activeCode;
      selectEl.appendChild(option);
    });
  }

  function syncLanguageSelectors() {
    const selectors = document.querySelectorAll(".global-language-select");
    selectors.forEach((selectEl) => {
      buildLanguageOptions(selectEl);
      selectEl.value = state.languageCode || DEFAULT_LANGUAGE;

      if (selectEl.dataset.bound === "1") return;
      selectEl.dataset.bound = "1";
      selectEl.addEventListener("change", (event) => {
        const target = event?.target;
        const selectedLang = target?.value || DEFAULT_LANGUAGE;
        setLanguage(selectedLang).catch((error) => {
          console.error("Language change failed", error);
        });
      });
    });
  }

  function applyTranslations() {
    const textNodes = document.querySelectorAll("[data-i18n]");
    textNodes.forEach((node) => {
      const key = node.getAttribute("data-i18n");
      const fallback = node.getAttribute("data-i18n-fallback") || node.textContent || "";
      if (!key) return;
      node.textContent = t(key, fallback);
    });

    const placeholderNodes = document.querySelectorAll("[data-i18n-placeholder]");
    placeholderNodes.forEach((node) => {
      const key = node.getAttribute("data-i18n-placeholder");
      const fallback = node.getAttribute("placeholder") || "";
      if (!key) return;
      node.setAttribute("placeholder", t(key, fallback));
    });

    const ariaNodes = document.querySelectorAll("[data-i18n-aria-label]");
    ariaNodes.forEach((node) => {
      const key = node.getAttribute("data-i18n-aria-label");
      const fallback = node.getAttribute("aria-label") || "";
      if (!key) return;
      node.setAttribute("aria-label", t(key, fallback));
    });

    const titleNodes = document.querySelectorAll("[data-i18n-title]");
    titleNodes.forEach((node) => {
      const key = node.getAttribute("data-i18n-title");
      const fallback = node.getAttribute("title") || "";
      if (!key) return;
      node.setAttribute("title", t(key, fallback));
    });
  }

  function emitLanguageChanged() {
    window.dispatchEvent(
      new CustomEvent("leaflens:language-changed", {
        detail: {
          languageCode: state.languageCode,
          currentLang: state.languageCode,
          translations: state.translations,
          languages: state.languages,
        },
      }),
    );
  }

  async function setLanguage(languageCode, options = {}) {
    const { emitEvent = true } = options;
    await loadTranslations(languageCode || DEFAULT_LANGUAGE);
    setStoredLang(state.languageCode);
    syncLanguageSelectors();
    applyTranslations();

    if (emitEvent) {
      emitLanguageChanged();
    }
    return state.languageCode;
  }

  async function init() {
    await loadLanguages();
    const initial = currentLang in state.languages ? currentLang : DEFAULT_LANGUAGE;
    await setLanguage(initial, { emitEvent: true });
  }

  function getLlmLanguageName(languageCode) {
    const code = languageCode || state.languageCode || DEFAULT_LANGUAGE;
    return state.languages?.[code]?.llm_name || "English";
  }

  window.LeafLensI18n = {
    ready: init(),
    t,
    setLanguage,
    applyTranslations,
    getLanguageCode: () => state.languageCode || DEFAULT_LANGUAGE,
    getCurrentLang: () => state.languageCode || DEFAULT_LANGUAGE,
    getTranslations: () => state.translations,
    getLanguages: () => state.languages,
    getLlmLanguageName,
  };

  window.addEventListener("load", () => {
    window.LeafLensI18n?.applyTranslations?.();
  });
})();
