/**
 * Simple, Clean Groq Prompt Shot Checker & Playground
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const apiKeyInput = document.getElementById("api-key-input");
  const toggleKeyBtn = document.getElementById("toggle-key-btn");
  const testKeyBtn = document.getElementById("test-key-btn");
  const keyStatus = document.getElementById("key-status");
  const apiKeyBox = document.getElementById("api-key-box");

  const promptInput = document.getElementById("prompt-input");
  const modelSelect = document.getElementById("model-select");
  const liveShotTag = document.getElementById("live-shot-tag");
  const submitBtn = document.getElementById("submit-btn");
  const clearBtn = document.getElementById("clear-btn");
  const presetPills = document.querySelectorAll(".preset-pill");

  const resultSection = document.getElementById("result-section");
  const shotBanner = document.getElementById("shot-banner");
  const shotBadgeLarge = document.getElementById("shot-badge-large");
  const shotExplanation = document.getElementById("shot-explanation");
  const examplesBlock = document.getElementById("examples-block");
  const examplesList = document.getElementById("examples-list");
  const responseText = document.getElementById("response-text");
  const responseStats = document.getElementById("response-stats");

  // Sample Presets
  const SAMPLES = {
    zero: "Explain the theory of relativity in simple terms for a high school student.",
    one: "Translate English to French:\n\nEnglish: Good morning, have a great day!\nFrench: Bonjour, passe une excellente journée !\n\nEnglish: Where is the nearest train station?\nFrench:",
    few: "Classify customer sentiment as Positive, Negative, or Neutral:\n\nReview: The shipping was fast and the quality was top notch.\nSentiment: Positive\n\nReview: It broke on the second day and support didn't reply.\nSentiment: Negative\n\nReview: It's okay, does what it says.\nSentiment: Neutral\n\nReview: Absolutely in love with this product, highly recommended!\nSentiment:"
  };

  // State
  let userApiKey = localStorage.getItem("groq_api_key") || "";
  let debounceTimer = null;

  init();

  function init() {
    // Load saved key from localStorage
    if (userApiKey) {
      apiKeyInput.value = userApiKey;
      setKeyStatus("Key saved locally", "success");
    }

    bindEvents();

    // Run initial heuristic check on whatever is in promptInput
    if (!promptInput.value.trim()) {
      promptInput.value = SAMPLES.few;
    }
    detectShotLive();
  }

  function bindEvents() {
    // API Key input
    apiKeyInput.addEventListener("input", () => {
      const key = apiKeyInput.value.trim();
      userApiKey = key;
      localStorage.setItem("groq_api_key", key);
      if (key) {
        setKeyStatus("Key saved", "success");
      } else {
        setKeyStatus("No key saved", "");
      }
    });

    // Show / Hide Key
    toggleKeyBtn.addEventListener("click", () => {
      if (apiKeyInput.type === "password") {
        apiKeyInput.type = "text";
        toggleKeyBtn.textContent = "🙈";
      } else {
        apiKeyInput.type = "password";
        toggleKeyBtn.textContent = "👁️";
      }
    });

    // Test API Key
    testKeyBtn.addEventListener("click", testGroqKey);

    // Live prompt shot detection as user types
    promptInput.addEventListener("input", () => {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(detectShotLive, 200);
    });

    // Presets
    presetPills.forEach(pill => {
      pill.addEventListener("click", () => {
        const type = pill.dataset.type;
        if (SAMPLES[type]) {
          promptInput.value = SAMPLES[type];
          detectShotLive();
        }
      });
    });

    // Clear
    clearBtn.addEventListener("click", () => {
      promptInput.value = "";
      detectShotLive();
      resultSection.classList.add("hidden");
    });

    // Submit (Click or Cmd/Ctrl+Enter)
    submitBtn.addEventListener("click", handleSubmit);
    promptInput.addEventListener("keydown", (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        e.preventDefault();
        handleSubmit();
      }
    });
  }

  function setKeyStatus(msg, className) {
    keyStatus.textContent = msg;
    keyStatus.className = "key-status " + (className || "");
  }

  function validateKey(key) {
    if (!key) return "Please paste your Groq API key first.";
    if (!key.startsWith("gsk_")) return "Invalid format — Groq keys start with 'gsk_'. Check you copied the full key.";
    if (key.length < 20) return "Key looks too short. Make sure you copied the full key.";
    return null; // valid
  }

  async function testGroqKey() {
    const key = apiKeyInput.value.trim();
    const validationError = validateKey(key);
    if (validationError) {
      highlightApiKeyBox(validationError);
      return;
    }

    testKeyBtn.textContent = "Testing...";
    testKeyBtn.disabled = true;

    try {
      const res = await fetch("/api/test-key", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ api_key: key })
      });
      const data = await res.json();
      if (data.success) {
        setKeyStatus(`✓ Connected! (${data.latency_ms}ms round-trip)`, "success");
        if (data.available_models && data.available_models.length > 0) {
          updateModelDropdown(data.available_models);
        }
      } else {
        setKeyStatus(`✕ ${data.error || "Connection failed"}`, "error");
      }
    } catch (e) {
      setKeyStatus("✕ Could not reach server. Is it running?", "error");
    } finally {
      testKeyBtn.textContent = "Test Key";
      testKeyBtn.disabled = false;
    }
  }

  function highlightApiKeyBox(msg) {
    apiKeyBox.classList.add("highlight");
    apiKeyInput.focus();
    if (msg) setKeyStatus(msg, "error");
    setTimeout(() => {
      apiKeyBox.classList.remove("highlight");
    }, 2000);
  }

  function updateModelDropdown(availableModels) {
    if (!availableModels || !availableModels.length) return;

    // Filter strictly for standard chat text LLMs
    const cleanList = availableModels.filter(m => {
      const lower = m.toLowerCase();
      return (lower.startsWith("llama") || lower.startsWith("mixtral") || lower.startsWith("gemma") || lower.startsWith("deepseek") || lower.startsWith("qwen"))
        && !lower.includes("canopy") && !lower.includes("whisper") && !lower.includes("orpheus") && !lower.includes("tts") && !lower.includes("vision");
    });

    if (!cleanList.length) return;

    modelSelect.innerHTML = "";
    cleanList.forEach(mId => {
      const opt = document.createElement("option");
      opt.value = mId;
      opt.textContent = mId;
      modelSelect.appendChild(opt);
    });

    // Always prefer llama-3.3-70b-versatile as default
    if (cleanList.includes("llama-3.3-70b-versatile")) {
      modelSelect.value = "llama-3.3-70b-versatile";
    } else {
      modelSelect.selectedIndex = 0;
    }
  }

  // Fast client-side prompt shot detection
  async function detectShotLive() {
    const prompt = promptInput.value.trim();
    if (!prompt) {
      updateShotBadge({ classification: "zero-shot", shot_count: 0 });
      return;
    }

    try {
      const res = await fetch("/api/analyze-heuristic", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt })
      });
      const data = await res.json();
      updateShotBadge(data);
      return data;
    } catch (e) {
      console.error("Heuristic detector failed:", e);
    }
  }

  function updateShotBadge(data) {
    const classification = (data.classification || "zero-shot").toLowerCase();
    const count = data.shot_count || 0;

    liveShotTag.className = "shot-tag tag-" + classification.replace("-shot", "");
    liveShotTag.textContent = `${classification.toUpperCase()} (${count} ${count === 1 ? 'Example' : 'Examples'})`;
  }

  // Main Submit Action
  async function handleSubmit() {
    const prompt = promptInput.value.trim();
    if (!prompt) {
      alert("Please enter a prompt first.");
      promptInput.focus();
      return;
    }

    const key = apiKeyInput.value.trim() || userApiKey;
    const keyError = validateKey(key);
    if (keyError) {
      highlightApiKeyBox(keyError);
      return;
    }

    // 1. Detect shot
    const detection = await detectShotLive();

    // 2. Reveal results section with loading state
    resultSection.classList.remove("hidden");
    shotBanner.className = "shot-banner " + (detection ? detection.classification : "zero-shot");
    shotBadgeLarge.textContent = (detection ? detection.classification.toUpperCase() : "ZERO-SHOT") + " PROMPT";
    shotExplanation.textContent = detection ? detection.reasoning : "";

    // Render examples if any
    if (detection && detection.examples && detection.examples.length > 0) {
      examplesBlock.classList.remove("hidden");
      examplesList.innerHTML = "";
      detection.examples.forEach(ex => {
        const item = document.createElement("div");
        item.className = "example-item";
        item.innerHTML = `
          <span class="ex-header">Demonstration #${ex.id || 1}:</span>
          ${ex.input ? `<div class="ex-in"><strong>Input:</strong> ${escapeHtml(ex.input)}</div>` : ''}
          ${ex.output ? `<div class="ex-out"><strong>Output:</strong> ${escapeHtml(ex.output)}</div>` : ''}
          ${!ex.input && !ex.output ? `<div>${escapeHtml(ex.content || '')}</div>` : ''}
        `;
        examplesList.appendChild(item);
      });
    } else {
      examplesBlock.classList.add("hidden");
    }

    // 3. Call Groq
    submitBtn.classList.add("loading");
    submitBtn.disabled = true;
    responseText.textContent = "Generating response via Groq LPU™...";
    responseStats.textContent = "";

    try {
      const res = await fetch("/api/execute", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Groq-API-Key": key
        },
        body: JSON.stringify({
          prompt,
          model: modelSelect.value,
          api_key: key
        })
      });

      const data = await res.json();
      if (!res.ok || data.error) {
        const errMsg = data.error || `HTTP ${res.status}`;
        responseText.innerHTML = `<div style="color:#f87171;font-weight:600">⚠️ Error from Groq:</div><div style="color:#fca5a5;margin-top:6px">${escapeHtml(errMsg)}</div>`;
        // If it's an auth error, highlight the key box
        if (errMsg.toLowerCase().includes("key") || errMsg.includes("401") || errMsg.toLowerCase().includes("auth") || errMsg.toLowerCase().includes("invalid")) {
          highlightApiKeyBox(errMsg);
        }
        return;
      }

      responseText.textContent = data.output;
      if (data.metrics) {
        responseStats.textContent = `${data.metrics.tokens_per_second || 0} tok/s • ${data.metrics.latency_ms || 0} ms • ${data.model || modelSelect.value}`;
      }
    } catch (err) {
      responseText.innerHTML = `<div style="color:#f87171;font-weight:600">⚠️ Error:</div><div style="color:#fca5a5;margin-top:6px">${escapeHtml(err.message)}</div>`;
    } finally {
      submitBtn.classList.remove("loading");
      submitBtn.disabled = false;
    }
  }

  function escapeHtml(str) {
    return (str || '')
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});
