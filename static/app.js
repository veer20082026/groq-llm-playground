/**
 * Groq LLM Playground - Client Application
 * Handles live prompt shot detection, Groq inference, deep diagnostics, and presets.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const promptInput = document.getElementById("prompt-input");
  const systemPromptInput = document.getElementById("system-prompt-input");
  const systemPromptWrapper = document.getElementById("system-prompt-wrapper");
  const toggleSystemBtn = document.getElementById("toggle-system-btn");
  const editorStats = document.getElementById("editor-stats");
  const liveShotBadge = document.getElementById("live-shot-badge");
  
  // Actions
  const analyzeBtn = document.getElementById("analyze-btn");
  const runBtn = document.getElementById("run-btn");
  const clearBtn = document.getElementById("clear-btn");
  const modelSelect = document.getElementById("model-select");
  const presetSelect = document.getElementById("preset-select");
  
  // Transform
  const transformDropdownBtn = document.getElementById("transform-dropdown-btn");
  const transformMenu = document.getElementById("transform-menu");
  const transformTargetBtns = document.querySelectorAll(".transform-target-btn");
  const transformedOutput = document.getElementById("transformed-output");
  const transformExplanation = document.getElementById("transform-explanation");
  const applyTransformedBtn = document.getElementById("apply-transformed-btn");
  const copyTransformedBtn = document.getElementById("copy-transformed-btn");
  
  // Tabs
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  // Diagnostics elements
  const mainShotBadge = document.getElementById("main-shot-badge");
  const confidencePill = document.getElementById("confidence-pill");
  const classificationReasoning = document.getElementById("classification-reasoning");
  const examplesContainer = document.getElementById("examples-container");
  const exampleCounter = document.getElementById("example-counter");
  const targetQueryText = document.getElementById("target-query-text");
  const qualityScoreVal = document.getElementById("quality-score-val");
  const examplesCountVal = document.getElementById("examples-count-val");
  const deepLatencyVal = document.getElementById("deep-latency-val");
  const techTagsList = document.getElementById("tech-tags-list");
  const improvementsList = document.getElementById("improvements-list");
  const deepStatusTag = document.getElementById("deep-status-tag");

  // Output elements
  const responseOutput = document.getElementById("response-output");
  const copyOutputBtn = document.getElementById("copy-output-btn");
  const hudSpeed = document.getElementById("hud-speed");
  const hudLatency = document.getElementById("hud-latency");
  const hudTokens = document.getElementById("hud-tokens");
  const hudModel = document.getElementById("hud-model");

  // Settings & Parameters
  const settingsToggleBtn = document.getElementById("settings-toggle-btn");
  const settingsDrawer = document.getElementById("settings-drawer");
  const closeSettingsBtn = document.getElementById("close-settings-btn");
  const tempSlider = document.getElementById("temp-slider");
  const tempVal = document.getElementById("temp-val");
  const tokensSlider = document.getElementById("tokens-slider");
  const tokensVal = document.getElementById("tokens-val");
  const topPSlider = document.getElementById("top-p-slider");
  const topPVal = document.getElementById("top-p-val");

  // API Key Inline Banner & Modal
  const apiKeyBtn = document.getElementById("api-key-btn");
  const apiModal = document.getElementById("api-modal");
  const closeModalBtn = document.getElementById("close-modal-btn");
  const modalKeyInput = document.getElementById("modal-key-input");
  const toggleKeyVisibility = document.getElementById("toggle-key-visibility");
  const saveKeyBtn = document.getElementById("save-key-btn");
  const testKeyBtn = document.getElementById("test-key-btn");
  const modalTestResult = document.getElementById("modal-test-result");
  const keyIndicator = document.getElementById("key-indicator");
  const keyLabel = document.getElementById("key-label");

  // Inline Banner Elements
  const inlineKeyInput = document.getElementById("inline-key-input");
  const inlineToggleKey = document.getElementById("inline-toggle-key");
  const inlineTestKeyBtn = document.getElementById("inline-test-key-btn");
  const keyInputWrapper = document.getElementById("key-input-wrapper");
  const bannerStatusDot = document.getElementById("banner-status-dot");
  const bannerStatusMsg = document.getElementById("banner-status-msg");

  // State
  let presetsData = [];
  let userApiKey = localStorage.getItem("groq_api_key") || "";
  let debounceTimer = null;
  let currentTargetShot = "few-shot";

  // Initialize
  initApp();

  async function initApp() {
    loadSavedKey();
    bindEvents();
    await checkServerStatus();
    await loadPresets();
    
    // If empty, load default sample
    if (!promptInput.value.trim() && presetsData.length > 0) {
      loadPreset(presetsData[0].id);
    } else {
      runLiveHeuristicCheck();
    }
  }

  function loadSavedKey() {
    if (userApiKey) {
      if (inlineKeyInput) inlineKeyInput.value = userApiKey;
      if (modalKeyInput) modalKeyInput.value = userApiKey;
      updateKeyStatus(true, "Key Ready (Local)");
    } else {
      updateKeyStatus(false, "No key entered");
    }
  }

  function updateKeyStatus(connected, labelText) {
    if (connected) {
      keyIndicator.classList.add("connected");
      keyLabel.textContent = labelText || "Groq Connected";
      if (bannerStatusDot) bannerStatusDot.classList.add("connected");
      if (bannerStatusMsg) {
        bannerStatusMsg.textContent = "API Key Active (Ready)";
        bannerStatusMsg.style.color = "#34d399";
      }
    } else {
      keyIndicator.classList.remove("connected");
      keyLabel.textContent = labelText || "Set Groq Key";
      if (bannerStatusDot) bannerStatusDot.classList.remove("connected");
      if (bannerStatusMsg) {
        bannerStatusMsg.textContent = "No key entered (offline mode)";
        bannerStatusMsg.style.color = "#9ca3af";
      }
    }
  }

  function highlightKeyInput() {
    if (keyInputWrapper) {
      keyInputWrapper.classList.add("highlight-needed");
      if (inlineKeyInput) inlineKeyInput.focus();
      setTimeout(() => {
        keyInputWrapper.classList.remove("highlight-needed");
      }, 2000);
    }
  }

  async function checkServerStatus() {
    try {
      const res = await fetch("/api/status");
      const data = await res.json();
      if (data.has_env_key && !userApiKey) {
        updateKeyStatus(true, `Env Key (${data.masked_env_key})`);
      }
    } catch (e) {
      console.error("Could not fetch server status:", e);
    }
  }

  async function loadPresets() {
    try {
      const res = await fetch("/api/presets");
      const data = await res.json();
      presetsData = data.presets || [];

      presetSelect.innerHTML = '<option value="">-- Load a Sample Prompt --</option>';
      presetsData.forEach(p => {
        const opt = document.createElement("option");
        opt.value = p.id;
        opt.textContent = `[${p.expected_shot.toUpperCase()}] ${p.title}`;
        presetSelect.appendChild(opt);
      });
    } catch (e) {
      console.error("Failed to load presets:", e);
    }
  }

  function loadPreset(presetId) {
    const preset = presetsData.find(p => p.id === presetId);
    if (!preset) return;

    promptInput.value = preset.prompt;
    if (preset.system_prompt) {
      systemPromptInput.value = preset.system_prompt;
      systemPromptWrapper.classList.remove("collapsed");
    } else {
      systemPromptInput.value = "";
    }
    updateStats();
    runLiveHeuristicCheck();
    switchTab("tab-analysis");
  }

  function bindEvents() {
    // Prompt input typing (debounced live detection)
    promptInput.addEventListener("input", () => {
      updateStats();
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(runLiveHeuristicCheck, 250);
    });

    // Preset selection change
    presetSelect.addEventListener("change", (e) => {
      if (e.target.value) {
        loadPreset(e.target.value);
      }
    });

    // Toggle System Prompt
    toggleSystemBtn.addEventListener("click", () => {
      systemPromptWrapper.classList.toggle("collapsed");
    });

    // Analyze Button
    analyzeBtn.addEventListener("click", runFullAnalysis);

    // Run Button
    runBtn.addEventListener("click", executeOnGroq);

    // Clear Button
    clearBtn.addEventListener("click", () => {
      promptInput.value = "";
      systemPromptInput.value = "";
      updateStats();
      runLiveHeuristicCheck();
    });

    // Keyboard shortcuts
    document.addEventListener("keydown", (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        e.preventDefault();
        if (e.shiftKey) {
          runFullAnalysis();
        } else {
          executeOnGroq();
        }
      }
    });

    // Tabs
    tabBtns.forEach(btn => {
      btn.addEventListener("click", () => {
        switchTab(btn.dataset.tab);
      });
    });

    // Transform Dropdown
    transformDropdownBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      transformMenu.classList.toggle("hidden");
    });

    document.addEventListener("click", () => {
      transformMenu.classList.add("hidden");
    });

    transformMenu.querySelectorAll(".dropdown-item").forEach(item => {
      item.addEventListener("click", () => {
        const targetShot = item.dataset.shot;
        switchTab("tab-transform");
        triggerTransform(targetShot);
      });
    });

    // Transform tab target buttons
    transformTargetBtns.forEach(btn => {
      btn.addEventListener("click", () => {
        transformTargetBtns.forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        currentTargetShot = btn.dataset.target;
        triggerTransform(currentTargetShot);
      });
    });

    // Apply & Copy transformed prompt
    applyTransformedBtn.addEventListener("click", () => {
      const text = transformedOutput.dataset.rawPrompt;
      if (text) {
        promptInput.value = text;
        updateStats();
        runLiveHeuristicCheck();
        switchTab("tab-analysis");
      }
    });

    copyTransformedBtn.addEventListener("click", () => {
      const text = transformedOutput.dataset.rawPrompt || transformedOutput.textContent;
      navigator.clipboard.writeText(text);
      copyTransformedBtn.textContent = "Copied!";
      setTimeout(() => copyTransformedBtn.textContent = "Copy", 2000);
    });

    copyOutputBtn.addEventListener("click", () => {
      navigator.clipboard.writeText(responseOutput.textContent);
      copyOutputBtn.textContent = "Copied!";
      setTimeout(() => copyOutputBtn.textContent = "Copy Response", 2000);
    });

    // Sliders
    tempSlider.addEventListener("input", (e) => tempVal.textContent = e.target.value);
    tokensSlider.addEventListener("input", (e) => tokensVal.textContent = e.target.value);
    topPSlider.addEventListener("input", (e) => topPVal.textContent = e.target.value);

    // Settings drawer
    settingsToggleBtn.addEventListener("click", () => settingsDrawer.classList.add("open"));
    closeSettingsBtn.addEventListener("click", () => settingsDrawer.classList.remove("open"));

    // Modal
    apiKeyBtn.addEventListener("click", () => apiModal.classList.remove("hidden"));
    closeModalBtn.addEventListener("click", () => apiModal.classList.add("hidden"));
    toggleKeyVisibility.addEventListener("click", () => {
      if (modalKeyInput.type === "password") {
        modalKeyInput.type = "text";
        toggleKeyVisibility.textContent = "Hide";
      } else {
        modalKeyInput.type = "password";
        toggleKeyVisibility.textContent = "Show";
      }
    });

    saveKeyBtn.addEventListener("click", () => {
      const key = modalKeyInput.value.trim();
      userApiKey = key;
      localStorage.setItem("groq_api_key", key);
      if (inlineKeyInput) inlineKeyInput.value = key;
      updateKeyStatus(Boolean(key), key ? "Key Configured" : "Set Groq Key");
      apiModal.classList.add("hidden");
    });

    // Inline API Key Banner inputs
    if (inlineKeyInput) {
      inlineKeyInput.addEventListener("input", () => {
        const key = inlineKeyInput.value.trim();
        userApiKey = key;
        localStorage.setItem("groq_api_key", key);
        if (modalKeyInput) modalKeyInput.value = key;
        updateKeyStatus(Boolean(key), key ? "Key Active" : "No key entered");
      });
    }

    if (inlineToggleKey) {
      inlineToggleKey.addEventListener("click", () => {
        if (inlineKeyInput.type === "password") {
          inlineKeyInput.type = "text";
          inlineToggleKey.textContent = "🙈";
        } else {
          inlineKeyInput.type = "password";
          inlineToggleKey.textContent = "👁️";
        }
      });
    }

    if (inlineTestKeyBtn) {
      inlineTestKeyBtn.addEventListener("click", async () => {
        const key = (inlineKeyInput ? inlineKeyInput.value.trim() : "") || userApiKey;
        if (!key) {
          highlightKeyInput();
          if (bannerStatusMsg) {
            bannerStatusMsg.textContent = "Please paste an API key first!";
            bannerStatusMsg.style.color = "#f87171";
          }
          return;
        }

        inlineTestKeyBtn.textContent = "Testing...";
        inlineTestKeyBtn.disabled = true;

        try {
          const res = await fetch("/api/test-key", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ api_key: key })
          });
          const data = await res.json();
          if (data.success) {
            updateKeyStatus(true, "Groq Connected");
            if (bannerStatusMsg) {
              bannerStatusMsg.textContent = `✓ Connected (${data.latency_ms}ms)`;
              bannerStatusMsg.style.color = "#34d399";
            }
          } else {
            updateKeyStatus(false, "Invalid Key");
            if (bannerStatusMsg) {
              const errSnippet = data.error ? data.error.substring(0, 45) : "Auth error";
              bannerStatusMsg.textContent = `✕ Failed: ${errSnippet}`;
              bannerStatusMsg.style.color = "#f87171";
            }
          }
        } catch (err) {
          if (bannerStatusMsg) {
            bannerStatusMsg.textContent = `✕ Error: ${err.message}`;
            bannerStatusMsg.style.color = "#f87171";
          }
        } finally {
          inlineTestKeyBtn.textContent = "Test Connection";
          inlineTestKeyBtn.disabled = false;
        }
      });
    }

    testKeyBtn.addEventListener("click", async () => {
      const key = modalKeyInput.value.trim();
      modalTestResult.className = "modal-test-result";
      modalTestResult.textContent = "Testing connection to Groq API...";
      modalTestResult.style.display = "block";

      try {
        const res = await fetch("/api/test-key", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ api_key: key })
        });
        const data = await res.json();
        if (data.success) {
          modalTestResult.className = "modal-test-result success";
          modalTestResult.textContent = `✓ ${data.message} (${data.latency_ms}ms)`;
        } else {
          modalTestResult.className = "modal-test-result error";
          modalTestResult.textContent = `✕ Connection failed: ${data.error}`;
        }
      } catch (err) {
        modalTestResult.className = "modal-test-result error";
        modalTestResult.textContent = `✕ Error: ${err.message}`;
      }
    });
  }

  function switchTab(tabId) {
    tabBtns.forEach(b => b.classList.toggle("active", b.dataset.tab === tabId));
    tabPanes.forEach(p => p.classList.toggle("active", p.id === tabId));
  }

  function updateStats() {
    const text = promptInput.value;
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;
    const chars = text.length;
    editorStats.textContent = `${words} words | ${chars} chars`;
  }

  // Fast Instant Heuristic Check (Runs on typing, 0 ms latency)
  async function runLiveHeuristicCheck() {
    const prompt = promptInput.value.trim();
    if (!prompt) {
      applyClassification({
        classification: "zero-shot",
        shot_count: 0,
        confidence: 1.0,
        reasoning: "Prompt is currently empty.",
        examples: [],
        target_query: ""
      });
      return;
    }

    try {
      const res = await fetch("/api/analyze-heuristic", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt })
      });
      const data = await res.json();
      applyClassification(data);
    } catch (e) {
      console.error("Heuristic check failed:", e);
    }
  }

  function applyClassification(data) {
    const classification = (data.classification || "zero-shot").toLowerCase();
    const count = data.shot_count || 0;
    const confidencePct = Math.round((data.confidence || 0.95) * 100);

    // Update live badge in editor
    liveShotBadge.className = `badge badge-${classification.replace("-shot", "")}`;
    liveShotBadge.textContent = `${classification.toUpperCase()} (${count} ${count === 1 ? 'Example' : 'Examples'})`;

    // Update Diagnostics Card
    mainShotBadge.className = `classification-badge-large badge-${classification.replace("-shot", "")}`;
    mainShotBadge.textContent = classification.toUpperCase();
    confidencePill.textContent = `Confidence: ${confidencePct}%`;
    classificationReasoning.textContent = data.reasoning || "";

    // Update Examples list
    exampleCounter.textContent = count;
    examplesCountVal.textContent = count;

    if (data.examples && data.examples.length > 0) {
      examplesContainer.innerHTML = "";
      data.examples.forEach(ex => {
        const card = document.createElement("div");
        card.className = "example-card";
        card.innerHTML = `
          <div class="example-card-header">Demonstration #${ex.id || 1}</div>
          ${ex.input ? `
            <div class="example-row">
              <span class="example-pill-label">Input / Premise</span>
              <div class="example-row-content">${escapeHtml(ex.input)}</div>
            </div>` : ''}
          ${ex.output ? `
            <div class="example-row">
              <span class="example-pill-label">Solved Output / Label</span>
              <div class="example-row-content">${escapeHtml(ex.output)}</div>
            </div>` : ''}
          ${!ex.input && !ex.output ? `
            <div class="example-row-content">${escapeHtml(ex.content || '')}</div>
          ` : ''}
        `;
        examplesContainer.appendChild(card);
      });
    } else {
      examplesContainer.innerHTML = `
        <div class="empty-state">
          No demonstration examples detected in this prompt. It directly instructs the model without solved examples (Zero-shot).
        </div>
      `;
    }

    // Target query
    targetQueryText.textContent = data.target_query || "(None identified - entire prompt is instruction)";
  }

  // Deep Prompt Analysis with Groq
  async function runFullAnalysis() {
    const prompt = promptInput.value.trim();
    if (!prompt) {
      alert("Please enter a prompt to analyze.");
      return;
    }

    switchTab("tab-analysis");
    analyzeBtn.classList.add("loading");
    analyzeBtn.disabled = true;

    try {
      const res = await fetch("/api/analyze-groq", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Groq-API-Key": userApiKey
        },
        body: JSON.stringify({
          prompt,
          model: modelSelect.value,
          api_key: userApiKey
        })
      });

      const data = await res.json();
      if (data.heuristic) {
        applyClassification(data.heuristic);
      }

      if (data.deep) {
        renderDeepDiagnostics(data.deep);
      } else if (data.groq_error) {
        alert("Deep Groq analysis notice: " + data.groq_error + "\n(Heuristic analysis was completed)");
      }
    } catch (e) {
      alert("Failed to analyze prompt: " + e.message);
    } finally {
      analyzeBtn.classList.remove("loading");
      analyzeBtn.disabled = false;
    }
  }

  function renderDeepDiagnostics(deep) {
    qualityScoreVal.textContent = deep.prompt_quality_score ? `${deep.prompt_quality_score}/100` : "N/A";
    if (deep.meta) {
      deepLatencyVal.textContent = `${deep.meta.latency_ms} ms`;
      deepStatusTag.textContent = `${deep.meta.model_used} • ${deep.meta.tokens_per_second} tok/s`;
    }

    // Techniques
    techTagsList.innerHTML = "";
    const techniques = deep.techniques_detected || ["Zero-shot"];
    techniques.forEach(t => {
      const span = document.createElement("span");
      span.className = "tag";
      span.textContent = t;
      techTagsList.appendChild(span);
    });

    // Improvement suggestions
    improvementsList.innerHTML = "";
    const suggestions = deep.improvement_suggestions || [];
    if (suggestions.length === 0) {
      improvementsList.innerHTML = "<li>No changes needed. Prompt is clear and well-structured.</li>";
    } else {
      suggestions.forEach(s => {
        const li = document.createElement("li");
        li.textContent = s;
        improvementsList.appendChild(li);
      });
    }

    // If deep classification provides more detailed reasoning
    if (deep.explanation) {
      classificationReasoning.textContent = deep.explanation;
    }
  }

  // Execute Prompt on Groq LPU
  async function executeOnGroq() {
    const prompt = promptInput.value.trim();
    if (!prompt) {
      alert("Please enter a prompt to run.");
      return;
    }

    if (!userApiKey) {
      highlightKeyInput();
      alert("Please paste your Groq API Key into the placeholder at the top of the page to run completions.\n\n(Your key is saved only in your local browser and is never stored on any server)");
      return;
    }

    switchTab("tab-output");
    runBtn.classList.add("loading");
    runBtn.disabled = true;
    responseOutput.innerHTML = '<span class="loading">Streaming from Groq high-speed LPU...</span>';

    try {
      const res = await fetch("/api/execute", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Groq-API-Key": userApiKey
        },
        body: JSON.stringify({
          prompt,
          system_prompt: systemPromptInput.value,
          model: modelSelect.value,
          temperature: parseFloat(tempSlider.value),
          max_tokens: parseInt(tokensSlider.value),
          top_p: parseFloat(topPSlider.value),
          api_key: userApiKey
        })
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || "Execution failed");
      }

      responseOutput.textContent = data.output;
      
      const metrics = data.metrics || {};
      hudSpeed.innerHTML = `${metrics.tokens_per_second || 0} <span class="unit">tok/s</span>`;
      hudLatency.innerHTML = `${metrics.latency_ms || 0} <span class="unit">ms</span>`;
      hudTokens.innerHTML = `${metrics.completion_tokens || 0} <span class="unit">tok</span>`;
      hudModel.textContent = data.model || modelSelect.value;
    } catch (err) {
      responseOutput.innerHTML = `<span style="color:#f87171">Error executing on Groq: ${escapeHtml(err.message)}</span>`;
    } finally {
      runBtn.classList.remove("loading");
      runBtn.disabled = false;
    }
  }

  // Transform Prompt (Convert Zero-shot to Few-shot)
  async function triggerTransform(targetShot) {
    const prompt = promptInput.value.trim();
    if (!prompt) {
      alert("Please enter a prompt to transform.");
      return;
    }

    if (!userApiKey) {
      highlightKeyInput();
      alert("Please paste your Groq API Key into the placeholder at the top of the page to use prompt transformation.");
      return;
    }

    transformedOutput.textContent = `Rewriting prompt into ${targetShot} format using Groq...`;
    transformExplanation.textContent = "";

    try {
      const res = await fetch("/api/transform", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Groq-API-Key": userApiKey
        },
        body: JSON.stringify({
          prompt,
          target_shot: targetShot,
          model: modelSelect.value,
          api_key: userApiKey
        })
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || "Transformation failed");
      }

      const transformedText = data.transformed_prompt || "";
      transformedOutput.dataset.rawPrompt = transformedText;
      transformedOutput.textContent = transformedText;
      transformExplanation.textContent = `Changes: ${data.explanation_of_changes || 'Prompt rewritten.'} (Examples added: ${data.added_examples_count || 0})`;
    } catch (err) {
      transformedOutput.textContent = `Error: ${err.message}`;
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
