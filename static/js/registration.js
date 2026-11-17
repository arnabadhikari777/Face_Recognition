/**
 * Registration workflow: consent → camera → 3 angle samples → POST /api/register-face
 */
const Registration = (() => {
  const STEPS = ["front", "left", "right"];
  const STEP_LABELS = {
    front: "Look straight at the camera",
    left: "Turn your head slightly to the left",
    right: "Turn your head slightly to the right",
  };

  let state = {
    active: false,
    stepIndex: 0,
    embeddings: [],
    name: "",
  };

  function $(id) {
    return document.getElementById(id);
  }

  function showStatus(msg, type = "info") {
    const el = $("statusBox");
    if (!el) return;
    el.className = "status-box " + type;
    el.innerHTML = msg;
    el.classList.remove("hidden");
  }

  function hideStatus() {
    const el = $("statusBox");
    if (el) el.classList.add("hidden");
  }

  function updateProgress() {
    const steps = document.querySelectorAll(".progress-steps .step");
    steps.forEach((el, i) => {
      el.classList.remove("done", "active");
      if (i < state.stepIndex) el.classList.add("done");
      else if (i === state.stepIndex) el.classList.add("active");
    });
    const label = $("stepLabel");
    if (label) {
      label.textContent =
        state.stepIndex < STEPS.length
          ? `Step ${state.stepIndex + 1}/3: ${STEP_LABELS[STEPS[state.stepIndex]]}`
          : "Processing…";
    }
  }

  function showPanel() {
    $("mainButtons").classList.add("hidden");
    $("registerPanel").classList.remove("hidden");
    $("recognizePanel").classList.add("hidden");
  }

  function resetUI() {
    state = { active: false, stepIndex: 0, embeddings: [], name: "" };
    Camera.stop();
    $("registerPanel").classList.add("hidden");
    $("mainButtons").classList.remove("hidden");
    $("nameInput").value = "";
    $("consentRegister").checked = false;
    $("captureBtn").disabled = true;
    hideStatus();
    updateProgress();
  }

  async function start() {
    showPanel();
    showStatus("Enter a name and confirm consent, then start the camera.", "info");
    $("nameInput").focus();
  }

  async function startCamera() {
    const name = ($("nameInput").value || "").trim();
    if (!name) {
      showStatus("Please enter a name.", "error");
      return;
    }
    if (!$("consentRegister").checked) {
      showStatus("Please confirm informed consent before continuing.", "error");
      return;
    }

    showStatus('<span class="spinner"></span> Loading face models…', "info");
    const init = await FaceEngine.init();
    if (!init.ok) {
      showStatus(init.error, "error");
      return;
    }

    const video = $("video");
    const result = await Camera.start(video, "user");
    if (!result.ok) {
      showStatus(result.error, "error");
      return;
    }

    state.active = true;
    state.name = name;
    state.stepIndex = 0;
    state.embeddings = [];
    $("cameraSection").classList.remove("hidden");
    $("captureBtn").disabled = false;
    updateProgress();
    showStatus(STEP_LABELS.front, "info");
  }

  async function captureSample() {
    if (!state.active) return;
    $("captureBtn").disabled = true;
    showStatus('<span class="spinner"></span> Capturing…', "info");

    const video = $("video");
    const canvas = $("captureCanvas");
    const imageData = Camera.captureFrame(video, canvas);
    if (!imageData) {
      showStatus("Could not capture frame. Try again.", "error");
      $("captureBtn").disabled = false;
      return;
    }

    try {
      const result = await FaceEngine.getEmbedding(imageData);
      if (!result.ok) {
        showStatus(result.error, "warning");
        $("captureBtn").disabled = false;
        return;
      }
      if (result.faces > 1) {
        showStatus("Please make sure only one person is in the camera.", "warning");
        $("captureBtn").disabled = false;
        return;
      }

      state.embeddings.push(result.embedding);
      state.stepIndex += 1;
      updateProgress();

      if (state.stepIndex >= STEPS.length) {
        await submitRegistration();
      } else {
        showStatus(STEP_LABELS[STEPS[state.stepIndex]], "info");
        $("captureBtn").disabled = false;
      }
    } catch (e) {
      console.error(e);
      showStatus("Processing failed: " + (e.message || "unknown error"), "error");
      $("captureBtn").disabled = false;
    }
  }

  async function submitRegistration() {
    showStatus('<span class="spinner"></span> Registering face…', "info");
    Camera.stop();
    $("cameraSection").classList.add("hidden");

    const meta = FaceEngine.getMeta();
    const body = {
      name: state.name,
      embeddings: state.embeddings,
      consent: true,
      model_name: meta.model_name,
      model_version: meta.model_version,
    };

    try {
      const res = await fetch("/api/register-face", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();

      if (res.status === 201 && data.success) {
        showStatus(
          `<strong>Face registered successfully.</strong><br>Name: ${escapeHtml(data.name)}`,
          "success"
        );
      } else if (res.status === 409 && data.duplicate) {
        showStatus(
          `<strong>Duplicate face</strong><br>${escapeHtml(data.message || "This face is already registered.")}`,
          "warning"
        );
      } else {
        showStatus(escapeHtml(data.error || data.message || "Registration failed."), "error");
      }
    } catch (e) {
      showStatus("Network error. Please check your connection and try again.", "error");
    }

    // Allow restart after a moment
    setTimeout(() => {
      $("doneRegisterBtn").classList.remove("hidden");
    }, 400);
  }

  function escapeHtml(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  function cancel() {
    Camera.stop();
    resetUI();
  }

  return { start, startCamera, captureSample, cancel, resetUI };
})();

window.Registration = Registration;
