/**
 * Recognition workflow: consent → camera → single capture → POST /api/recognize-face
 */
const Recognition = (() => {
  let active = false;

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

  function showPanel() {
    $("mainButtons").classList.add("hidden");
    $("recognizePanel").classList.remove("hidden");
    $("registerPanel").classList.add("hidden");
  }

  function resetUI() {
    active = false;
    Camera.stop();
    $("recognizePanel").classList.add("hidden");
    $("mainButtons").classList.remove("hidden");
    $("consentRecognize").checked = false;
    $("recognizeCaptureBtn").disabled = true;
    $("cameraSection").classList.add("hidden");
    hideStatus();
  }

  async function start() {
    showPanel();
    showStatus("Confirm permission, then open the camera to recognize a face.", "info");
  }

  async function startCamera() {
    if (!$("consentRecognize").checked) {
      showStatus("Please confirm you have permission to capture and check this face.", "error");
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

    active = true;
    $("cameraSection").classList.remove("hidden");
    $("recognizeCaptureBtn").disabled = false;
    showStatus("Position a single face in the frame, then capture.", "info");
  }

  async function captureAndRecognize() {
    if (!active) return;
    $("recognizeCaptureBtn").disabled = true;
    showStatus('<span class="spinner"></span> Recognizing…', "info");

    const video = $("video");
    const canvas = $("captureCanvas");
    const imageData = Camera.captureFrame(video, canvas);
    if (!imageData) {
      showStatus("Could not capture frame. Try again.", "error");
      $("recognizeCaptureBtn").disabled = false;
      return;
    }

    try {
      const result = await FaceEngine.getEmbedding(imageData);
      if (!result.ok) {
        showStatus(result.error, "warning");
        $("recognizeCaptureBtn").disabled = false;
        return;
      }
      if (result.faces === 0) {
        showStatus("No face detected. Please try again.", "warning");
        $("recognizeCaptureBtn").disabled = false;
        return;
      }
      if (result.faces > 1) {
        showStatus("Please make sure only one person is in the camera.", "warning");
        $("recognizeCaptureBtn").disabled = false;
        return;
      }

      const meta = FaceEngine.getMeta();
      const body = {
        embedding: result.embedding,
        consent: true,
        model_name: meta.model_name,
        model_version: meta.model_version,
      };

      const res = await fetch("/api/recognize-face", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();

      if (!res.ok || !data.success) {
        showStatus(escapeHtml(data.error || "Recognition failed."), "error");
        $("recognizeCaptureBtn").disabled = false;
        return;
      }

      if (data.status === "match") {
        showStatus(
          `<strong>Match Found</strong>
           <div class="result-name">${escapeHtml(data.name)}</div>
           <div class="result-score">Similarity Score: ${data.score}</div>
           <p style="margin-top:0.5rem;font-size:0.8rem;color:var(--muted)">
             Similarity is not a probability of identity. Results are not 100% accurate.
           </p>`,
          "success"
        );
      } else if (data.status === "uncertain") {
        showStatus(
          `<strong>Unable to identify reliably. Please try again.</strong>
           ${data.closest_name ? `<div class="result-score">Closest: ${escapeHtml(data.closest_name)} (${data.score})</div>` : ""}`,
          "warning"
        );
      } else {
        showStatus(
          `<strong>Unknown Face</strong><br>No matching registered face found.`,
          "info"
        );
      }
    } catch (e) {
      console.error(e);
      showStatus("Network or processing error. Please try again.", "error");
    }

    $("recognizeCaptureBtn").disabled = false;
  }

  function escapeHtml(s) {
    const d = document.createElement("div");
    d.textContent = s || "";
    return d.innerHTML;
  }

  function cancel() {
    Camera.stop();
    resetUI();
  }

  return { start, startCamera, captureAndRecognize, cancel, resetUI };
})();

window.Recognition = Recognition;
