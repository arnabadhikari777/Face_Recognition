/**
 * Camera helpers: permission, start/stop, front/rear switch.
 */
const Camera = (() => {
  let stream = null;
  let currentFacing = "user"; // user = front, environment = rear

  async function listDevices() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      return devices.filter((d) => d.kind === "videoinput");
    } catch {
      return [];
    }
  }

  async function start(videoEl, facingMode = currentFacing) {
    stop();
    currentFacing = facingMode;
    const constraints = {
      audio: false,
      video: {
        facingMode: { ideal: facingMode },
        width: { ideal: 640 },
        height: { ideal: 480 },
      },
    };
    try {
      stream = await navigator.mediaDevices.getUserMedia(constraints);
      videoEl.srcObject = stream;
      await videoEl.play();
      return { ok: true };
    } catch (err) {
      let message = "Unable to access camera.";
      if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
        message = "Camera permission denied. Please allow camera access and try again.";
      } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
        message = "No camera found on this device.";
      } else if (err.name === "NotReadableError") {
        message = "Camera is already in use by another application.";
      }
      return { ok: false, error: message, name: err.name };
    }
  }

  function stop() {
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
    }
  }

  function isActive() {
    return !!stream && stream.active;
  }

  async function switchCamera(videoEl) {
    const next = currentFacing === "user" ? "environment" : "user";
    return start(videoEl, next);
  }

  function captureFrame(videoEl, canvasEl) {
    const w = videoEl.videoWidth;
    const h = videoEl.videoHeight;
    if (!w || !h) return null;
    canvasEl.width = w;
    canvasEl.height = h;
    const ctx = canvasEl.getContext("2d");
    // Mirror front camera for natural selfie feel in preview; capture as-is
    ctx.drawImage(videoEl, 0, 0, w, h);
    return ctx.getImageData(0, 0, w, h);
  }

  return {
    start,
    stop,
    isActive,
    switchCamera,
    captureFrame,
    listDevices,
    get facing() {
      return currentFacing;
    },
  };
})();

window.Camera = Camera;
