/** Camera access and cleanup. Every failure is mapped to a plain-English message. */
export class CameraError extends Error {
  constructor(code, message) { super(message); this.code = code; }
}

function mapError(e) {
  switch (e && e.name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return new CameraError('denied', 'Camera permission was denied. Allow camera access in your browser settings and try again.');
    case 'NotFoundError':
    case 'DevicesNotFoundError':
      return new CameraError('not_found', 'No camera was found on this device.');
    case 'NotReadableError':
    case 'TrackStartError':
      return new CameraError('in_use', 'The camera is in use by another app or cannot be started.');
    default:
      return new CameraError('unknown', 'The camera could not be started.');
  }
}

export class Camera {
  constructor(video) {
    this.video = video;
    this.stream = null;
    this.facing = 'user';
  }

  get active() { return !!this.stream; }

  async start(facing = this.facing) {
    this.stop();
    if (!window.isSecureContext) {
      throw new CameraError('insecure', 'The camera requires a secure (HTTPS) connection.');
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new CameraError('unsupported', 'This browser does not support camera access.');
    }
    const base = { width: { ideal: 1280 }, height: { ideal: 720 } };
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { ...base, facingMode: { ideal: facing } }, audio: false });
    } catch (e) {
      if (e && (e.name === 'OverconstrainedError' || e.name === 'ConstraintNotSatisfiedError')) {
        try {
          this.stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        } catch (e2) { throw mapError(e2); }
      } else { throw mapError(e); }
    }
    this.facing = facing;
    const reported = this.stream.getVideoTracks()[0].getSettings().facingMode;
    // Webcams usually report no facingMode; treat them as user-facing (mirrored preview).
    this.video.classList.toggle('mirror', !reported || reported === 'user');
    this.video.srcObject = this.stream;
    try { await this.video.play(); } catch (e) { this.stop(); throw mapError(e); }
    await this._waitForFrame();
  }

  _waitForFrame() {
    return new Promise((resolve) => {
      const v = this.video;
      if (v.videoWidth > 0) return resolve();
      v.addEventListener('loadeddata', () => resolve(), { once: true });
    });
  }

  async hasMultipleCameras() {
    try {
      const devs = await navigator.mediaDevices.enumerateDevices();
      return devs.filter((d) => d.kind === 'videoinput').length > 1;
    } catch { return false; }
  }

  async switch() {
    await this.start(this.facing === 'user' ? 'environment' : 'user');
  }

  stop() {
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.video.pause();
    this.video.srcObject = null;
  }
}
