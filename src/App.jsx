import { useEffect, useRef, useState } from 'react';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

// ─── MediaPipe CDN ────────────────────────────────────────────────────────────
const WASM_CDN  = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.34/wasm';
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

// ─── Eye landmark indices (MediaPipe 478-point mesh) ─────────────────────────
// Layout: [outer, upper-outer, upper-inner, inner, lower-inner, lower-outer]
const LEFT_EYE  = [33,  160, 158, 133, 153, 144];
const RIGHT_EYE = [362, 385, 387, 263, 373, 380];

const EAR_THRESHOLD    = 0.20;  // below this → eyes closed
const CLOSED_DEBOUNCE  = 500;   // ms both eyes must stay closed before playback

// ─── Helpers ─────────────────────────────────────────────────────────────────
function dist(a, b) {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

function computeEAR(lm, idx) {
  const [i1, i2, i3, i4, i5, i6] = idx;
  const v1 = dist(lm[i2], lm[i6]);
  const v2 = dist(lm[i3], lm[i5]);
  const h  = dist(lm[i1], lm[i4]);
  return h === 0 ? 1 : (v1 + v2) / (2 * h);
}

// ─── Component ───────────────────────────────────────────────────────────────
export default function App() {
  // ── Refs ──────────────────────────────────────────────────────────────────
  const webcamRef          = useRef(null);
  const hiddenVideoRef     = useRef(null); // audio-routed, hidden from UI
  const faceLandmarkerRef  = useRef(null);
  const eyesClosedSinceRef = useRef(null);
  const committedStateRef  = useRef('none');
  const audioInitRef       = useRef(false); // Web Audio graph created once per element

  // Web Audio API
  const audioCtxRef  = useRef(null);
  const gainNodeRef  = useRef(null);
  const analyserRef  = useRef(null);
  const dataArrayRef = useRef(null);

  // DOM refs for imperatively updating meter (no re-render per frame)
  const meterFillRef = useRef(null);

  // ── State ─────────────────────────────────────────────────────────────────
  const [eyeStatus,   setEyeStatus]   = useState('Initializing model…');
  const [modelReady,  setModelReady]  = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [fileName,    setFileName]    = useState('');
  const [gainValue,   setGainValue]   = useState(1.0);

  // Ref mirror of gainValue so handlers always read current value
  const gainValueRef = useRef(1.0);

  // ── 1. Load MediaPipe Face Landmarker ─────────────────────────────────────
  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(WASM_CDN);
        const fl     = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
          outputFaceBlendshapes: false,
          runningMode: 'VIDEO',
          numFaces: 1,
        });
        if (!cancelled) {
          faceLandmarkerRef.current = fl;
          setModelReady(true);
        }
      } catch (err) {
        console.error('[SightShift] Model load failed:', err);
        if (!cancelled) setEyeStatus('⚠ Model failed to load');
      }
    })();

    return () => { cancelled = true; };
  }, []);

  // ── 2. Webcam (after model ready) ─────────────────────────────────────────
  useEffect(() => {
    if (!modelReady) return;

    let stream = null;

    (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'user', width: 640, height: 480 },
          audio: false,
        });
        const vid = webcamRef.current;
        if (vid) {
          vid.srcObject = stream;
          vid.onloadedmetadata = () => {
            vid.play();
            setCameraReady(true);
            setEyeStatus('Eyes Open');
          };
        }
      } catch (err) {
        console.error('[SightShift] Camera denied:', err);
        setEyeStatus('⚠ Camera access denied');
      }
    })();

    return () => {
      setCameraReady(false);
      if (stream) stream.getTracks().forEach(t => t.stop());
    };
  }, [modelReady]);

  // ── 3. Detection + volume meter loop (Web Worker timer) ──────────────────
  //
  // requestAnimationFrame is throttled to ≤1 fps by browsers when the tab is
  // hidden or the window is minimized. A Web Worker's setInterval is NOT
  // throttled — it keeps firing at full rate regardless of visibility. The
  // worker just sends a tick message; all DOM/ML work stays on the main thread.
  useEffect(() => {
    if (!modelReady || !cameraReady) return;

    // Inline worker via Blob URL — no separate file needed, works on Vercel.
    const blob      = new Blob(['setInterval(()=>postMessage(1),33);'], { type: 'application/javascript' });
    const workerUrl = URL.createObjectURL(blob);
    const worker    = new Worker(workerUrl);

    // Keep track of the last MediaPipe timestamp to avoid duplicate frames.
    let lastTimestamp = -1;

    function commitStatus(next) {
      if (committedStateRef.current !== next) {
        committedStateRef.current = next;
        setEyeStatus(next === 'closed' ? 'Eyes Closed' : 'Eyes Open');
      }
    }

    function updateMeter() {
      if (!analyserRef.current || !meterFillRef.current || !dataArrayRef.current) return;
      analyserRef.current.getByteFrequencyData(dataArrayRef.current);
      let sum = 0;
      for (let i = 0; i < dataArrayRef.current.length; i++) sum += dataArrayRef.current[i];
      const avg = sum / dataArrayRef.current.length;
      const pct = Math.min(100, (avg / 255) * 100 * 3);
      meterFillRef.current.style.width = pct + '%';
      const r = pct < 50 ? Math.round((pct / 50) * 255) : 255;
      const g = pct < 50 ? 210 : Math.round(210 * (1 - (pct - 50) / 50));
      meterFillRef.current.style.backgroundColor = `rgb(${r},${g},40)`;
    }

    worker.onmessage = function () {
      updateMeter();

      const webcam = webcamRef.current;
      const fl     = faceLandmarkerRef.current;
      const player = hiddenVideoRef.current;

      if (!webcam || webcam.readyState < 2 || !fl) return;

      const nowMs = performance.now();
      // MediaPipe requires strictly increasing timestamps; skip if same frame.
      if (nowMs <= lastTimestamp) return;
      lastTimestamp = nowMs;

      const results = fl.detectForVideo(webcam, nowMs);

      if (results.faceLandmarks?.length > 0) {
        const lm       = results.faceLandmarks[0];
        const leftEAR  = computeEAR(lm, LEFT_EYE);
        const rightEAR = computeEAR(lm, RIGHT_EYE);
        const avgEAR   = (leftEAR + rightEAR) / 2;
        const closed   = avgEAR < EAR_THRESHOLD;

        console.log(
          `[SightShift] EAR L=${leftEAR.toFixed(3)} R=${rightEAR.toFixed(3)} avg=${avgEAR.toFixed(3)} → ${closed ? 'CLOSED' : 'OPEN'}`
        );

        if (closed) {
          if (eyesClosedSinceRef.current === null) eyesClosedSinceRef.current = performance.now();
          if (performance.now() - eyesClosedSinceRef.current >= CLOSED_DEBOUNCE) {
            commitStatus('closed');
            if (player && player.src) {
              // AudioContext may auto-suspend when page is hidden; resume it.
              if (audioCtxRef.current?.state === 'suspended') audioCtxRef.current.resume();
              player.play().catch(e => console.warn('[SightShift] play():', e));
            }
          }
        } else {
          eyesClosedSinceRef.current = null;
          commitStatus('open');
          if (player) player.pause();
        }
      } else {
        eyesClosedSinceRef.current = null;
        if (committedStateRef.current !== 'no-face') {
          committedStateRef.current = 'no-face';
          setEyeStatus('No face detected');
        }
      }
    };

    return () => {
      worker.terminate();
      URL.revokeObjectURL(workerUrl);
    };
  }, [modelReady, cameraReady]);

  // ── Handlers ──────────────────────────────────────────────────────────────
  function handleFileChange(e) {
    const file = e.target.files[0];
    if (!file) return;

    const url = URL.createObjectURL(file);
    setFileName(file.name);

    const vid = hiddenVideoRef.current;
    if (!vid) return;

    // Update src directly – avoids waiting for React re-render
    vid.src = url;

    // Wire up the Web Audio graph exactly once per hidden video element.
    // createMediaElementSource() may only be called once per element.
    if (!audioInitRef.current) {
      const ctx      = new AudioContext();
      const source   = ctx.createMediaElementSource(vid);
      const gain     = ctx.createGain();
      gain.gain.value = gainValueRef.current;
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.75;

      source.connect(gain);
      gain.connect(analyser);
      analyser.connect(ctx.destination);

      audioCtxRef.current  = ctx;
      gainNodeRef.current  = gain;
      analyserRef.current  = analyser;
      dataArrayRef.current = new Uint8Array(analyser.frequencyBinCount);
      audioInitRef.current = true;
    }

    // Resume if browser auto-suspended the context
    if (audioCtxRef.current?.state === 'suspended') audioCtxRef.current.resume();
  }

  function handleGainChange(e) {
    const val = parseFloat(e.target.value);
    setGainValue(val);
    gainValueRef.current = val;
    if (gainNodeRef.current) gainNodeRef.current.gain.value = val;
    // Resume on slider touch (user gesture)
    if (audioCtxRef.current?.state === 'suspended') audioCtxRef.current.resume();
  }

  // ── Status colour ─────────────────────────────────────────────────────────
  const badgeColor =
    eyeStatus === 'Eyes Open'        ? '#4ade80' :
    eyeStatus === 'Eyes Closed'      ? '#f87171' :
    eyeStatus === 'No face detected' ? '#fb923c' :
                                       '#facc15';

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div style={s.page}>
      <header style={s.header}>
        <div style={s.logoRow}>
          {/* Eye logomark */}
          <svg width="38" height="38" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
            <defs>
              <linearGradient id="lg" x1="0" y1="0" x2="32" y2="32" gradientUnits="userSpaceOnUse">
                <stop stopColor="#a78bfa"/>
                <stop offset="1" stopColor="#38bdf8"/>
              </linearGradient>
            </defs>
            <rect width="32" height="32" rx="9" fill="#12131a"/>
            <path d="M4 16 Q16 7 28 16 Q16 25 4 16Z" fill="url(#lg)" opacity="0.18"/>
            <path d="M4 16 Q16 7 28 16 Q16 25 4 16Z" fill="none" stroke="url(#lg)" strokeWidth="1.6"/>
            <circle cx="16" cy="16" r="4.5" fill="url(#lg)" opacity="0.85"/>
            <circle cx="16" cy="16" r="2" fill="#090a0f"/>
            <circle cx="17.4" cy="14.6" r="0.9" fill="white" opacity="0.45"/>
          </svg>
          <h1 style={s.title}>Sight Shift</h1>
        </div>
        <p style={s.subtitle}>Close your eyes to play · Open to pause</p>
      </header>

      {/* Status badge */}
      <div style={{ ...s.badge, background: badgeColor }}>{eyeStatus}</div>

      {/* Loading notice */}
      {!modelReady && <p style={s.notice}>Loading face detection model…</p>}

      {/* ── Webcam ── */}
      <section style={{ ...s.card, maxWidth: '700px' }}>
        <span style={s.label}>Live Webcam</span>
        <video ref={webcamRef} autoPlay muted playsInline style={s.webcam} />
      </section>

      {/* ── Volume ── */}
      <section style={s.card}>
        <div style={s.volRow}>
          <span style={s.label}>Volume</span>
          <span style={s.volPct}>{Math.round(gainValue * 100)}%</span>
        </div>
        <input
          type="range"
          min="0"
          max="10"
          step="0.05"
          value={gainValue}
          onChange={handleGainChange}
          style={s.slider}
        />
        <div style={s.meterTrack}>
          <div ref={meterFillRef} style={s.meterFill} />
        </div>
        {gainValue > 3 && (
          <p style={s.warnText}>⚠ High gain — protect your hearing</p>
        )}
      </section>

      {/* Hidden audio/video — always in DOM so ref is stable */}
      <video ref={hiddenVideoRef} loop playsInline style={{ display: 'none' }} />

      {/* ── Upload (bottom) ── */}
      <section style={s.card}>
        <span style={s.label}>Audio / Video File</span>
        <label style={s.fileBtn}>
          {fileName ? `✓  ${fileName}` : 'Choose file'}
          <input
            type="file"
            accept="video/*,audio/*"
            onChange={handleFileChange}
            style={s.fileInput}
          />
        </label>
        {!fileName && <p style={s.hint}>MP4 · WebM · MP3 · AAC and more</p>}
      </section>
    </div>
  );
}

// ─── Styles ──────────────────────────────────────────────────────────────────
const s = {
  page: {
    minHeight: '100vh',
    background: '#090a0f',
    color: '#e2e2e6',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    padding: '2.5rem 1.25rem 5rem',
    fontFamily: "system-ui, 'Segoe UI', Roboto, sans-serif",
  },

  header: {
    textAlign: 'center',
    marginBottom: '1.5rem',
  },

  logoRow: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '0.6rem',
    marginBottom: '0.25rem',
  },

  title: {
    fontSize: '2.8rem',
    fontWeight: 800,
    letterSpacing: '-0.05em',
    // padding gives the gradient background room so ascenders/descenders aren't clipped
    padding: '0.06em 0.12em',
    margin: 0,
    display: 'inline-block',
    background: 'linear-gradient(135deg, #a78bfa 0%, #38bdf8 100%)',
    WebkitBackgroundClip: 'text',
    backgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    color: 'transparent',
  },

  subtitle: {
    fontSize: '0.9rem',
    color: '#555',
    margin: '0.4rem 0 0',
  },

  badge: {
    padding: '0.45rem 1.8rem',
    borderRadius: '999px',
    fontWeight: 700,
    fontSize: '1rem',
    color: '#08090f',
    marginBottom: '2rem',
    transition: 'background 0.3s ease',
    letterSpacing: '0.03em',
  },

  notice: {
    color: '#facc15',
    fontSize: '0.85rem',
    marginBottom: '1rem',
    margin: '0 0 1rem',
  },

  card: {
    width: '100%',
    maxWidth: '520px',
    background: '#12131a',
    border: '1px solid #1e1f2e',
    borderRadius: '14px',
    padding: '1.25rem 1.5rem',
    marginBottom: '1.2rem',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '0.9rem',
    boxSizing: 'border-box',
  },

  label: {
    alignSelf: 'flex-start',
    fontSize: '0.72rem',
    fontWeight: 700,
    letterSpacing: '0.12em',
    textTransform: 'uppercase',
    color: '#444',
  },

  webcam: {
    width: '100%',
    aspectRatio: '16 / 9',
    objectFit: 'cover',
    borderRadius: '10px',
    background: '#000',
    display: 'block',
  },

  volRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    width: '100%',
  },

  volPct: {
    fontSize: '1.15rem',
    fontWeight: 700,
    color: '#a78bfa',
    fontVariantNumeric: 'tabular-nums',
  },

  slider: {
    width: '100%',
    accentColor: '#a78bfa',
    cursor: 'pointer',
    height: '4px',
  },

  meterTrack: {
    width: '100%',
    height: '10px',
    background: '#1a1b26',
    borderRadius: '999px',
    overflow: 'hidden',
  },

  meterFill: {
    height: '100%',
    width: '0%',
    borderRadius: '999px',
    backgroundColor: 'rgb(80,200,40)',
    // intentionally no CSS transition — we update every RAF frame directly
  },

  warnText: {
    margin: 0,
    fontSize: '0.78rem',
    color: '#f87171',
    alignSelf: 'flex-start',
  },

  fileBtn: {
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: '#4f46e5',
    color: '#fff',
    padding: '0.65rem 1.8rem',
    borderRadius: '8px',
    cursor: 'pointer',
    fontSize: '0.9rem',
    fontWeight: 600,
    letterSpacing: '0.01em',
    userSelect: 'none',
    maxWidth: '100%',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    position: 'relative',
  },

  fileInput: {
    position: 'absolute',
    opacity: 0,
    width: 0,
    height: 0,
    pointerEvents: 'none',
  },

  hint: {
    margin: 0,
    color: '#2f303f',
    fontSize: '0.8rem',
  },
};
