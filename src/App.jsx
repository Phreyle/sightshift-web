import { useEffect, useRef, useState } from 'react';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Slider } from '@/components/ui/slider';
import { Button } from '@/components/ui/button';

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

  function handleGainChange(values) {
    const val = values[0];
    setGainValue(val);
    gainValueRef.current = val;
    if (gainNodeRef.current) gainNodeRef.current.gain.value = val;
    // Resume on slider touch (user gesture)
    if (audioCtxRef.current?.state === 'suspended') audioCtxRef.current.resume();
  }

  // ── Status badge colour ───────────────────────────────────────────────────
  const badgeClass =
    eyeStatus === 'Eyes Open'        ? 'bg-green-400 text-gray-950 border-green-400' :
    eyeStatus === 'Eyes Closed'      ? 'bg-red-400 text-gray-950 border-red-400' :
    eyeStatus === 'No face detected' ? 'bg-orange-400 text-gray-950 border-orange-400' :
                                       'bg-yellow-400 text-gray-950 border-yellow-400';

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col items-center px-5 py-10 pb-20">

      {/* ── Header ── */}
      <header className="text-center mb-8">
        <div className="flex items-center justify-center gap-3 mb-1">
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
          <h1 className="text-5xl font-extrabold tracking-tighter bg-gradient-to-br from-violet-400 to-sky-400 bg-clip-text text-transparent py-2 pb-3 leading-none inline-block">
            Sight Shift
          </h1>
        </div>
        <p className="text-sm text-muted-foreground mt-2">Close your eyes to play · Open to pause</p>
      </header>

      {/* ── Status badge ── */}
      <Badge
        variant="outline"
        className={`px-6 py-2 text-base font-bold mb-8 transition-colors duration-300 ${badgeClass}`}
      >
        {eyeStatus}
      </Badge>

      {/* ── Loading notice ── */}
      {!modelReady && (
        <p className="text-yellow-400 text-sm mb-4">Loading face detection model…</p>
      )}

      {/* ── Webcam card ── */}
      <Card className="w-full max-w-2xl mb-5">
        <CardHeader className="pb-3">
          <CardTitle className="text-xs font-bold tracking-widest uppercase text-muted-foreground">
            Live Webcam
          </CardTitle>
        </CardHeader>
        <CardContent className="pt-0">
          <video
            ref={webcamRef}
            autoPlay
            muted
            playsInline
            className="w-full aspect-video object-cover rounded-lg bg-black block"
          />
        </CardContent>
      </Card>

      {/* ── Volume card ── */}
      <Card className="w-full max-w-xl mb-5">
        <CardHeader className="pb-3">
          <div className="flex justify-between items-center w-full">
            <CardTitle className="text-xs font-bold tracking-widest uppercase text-muted-foreground">
              Volume
            </CardTitle>
            <span className="text-lg font-bold text-primary tabular-nums">
              {Math.round(gainValue * 100)}%
            </span>
          </div>
        </CardHeader>
        <CardContent className="pt-0 space-y-3">
          <Slider
            min={0}
            max={10}
            step={0.05}
            value={[gainValue]}
            onValueChange={handleGainChange}
          />
          <div className="w-full h-2.5 bg-muted rounded-full overflow-hidden">
            <div ref={meterFillRef} className="h-full w-0 rounded-full bg-green-400" />
          </div>
          {gainValue > 3 && (
            <p className="text-xs text-red-400">⚠ High gain — protect your hearing</p>
          )}
        </CardContent>
      </Card>

      {/* Hidden audio/video — always in DOM so ref is stable */}
      <video ref={hiddenVideoRef} loop playsInline className="hidden" />

      {/* ── Upload card ── */}
      <Card className="w-full max-w-xl">
        <CardHeader className="pb-3">
          <CardTitle className="text-xs font-bold tracking-widest uppercase text-muted-foreground">
            Audio / Video File
          </CardTitle>
        </CardHeader>
        <CardContent className="pt-0 flex flex-col items-center gap-3">
          <Button variant="default" size="lg" className="w-full relative overflow-hidden" asChild>
            <label className="cursor-pointer truncate">
              {fileName ? `✓  ${fileName}` : 'Choose file'}
              <input
                type="file"
                accept="video/*,audio/*"
                onChange={handleFileChange}
                className="absolute opacity-0 w-0 h-0 pointer-events-none"
              />
            </label>
          </Button>
          {!fileName && (
            <p className="text-xs text-muted-foreground">MP4 · WebM · MP3 · AAC and more</p>
          )}
        </CardContent>
      </Card>

    </div>
  );
}
