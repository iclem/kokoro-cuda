"""Simplified Kokoro TTS API with streaming support."""

import io
import os
import struct
from contextlib import asynccontextmanager
from enum import Enum

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel
from pydub import AudioSegment

SAMPLE_RATE = 24000
MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")
VOICES_DIR = os.getenv("VOICES_DIR", "/app/voices")

model = None
pipelines = {}
VOICE_PREFIX_TO_LANG_CODE = {
    "a": "a",
    "b": "b",
    "e": "e",
    "f": "f",
    "h": "h",
    "i": "i",
    "j": "j",
    "p": "p",
    "z": "z",
}
SUPPORTED_LANG_CODES = set(VOICE_PREFIX_TO_LANG_CODE.values())


class AudioFormat(str, Enum):
    wav = "wav"
    mp3 = "mp3"
    opus = "opus"
    flac = "flac"
    pcm = "pcm"


class TTSRequest(BaseModel):
    input: str
    voice: str = "af_heart"
    lang_code: str | None = None
    speed: float = 1.0
    response_format: AudioFormat = AudioFormat.wav
    bitrate: str = "192k"


def infer_lang_code(voice: str) -> str:
    """Infer Kokoro language code from the voice prefix."""
    prefix = voice.split("_", maxsplit=1)[0].strip().lower()
    if not prefix:
        raise HTTPException(400, "Voice cannot be empty")

    lang_code = VOICE_PREFIX_TO_LANG_CODE.get(prefix[0])
    if not lang_code:
        raise HTTPException(400, f"Unable to infer language from voice '{voice}'")

    return lang_code


def get_pipeline(lang_code: str):
    """Reuse one pipeline per language code."""
    if lang_code not in SUPPORTED_LANG_CODES:
        raise HTTPException(400, f"Unsupported lang_code '{lang_code}'")

    pipeline = pipelines.get(lang_code)
    if pipeline is not None:
        return pipeline

    from kokoro import KPipeline

    pipeline = KPipeline(lang_code=lang_code, model=model, device="cuda")
    pipelines[lang_code] = pipeline
    return pipeline


def make_wav_header(data_size: int = 0xFFFFFFFF) -> bytes:
    """Create a 44-byte WAV header for 16-bit mono PCM at 24kHz."""
    channels = 1
    bits_per_sample = 16
    byte_rate = SAMPLE_RATE * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        data_size + 36 if data_size != 0xFFFFFFFF else 0xFFFFFFFF,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM
        channels,
        SAMPLE_RATE,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )


def audio_to_pcm(audio: np.ndarray) -> bytes:
    """Convert float32 audio [-1, 1] to int16 PCM bytes."""
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def encode_audio(pcm_data: bytes, fmt: AudioFormat, bitrate: str) -> tuple[bytes, str]:
    """Encode raw PCM to the requested format. Returns (bytes, media_type)."""
    if fmt == AudioFormat.pcm:
        return pcm_data, "audio/pcm"

    if fmt == AudioFormat.wav:
        buf = io.BytesIO()
        buf.write(make_wav_header(len(pcm_data)))
        buf.write(pcm_data)
        return buf.getvalue(), "audio/wav"

    if fmt == AudioFormat.flac:
        buf = io.BytesIO()
        samples = np.frombuffer(pcm_data, dtype=np.int16)
        sf.write(buf, samples, SAMPLE_RATE, format="FLAC")
        return buf.getvalue(), "audio/flac"

    # MP3 and Opus via pydub (requires ffmpeg)
    segment = AudioSegment(
        data=pcm_data,
        sample_width=2,
        frame_rate=SAMPLE_RATE,
        channels=1,
    )
    buf = io.BytesIO()
    if fmt == AudioFormat.mp3:
        segment.export(buf, format="mp3", bitrate=bitrate)
        return buf.getvalue(), "audio/mpeg"
    if fmt == AudioFormat.opus:
        segment.export(buf, format="opus", bitrate=bitrate)
        return buf.getvalue(), "audio/opus"

    return pcm_data, "application/octet-stream"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, pipelines
    from kokoro import KModel

    config_path = os.path.join(MODEL_DIR, "config.json")
    model_path = os.path.join(MODEL_DIR, "kokoro-v1_0.pth")

    if not os.path.exists(model_path) or not os.path.exists(config_path):
        raise RuntimeError(
            f"Model files not found in {MODEL_DIR}. "
            "Run download_model.py first or set MODEL_DIR."
        )

    print(f"Loading Kokoro model from {MODEL_DIR}...")
    model = KModel(config=config_path, model=model_path).eval().cuda()
    pipelines = {}
    print("Model loaded and ready.")
    yield


app = FastAPI(title="Kokoro TTS", version="0.1.0", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def web_ui():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Kokoro TTS</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #0a0a0a; color: #e0e0e0; display: flex; justify-content: center; padding: 2rem; }
  .container { width: 100%; max-width: 640px; }
  h1 { font-size: 1.5rem; margin-bottom: 1.5rem; }
  textarea { width: 100%; height: 120px; padding: 0.75rem; border: 1px solid #333; border-radius: 8px; background: #1a1a1a; color: #e0e0e0; font-size: 1rem; resize: vertical; }
  .controls { display: flex; gap: 0.75rem; margin-top: 0.75rem; flex-wrap: wrap; align-items: end; }
  .field { display: flex; flex-direction: column; gap: 0.25rem; }
  .field label { font-size: 0.75rem; color: #888; }
  select, input[type="number"] { padding: 0.5rem; border: 1px solid #333; border-radius: 6px; background: #1a1a1a; color: #e0e0e0; font-size: 0.875rem; }
  input[type="number"] { width: 5rem; }
  button { padding: 0.5rem 1.5rem; border: none; border-radius: 6px; background: #2563eb; color: white; font-size: 0.875rem; cursor: pointer; }
  button:hover { background: #1d4ed8; }
  button:disabled { background: #333; cursor: not-allowed; }
  .status { margin-top: 1rem; font-size: 0.875rem; color: #888; min-height: 1.25rem; }
  .status.error { color: #ef4444; }
  audio { margin-top: 1rem; width: 100%; }
</style>
</head>
<body>
<div class="container">
  <h1>Kokoro TTS</h1>
  <textarea id="text" placeholder="Type text to synthesize...">Hello, this is a test of the Kokoro text to speech system.</textarea>
  <div class="controls">
    <div class="field">
      <label>Voice</label>
      <select id="voice"></select>
    </div>
    <div class="field">
      <label>Speed</label>
      <input type="number" id="speed" value="1.0" min="0.5" max="2.0" step="0.1">
    </div>
    <div class="field">
      <label>Format</label>
      <select id="format">
        <option value="wav">WAV</option>
        <option value="mp3">MP3</option>
        <option value="opus">Opus</option>
        <option value="flac">FLAC</option>
      </select>
    </div>
    <button id="btn" onclick="speak()">Speak</button>
  </div>
  <div id="status" class="status"></div>
  <audio id="player" controls hidden></audio>
</div>
<script>
async function loadVoices() {
  try {
    const res = await fetch('/v1/voices');
    const data = await res.json();
    const sel = document.getElementById('voice');
    sel.innerHTML = '';
    data.voices.forEach(v => {
      const opt = document.createElement('option');
      opt.value = v;
      opt.textContent = v;
      if (v === 'af_heart') opt.selected = true;
      sel.appendChild(opt);
    });
  } catch (e) { console.error('Failed to load voices:', e); }
}
loadVoices();

async function speak() {
  const btn = document.getElementById('btn');
  const status = document.getElementById('status');
  const player = document.getElementById('player');
  const text = document.getElementById('text').value.trim();
  if (!text) return;

  btn.disabled = true;
  status.className = 'status';
  status.textContent = 'Generating...';
  player.hidden = true;

  try {
    const t0 = performance.now();
    const res = await fetch('/v1/audio/speech', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        input: text,
        voice: document.getElementById('voice').value,
        speed: parseFloat(document.getElementById('speed').value),
        response_format: document.getElementById('format').value,
      }),
    });
    if (!res.ok) throw new Error(await res.text());

    const blob = await res.blob();
    const ms = Math.round(performance.now() - t0);
    const url = URL.createObjectURL(blob);

    player.src = url;
    player.hidden = false;
    player.play();
    status.textContent = `Generated in ${ms}ms (${(blob.size / 1024).toFixed(1)} KB)`;
  } catch (e) {
    status.className = 'status error';
    status.textContent = 'Error: ' + e.message;
  } finally {
    btn.disabled = false;
  }
}
</script>
</body>
</html>"""


@app.get("/v1/voices")
async def list_voices():
    """List available voice packs."""
    voices = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(VOICES_DIR)
        if f.endswith(".pt")
    )
    return {"voices": voices}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "cached_language_pipelines": sorted(pipelines),
    }


@app.post("/v1/audio/speech")
async def synthesize(req: TTSRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded")

    if not req.input.strip():
        raise HTTPException(400, "Input text is empty")

    voice_path = os.path.join(VOICES_DIR, f"{req.voice}.pt")
    if not os.path.exists(voice_path):
        raise HTTPException(404, f"Voice '{req.voice}' not found")

    lang_code = (req.lang_code or infer_lang_code(req.voice)).strip().lower()
    pipeline = get_pipeline(lang_code)

    # For WAV and PCM, stream chunks as they arrive
    if req.response_format in (AudioFormat.wav, AudioFormat.pcm):

        async def stream():
            header_sent = False
            for result in pipeline(req.input, voice=voice_path, speed=req.speed):
                if result.audio is None:
                    continue
                pcm = audio_to_pcm(result.audio.numpy())
                if not header_sent and req.response_format == AudioFormat.wav:
                    yield make_wav_header(0xFFFFFFFF)
                    header_sent = True
                yield pcm

        media_type = "audio/wav" if req.response_format == AudioFormat.wav else "audio/pcm"
        return StreamingResponse(stream(), media_type=media_type)

    # For MP3, Opus, FLAC: collect all audio then encode
    chunks = []
    for result in pipeline(req.input, voice=voice_path, speed=req.speed):
        if result.audio is None:
            continue
        chunks.append(result.audio.numpy())

    if not chunks:
        raise HTTPException(500, "No audio generated")

    full_audio = np.concatenate(chunks)
    pcm_data = audio_to_pcm(full_audio)
    encoded, media_type = encode_audio(pcm_data, req.response_format, req.bitrate)

    return Response(content=encoded, media_type=media_type)
