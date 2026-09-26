# Kokoro CUDA

[![Docker Image](https://github.com/ffaerber/kokoro-cuda/actions/workflows/docker.yml/badge.svg)](https://github.com/ffaerber/kokoro-cuda/actions/workflows/docker.yml)

Minimal Kokoro TTS API with CUDA support for NVIDIA GPUs (Ada Lovelace and newer).

**Source:** https://github.com/ffaerber/kokoro-cuda

## Features

- Single FastAPI endpoint for text-to-speech
- Streaming audio (WAV/PCM) and encoded formats (MP3, Opus, FLAC)
- 49 built-in voices (American/British English, Spanish, French, Hindi, Italian, Japanese, Portuguese, Chinese)
- Configurable voice, speed, and bitrate
- Web UI at `/` — type text, pick a voice, and hear audio instantly
- Swagger UI at `/docs`
- Automatic model + voice download on first start (persisted via Docker volumes)

## Quick Start

```bash
docker build -t kokoro-cuda .
docker run --gpus all -p 8880:8880 -v kokoro-models:/app/models -v kokoro-voices:/app/voices kokoro-cuda
```

## API

### `POST /v1/audio/speech`

```json
{
  "input": "Hello, this is a test.",
  "voice": "af_heart",
  "lang_code": "a",
  "speed": 1.0,
  "response_format": "wav",
  "bitrate": "192k"
}
```

| Parameter | Default | Options |
|-----------|---------|---------|
| input | (required) | Any text |
| voice | af_heart | 49 voices — see `GET /v1/voices` |
| lang_code | inferred from voice | Optional Kokoro language code override |
| speed | 1.0 | 0.5 - 2.0 |
| response_format | wav | wav, mp3, opus, flac, pcm |
| bitrate | 192k | 128k, 192k, 320k |

By default, the service infers the Kokoro `lang_code` from the voice prefix. For example, `ff_` voices use the French pipeline and `af_` voices use the American English pipeline. You can still provide `lang_code` explicitly when needed; explicit `lang_code` takes priority over inference.

### `GET /v1/voices`

Returns available voice packs.

### `GET /health`

Returns model status.

## Examples

```bash
# WAV
curl -X POST http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world"}' -o hello.wav

# MP3 at 320k
curl -X POST http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "response_format": "mp3", "bitrate": "320k"}' -o hello.mp3
```

## Web UI

Open `http://localhost:8880/` in your browser. Select a voice, adjust speed, type text, and click **Speak** to hear the audio immediately.

## Voices

49 voice packs are downloaded automatically on first start. Naming convention: `{lang}{gender}_{name}`

| Prefix | Language | Voices |
|--------|----------|--------|
| `af_` / `am_` | American English | 20 |
| `bf_` / `bm_` | British English | 8 |
| `ef_` / `em_` | Spanish | 3 |
| `ff_` | French | 1 |
| `hf_` / `hm_` | Hindi | 4 |
| `if_` / `im_` | Italian | 2 |
| `jf_` / `jm_` | Japanese | 5 |
| `pf_` / `pm_` | Portuguese | 3 |
| `zf_` | Chinese | 4 |

These prefixes also determine the default synthesis language unless you override it with `lang_code` in the request.

## Benchmark

Run the full benchmark suite (Kokoro + Whisper validation):

```bash
cd benchmark
docker compose up --build --abort-on-container-exit --exit-code-from benchmark
```

Results are saved to `benchmark/output/<GPU_NAME>/report.md`. See [latest results](benchmark/output/NVIDIAGeForceRTX5090/report.md).

## Architecture

```
main.py            — FastAPI app, model loading, streaming TTS endpoint
download_model.py  — Downloads model + voice pack on first start
entrypoint.sh      — Runs download then starts uvicorn
Dockerfile         — CUDA 12.8 runtime image (Ada Lovelace + Blackwell)
benchmark/         — Benchmark suite with Whisper validation
```

## Requirements

- NVIDIA GPU with CUDA 12.8+ (RTX 4090 / RTX 5090 and similar)
- Docker with NVIDIA Container Toolkit
