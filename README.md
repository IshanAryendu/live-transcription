# AI4Bharat Gujarati STT (FastAPI + Terminal CLI)

Gujarati speech-to-text using Hugging Face Transformers (`ai4bharat/indicwav2vec_v1_gujarati`) with:

- **FastAPI WebSocket server** for near-real-time chunked transcription
- **Terminal-only CLI** (`cli_transcribe.py`) for quick local testing (no UI)
- **VAD-gated chunking + overlap-aware merging + silence finalization** shared in `asr_core.py`

> Note: This is chunked “near-real-time” inference, not a model with native streaming logits.

## Requirements

- **Python**: 3.10+ recommended
- **ffmpeg**: required (used to resample/convert audio)
- **Model access**: the default model may be **gated** on Hugging Face (you may need to request access and login)

## Install

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install ffmpeg (examples):

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

## Terminal-only test (recommended first)

Run the exact test command (note the quotes because the path contains a space):

```bash
python cli_transcribe.py --device cpu "path/to/gujarati_test.wav"
```

If `python` is not available on your system, use:

```bash
python3 cli_transcribe.py --device cpu "path/to/gujarati_test.wav"
```

### Switching model

```bash
ASR_MODEL="ai4bharat/indicwav2vec_v1_gujarati" python3 cli_transcribe.py --device cpu "/path/to/audio.wav"
```

### CPU vs CUDA

CPU is the safest default (especially for older GPUs):

```bash
ASR_DEVICE=cpu python3 cli_transcribe.py "/path/to/audio.wav"
```

CUDA (only if your installed PyTorch supports your GPU):

```bash
ASR_DEVICE=cuda python3 cli_transcribe.py --device cuda "/path/to/audio.wav"
```

## Run the FastAPI server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000`.

## Configuration

Environment variables (defaults shown):

- **Model/device**
  - `ASR_MODEL=ai4bharat/indicwav2vec_v1_gujarati`
  - `ASR_DEVICE=cpu`
- **Audio**
  - `CHUNK_MS=250`
  - `OVERLAP_MS=500`
  - `MIN_SPEECH_MS=900`
  - `MAX_UTTERANCE_MS=10000`
  - `FINAL_SILENCE_MS=900`
  - `VAD_RMS_THRESHOLD=550`
  - `PARTIAL_LOOKBACK_WORDS=8`

## Troubleshooting

### Hugging Face 401 / gated repo

If you see `401 Unauthorized` / “gated repo” errors for `ai4bharat/indicwav2vec_v1_gujarati`:

- Request/accept access on the model page: `https://huggingface.co/ai4bharat/indicwav2vec_v1_gujarati`
- Login locally:

```bash
huggingface-cli login
```

Or set a token:

```bash
export HF_TOKEN="hf_..."
```

## Project layout

- `asr_core.py`: shared VAD + merge + finalize logic
- `cli_transcribe.py`: terminal transcription using the same logic as the server
- `app.py`: FastAPI server + WebSocket endpoint
