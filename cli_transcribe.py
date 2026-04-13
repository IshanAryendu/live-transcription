import argparse
import os

try:
    from asr_core import (
        MODEL_NAME,
        build_asr,
        decode_audio_file_to_float32,
        run_streaming_logic_over_audio,
    )
except ModuleNotFoundError as e:
    raise SystemExit(
        "Missing dependency or module import failed.\n"
        "Make sure you're running from the project directory and deps are installed:\n"
        "  python3 -m venv .venv\n"
        "  source .venv/bin/activate\n"
        "  pip install -r requirements.txt\n"
    ) from e


DEFAULT_DEVICE = os.getenv("ASR_DEVICE", "cpu").lower().strip()


def main() -> int:
    ap = argparse.ArgumentParser(description="Transcribe an audio file from the terminal (no UI).")
    ap.add_argument("audio_path", help="Path to audio file (wav/mp3/m4a/webm/etc).")
    ap.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        choices=["cpu", "cuda"],
        help="Inference device. Default: %(default)s (set ASR_DEVICE to change).",
    )
    args = ap.parse_args()

    asr = build_asr(model_name=MODEL_NAME, device=args.device)
    audio = decode_audio_file_to_float32(args.audio_path)
    state = run_streaming_logic_over_audio(asr, audio)
    print("\n".join(state.finalized))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

