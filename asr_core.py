import audioop
import os
import re
import subprocess
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from transformers import pipeline


MODEL_NAME = os.getenv("ASR_MODEL", "ai4bharat/indicwav2vec_v1_gujarati")
TARGET_SR = 16000

CHUNK_MS = int(os.getenv("CHUNK_MS", "250"))
OVERLAP_MS = int(os.getenv("OVERLAP_MS", "500"))
MIN_SPEECH_MS = int(os.getenv("MIN_SPEECH_MS", "900"))
MAX_UTTERANCE_MS = int(os.getenv("MAX_UTTERANCE_MS", "10000"))
FINAL_SILENCE_MS = int(os.getenv("FINAL_SILENCE_MS", "900"))
VAD_RMS_THRESHOLD = int(os.getenv("VAD_RMS_THRESHOLD", "550"))
PARTIAL_LOOKBACK_WORDS = int(os.getenv("PARTIAL_LOOKBACK_WORDS", "8"))


@dataclass
class StreamState:
    committed: str = ""
    partial: str = ""
    finalized: List[str] = None
    in_speech: bool = False
    speech_audio: np.ndarray = None
    trailing_silence_ms: int = 0

    def __post_init__(self):
        if self.finalized is None:
            self.finalized = []
        if self.speech_audio is None:
            self.speech_audio = np.array([], dtype=np.float32)


def normalize_gujarati_text(text: str) -> str:
    text = text.replace("\u200c", "").replace("\u200d", "")
    text = text.replace("઼", "")
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"\s+([,.;!?])", r"\1", text)
    text = re.sub(r"([૦-૯])\s+([૦-૯])", r"\1\2", text)
    text = re.sub(r"([અ-હા-ૅેૈૉોૌ્ૠૡૢૣૐઽ]+)\s+([ા-્ૅેૈૉોૌૢૣ]+)", r"\1\2", text)
    return text


def words(text: str) -> List[str]:
    return normalize_gujarati_text(text).split()


def longest_suffix_prefix(a: List[str], b: List[str], min_match: int = 1) -> int:
    max_k = min(len(a), len(b))
    for k in range(max_k, min_match - 1, -1):
        if a[-k:] == b[:k]:
            return k
    return 0


def merge_partial(committed: str, incoming: str, lookback: int = PARTIAL_LOOKBACK_WORDS) -> Tuple[str, str]:
    committed = normalize_gujarati_text(committed)
    incoming = normalize_gujarati_text(incoming)
    if not incoming:
        return committed, ""
    if not committed:
        inc = words(incoming)
        stable = " ".join(inc[:-3]) if len(inc) > 3 else ""
        partial = " ".join(inc[-3:]) if inc else ""
        return normalize_gujarati_text(stable), partial

    cw = words(committed)
    iw = words(incoming)
    anchor = cw[-lookback:] if len(cw) > lookback else cw
    overlap = longest_suffix_prefix(anchor, iw, min_match=1)
    addition = iw[overlap:]
    stable = " ".join(addition[:-3]) if len(addition) > 3 else ""
    partial = " ".join(addition[-3:]) if addition else ""
    if stable:
        committed = normalize_gujarati_text(committed + " " + stable)
    return committed, partial


def dedupe_finalized(lines: List[str], candidate: str) -> List[str]:
    candidate = normalize_gujarati_text(candidate)
    if not candidate:
        return lines
    if not lines:
        return [candidate]
    last = lines[-1]
    if candidate == last or candidate in last or last in candidate:
        lines[-1] = candidate if len(candidate) > len(last) else last
        return lines
    last_words = words(last)
    cand_words = words(candidate)
    overlap = (
        longest_suffix_prefix(last_words, cand_words, min_match=2)
        if min(len(last_words), len(cand_words)) >= 2
        else 0
    )
    if overlap:
        merged = normalize_gujarati_text(last + " " + " ".join(cand_words[overlap:]))
        lines[-1] = merged
        return lines
    lines.append(candidate)
    return lines


def duration_ms(audio: np.ndarray, sample_rate: int = TARGET_SR) -> int:
    return int(audio.size * 1000 / sample_rate)


def tail_audio(audio: np.ndarray, ms: int, sample_rate: int = TARGET_SR) -> np.ndarray:
    n = int(sample_rate * ms / 1000)
    return audio[-n:] if audio.size > n else audio


def is_speech(audio: np.ndarray) -> bool:
    if audio.size == 0:
        return False
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16).tobytes()
    rms = audioop.rms(pcm, 2)
    return rms >= VAD_RMS_THRESHOLD


def decode_audio_file_to_float32(path: str, sample_rate: int = TARGET_SR) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-i",
        path,
        "-f",
        "s16le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    audio = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


def build_asr(model_name: str = MODEL_NAME, device: str = "cpu"):
    device_index = -1 if device == "cpu" else 0
    return pipeline("automatic-speech-recognition", model=model_name, device=device_index)


def transcribe_audio(asr, audio: np.ndarray) -> str:
    if audio.size == 0:
        return ""
    result = asr(audio)
    text = result.get("text", "") if isinstance(result, dict) else str(result)
    return normalize_gujarati_text(text)


def run_streaming_logic_over_audio(asr, audio: np.ndarray) -> StreamState:
    """
    Run the same VAD + partial merge + finalize logic as the websocket path,
    but over a pre-loaded audio array.
    """
    state = StreamState()
    chunk_len = int(TARGET_SR * CHUNK_MS / 1000)

    idx = 0
    while idx < audio.size:
        chunk = audio[idx : idx + chunk_len]
        idx += chunk_len

        speech = is_speech(chunk)
        chunk_dur = duration_ms(chunk)

        if speech:
            state.in_speech = True
            state.trailing_silence_ms = 0
            state.speech_audio = np.concatenate([state.speech_audio, chunk])
            if duration_ms(state.speech_audio) > MAX_UTTERANCE_MS:
                state.speech_audio = tail_audio(state.speech_audio, MAX_UTTERANCE_MS)

            if duration_ms(state.speech_audio) >= MIN_SPEECH_MS:
                audio_window = tail_audio(
                    state.speech_audio,
                    min(duration_ms(state.speech_audio), MIN_SPEECH_MS + OVERLAP_MS),
                )
                text = transcribe_audio(asr, audio_window)
                state.committed, state.partial = merge_partial(state.committed, text)
            continue

        if state.in_speech:
            state.trailing_silence_ms += chunk_dur
            state.speech_audio = np.concatenate([state.speech_audio, chunk])

            if state.trailing_silence_ms >= FINAL_SILENCE_MS:
                text = transcribe_audio(asr, state.speech_audio)
                full = normalize_gujarati_text(text)
                if full:
                    state.finalized = dedupe_finalized(state.finalized, full)
                state.committed = ""
                state.partial = ""
                state.in_speech = False
                state.speech_audio = np.array([], dtype=np.float32)
                state.trailing_silence_ms = 0
            continue

    # End-of-file flush: if we were mid-utterance, finalize once.
    if state.in_speech and state.speech_audio.size:
        text = transcribe_audio(asr, state.speech_audio)
        full = normalize_gujarati_text(text)
        if full:
            state.finalized = dedupe_finalized(state.finalized, full)
        state.committed = ""
        state.partial = ""
        state.in_speech = False
        state.speech_audio = np.array([], dtype=np.float32)
        state.trailing_silence_ms = 0

    return state

