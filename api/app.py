from __future__ import annotations


import os
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import torch
from TTS.api import TTS
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsArgs, XttsAudioConfig

# -------------------- System Optimization --------------------
torch.set_num_threads(max(1, int(os.cpu_count() * 0.8)))
torch.set_num_interop_threads(max(1, int(os.cpu_count() * 0.8)))

# -------------------- Paths and Constants --------------------
INPUT_VIDEO = Path("video_exemplo3.mp4")
REFERENCE_AUDIO = Path("ref_speaker.wav")
TEMP_OUTPUT = Path("outputs/output_temp.wav")
FINAL_OUTPUT = Path("outputs/output.wav")
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
TEXT = "Hello, Simon!"
TARGET_LANGUAGE = "en"
TARGET_SAMPLE_RATE = 24_000

# -------------------- Synthesis Parameters --------------------
SYNTHESIS_PARAMETERS = {
    "temperature": 0.7,
    "length_penalty": 1.3,
    "repetition_penalty": 2.0,
    "top_p": 0.7,
    "split_sentences": False,
}


def allow_xtts_globals() -> None:
    safe_objects = [XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs]
    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if callable(add_safe_globals):
        add_safe_globals(safe_objects)
        return
    add_safe_class = getattr(torch.serialization, "add_safe_class", None)
    if callable(add_safe_class):
        for obj in safe_objects:
            add_safe_class(obj)


def timed_step(description: str, func, *args, **kwargs):
    print(f"[>] {description}...")
    start = perf_counter()
    result = func(*args, **kwargs)
    elapsed = perf_counter() - start
    print(f"[✓] {description} ({elapsed:.2f}s)")
    return result


def ensure_prerequisites() -> None:
    if shutil.which("ffmpeg") is None:
        print("FFmpeg executable not found in PATH.", file=sys.stderr)
        sys.exit(1)
    if not INPUT_VIDEO.is_file():
        print(f"Input video not found: {INPUT_VIDEO}", file=sys.stderr)
        sys.exit(1)


def run_ffmpeg(command: list[str], fast: bool = False) -> None:
    if fast:
        command.insert(1, "-threads")
        command.insert(2, str(max(1, int(os.cpu_count() * 0.8))))
        command.insert(3, "-preset")
        command.insert(4, "ultrafast")

    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="ignore") if exc.stderr else ""
        if stderr:
            print(stderr.strip(), file=sys.stderr)
        sys.exit(exc.returncode or 1)


def extract_reference_audio() -> None:
    run_ffmpeg([
    "ffmpeg",
    "-y",
    "-hwaccel", "cuda",
    "-i", str(INPUT_VIDEO),
    "-vn",
    "-ac", "1",
    "-ar", str(TARGET_SAMPLE_RATE),
    str(REFERENCE_AUDIO),
])



def load_tts_model(use_cuda: bool) -> TTS:
    allow_xtts_globals()
    try:
        model = TTS(
            model_name=MODEL_NAME,
            progress_bar=False,
            gpu=use_cuda,
        )
    except Exception as exc:
        print(f"Failed to load TTS model: {exc}", file=sys.stderr)
        sys.exit(1)
    return model


def synthesize_speech(model: TTS) -> None:
    if TEMP_OUTPUT.exists():
        TEMP_OUTPUT.unlink()
    try:
        model.tts_to_file(
            text=TEXT,
            speaker_wav=str(REFERENCE_AUDIO),
            language=TARGET_LANGUAGE,
            file_path=str(TEMP_OUTPUT),
            **SYNTHESIS_PARAMETERS,
        )
    except Exception as exc:
        print(f"Speech generation failed: {exc}", file=sys.stderr)
        sys.exit(1)


def save_final_output() -> None:
    if not TEMP_OUTPUT.is_file():
        print("Temporary synthesis file was not created.", file=sys.stderr)
        sys.exit(1)
    shutil.move(TEMP_OUTPUT, FINAL_OUTPUT)


def main() -> None:
    timed_step("Validating environment", ensure_prerequisites)
    timed_step(f"Extracting audio ({INPUT_VIDEO.name} -> {REFERENCE_AUDIO.name})", extract_reference_audio)
    use_cuda = torch.cuda.is_available()
    device_label = "GPU" if use_cuda else "CPU"
    tts_model = timed_step(
        f"Loading Coqui XTTS model on {device_label}",
        load_tts_model,
        use_cuda,
    )
    timed_step("Generating cloned speech", synthesize_speech, tts_model)
    timed_step(f"Saving final audio ({FINAL_OUTPUT.name})", save_final_output)
    print("All done. Output saved successfully.")


if __name__ == "__main__":
    main()
