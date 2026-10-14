"""Transcribe German voice messages locally with faster-whisper.

The model is loaded from models/large-v3-turbo next to this script, so no network
access happens. Audio never leaves the machine.
"""

import argparse
import sys
import time
from pathlib import Path

from faster_whisper import WhisperModel

DEFAULT_MODEL = Path(__file__).parent / "models" / "large-v3-turbo"
AUDIO_EXTENSIONS = {".mp3", ".m4a", ".ogg", ".opus", ".oga", ".wav", ".flac", ".aac", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe German voice messages to text.")
    parser.add_argument("paths", nargs="+", type=Path, help="audio files or folders")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="model folder or Hugging Face model name (default: models/large-v3-turbo)")
    parser.add_argument("--out", choices=["txt", "srt", "stdout"], default="txt", help="output format (default: txt next to the audio, also printed)")
    parser.add_argument("--timestamps", action="store_true", help="prefix each segment with its start time")
    parser.add_argument("--prompt", help="hint for names or technical terms, e.g. 'Namen: Anna, Jörg'")
    return parser.parse_args()


def collect_files(paths: list[Path]) -> list[Path]:
    files = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(p for p in path.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS))
        elif path.is_file():
            files.append(path)
        else:
            print(f"Not found: {path}", file=sys.stderr)
    return files


def format_time(seconds: float, srt: bool = False) -> str:
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if srt:
        return f"{int(hours):02}:{int(minutes):02}:{int(secs):02},{int(secs % 1 * 1000):03}"
    return f"{int(minutes):02}:{int(secs):02}" if hours < 1 else f"{int(hours)}:{int(minutes):02}:{int(secs):02}"


def transcribe(model: WhisperModel, path: Path, prompt: str | None) -> list:
    segments, _ = model.transcribe(
        str(path),
        language="de",
        beam_size=5,
        vad_filter=True,
        initial_prompt=prompt,
    )
    return list(segments)


def render(segments: list, fmt: str, timestamps: bool) -> str:
    if fmt == "srt":
        blocks = [
            f"{i}\n{format_time(s.start, srt=True)} --> {format_time(s.end, srt=True)}\n{s.text.strip()}\n"
            for i, s in enumerate(segments, start=1)
        ]
        return "\n".join(blocks)
    if timestamps:
        return "\n".join(f"[{format_time(s.start)}] {s.text.strip()}" for s in segments)
    return " ".join(s.text.strip() for s in segments)


def main() -> int:
    args = parse_args()
    files = collect_files(args.paths)
    if not files:
        print("No audio files given.", file=sys.stderr)
        return 1

    print(f"Loading model {args.model} ...", file=sys.stderr)
    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    failures = 0
    for path in files:
        started = time.perf_counter()
        try:
            segments = transcribe(model, path, args.prompt)
        except Exception as exc:
            print(f"Failed: {path}: {exc}", file=sys.stderr)
            failures += 1
            continue

        text = render(segments, args.out, args.timestamps)
        print(f"\n== {path.name} ({time.perf_counter() - started:.1f}s) ==", file=sys.stderr)
        print(text)
        if args.out != "stdout":
            target = path.with_suffix(f".{args.out}")
            target.write_text(text + "\n", encoding="utf-8")
            print(f"-> {target}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
