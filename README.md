# audiosanity

Transcribes German voice messages (WhatsApp, Telegram, iMessage, MP3, …) to text, fully
offline. Speech recognition runs locally on the CPU with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) and the Whisper
`large-v3-turbo` model. No audio leaves the machine and no API is called.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

1. Install dependencies:

   ```
   uv sync
   ```

2. Download the model (~1.6 GB) once. Save these files into `models/large-v3-turbo/`:

   - [model.bin](https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo/resolve/main/model.bin)
   - [config.json](https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo/resolve/main/config.json)
   - [preprocessor_config.json](https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo/resolve/main/preprocessor_config.json)
   - [tokenizer.json](https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo/resolve/main/tokenizer.json)
   - [vocabulary.json](https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo/resolve/main/vocabulary.json)

   If `huggingface.co` is blocked on your network (company firewall/VPN), download over a
   different connection.

## Usage

```
uv run transcribe.py "path/to/voice message.ogg"
uv run transcribe.py path/to/folder/
```

The transcript is printed and saved as a `.txt` next to the audio file. Folders are scanned
for `.mp3`, `.m4a`, `.ogg`, `.opus`, `.oga`, `.wav`, `.flac`, `.aac` and `.webm`.

| Option | Effect |
|---|---|
| `--out txt\|srt\|stdout` | Output format. `txt` (default) writes a file next to the audio, `srt` writes subtitles, `stdout` only prints |
| `--timestamps` | Prefix each segment with its start time |
| `--prompt "Namen: …"` | Hint for names and technical terms the model would otherwise mishear |
| `--model <folder or name>` | A different model folder, or a Hugging Face model name such as `small` (downloaded on first use) |

### Names and technical terms

Whisper spells unknown names phonetically, so rarer names or abbreviations often come out
wrong. `--prompt` passes text that the model treats as preceding context, which nudges it
toward the right spelling:

```
uv run transcribe.py --prompt "Namen: Anna, Jörg, Wolfsburg. Begriffe: Kubernetes, Sprint-Review" message.ogg
```

List only words that actually occur in the recordings; a long or unrelated prompt can make
the model insert those words where they were not spoken. The prompt applies to every file of
the run, so group files by sender when the names differ.

A one-minute voice message takes roughly 5–30 seconds on a modern laptop CPU.

## Notes

- In Git Bash, write Windows paths with forward slashes (`C:/Users/...`) or quote them;
  unquoted backslashes are stripped.
- `av` is pinned below 19 because PyAV 19 dropped the `metadata_errors` argument that
  faster-whisper 1.2.1 passes to `av.open()`.
- The language is fixed to German. Whisper can hallucinate plausible-sounding text on
  noise or unclear speech, so check transcripts where accuracy matters.
