"""Speech recognition with speaker diarisation for the primary corpus (Whisper large-v3 + pyannote).

Writes one <id>/<id>_transcript.csv per recording with columns speaker, start_time, transcript,
the format read by scoring/prepare.py. No language is forced, so Hindi, English and code-mixed
speech are transcribed as spoken (no translation). Speakers are left as SPEAKER_00, SPEAKER_01, ...;
roles are assigned later in prepare.py.

    HF_TOKEN=... python transcription/transcribe.py recordings/ data/raw/converseAI/
    (recordings/<id>.wav|.mp3|.mp4 -> data/raw/converseAI/<id>/<id>_transcript.csv)

Recordings must be de-identified before transcripts leave the secure environment.
"""
import argparse
import os

import pandas as pd
import torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

AUDIO = (".wav", ".mp3", ".m4a", ".mp4", ".flac")


def speaker_for(seg, turns):
    """Diarisation speaker with the largest time overlap with an ASR segment."""
    best, top = None, 0.0
    for s, e, spk in turns:
        ov = min(e, seg.end) - max(s, seg.start)
        if ov > top:
            best, top = spk, ov
    return best or "SPEAKER_UNK"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--asr", default="large-v3")
    ap.add_argument("--diariser", default="pyannote/speaker-diarization-3.1")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    asr = WhisperModel(a.asr, device=dev, compute_type="float16" if dev == "cuda" else "int8")
    dia = Pipeline.from_pretrained(a.diariser, use_auth_token=os.environ.get("HF_TOKEN")).to(torch.device(dev))

    for f in sorted(os.listdir(a.audio_dir)):
        if not f.lower().endswith(AUDIO):
            continue
        sid = os.path.splitext(f)[0]
        path = os.path.join(a.audio_dir, f)
        turns = [(t.start, t.end, spk) for t, _, spk in dia(path).itertracks(yield_label=True)]
        segs, _ = asr.transcribe(path, task="transcribe", beam_size=5, vad_filter=True)
        rows = [dict(speaker=speaker_for(s, turns), start_time=round(s.start, 3), transcript=s.text.strip())
                for s in segs if s.text.strip()]
        os.makedirs(os.path.join(a.out_dir, sid), exist_ok=True)
        pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, sid, f"{sid}_transcript.csv"), index=False)
        print(sid, len(rows), "segments,", len({r["speaker"] for r in rows}), "speakers")


if __name__ == "__main__":
    main()
