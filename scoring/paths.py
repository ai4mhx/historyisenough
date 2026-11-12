"""Locations of the raw corpora, prepared inputs and model outputs.

Override with environment variables:
    HP_AIIMS_DIR   primary corpus: <id>-P/<id>-P_transcript.csv per session, plus PHQ9.csv, HAMD.csv, HAMA.csv, MADRS.csv
    HP_EDAIC_DIR   E-DAIC release: transcripts/<id>_Transcript.csv and labels/{train,dev,test}_split.csv, detailed_lables.csv
    HP_DATA_DIR    prepared inputs (inputs.jsonl, refs.csv, masking.csv, transcripts_txt/)
    HP_RESULTS_DIR model outputs and evaluation tables
"""
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AIIMS = os.environ.get("HP_AIIMS_DIR", os.path.join(REPO, "data", "raw", "converseAI"))
EDAIC = os.environ.get("HP_EDAIC_DIR", os.path.join(REPO, "data", "raw", "EDAIC"))
DATA = os.environ.get("HP_DATA_DIR", os.path.join(REPO, "data"))
RESULTS = os.environ.get("HP_RESULTS_DIR", os.path.join(REPO, "results"))
