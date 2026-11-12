# Confidence-aware inference of depression and anxiety rating scales from psychiatric history transcripts

Code for the paper by Adhikary, Chauhan, Sinha Deb and Chakraborty (submitted to *npj Mental Health Research*).

A language model reads a psychiatric history-taking transcript and, for every item of the HAM-D, HAM-A, MADRS and
PHQ-9 (PHQ-8 on E-DAIC), returns a severity score, an evidence tag (**E**xplicit, **I**nferred, **A**bsent), a
confidence value and a short rationale. Scores are compared with clinician ratings under full, masked and partial
context.

## Repository layout

| Path | What it does | Paper section |
|---|---|---|
| `transcription/transcribe.py` | Whisper large-v3 speech recognition with pyannote speaker diarisation | Primary corpus |
| `scoring/instruments.py` | Item definitions, anchors and the four-part prompt template for all five instruments | Prompt design |
| `scoring/prepare.py` | Builds model inputs (full and partial context), role labelling, the clinician reference table | Inference conditions; Comparison models |
| `scoring/mask.py` | Automated masked-context condition (Qwen3-32B selects 3–5 symptom spans, replaced by `[MASKED]`) | Comparison models |
| `scoring/run.py` | Zero-shot, greedy-decoded item scoring with vLLM; parses the JSON output | Model and inference configuration |
| `scoring/parsing.py` | JSON extraction and validation of model output | Comparison models |
| `scoring/to_analysis.py` | Keeps transcript–instrument pairs with complete references and parsable output | Comparison models |
| `analysis/evaluate.py` | Weighted κ, Pearson r, Spearman ρ, MAE/RMSE, ICC(2,1), Bland–Altman, severity bands, AUC-ROC, ECE and reliability diagrams, risk–coverage, evidence-tag accuracy, verbatim-quote check, suicidality sensitivity, paired-bootstrap degradation tests; transcript-level bootstrap CIs | Statistical analysis; Uncertainty, agreement and safety analyses |
| `analysis/make_table3.py` | Table 3 (comparison with general-purpose models) | Results |
| `figures/make_figures.py` | Figs 1–3 (Fig. 4 is written by `evaluate.py`) | Figures |
| `scripts/run_comparison.sh` | End-to-end runs for each model | — |
| `scripts/analyse_all.sh` | Paired analysis of several models on the same transcripts, then Table 3 | — |

## Data

* **Primary corpus** (46 history-taking sessions, AIIMS Delhi). Not public: it contains identifiable clinical
  information. De-identified derived data are available from the corresponding author on reasonable request and with
  the permission of the Department of Psychiatry, AIIMS Delhi. Expected layout: `<id>-P/<id>-P_transcript.csv`
  (`speaker,start_time,transcript`) per session, plus rating sheets `PHQ9.csv`, `HAMD.csv`, `HAMA.csv`, `MADRS.csv`
  (one row per patient, `Patient ID` then the item columns in order).
* **E-DAIC** is available from the USC Institute for Creative Technologies under its data-use agreement. Expected
  layout: `transcripts/<id>_Transcript.csv`, `labels/{train,dev,test}_split.csv`, `labels/detailed_lables.csv`
  (item-level PHQ-8, file name as released).

Point the code at the data with environment variables (defaults in brackets):

```bash
export HP_AIIMS_DIR=/secure/converseAI      # [data/raw/converseAI]
export HP_EDAIC_DIR=/secure/EDAIC           # [data/raw/EDAIC]
export HP_DATA_DIR=$PWD/data                # prepared inputs
export HP_RESULTS_DIR=$PWD/results          # model outputs and tables
```

`data/`, model outputs and per-transcript predictions are git-ignored. Only aggregate tables are meant to be committed.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # vLLM needs a CUDA GPU; analysis and figures run on CPU
```

Experiments in the paper used vLLM 0.10.2 on NVIDIA RTX A6000 GPUs (48 GB), a 65,536-token context and greedy decoding.

## Reproducing the paper

```bash
# 1. Transcribe recordings (primary corpus only; HF_TOKEN gives access to the pyannote model)
python transcription/transcribe.py recordings/ $HP_AIIMS_DIR

# 2. Inputs: full and partial (first 40% of patient utterances) context, references
./scripts/run_comparison.sh prepare

# 3. Masked context (automated)
GPUS=0 ./scripts/run_comparison.sh mask

# 4. Score with each model (primary corpus: full/masked/partial; E-DAIC: full/partial)
GPUS=0   ./scripts/run_comparison.sh 3b       # LLaMA 3.2-3B-Instruct
GPUS=1   ./scripts/run_comparison.sh 8b       # LLaMA 3.1-8B-Instruct
GPUS=2,3 ./scripts/run_comparison.sh 70b      # LLaMA 3.1-70B-Instruct, 4-bit AWQ
GPUS=0   ./scripts/run_comparison.sh study /path/to/domain-adapted-checkpoint

# 5. Paired statistics, 2,000 bootstrap resamples, and Table 3
./scripts/analyse_all.sh llama3b llama8b llama70b

# 6. Figures 1-3
python figures/make_figures.py
```

Each model's results land in `results/<model>_<corpus>/`:
`table2_scale_level.csv`, `supp_table1_item_kappa.csv`, `calibration_ece.csv`, `calibration_bins.csv`,
`risk_coverage.csv`, `evidence_tags.csv`, `quote_check.csv`, `suicidality.csv`, `degradation_tests.csv`,
plus `Fig4_calibration` and `SuppFig_bland_altman`.

`analysis/evaluate.py` also runs on its own on any long-format prediction file
(`transcript_id, instrument, item, condition, pred, ref, tag, confidence[, rationale]`):

```bash
python analysis/evaluate.py predictions.csv --out results/my_run --weights quadratic --bins 10
python analysis/evaluate.py --demo --out demo_results     # synthetic data, no GPU or clinical data needed
```

## Notes

* The domain-adapted LLaMA 3.2-3B study model is not distributed here; `scoring/run.py` accepts its checkpoint
  directory through `--model`.
* In the paper the masked condition for the study model was prepared by a psychiatrist; `scoring/mask.py` is the
  automated procedure used for the comparison models.
* Long-form, code-mixed transcripts are scored whole: no transcript exceeded the 65,536-token window. `run.py`
  records a `truncated` flag in case a longer transcript is supplied.

## Licence

MIT (see `LICENSE`). E-DAIC and the primary corpus are covered by their own data-use terms.
