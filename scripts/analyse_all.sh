#!/usr/bin/env bash
# Paired statistics for the comparison models: every model is analysed on the same transcripts
# (complete clinician ratings + parsable output from all models in all conditions).
#   ./scripts/analyse_all.sh llama3b llama8b llama70b
set -e
cd "$(dirname "$0")/.."
REPO=$PWD
RES=${HP_RESULTS_DIR:-$REPO/results}
DATA=${HP_DATA_DIR:-$REPO/data}
for m in "$@"; do
  python3 scoring/to_analysis.py $RES/${m}_AIIMS_preds.csv $RES/${m}_AIIMS_eval.csv
  python3 scoring/to_analysis.py $RES/${m}_E-DAIC_preds.csv $RES/${m}_EDAICtest_eval.csv --split test
done
python3 - "$RES" "$@" <<'PY'
import sys
import pandas as pd
res, models = sys.argv[1], sys.argv[2:]
for corpus in ("AIIMS", "EDAICtest"):
    ev = {m: pd.read_csv(f"{res}/{m}_{corpus}_eval.csv", dtype={"transcript_id": str}) for m in models}
    keys = set.intersection(*[set(map(tuple, e[["transcript_id", "instrument"]].drop_duplicates().values)) for e in ev.values()])
    for m, e in ev.items():
        e = e[[k in keys for k in map(tuple, e[["transcript_id", "instrument"]].values)]]
        e.to_csv(f"{res}/{m}_{corpus}_eval.csv", index=False)
        print(corpus, m, e.groupby("instrument").transcript_id.nunique().to_dict())
PY
for m in "$@"; do
  for c in AIIMS EDAICtest; do
    python3 analysis/evaluate.py $RES/${m}_${c}_eval.csv --out $RES/${m}_${c} \
      --transcripts $DATA/transcripts_txt --boot 2000 2>&1 | grep -E "Wrote|Error"
  done
done
python3 analysis/make_table3.py "$@" --results $RES
