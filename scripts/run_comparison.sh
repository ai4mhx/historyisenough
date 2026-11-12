#!/usr/bin/env bash
# Comparison-model runs reported in the paper (Comparison models; Table 3).
#   GPUS=0   ./scripts/run_comparison.sh prepare     # inputs.jsonl, refs.csv, transcripts_txt/
#   GPUS=0   ./scripts/run_comparison.sh mask        # automated masked condition (Qwen3-32B-AWQ)
#   GPUS=0   ./scripts/run_comparison.sh 3b
#   GPUS=1   ./scripts/run_comparison.sh 8b
#   GPUS=2,3 ./scripts/run_comparison.sh 70b         # 4-bit AWQ, tensor-parallel over two 48 GB GPUs
#   GPUS=0   ./scripts/run_comparison.sh study /path/to/domain-adapted-checkpoint
set -e
cd "$(dirname "$0")/../scoring"
export CUDA_VISIBLE_DEVICES=${GPUS:-0} VLLM_USE_FLASHINFER_SAMPLER=0
LOGS=../logs
mkdir -p $LOGS
case $1 in
  prepare) python prepare.py; exit ;;
  mask)    python mask.py --model Qwen/Qwen3-32B-AWQ > $LOGS/mask.log 2>&1; exit ;;
  3b)      M=meta-llama/Llama-3.2-3B-Instruct; T=llama3b; X="" ;;
  8b)      M=meta-llama/Llama-3.1-8B-Instruct; T=llama8b; X="" ;;
  70b)     M=hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4; T=llama70b; X="--tp 2"; export NCCL_P2P_DISABLE=1 ;;
  study)   M=${2:?checkpoint path}; T=study; X="" ;;
  *)       echo "usage: $0 prepare|mask|3b|8b|70b|study <ckpt>"; exit 1 ;;
esac
python run.py --corpus AIIMS --tag $T --model $M $X > $LOGS/${T}_aiims.log 2>&1
python run.py --corpus E-DAIC --tag $T --model $M $X --conditions full,partial > $LOGS/${T}_edaic.log 2>&1
echo "done: results/${T}_AIIMS_preds.csv, results/${T}_E-DAIC_preds.csv"
