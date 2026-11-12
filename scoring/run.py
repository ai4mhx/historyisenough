"""Zero-shot item-level scoring with vLLM (greedy decoding; JSON parsed from the output).

    CUDA_VISIBLE_DEVICES=0 python scoring/run.py --corpus AIIMS --model meta-llama/Llama-3.2-3B-Instruct --tag llama3b

--model accepts a Hugging Face ID or a local checkpoint directory (for example the domain-adapted
study model). Writes <results>/<tag>_<corpus>_raw.jsonl (one row per prompt) and
<results>/<tag>_<corpus>_preds.csv in the long format read by analysis/evaluate.py (ref merged in).
"""
import argparse
import json
import os

import pandas as pd
from vllm import LLM, SamplingParams

from instruments import build_prompt, items
from parsing import parse, parse_json  # noqa: F401  (parse_json is used by mask.py)
from paths import DATA, RESULTS

INSTR = {"AIIMS": ["PHQ-9", "HAM-D", "HAM-A", "MADRS"], "E-DAIC": ["PHQ-8"]}


def engine_kwargs(model, quant, max_len, tp=1):
    """vLLM settings; Qwen3 needs YaRN beyond its native 32k window."""
    kw = dict(max_model_len=max_len, gpu_memory_utilization=0.92, seed=0, tensor_parallel_size=tp)
    if tp > 1:  # GPU P2P can hang on PCIe-only servers: keep all-reduce on NCCL with P2P off
        kw["disable_custom_all_reduce"] = True
    if quant:
        kw["quantization"] = quant
    if "Qwen3" in model and max_len > 32768:
        kw["hf_overrides"] = {"rope_scaling": {"rope_type": "yarn", "factor": max_len / 32768,
                                               "original_max_position_embeddings": 32768}}
    return kw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=list(INSTR))
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--tag", default="llama3b")
    ap.add_argument("--max-len", type=int, default=65536)
    ap.add_argument("--conditions", default="full,masked,partial")
    ap.add_argument("--quant", help="vLLM quantization method, e.g. fp8")
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel GPUs")
    ap.add_argument("--limit", type=int, help="first N inputs only (smoke test)")
    ap.add_argument("--reparse", action="store_true", help="rebuild preds.csv from an existing raw.jsonl")
    a = ap.parse_args()
    if a.reparse:
        return collect(os.path.join(RESULTS, f"{a.tag}_{a.corpus}"), a.corpus, a.tag)

    inputs = [json.loads(l) for l in open(os.path.join(DATA, "inputs.jsonl"))]
    inputs = [r for r in inputs if r["corpus"] == a.corpus and r["condition"] in a.conditions.split(",")]
    inputs = inputs[: a.limit] if a.limit else inputs
    llm = LLM(a.model, **engine_kwargs(a.model, a.quant, a.max_len, a.tp))
    tok = llm.get_tokenizer()

    jobs, prompts, params = [], [], []
    for r in inputs:
        for instr in INSTR[a.corpus]:
            msgs = [{"role": "user", "content": build_prompt(instr, r["text"])}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
            out_budget = 160 * len(items(instr)) + 200
            trunc = len(ids) + out_budget > a.max_len
            if trunc:  # keep the head of the prompt (instructions) and the tail (output spec)
                keep = a.max_len - out_budget
                ids = ids[: keep // 2] + ids[-(keep - keep // 2):]
            jobs.append(dict(r, instrument=instr, prompt_tokens=len(ids), truncated=trunc))
            prompts.append({"prompt_token_ids": ids})
            params.append(SamplingParams(temperature=0, max_tokens=out_budget))
    outs = llm.generate(prompts, params)

    os.makedirs(RESULTS, exist_ok=True)
    base = os.path.join(RESULTS, f"{a.tag}_{a.corpus}")
    with open(base + "_raw.jsonl", "w") as f:
        for j, o in zip(jobs, outs):
            j = {k: v for k, v in j.items() if k != "text"}
            f.write(json.dumps(dict(j, output=o.outputs[0].text), ensure_ascii=False) + "\n")
    collect(base, a.corpus, a.tag)


def collect(base, corpus, tag):
    """Parse <base>_raw.jsonl into <base>_preds.csv (can be rerun without the model)."""
    raw = [json.loads(l) for l in open(base + "_raw.jsonl")]
    rows, ok = [], 0
    for j in raw:
        parsed = parse(j["output"])
        ok += parsed is not None
        its = items(j["instrument"])
        for k, it in enumerate((parsed or [])[:len(its)], 1):
            rows.append(dict(corpus=corpus, transcript_id=j["transcript_id"], instrument=j["instrument"],
                             item=k, condition=j["condition"], split=j.get("split"),
                             pred=min(max(it["score"], 0), its[k - 1][2]), tag=it["evidence_tag"],
                             confidence=min(max(it["confidence"], 0.0), 1.0), rationale=it["rationale"],
                             truncated=j["truncated"], model=tag))
    p = pd.DataFrame(rows)
    refs = pd.read_csv(os.path.join(DATA, "refs.csv"), dtype={"transcript_id": str})
    p = p.merge(refs, on=["corpus", "transcript_id", "instrument", "item"], how="left")
    p.to_csv(base + "_preds.csv", index=False)
    print(f"prompts {len(raw)}, parsed {ok}, truncated {sum(j['truncated'] for j in raw)}, rows {len(p)}")


if __name__ == "__main__":
    main()
