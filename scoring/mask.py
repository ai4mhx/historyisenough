"""Automated masked-context condition used for the comparison models.

An LLM is shown the numbered patient utterances of each primary-corpus transcript and asked for the 3-5
contiguous spans that most directly state symptoms covered by the four scales. Those utterances are
replaced by [MASKED]; the rest of the dialogue is kept. Appends condition="masked" rows to
inputs.jsonl and writes masking.csv (spans, domains, share of patient tokens masked).

    CUDA_VISIBLE_DEVICES=0 python scoring/mask.py --model Qwen/Qwen3-32B-AWQ
"""
import argparse
import json
import os

import pandas as pd
from vllm import LLM, SamplingParams

from parsing import parse_json
from paths import DATA
from run import engine_kwargs

PROMPT = """You are a consultant psychiatrist preparing a masked-context experiment. Below are the numbered PATIENT utterances from a psychiatric history-taking interview (Hindi, English or code-mixed; automatic speech recognition, so expect errors).

Select between 3 and 5 non-overlapping spans of consecutive utterances that most directly and explicitly state symptoms scored by the HAM-D, HAM-A, MADRS or PHQ-9 (for example low mood, loss of interest, sleep, appetite, energy, concentration, guilt or worthlessness, suicidal thoughts, worry or tension, somatic anxiety). Each span should cover 1 to 6 utterances. Prefer the clearest, most informative statements.

Respond ONLY with a JSON array of objects with keys: start (first utterance number), end (last utterance number, inclusive), domain (the symptom stated).

PATIENT UTTERANCES:
{utts}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-32B-AWQ")
    ap.add_argument("--quant")
    ap.add_argument("--max-len", type=int, default=65536)
    a = ap.parse_args()

    path = os.path.join(DATA, "inputs.jsonl")
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if r["condition"] != "masked"]
    full = [r for r in rows if r["corpus"] == "AIIMS" and r["condition"] == "full"]

    llm = LLM(a.model, **engine_kwargs(a.model, a.quant, a.max_len))
    tok = llm.get_tokenizer()
    prompts, pats = [], []
    for r in full:
        lines = r["text"].split("\n")
        pat = [i for i, l in enumerate(lines) if l.startswith("PATIENT:")]
        pats.append((lines, pat))
        utts = "\n".join(f"[{k}] {lines[i][len('PATIENT: '):]}" for k, i in enumerate(pat))
        ids = tok.apply_chat_template([{"role": "user", "content": PROMPT.format(utts=utts)}],
                                      add_generation_prompt=True, enable_thinking=False)
        if len(ids) > a.max_len - 600:
            ids = ids[: (a.max_len - 600) // 2] + ids[-((a.max_len - 600) // 2):]
        prompts.append({"prompt_token_ids": ids})
    outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=600))

    log, new = [], []
    for r, (lines, pat), o in zip(full, pats, outs):
        spans = parse_json(o.outputs[0].text) or []
        masked, kept = set(), []
        for s in spans:
            try:
                st, en = int(s["start"]), int(s["end"])
            except (KeyError, TypeError, ValueError):
                continue
            st, en = max(st, 0), min(en, st + 5, len(pat) - 1)
            if en < st or any(k in masked for k in range(st, en + 1)) or len(kept) == 5:
                continue
            masked.update(range(st, en + 1))
            kept.append(dict(start=st, end=en, domain=str(s.get("domain", ""))))
        out = list(lines)
        for k in masked:
            out[pat[k]] = "PATIENT: [MASKED]"
        ntok = lambda ls: sum(len(tok(l).input_ids) for l in ls)
        share = ntok([lines[pat[k]] for k in masked]) / max(ntok([lines[i] for i in pat]), 1)
        new.append(dict(corpus="AIIMS", transcript_id=r["transcript_id"], condition="masked", text="\n".join(out)))
        log.append(dict(transcript_id=r["transcript_id"], n_spans=len(kept), n_utts_masked=len(masked),
                        n_patient_utts=len(pat), share_patient_tokens=round(share, 4),
                        domains="; ".join(s["domain"] for s in kept), raw=o.outputs[0].text))
    with open(path, "w") as f:
        for r in rows + new:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    m = pd.DataFrame(log)
    m.to_csv(os.path.join(DATA, "masking.csv"), index=False)
    print(m[["n_spans", "n_utts_masked", "share_patient_tokens"]].describe().round(4).to_string())


if __name__ == "__main__":
    main()
