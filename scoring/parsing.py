"""Extract item predictions from free-text model output."""
import json
import re


def parse_json(txt):
    """First JSON array in the output (after any <think> block); None if absent or malformed."""
    txt = txt.split("</think>")[-1]
    s, e = txt.find("["), txt.rfind("]")
    if s < 0 or e <= s:
        return None
    body = re.sub(r'("evidence_tag"\s*:\s*)([EIA])\b', r'\1"\2"', txt[s:e + 1])  # unquoted tags
    try:
        out = json.loads(body)
    except json.JSONDecodeError:
        return None
    return out if isinstance(out, list) else None


def parse(txt):
    """Item predictions from the model output; None if missing or malformed."""
    out = parse_json(txt)
    if out is None:
        return None
    good = []
    for it in out:
        try:
            good.append(dict(score=int(it["score"]), evidence_tag=str(it["evidence_tag"])[:1].upper(),
                             confidence=float(it["confidence"]), rationale=str(it.get("rationale", ""))))
        except (KeyError, TypeError, ValueError):
            return None
    return good
