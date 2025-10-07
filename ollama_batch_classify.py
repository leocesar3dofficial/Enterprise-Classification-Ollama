#!/usr/bin/env python3
"""
ollama_batch_classify_fixed.py

More robust Ollama caller that handles:
 - single JSON responses
 - streaming / NDJSON responses (one JSON object per line or "data: {..}" SSE style)
 - plain text responses

Usage:
  export OLLAMA_ENDPOINT="http://localhost:11434/api/generate"
  export MODEL="your-model-name"
  python ollama_batch_classify_fixed.py --input enterprise_classification_dataset.csv --output results.csv
"""

import os
import argparse
import csv
import requests
import pandas as pd
import time
import json
from json import JSONDecodeError
from typing import Optional
import difflib
import re

DEFAULT_PROMPT_TEMPLATE = (
    "You are a classification assistant. Return **only** the label from the list below "
    "that best matches the text. If none match, return 'Other'.\n\n"
    "Labels: {labels}\n\n"
    "Text:\n{text}\n\n"
    "Answer:"
)

# -------------------------
# Robust response parsing
# -------------------------
def extract_json_objects_from_text(text):
    """
    Try to extract JSON objects from text that may contain:
      - multiple JSON objects separated by newlines (NDJSON)
      - SSE style lines like 'data: {...}'
      - extra leading/trailing text
    Returns list of parsed JSON objects (could be empty).
    """
    objs = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Remove "data: " prefix often present in SSE-like streams
        if line.startswith("data:"):
            line = line[len("data:"):].strip()
        # Try to parse the line as JSON
        try:
            obj = json.loads(line)
            objs.append(obj)
            continue
        except JSONDecodeError:
            # sometimes streaming fragments are concatenated; try to find {...}
            # we'll attempt to find first { ... } substring and parse it
            start = line.find("{")
            end = line.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = line[start:end+1]
                try:
                    obj = json.loads(candidate)
                    objs.append(obj)
                    continue
                except JSONDecodeError:
                    pass
            # not JSON - skip
            continue
    return objs

def extract_text_from_response_json(j):
    """
    Given a parsed JSON object, attempt to extract model text from common shapes.
    """
    if j is None:
        return ""
    # If dict, check common keys
    if isinstance(j, dict):
        # possible top-level raw text
        for key in ("text", "output", "result", "content", "answer"):
            if key in j and isinstance(j[key], str) and j[key].strip():
                return j[key].strip()
        # choices style
        if "choices" in j and isinstance(j["choices"], list) and j["choices"]:
            first = j["choices"][0]
            if isinstance(first, dict):
                for key in ("text", "message", "content"):
                    if key in first and isinstance(first[key], str) and first[key].strip():
                        return first[key].strip()
                # sometimes delta fragments are present
                if "delta" in first and isinstance(first["delta"], dict):
                    parts = []
                    for v in first["delta"].values():
                        if isinstance(v, str):
                            parts.append(v)
                    if parts:
                        return "".join(parts).strip()
        # some Ollama responses embed an array of outputs
        if "outputs" in j and isinstance(j["outputs"], list) and j["outputs"]:
            first = j["outputs"][0]
            if isinstance(first, dict):
                for key in ("text", "content"):
                    if key in first and isinstance(first[key], str):
                        return first[key].strip()
        # if nothing matched, return stringified dict
        return str(j).strip()
    # if it's a list, try to inspect first element
    if isinstance(j, list) and j:
        return extract_text_from_response_json(j[0])
    # fallback
    return str(j)

# -------------------------
# Call Ollama robustly
# -------------------------
def call_ollama(endpoint: str, model: str, prompt: str, timeout=60) -> (str, str):
    """
    Calls the provided endpoint with {"model": model, "prompt": prompt}.
    Returns (extracted_text, raw_response_text).
    """
    payload = {
        "model": model,
        "prompt": prompt
    }
    headers = {"Content-Type": "application/json"}
    resp = requests.post(endpoint, json=payload, headers=headers, timeout=timeout, stream=False)

    raw = resp.text or ""
    # first try standard JSON decode
    try:
        j = resp.json()
        extracted = extract_text_from_response_json(j)
        return extracted, raw
    except JSONDecodeError:
        # try to extract JSON objects from each line (NDJSON or SSE)
        objs = extract_json_objects_from_text(raw)
        if objs:
            # pick the last or first object containing a likely "text" field.
            # We'll prioritize the last because streaming endpoints often append final content last.
            for candidate in reversed(objs):
                extracted = extract_text_from_response_json(candidate)
                if extracted:
                    return extracted, raw
            # if none extracted text, return stringified last obj
            return json.dumps(objs[-1]), raw
        # If still nothing, fallback to raw text
        return raw.strip(), raw

# -------------------------
# Utility label helpers
# -------------------------
def normalize_label(s):
    if s is None:
        return ""
    return " ".join(s.strip().lower().split())

# --- add at top of file ---
import difflib
import re

# --- replace pick_label_from_response with this ---
def pick_label_from_response(resp_text, label_list, synonyms=None, fuzzy_thresh=0.6):
    """
    Robust mapping from model text -> one label from label_list.
    Order of attempts:
      1) exact (case-insensitive)
      2) first non-empty line
      3) tokenized first token / punctuation-trim
      4) substring match (prefer longer labels)
      5) synonyms mapping (dict mapping alt -> canonical label)
      6) fuzzy match via difflib (ratio threshold)
      7) fallback: return resp_text (raw)
    """
    if resp_text is None:
        return ""

    r = resp_text.strip()

    # 1) exact match (case-ins)
    for lab in label_list:
        if r.lower() == lab.lower():
            return lab

    # 2) check each non-empty line
    for line in r.splitlines():
        s = line.strip()
        if not s:
            continue
        for lab in label_list:
            if s.lower() == lab.lower():
                return lab

    # 3) first token / strip punctuation
    first_token = re.split(r'[\s,:;-]+', r)[0].strip()
    for lab in label_list:
        if first_token.lower() == lab.lower():
            return lab

    # 4) substring match (prefer longer labels)
    sorted_labels = sorted(label_list, key=lambda x: -len(x))
    for lab in sorted_labels:
        if lab.lower() in r.lower():
            return lab

    # 5) synonyms map
    if synonyms:
        # check exact synonym matches & substring matches
        for alt, canonical in synonyms.items():
            if alt.lower() == r.lower():
                return canonical
            if alt.lower() in r.lower():
                return canonical

    # 6) fuzzy matching
    # compare against labels and synonyms keys
    candidates = label_list[:]
    if synonyms:
        # include synonyms keys but map later
        candidates += list(synonyms.keys())
    # use difflib to find close matches
    match = difflib.get_close_matches(r, candidates, n=1, cutoff=fuzzy_thresh)
    if match:
        m = match[0]
        # if m is a synonym, map to canonical
        if synonyms and m in synonyms:
            return synonyms[m]
        # otherwise return label with exact case from label_list
        for lab in label_list:
            if lab.lower() == m.lower():
                return lab
        return m

    # 7) fallback - try to return first meaningful token line
    if r:
        return r

    return ""

def _extract_from_obj(obj):
    """
    Try to pull text fragments from a parsed JSON object.
    Prefer 'response', then 'text', 'output', 'content', 'answer', and common nested shapes.
    Returns a string (possibly empty).
    """
    if obj is None:
        return ""
    # direct string fields
    for key in ("response", "text", "output", "content", "answer"):
        if key in obj and isinstance(obj[key], str) and obj[key].strip() != "":
            return obj[key]
    # choices / outputs arrays
    if "choices" in obj and isinstance(obj["choices"], list):
        for choice in obj["choices"]:
            if isinstance(choice, dict):
                for key in ("text", "message", "content", "response"):
                    if key in choice and isinstance(choice[key], str) and choice[key].strip() != "":
                        return choice[key]
                # delta fragments
                if "delta" in choice and isinstance(choice["delta"], dict):
                    parts = []
                    for v in choice["delta"].values():
                        if isinstance(v, str):
                            parts.append(v)
                    if parts:
                        return "".join(parts)
    if "outputs" in obj and isinstance(obj["outputs"], list):
        for out in obj["outputs"]:
            if isinstance(out, dict):
                for key in ("text", "content", "response"):
                    if key in out and isinstance(out[key], str) and out[key].strip() != "":
                        return out[key]
    # nested response object
    if "response" in obj and isinstance(obj["response"], dict):
        for key in ("text", "content"):
            if key in obj["response"] and isinstance(obj["response"][key], str) and obj["response"][key].strip() != "":
                return obj["response"][key]
    return ""

def call_ollama_streaming(endpoint: str, model: str, prompt: str, timeout=60):
    """
    Robust call for endpoints that may stream NDJSON fragments.
    Returns (extracted_text, raw_response_text).
    """
    payload = {"model": model, "prompt": prompt}
    headers = {"Content-Type": "application/json"}
    resp = requests.post(endpoint, json=payload, headers=headers, timeout=timeout, stream=False)
    raw = resp.text or ""

    fragments = []  # will collect fragments in order

    # Try parse the whole response as JSON first
    try:
        j = resp.json()
        # If it's a single object with direct text, extract and return
        text = _extract_from_obj(j)
        if text:
            # normalize whitespace, return
            return " ".join(text.split()), raw
        # if JSON but no text, fallback to stringified JSON
        return json.dumps(j), raw
    except JSONDecodeError:
        # Likely NDJSON or SSE-like stream. Parse line by line.
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            # remove SSE 'data:' prefix if present
            if line.startswith("data:"):
                line = line[len("data:"):].strip()
            # attempt to load JSON from the line
            try:
                obj = json.loads(line)
            except JSONDecodeError:
                # try to extract a {...} substring and parse that
                start = line.find("{")
                end = line.rfind("}")
                if start != -1 and end != -1 and end > start:
                    candidate = line[start:end+1]
                    try:
                        obj = json.loads(candidate)
                    except JSONDecodeError:
                        obj = None
                else:
                    obj = None

            if obj:
                # get fragment from object
                frag = _extract_from_obj(obj)
                if frag:
                    fragments.append(frag)
                # also check nested choices/outputs arrays for fragments
                # above function already checks those for the first non-empty fragment
            else:
                # no JSON object for this line; skip
                continue

    # join fragments found (preserving order), normalize whitespace and return
    if fragments:
        joined = "".join(fragments)  # fragments often include intended spacing
        # normalize spaces and newlines to single spaces, then strip
        normalized = " ".join(joined.split())
        return normalized, raw

    # last resort: return raw response trimmed
    return raw.strip(), raw

# -------------------------
# Main
# -------------------------
def main():

    SYNONYMS = {
    "payment": "Payment Issue",
    "payment issue": "Payment Issue",
    "billing": "Billing",
    "invoice": "Invoice",
    "bug": "Bug Report",
    "error": "Bug Report",
    "refund request": "Refund",
    "nda": "Legal",
    "contract document": "Contract",
    "security alert": "Security Alert",
    "data security": "Data Security",
    "hr": "HR Request",
    "human resources": "HR Request",
    "procurement policy": "Procurement Policy",
    # Add more mappings you see in raw_response -> gold label mismatches
    }

    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", default="input.csv", help="Input CSV path (id,text,category)")
    p.add_argument("--output", "-o", default="results.csv", help="Output CSV path")
    p.add_argument("--prompt-template-file", "-t", default=None, help="Optional file with prompt template. Use {labels} and {text}.")
    p.add_argument("--labels", "-l", default=None, help="Comma-separated list of labels to provide to model (overrides labels from data).")
    p.add_argument("--pause", type=float, default=0.2, help="Pause (seconds) between requests to avoid rate limits")
    args = p.parse_args()

    endpoint = os.environ.get("OLLAMA_ENDPOINT")
    if not endpoint:
        raise SystemExit("Set OLLAMA_ENDPOINT environment variable (e.g. export OLLAMA_ENDPOINT='http://localhost:11434/api/generate')")

    model = os.environ.get("MODEL")
    if not model:
        raise SystemExit("Set MODEL environment variable (e.g. export MODEL='your-model-name')")

    df = pd.read_csv(args.input, dtype=str).fillna("")
    required = {"id", "text", "category"}
    if not required.issubset(set(df.columns)):
        raise SystemExit(f"Input CSV must contain columns: {required}. Found: {list(df.columns)}")

    if args.labels:
        label_list = [s.strip() for s in args.labels.split(",") if s.strip()]
    else:
        label_list = sorted(df["category"].dropna().unique().tolist())

    if args.prompt_template_file:
        with open(args.prompt_template_file, "r", encoding="utf-8") as fh:
            template = fh.read()
    else:
        template = DEFAULT_PROMPT_TEMPLATE

    results = []
    total = len(df)
    correct_count = 0

    for idx, row in df.iterrows():
        row_id = row["id"]
        text = row["text"]
        gold = row["category"]

        prompt = template.format(labels=", ".join(label_list), text=text)

        try:
            extracted_text, raw_resp = call_ollama_streaming(endpoint, model, prompt)
        except Exception as e:
            pred_label = f"ERROR: {e}"
            correct = 0
            print(f"[{idx+1}/{total}] id={row_id} CALL ERROR: {e}")
            results.append({
                "id": row_id,
                "text": text,
                "category": gold,
                "prediction": pred_label,
                "correct": correct,
                "raw_response": str(e)
            })
            time.sleep(args.pause)
            continue

        mapped = pick_label_from_response(extracted_text, label_list, synonyms=SYNONYMS, fuzzy_thresh=0.6)
        correct = 1 if normalize_label(mapped) == normalize_label(gold) else 0
        if correct:
            correct_count += 1

        print(f"[{idx+1}/{total}] id={row_id} gold='{gold}' pred='{mapped}' correct={correct}")

        results.append({
            "id": row_id,
            "text": text,
            "category": gold,
            "prediction": mapped,
            "correct": correct,
            "raw_response": raw_resp
        })

        time.sleep(args.pause)

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output, index=False)

    # compute per-label metrics and confusion matrix
    labels = sorted(label_list)
    # mapping label -> index
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)
    conf = [[0]*n for _ in range(n)]  # conf[true][pred]

    tp = {lab:0 for lab in labels}
    fp = {lab:0 for lab in labels}
    fn = {lab:0 for lab in labels}

    for _, row in out_df.iterrows():
        true = row['category']
        pred = row['prediction']
        # If predicted value isn't one of label_list, treat as other/unseen
        if pred not in label_to_idx:
            # optionally map via synonyms reverse mapping
            pred_idx = None
        else:
            pred_idx = label_to_idx[pred]
        true_idx = label_to_idx.get(true, None)
        if true_idx is None:
            continue
        if pred_idx is None:
            # count as predicted "Other" (not in labels)
            # skip incrementing conf matrix but count as FN for true
            fn[true] += 1
        else:
            conf[true_idx][pred_idx] += 1
            if true == pred:
                tp[true] += 1
            else:
                fp[pred] += 1
                fn[true] += 1

    # calculate precision/recall/F1 per label
    print("\nPer-label metrics:")
    print(f"{'Label':30s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'Support':>8s}")
    for lab in labels:
        precision = tp[lab] / (tp[lab] + fp[lab]) if (tp[lab] + fp[lab])>0 else 0.0
        recall = tp[lab] / (tp[lab] + fn[lab]) if (tp[lab] + fn[lab])>0 else 0.0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
        support = tp[lab] + fn[lab]
        print(f"{lab:30s} {precision:6.2f} {recall:6.2f} {f1:6.2f} {support:8d}")

    # print confusion matrix (rows=true, cols=pred)
    print("\nConfusion matrix (rows=true, cols=pred):")
    print("," + ",".join(labels))
    for i, lab in enumerate(labels):
        row = ",".join(str(x) for x in conf[i])
        print(f"{lab},{row}")

    accuracy = 100.0 * out_df["correct"].astype(int).sum() / len(out_df)
    print(f"\nWrote {args.output}. Accuracy: {accuracy:.2f}% ({int(out_df['correct'].astype(int).sum())}/{len(out_df)})")

if __name__ == "__main__":
    main()
