#!/usr/bin/env python3
"""
Simplified Ollama batch classifier for CSV data.

Usage:
  export OLLAMA_ENDPOINT="http://localhost:11434/api/generate"
  export MODEL="your-model-name"
  python3 ollama_batch_classify.py --input enterprise_classification_dataset.csv --output results.csv
"""

import os
import argparse
import time
import json
import requests
import pandas as pd
import difflib

DEFAULT_PROMPT = (
    "You are a strict classifier.\n"
    "Choose exactly one label from the list below and output exactly that label and NOTHING ELSE\n"
    "(no punctuation, no explanation, no quotes, just the single label).\n\n"
    "Labels: {labels}\n\nText: {text}\n\nAnswer:"
)

SYNONYMS = {
    "payment": "Payment Issue",
    "billing": "Billing",
    "bug": "Bug Report",
    "error": "Bug Report",
    "refund request": "Refund",
    "nda": "Legal",
    "security alert": "Security Alert",
    "hr": "HR Request",
}


def call_ollama(endpoint, model, prompt, timeout=60):
    """Call Ollama API and extract response text."""
    payload = {"model": model, "prompt": prompt}
    resp = requests.post(endpoint, json=payload, timeout=timeout)
    raw = resp.text
    
    # Try parsing as single JSON
    try:
        data = resp.json()
        text = extract_text(data)
        return text, raw
    except json.JSONDecodeError:
        pass
    
    # Try parsing as NDJSON (line-by-line)
    fragments = []
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            line = line[5:].strip()
        try:
            obj = json.loads(line)
            text = extract_text(obj)
            if text:
                fragments.append(text)
        except json.JSONDecodeError:
            continue
    
    return " ".join(fragments) if fragments else raw.strip(), raw


def extract_text(obj):
    """Extract text from common JSON response structures."""
    if not isinstance(obj, dict):
        return str(obj)
    
    # Check common keys
    for key in ["response", "text", "output", "content", "answer"]:
        if key in obj and isinstance(obj[key], str) and obj[key].strip():
            return obj[key].strip()
    
    # Check choices array (OpenAI-style)
    if "choices" in obj and isinstance(obj["choices"], list) and obj["choices"]:
        choice = obj["choices"][0]
        if isinstance(choice, dict):
            for key in ["text", "message", "content"]:
                if key in choice and isinstance(choice[key], str):
                    return choice[key].strip()
    
    return ""


def match_label(response, labels, synonyms=None, threshold=0.6):
    """Match model response to one of the expected labels."""
    response = response.strip().lower()
    
    # Exact match
    for label in labels:
        if response == label.lower():
            return label
    
    # Substring match (prefer longer labels)
    for label in sorted(labels, key=len, reverse=True):
        if label.lower() in response:
            return label
    
    # Synonym match
    if synonyms:
        for syn, canonical in synonyms.items():
            if syn.lower() in response:
                return canonical
    
    # Fuzzy match
    match = difflib.get_close_matches(response, labels, n=1, cutoff=threshold)
    if match:
        return match[0]
    
    # Synonym fuzzy match
    if synonyms:
        syn_keys = list(synonyms.keys())
        match = difflib.get_close_matches(response, syn_keys, n=1, cutoff=threshold)
        if match:
            return synonyms[match[0]]
    
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="input.csv")
    parser.add_argument("--output", "-o", default="results.csv")
    parser.add_argument("--labels", "-l", help="Comma-separated labels (optional)")
    parser.add_argument("--pause", type=float, default=0.2, help="Pause between requests")
    args = parser.parse_args()
    
    # Get environment variables
    endpoint = os.environ.get("OLLAMA_ENDPOINT")
    model = os.environ.get("MODEL")
    if not endpoint or not model:
        raise SystemExit("Set OLLAMA_ENDPOINT and MODEL environment variables")
    
    # Load data
    df = pd.read_csv(args.input, dtype=str).fillna("")
    if not {"id", "text", "category"}.issubset(df.columns):
        raise SystemExit("CSV must have: id, text, category columns")
    
    # Get labels
    labels = [l.strip() for l in args.labels.split(",")] if args.labels else sorted(df["category"].unique())
    
    # Process rows
    results = []
    correct = 0
    
    for idx, row in df.iterrows():
        prompt = DEFAULT_PROMPT.format(labels=", ".join(labels), text=row["text"])
        
        try:
            pred_text, raw = call_ollama(endpoint, model, prompt)
            pred_label = match_label(pred_text, labels, SYNONYMS)
            is_correct = 1 if pred_label.lower() == row["category"].lower() else 0
            correct += is_correct
            
            print(f"[{idx+1}/{len(df)}] id={row['id']} gold='{row['category']}' pred='{pred_label}' ✓={is_correct}")
            
            results.append({
                "id": row["id"],
                "text": row["text"],
                "category": row["category"],
                "prediction": pred_label,
                "correct": is_correct,
                "raw_response": raw
            })
        except Exception as e:
            print(f"[{idx+1}/{len(df)}] ERROR: {e}")
            results.append({
                "id": row["id"],
                "text": row["text"],
                "category": row["category"],
                "prediction": f"ERROR: {e}",
                "correct": 0,
                "raw_response": str(e)
            })
        
        time.sleep(args.pause)
    
    # Save results
    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output, index=False)
    
    # Calculate metrics
    accuracy = 100.0 * correct / len(df)
    print(f"\n✓ Saved to {args.output}")
    print(f"✓ Accuracy: {accuracy:.2f}% ({correct}/{len(df)})")
    
    # Per-label metrics
    print("\nPer-label metrics:")
    for label in labels:
        subset = out_df[out_df["category"] == label]
        if len(subset) > 0:
            label_acc = 100.0 * subset["correct"].sum() / len(subset)
            print(f"  {label:30s}: {label_acc:5.1f}% ({subset['correct'].sum()}/{len(subset)})")


if __name__ == "__main__":
    main()