#!/usr/bin/env python3
"""
Simplified Ollama batch classifier for CSV data.

Usage:
  export OLLAMA_ENDPOINT="http://localhost:11434/api/generate"
  export MODEL="granite4"
  python3 ollama_batch_classify.py --input benchmark_data.csv --output results.csv
"""

import os
import argparse
import time
import json
import requests
import pandas as pd
import difflib

DEFAULT_PROMPT = (
    "Classify this text into exactly ONE category.\n\n"
    "Categories:\n"
    "- Technical Support: bugs, errors, crashes, technical problems, system issues, "
    "connectivity problems, software malfunctions, error messages, things NOT WORKING as expected\n"
    "  Key: Something is BROKEN, FAILING, or producing ERRORS. Example: 'drag-and-drop doesn't work'\n\n"
    
    "- Billing: payments, invoices, refunds, subscriptions, charges, pricing, payment methods, "
    "plan changes (upgrades/downgrades), billing profile updates (VAT, tax info), promo codes, "
    "receipts, renewal reminders, billing ownership transfers\n"
    "  Key: Anything about MONEY, PAYMENT PLANS, SUBSCRIPTIONS, or BILLING INFORMATION. "
    "Even if it mentions 'account', if it's about billing info or plan changes, it's Billing.\n\n"
    
    "- Product Feedback: feature requests, suggestions, reviews, praise, complaints, "
    "user experience comments, UI/UX observations, performance issues (slow but working), "
    "enhancement ideas, customer service experiences, behavior preferences (e.g., 'should stay applied')\n"
    "  Key: OPINIONS and SUGGESTIONS about how the product works or could work better. "
    "If something works but the user wants it to behave differently, it's feedback.\n\n"
    
    "- Account Management: passwords, login credentials, user access permissions, "
    "account settings (non-billing), profile changes, user provisioning/deprovisioning, "
    "account ownership (non-billing), team member management, viewing account data (usage stats, activity logs)\n"
    "  Key: USER IDENTITY, ACCESS CONTROL, and viewing ACCOUNT-SPECIFIC DATA. "
    "NOT about billing info or plan changes.\n\n"
    
    "- General Inquiry: policy questions, company information, availability questions, "
    "service capabilities, deployment options, "
    "security questions, finding resources/downloads, API availability, data migration questions, "
    "testing/trial options, onboarding services, compliance questions, documentation locations\n"
    "  Key: INFORMATIONAL questions about what EXISTS, what's AVAILABLE, or CAPABILITIES.\n\n"
    
    "Text: {text}\n\n"
    "Return only the category name:"
)

SYNONYMS = {
    # Technical Support variations
    "technical support": "Technical Support",
    "tech support": "Technical Support",
    "technical": "Technical Support",
    "tech": "Technical Support",
    "bug": "Technical Support",
    "bugs": "Technical Support",
    "error": "Technical Support",
    "errors": "Technical Support",
    "crash": "Technical Support",
    "crashes": "Technical Support",
    "issue": "Technical Support",
    "issues": "Technical Support",
    "problem": "Technical Support",
    "problems": "Technical Support",
    "malfunction": "Technical Support",
    "not working": "Technical Support",
    "doesn't work": "Technical Support",
    "broken": "Technical Support",
    "fails": "Technical Support",
    "failure": "Technical Support",
    "timeout": "Technical Support",
    "connection": "Technical Support",
    "connectivity": "Technical Support",
    "api error": "Technical Support",
    "system error": "Technical Support",
    "not arriving": "Technical Support",
    "won't load": "Technical Support",
    "can't connect": "Technical Support",
    "unable to": "Technical Support",
    
    # Billing variations
    "billing": "Billing",
    "payment": "Billing",
    "payments": "Billing",
    "invoice": "Billing",
    "invoices": "Billing",
    "refund": "Billing",
    "refunds": "Billing",
    "subscription": "Billing",
    "subscriptions": "Billing",
    "charge": "Billing",
    "charges": "Billing",
    "charged": "Billing",
    "pricing": "Billing",
    "price": "Billing",
    "cost": "Billing",
    "costs": "Billing",
    "credit card": "Billing",
    "payment method": "Billing",
    "upgrade plan": "Billing",
    "upgrade to": "Billing",
    "downgrade": "Billing",
    "downgrade plan": "Billing",
    "downgrade to": "Billing",
    "cancel subscription": "Billing",
    "receipt": "Billing",
    "receipts": "Billing",
    "proration": "Billing",
    "enterprise plan": "Billing",
    "pro plan": "Billing",
    "team plan": "Billing",
    "plan": "Billing",
    "promo code": "Billing",
    "tax exemption": "Billing",
    "vat": "Billing",
    "vat id": "Billing",
    "billing profile": "Billing",
    "renewal": "Billing",
    "renewal reminders": "Billing",
    "billing ownership": "Billing",
    
    # Product Feedback variations
    "feedback": "Product Feedback",
    "feature request": "Product Feedback",
    "feature": "Product Feedback",
    "suggestion": "Product Feedback",
    "suggestions": "Product Feedback",
    "review": "Product Feedback",
    "complaint": "Product Feedback",
    "complaints": "Product Feedback",
    "praise": "Product Feedback",
    "love": "Product Feedback",
    "hate": "Product Feedback",
    "improvement": "Product Feedback",
    "enhance": "Product Feedback",
    "enhancement": "Product Feedback",
    "ux": "Product Feedback",
    "ui": "Product Feedback",
    "user interface": "Product Feedback",
    "user experience": "Product Feedback",
    "slow": "Product Feedback",
    "slowly": "Product Feedback",
    "loads slowly": "Product Feedback",
    "confusing": "Product Feedback",
    "excellent": "Product Feedback",
    "great": "Product Feedback",
    "appreciate": "Product Feedback",
    "customer service": "Product Feedback",
    "experience": "Product Feedback",
    "should": "Product Feedback",
    "should stay": "Product Feedback",
    "expires too soon": "Product Feedback",
    
    # Account Management variations
    "account management": "Account Management",
    "account": "Account Management",
    "password": "Account Management",
    "passwords": "Account Management",
    "reset password": "Account Management",
    "login": "Account Management",
    "log in": "Account Management",
    "username": "Account Management",
    "user access": "Account Management",
    "access": "Account Management",
    "permissions": "Account Management",
    "add user": "Account Management",
    "remove user": "Account Management",
    "delete account": "Account Management",
    "change email": "Account Management",
    "profile": "Account Management",
    "sso": "Account Management",
    "single sign-on": "Account Management",
    "ownership": "Account Management",
    "transfer account": "Account Management",
    "setup sso": "Account Management",
    "setup single sign-on": "Account Management",
    "enable two-factor": "Account Management",
    "enable 2fa": "Account Management",
    "usage statistics": "Account Management",
    "account usage": "Account Management",
    "activity logs": "Account Management",
    "user activity": "Account Management",
    "view account": "Account Management",
    "check user": "Account Management",
    
    # General Inquiry variations
    "general inquiry": "General Inquiry",
    "inquiry": "General Inquiry",
    "question": "General Inquiry",
    "questions": "General Inquiry",
    "info": "General Inquiry",
    "information": "General Inquiry",
    "general": "General Inquiry",
    "business hours": "General Inquiry",
    "office": "General Inquiry",
    "location": "General Inquiry",
    "policy": "General Inquiry",
    "policies": "General Inquiry",
    "compliance": "General Inquiry",
    "gdpr": "General Inquiry",
    "documentation": "General Inquiry",
    "docs": "General Inquiry",
    "api docs": "General Inquiry",
    "api documentation": "General Inquiry",
    "where can i find": "General Inquiry",
    "where to find": "General Inquiry",
    "where can i download": "General Inquiry",
    "plan comparison": "General Inquiry",
    "difference between": "General Inquiry",
    "discount": "General Inquiry",
    "student discount": "General Inquiry",
    "referral": "General Inquiry",
    "training": "General Inquiry",
    "webinar": "General Inquiry",
    "hiring": "General Inquiry",
    "jobs": "General Inquiry",
    "careers": "General Inquiry",
    "available": "General Inquiry",
    "availability": "General Inquiry",
    "do you offer": "General Inquiry",
    "can i use": "General Inquiry",
    "can i test": "General Inquiry",
    "can i migrate": "General Inquiry",
    "is there": "General Inquiry",
    "is there a": "General Inquiry",
    "do you provide": "General Inquiry",
    "do you send": "General Inquiry",
    "how secure": "General Inquiry",
    "api sandbox": "General Inquiry",
    "offline": "General Inquiry",
    "onboarding": "General Inquiry",
    "on-premise": "General Inquiry",
    "deployment": "General Inquiry",
    "premium features": "General Inquiry",
    "mobile version": "General Inquiry",
    "public api": "General Inquiry",
    "migrate data": "General Inquiry",
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
    
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        prompt = DEFAULT_PROMPT.format(labels=", ".join(labels), text=row.text)
        
        try:
            pred_text, raw = call_ollama(endpoint, model, prompt)
            pred_label = match_label(pred_text, labels, SYNONYMS)
            is_correct = 1 if str(pred_label).lower() == str(row.category).lower() else 0
            correct += is_correct
            
            print(f"[{idx}/{len(df)}] id={row.id} gold='{row.category}' pred='{pred_label}' ✓={is_correct}")
            
            results.append({
                "id": row.id,
                "text": row.text,
                "category": row.category,
                "prediction": pred_label,
                "correct": is_correct,
                "raw_response": raw
            })
        except Exception as e:
            print(f"[{idx}/{len(df)}] ERROR: {e}")
            results.append({
                "id": row.id,
                "text": row.text,
                "category": row.category,
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