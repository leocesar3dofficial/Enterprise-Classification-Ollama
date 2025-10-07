#!/usr/bin/env python3
"""
Analyze classification results and show confusion matrix + misclassified examples.

Usage:
  python3 analyze_results.py --input results.csv
"""

import argparse
import pandas as pd
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="results.csv")
    args = parser.parse_args()
    
    # Load results
    df = pd.read_csv(args.input)
    
    # Get unique categories
    categories = sorted(df["category"].unique())
    
    # Build confusion matrix
    confusion = defaultdict(lambda: defaultdict(int))
    for _, row in df.iterrows():
        true_cat = row["category"]
        pred_cat = row["prediction"]
        confusion[true_cat][pred_cat] += 1
    
    # Print confusion matrix
    print("=" * 80)
    print("CONFUSION MATRIX")
    print("=" * 80)
    print("Rows = True Category | Columns = Predicted Category")
    print()
    
    # Header
    print(f"{'True \\ Predicted':<25}", end="")
    for cat in categories:
        print(f"{cat[:12]:>13}", end="")
    print()
    print("-" * 80)
    
    # Matrix rows
    for true_cat in categories:
        print(f"{true_cat:<25}", end="")
        for pred_cat in categories:
            count = confusion[true_cat][pred_cat]
            if count > 0:
                if true_cat == pred_cat:
                    print(f"\033[92m{count:>13}\033[0m", end="")  # Green for correct
                else:
                    print(f"\033[91m{count:>13}\033[0m", end="")  # Red for errors
            else:
                print(f"{'-':>13}", end="")
        print()
    
    print()
    
    # Show misclassified examples
    print("=" * 80)
    print("MISCLASSIFIED EXAMPLES")
    print("=" * 80)
    
    errors = df[df["correct"] == 0].copy()
    
    if len(errors) == 0:
        print("🎉 Perfect accuracy! No misclassifications.")
        return
    
    # Group by true category
    for true_cat in categories:
        cat_errors = errors[errors["category"] == true_cat]
        if len(cat_errors) == 0:
            continue
        
        print(f"\n{'─' * 80}")
        print(f"TRUE CATEGORY: {true_cat}")
        print(f"{'─' * 80}")
        
        for _, row in cat_errors.iterrows():
            print(f"\nID: {row['id']}")
            print(f"Text: {row['text'][:120]}{'...' if len(row['text']) > 120 else ''}")
            print(f"❌ Predicted as: \033[91m{row['prediction']}\033[0m")
            print(f"Raw response: {row['raw_response'][:100]}{'...' if len(str(row['raw_response'])) > 100 else ''}")
    
    print()
    
    # Summary statistics
    print("=" * 80)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    
    # Find most common confusions
    confusions_list = []
    for true_cat in categories:
        for pred_cat in categories:
            if true_cat != pred_cat and confusion[true_cat][pred_cat] > 0:
                confusions_list.append((
                    confusion[true_cat][pred_cat],
                    true_cat,
                    pred_cat
                ))
    
    confusions_list.sort(reverse=True)
    
    if confusions_list:
        print("\nMost Common Confusions:")
        for count, true_cat, pred_cat in confusions_list[:5]:
            print(f"  • {true_cat} → {pred_cat}: {count} times")
        
        print("\n💡 Recommendations:")
        
        # Analyze confusion patterns
        for count, true_cat, pred_cat in confusions_list[:3]:
            if "General Inquiry" in [true_cat, pred_cat]:
                print(f"  • Consider adding more distinctive keywords for 'General Inquiry'")
                print(f"    vs '{pred_cat if true_cat == 'General Inquiry' else true_cat}'")
            
            if "Billing" in [true_cat, pred_cat] and "Account Management" in [true_cat, pred_cat]:
                print(f"  • Clarify distinction between payment issues (Billing) and")
                print(f"    account settings (Account Management)")
            
            if "Technical Support" in [true_cat, pred_cat] and "Product Feedback" in [true_cat, pred_cat]:
                print(f"  • Distinguish between reporting bugs (Technical Support) and")
                print(f"    suggesting improvements (Product Feedback)")
    
    # Overall stats
    total = len(df)
    correct = df["correct"].sum()
    accuracy = 100.0 * correct / total
    
    print(f"\n📊 Overall Accuracy: {accuracy:.1f}% ({correct}/{total})")
    print(f"📊 Total Errors: {total - correct}")


if __name__ == "__main__":
    main()