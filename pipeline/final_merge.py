"""
Final Merge - Fixed for Submission Format

Phase 4: Merge death and timeline contradictions into final predictions

Output format:
- Uses 0 (contradict) and 1 (consistent) instead of text labels
- Includes ALL backstories from train.csv
- Creates submission-ready format
"""

import json
import csv
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

DEATH_JSON = "results/death_contradictions.json"
TIMELINE_JSON = "results/timeline_inconsistencies.json"
TRAIN_CSV = "data/test.csv"
OUT_CSV = "results.csv"  # Changed to results.csv

PIPELINE_VERSION = "v4.0-final"

# ============================================================================
# UTILITIES
# ============================================================================

def load_json_map(path, key="dataset_id"):
    """Load JSON and return as dict keyed by dataset_id"""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            return {r[key]: r for r in data}
    except FileNotFoundError:
        print(f"[WARN] File not found: {path}")
        return {}

def load_train_data():
    """Load all training data with backstories"""
    train_data = {}
    with open(TRAIN_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            train_data[row["id"]] = {
                "id": row["id"],
                "book_name": row["book_name"],
                "character": row["char"],
                "backstory": row["content"]
            }
    return train_data

def format_evidence(entry):
    """Format evidence for display"""
    novel = entry.get("novel_evidence", "").strip()
    backstory = entry.get("backstory_evidence", "").strip()

    if novel and backstory:
        text = f"Novel: {novel} | Backstory: {backstory}"
    elif novel:
        text = f"Novel: {novel}"
    elif backstory:
        text = f"Backstory: {backstory}"
    else:
        text = ""

    return text[:500]  # Increased to 500 chars for more context

# ============================================================================
# MAIN MERGE
# ============================================================================

def main():
    print("=" * 70)
    print("FINAL MERGE - Phase 4")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    death_map = load_json_map(DEATH_JSON)
    timeline_map = load_json_map(TIMELINE_JSON)
    train_data = load_train_data()
    
    print(f"Death contradictions:    {len(death_map)}")
    print(f"Timeline contradictions: {len(timeline_map)}")
    print(f"Total test cases:        {len(train_data)}")
    
    # Deduplicate: if ID in both death and timeline, prefer death
    contradictions = {}
    
    # Add death contradictions first (higher priority)
    for dataset_id, data in death_map.items():
        contradictions[dataset_id] = {
            "type": "death",
            "confidence": data.get("confidence", data.get("death_confidence", 85)),
            "evidence": format_evidence(data),
            "novel_evidence": data.get("novel_evidence", ""),
            "backstory_evidence": data.get("backstory_evidence", "")
        }
    
    # Add timeline contradictions (only if not already in death)
    for dataset_id, data in timeline_map.items():
        if dataset_id not in contradictions:
            contradictions[dataset_id] = {
                "type": "timeline",
                "confidence": data.get("confidence", data.get("terminal_confidence", 65)),
                "evidence": format_evidence(data),
                "novel_evidence": data.get("novel_evidence", ""),
                "backstory_evidence": data.get("backstory_evidence", "")
            }
    
    print(f"\nAfter deduplication:")
    print(f"  Total unique contradictions: {len(contradictions)}")
    overlap = len(death_map) + len(timeline_map) - len(contradictions)
    if overlap > 0:
        print(f"  Overlapping cases (deduplicated): {overlap}")
    
    # Build final predictions
    final_predictions = []
    counts = {"death": 0, "timeline": 0, "consistent": 0}
    
    for dataset_id in sorted(train_data.keys()):
        row = train_data[dataset_id]
        
        if dataset_id in contradictions:
            # Contradiction found
            contradiction = contradictions[dataset_id]
            
            final_predictions.append({
                "id": dataset_id,
                "prediction": 1,  # 1 = contradict
                "book_name": row["book_name"],
                "character": row["character"],
                "backstory": row["backstory"],
                "confidence": contradiction["confidence"],
                "reasoning_type": contradiction["type"],
                "novel_evidence": contradiction.get("novel_evidence", "")[:400],
                "backstory_evidence": contradiction.get("backstory_evidence", "")[:400],
                "evidence_summary": contradiction["evidence"]
            })
            
            counts[contradiction["type"]] += 1
        else:
            # No contradiction found → consistent
            final_predictions.append({
                "id": dataset_id,
                "prediction": 0,  # 0 = consistent
                "book_name": row["book_name"],
                "character": row["character"],
                "backstory": row["backstory"],
                "confidence": 50,
                "reasoning_type": "no_contradiction_found",
                "novel_evidence": "",
                "backstory_evidence": "",
                "evidence_summary": ""
            })
            
            counts["consistent"] += 1
    
    # Write CSV output only
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "backstory", "prediction"])
        writer.writeheader()
        for pred in final_predictions:
            writer.writerow({
                "id": pred["id"],
                "backstory": pred["backstory"],
                "prediction": pred["prediction"]
            })
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL MERGE COMPLETE")
    print("=" * 70)
    print(f"Pipeline version: {PIPELINE_VERSION}")
    print(f"Total predictions: {len(final_predictions)}")
    print(f"\nBreakdown:")
    print(f"  Death contradictions:    {counts['death']}")
    print(f"  Timeline contradictions: {counts['timeline']}")
    print(f"  Consistent:              {counts['consistent']}")
    print(f"\nDistribution:")
    total = len(final_predictions)
    contradict_total = counts['death'] + counts['timeline']
    print(f"  Contradict (1): {contradict_total:3d} ({contradict_total/total*100:5.1f}%)")
    print(f"  Consistent (0): {counts['consistent']:3d} ({counts['consistent']/total*100:5.1f}%)")
    print(f"\nOutput files:")
    print(f"  CSV (with backstory): {OUT_CSV}")
    
    # Show sample predictions
    print(f"\nSample contradictions:")
    contradictions_list = [p for p in final_predictions if p["prediction"] == 1]
    for i, pred in enumerate(contradictions_list[:5], 1):
        print(f"\n  {i}. ID {pred['id']} - {pred['character']}")
        print(f"     Type: {pred['reasoning_type']}, Confidence: {pred['confidence']}%")
        print(f"     Backstory: {pred['backstory'][:100]}...")
        print(f"     Novel: {pred['novel_evidence'][:100]}...")
    
    print("\n" + "=" * 70)
    
    # Validation
    if len(final_predictions) != len(train_data):
        print(f"\n⚠️  WARNING: Prediction count mismatch!")
        print(f"   Expected: {len(train_data)}")
        print(f"   Got:      {len(final_predictions)}")
    else:
        print(f"\n✓ All {len(train_data)} test cases processed successfully!")

if __name__ == "__main__":
    main()