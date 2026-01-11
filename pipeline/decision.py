"""
FIX: decision.py Input Mismatch

PROBLEM: 
- embeddings.py writes to retrieval_features.json
- decision.py reads from scores.csv
- These don't match!

SOLUTION:
Either change decision.py to read retrieval_features.json,
OR add a step to embeddings.py to write scores.csv

Option 1: Update decision.py to read retrieval_features.json
"""

import json
import csv
import numpy as np
from typing import Dict, List

# ============================================================================
# FIXED: Load from retrieval_features.json instead of scores.csv
# ============================================================================

def load_similarity_scores_FIXED() -> Dict[str, dict]:
    """
    Load similarity scores from Phase 1 (embeddings.py)
    
    FIXED: Now reads from retrieval_features.json which is what embeddings.py actually outputs
    """
    scores = {}
    
    # Check which file exists
    import os
    retrieval_json = "results/retrieval_features.json"
    scores_csv = "results/scores.csv"
    
    if os.path.exists(retrieval_json):
        print(f"[INFO] Loading from {retrieval_json}")
        with open(retrieval_json, encoding="utf-8") as f:
            data = json.load(f)
            
        for entry in data:
            dataset_id = entry["dataset_id"]
            features = entry["features"]
            
            scores[dataset_id] = {
                "max_sim": features["max_similarity"],
                "mean_topk_sim": features["mean_similarity"],
                "novel_length": 0  # Not available in retrieval_features.json
            }
    
    elif os.path.exists(scores_csv):
        print(f"[INFO] Loading from {scores_csv}")
        with open(scores_csv) as f:
            for row in csv.DictReader(f):
                scores[row["dataset_id"]] = {
                    "max_sim": float(row["max_sim"]),
                    "mean_topk_sim": float(row["mean_topk_sim"]),
                    "novel_length": int(row.get("novel_length", 0)) if "novel_length" in row else 0
                }
    
    else:
        raise FileNotFoundError(
            "Neither retrieval_features.json nor scores.csv found! "
            "Run embeddings.py first."
        )
    
    return scores


# ============================================================================
# BETTER SOLUTION: Update embeddings.py to write scores.csv
# ============================================================================

# Add this to the END of embeddings.py main() function:

def write_scores_csv_from_retrieval_features():
    """
    Convert retrieval_features.json to scores.csv
    This ensures compatibility with decision.py
    """
    import json
    import csv
    import os
    
    retrieval_json = "results/retrieval_features.json"
    scores_csv = "results/scores.csv"
    
    if not os.path.exists(retrieval_json):
        print(f"[ERROR] {retrieval_json} not found")
        return
    
    print(f"[INFO] Converting {retrieval_json} to {scores_csv}...")
    
    with open(retrieval_json, encoding="utf-8") as f:
        data = json.load(f)
    
    with open(scores_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset_id", 
            "max_sim", 
            "mean_topk_sim", 
            "min_sim",
            "temporal_spread",
            "novel_length"
        ])
        writer.writeheader()
        
        for entry in data:
            features = entry["features"]
            writer.writerow({
                "dataset_id": entry["dataset_id"],
                "max_sim": features["max_similarity"],
                "mean_topk_sim": features["mean_similarity"],
                "min_sim": features["min_similarity"],
                "temporal_spread": features["temporal_spread"],
                "novel_length": 0  # Could calculate this if needed
            })
    
    print(f"[INFO] Wrote {len(data)} scores to {scores_csv}")


# ============================================================================
# COMPLETE FIXED decision.py
# ============================================================================

"""
Phase 2 Decision Engine - FIXED VERSION

Purpose: Use similarity scores to create predictions.csv for Phase 3
"""

import csv
import json
import os
import numpy as np
from typing import Dict, List

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Phase 2 configuration - similarity-based filtering"""
    
    # Similarity thresholds for initial assessment
    SIM_EXTREME_LOW = 0.25
    SIM_LOW = 0.40
    SIM_MEDIUM = 0.55
    SIM_HIGH = 0.70
    
    # Adaptive threshold modifiers
    ADAPTIVE_ENABLED = True
    NOVEL_LENGTH_PERCENTILES = [0.33, 0.67]

# ============================================================================
# LOAD SIMILARITY SCORES - FIXED
# ============================================================================

def load_similarity_scores() -> Dict[str, dict]:
    """
    Load similarity scores from Phase 1 (embeddings.py)
    
    FIXED: Now tries retrieval_features.json first, then scores.csv
    """
    scores = {}
    
    retrieval_json = "results/retrieval_features.json"
    scores_csv = "results/scores.csv"
    
    # Try retrieval_features.json first (what embeddings.py actually outputs)
    if os.path.exists(retrieval_json):
        print(f"[INFO] Loading from {retrieval_json}")
        with open(retrieval_json, encoding="utf-8") as f:
            data = json.load(f)
            
        for entry in data:
            dataset_id = entry["dataset_id"]
            features = entry["features"]
            
            scores[dataset_id] = {
                "max_sim": features["max_similarity"],
                "mean_topk_sim": features["mean_similarity"],
                "novel_length": 0
            }
        
        print(f"[INFO] Loaded {len(scores)} scores from retrieval_features.json")
    
    # Fallback to scores.csv
    elif os.path.exists(scores_csv):
        print(f"[INFO] Loading from {scores_csv}")
        with open(scores_csv) as f:
            for row in csv.DictReader(f):
                scores[row["dataset_id"]] = {
                    "max_sim": float(row["max_sim"]),
                    "mean_topk_sim": float(row["mean_topk_sim"]),
                    "novel_length": int(row.get("novel_length", 0)) if "novel_length" in row else 0
                }
        
        print(f"[INFO] Loaded {len(scores)} scores from scores.csv")
    
    else:
        raise FileNotFoundError(
            f"Neither {retrieval_json} nor {scores_csv} found!\n"
            "Run embeddings.py first to generate similarity scores."
        )
    
    return scores

# ============================================================================
# LOAD ALL TEST IDS - FIXED
# ============================================================================

def load_all_test_ids() -> set:
    """
    Load ALL test IDs from test.csv
    
    CRITICAL: This ensures we process ALL 80 test cases,
    even if some don't have similarity scores
    """
    test_ids = set()
    
    with open("data/test.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            test_ids.add(row["id"])
    
    print(f"[INFO] Loaded {len(test_ids)} test IDs from test.csv")
    return test_ids

# ============================================================================
# ADAPTIVE THRESHOLDS
# ============================================================================

class ThresholdAdapter:
    """Adjusts similarity thresholds based on novel characteristics"""
    
    def __init__(self, similarity_map: Dict[str, dict]):
        self.similarity_map = similarity_map
        self.novel_lengths = [
            v.get("novel_length", 0) 
            for v in similarity_map.values() 
            if v.get("novel_length", 0) > 0
        ]
        
        if self.novel_lengths and Config.ADAPTIVE_ENABLED:
            self.length_p33 = np.percentile(self.novel_lengths, 33)
            self.length_p67 = np.percentile(self.novel_lengths, 67)
        else:
            self.length_p33 = None
            self.length_p67 = None
    
    def adjust_thresholds(self, dataset_id: str, base_threshold: float) -> float:
        """Adjust threshold based on novel length"""
        if not Config.ADAPTIVE_ENABLED or not self.length_p33:
            return base_threshold
        
        data = self.similarity_map.get(dataset_id, {})
        novel_len = data.get("novel_length", 0)
        
        if novel_len == 0:
            return base_threshold
        
        if novel_len < self.length_p33:
            return base_threshold * 0.90
        elif novel_len > self.length_p67:
            return base_threshold * 1.10
        else:
            return base_threshold

# ============================================================================
# PHASE 2 PREDICTOR - FIXED
# ============================================================================

class Phase2Predictor:
    """
    Creates initial predictions based on similarity
    
    FIXED: Now processes ALL test cases from test.csv
    """
    
    def __init__(self):
        self.similarity_map = load_similarity_scores()
        self.all_test_ids = load_all_test_ids()
        self.threshold_adapter = ThresholdAdapter(self.similarity_map)
        
        # Validation check
        missing_scores = len(self.all_test_ids) - len(self.similarity_map)
        if missing_scores > 0:
            print(f"[WARN] {missing_scores} test cases have no similarity scores")
            print(f"       Total test cases: {len(self.all_test_ids)}")
            print(f"       Cases with scores: {len(self.similarity_map)}")
    
    def predict(self, dataset_id: str) -> dict:
        """Make Phase 2 prediction based on similarity"""
        sim_data = self.similarity_map.get(dataset_id)
        
        if not sim_data:
            # No similarity data - default to consistent
            return {
                "id": dataset_id,
                "prediction": 1,
                "confidence": 0.50,
                "phase2_signal": "no_similarity_data"
            }
        
        max_sim = sim_data["max_sim"]
        mean_sim = sim_data["mean_topk_sim"]
        combined = 0.7 * max_sim + 0.3 * mean_sim
        
        # Apply adaptive thresholds
        low_thresh = self.threshold_adapter.adjust_thresholds(dataset_id, Config.SIM_LOW)
        high_thresh = self.threshold_adapter.adjust_thresholds(dataset_id, Config.SIM_HIGH)
        
        # Classify based on similarity
        if combined < Config.SIM_EXTREME_LOW:
            signal = "extreme_low_similarity"
            prediction = 0
            confidence = 0.60
        elif combined < low_thresh:
            signal = "low_similarity"
            prediction = 1
            confidence = 0.40
        elif combined < high_thresh:
            signal = "medium_similarity"
            prediction = 1
            confidence = 0.50
        else:
            signal = "high_similarity"
            prediction = 1
            confidence = 0.70
        
        return {
            "id": dataset_id,
            "prediction": prediction,
            "confidence": confidence,
            "phase2_signal": signal,
            "similarity_score": round(combined, 3)
        }
    
    def predict_all(self) -> List[dict]:
        """
        Generate Phase 2 predictions for ALL test cases
        
        FIXED: Uses self.all_test_ids instead of self.similarity_map.keys()
        """
        predictions = []
        
        # Process ALL test IDs from test.csv
        for dataset_id in sorted(self.all_test_ids):
            predictions.append(self.predict(dataset_id))
        
        return predictions

# ============================================================================
# OUTPUT
# ============================================================================

def write_predictions(predictions: List[dict], path: str = "results/predictions.csv"):
    """Write predictions.csv for Phase 3"""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "prediction", "confidence"])
        writer.writeheader()
        
        for pred in predictions:
            writer.writerow({
                "id": pred["id"],
                "prediction": pred["prediction"],
                "confidence": pred["confidence"]
            })
    
    print(f"[INFO] Wrote {len(predictions)} predictions to: {path}")

def write_analysis(predictions: List[dict], path: str = "results/phase2_analysis.json"):
    """Write detailed Phase 2 analysis"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)
    
    print(f"[INFO] Wrote detailed analysis to: {path}")

def print_statistics(predictions: List[dict]):
    """Print Phase 2 statistics"""
    from collections import Counter
    
    print("\n" + "=" * 70)
    print("PHASE 2 STATISTICS (Similarity-Based)")
    print("=" * 70)
    
    # Signal distribution
    signals = Counter(p["phase2_signal"] for p in predictions)
    print(f"\nSimilarity signals:")
    for signal, count in signals.most_common():
        pct = count / len(predictions) * 100
        print(f"  {signal:30s}: {count:3d} ({pct:5.1f}%)")
    
    # Prediction distribution
    pred_counts = Counter(p["prediction"] for p in predictions)
    print(f"\nPhase 2 initial predictions:")
    print(f"  Contradict (0):  {pred_counts.get(0, 0):3d} ({pred_counts.get(0, 0)/len(predictions)*100:5.1f}%)")
    print(f"  Consistent (1):  {pred_counts.get(1, 0):3d} ({pred_counts.get(1, 0)/len(predictions)*100:5.1f}%)")
    
    # Confidence distribution
    confidences = [p["confidence"] for p in predictions]
    print(f"\nConfidence distribution:")
    print(f"  Mean:   {np.mean(confidences):.3f}")
    print(f"  Median: {np.median(confidences):.3f}")
    print(f"  Range:  {min(confidences):.3f} - {max(confidences):.3f}")
    
    print("\n" + "=" * 70)

# ============================================================================
# MAIN - FIXED
# ============================================================================

def main():
    print("=" * 70)
    print("PHASE 2: SIMILARITY-BASED PREDICTION (FIXED)")
    print("=" * 70)
    print("\nFIXES:")
    print("  ✓ Now reads from retrieval_features.json (what embeddings.py outputs)")
    print("  ✓ Processes ALL test IDs from test.csv")
    print("  ✓ Handles missing similarity scores gracefully")
    print(f"\nConfiguration:")
    print(f"  Adaptive thresholds: {Config.ADAPTIVE_ENABLED}")
    print(f"  Extreme low threshold: {Config.SIM_EXTREME_LOW}")
    print(f"  Low threshold: {Config.SIM_LOW}")
    print(f"  High threshold: {Config.SIM_HIGH}")
    
    # Initialize predictor
    predictor = Phase2Predictor()
    
    # Generate predictions
    predictions = predictor.predict_all()
    
    # Validation
    print(f"\n[INFO] Generated {len(predictions)} predictions")
    
    # Write outputs
    write_predictions(predictions)
    write_analysis(predictions)
    
    # Print statistics
    print_statistics(predictions)
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. Run: python pipeline/death_check.py")
    print("2. Run: python pipeline/timeline_reasoning.py")
    print("3. Run: python pipeline/final_merge.py")
    print("=" * 70)

if __name__ == "__main__":
    main()