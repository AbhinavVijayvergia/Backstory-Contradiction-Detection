import os
import csv
import json
import hashlib
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional

# Import shared chunking logic
import sys
sys.path.insert(0, os.path.dirname(__file__))
from chunk import (
    chunk_text,
    normalize,
    load_and_chunk_novels,
    CHUNK_SIZE,
    OVERLAP,
    STEP
)

# -----------------------------
# Safety + performance settings
# -----------------------------
torch.set_grad_enabled(False)

# -----------------------------
# Paths & constants
# -----------------------------
DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "test.csv")
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Model configuration
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Retrieval parameters
TOP_K_INITIAL = 5       # First pass: high-similarity chunks
TOP_K_CONSTRAINT = 10   # Second pass: constraint-bearing chunks
TOP_K_TEMPORAL = 5      # Third pass: temporal spread
BATCH_SIZE = 32

# Constraint detection patterns (for feature extraction, not decisions)
CONSTRAINT_MARKERS = {
    "terminal": ["died", "dead", "killed", "executed", "imprisoned", "exiled"],
    "temporal": ["after", "later", "years later", "eventually", "subsequently"],
    "irreversible": ["never returned", "lost forever", "fate sealed"]
}

# -----------------------------
# Cache key generation
# -----------------------------
def get_cache_key(prefix: str) -> str:
    """Generate cache key encoding all parameters that affect output"""
    config = {
        "model": MODEL_NAME,
        "chunk_size": CHUNK_SIZE,
        "overlap": OVERLAP,
        "top_k_initial": TOP_K_INITIAL,
        "top_k_constraint": TOP_K_CONSTRAINT,
        "top_k_temporal": TOP_K_TEMPORAL,
    }
    config_str = json.dumps(config, sort_keys=True)
    hash_suffix = hashlib.md5(config_str.encode()).hexdigest()[:8]
    return f"{prefix}_{hash_suffix}"

# -----------------------------
# Novel chunk organization
# -----------------------------
def organize_chunks_by_book():
    """Organize chunks by book using shared chunk.py logic"""
    cache_path = os.path.join(CACHE_DIR, get_cache_key("chunks_metadata") + ".json")
    
    if os.path.exists(cache_path):
        print("[INFO] Loading chunk metadata from cache...")
        with open(cache_path) as f:
            return json.load(f)
    
    print("[INFO] Organizing chunks by book...")
    all_chunks = load_and_chunk_novels()
    
    # Organize by novel_key
    chunks_by_book = {}
    for book, novel_key, chunk_id, start, end, text in all_chunks:
        if novel_key not in chunks_by_book:
            chunks_by_book[novel_key] = []
        
        # Calculate relative position in novel
        total_length = end if chunk_id == 0 else None  # Will update this
        relative_pos = start / (end + CHUNK_SIZE * 10)  # Approximate
        
        chunks_by_book[novel_key].append({
            "chunk_id": chunk_id,
            "start": start,
            "end": end,
            "text": text,
            "relative_position": relative_pos,
        })
    
    # Update relative positions with actual total length
    for novel_key, chunks in chunks_by_book.items():
        max_end = max(c["end"] for c in chunks)
        for chunk in chunks:
            chunk["relative_position"] = chunk["start"] / max_end
    
    # Cache for future runs
    with open(cache_path, "w") as f:
        json.dump(chunks_by_book, f)
    
    return chunks_by_book

# -----------------------------
# Load test data
# -----------------------------
def load_test_rows():
    """Load test data with validation"""
    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "id": r["id"],
                "book_key": normalize(r["book_name"]),
                "book_name_raw": r["book_name"],
                "backstory": r["content"],
            })
    return rows

# -----------------------------
# Embedding with parameter-aware caching
# -----------------------------
def load_or_compute_embeddings(chunks_by_book, model):
    """Load cached embeddings or compute them with proper cache key"""
    cache_path = os.path.join(CACHE_DIR, get_cache_key("embeddings") + ".npz")
    
    if os.path.exists(cache_path):
        print("[INFO] Loading embeddings from cache...")
        data = np.load(cache_path, allow_pickle=True)
        return {key: data[key] for key in data.files}
    
    print("[INFO] Computing chunk embeddings (first run)...")
    chunk_embeddings = {}

    for book_key, chunks in chunks_by_book.items():
        texts = [c["text"] for c in chunks]
        chunk_embeddings[book_key] = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
    
    # Cache with parameter-aware key
    np.savez_compressed(cache_path, **chunk_embeddings)
    print(f"[INFO] Cached embeddings to {cache_path}")
    return chunk_embeddings

# -----------------------------
# Constraint-aware feature extraction
# -----------------------------
def extract_constraint_features(chunk_text: str) -> Dict:
    """Extract constraint-related features from chunk text"""
    text_lower = chunk_text.lower()
    
    features = {
        "has_terminal_marker": any(m in text_lower for m in CONSTRAINT_MARKERS["terminal"]),
        "has_temporal_marker": any(m in text_lower for m in CONSTRAINT_MARKERS["temporal"]),
        "has_irreversible_marker": any(m in text_lower for m in CONSTRAINT_MARKERS["irreversible"]),
        "terminal_marker_count": sum(text_lower.count(m) for m in CONSTRAINT_MARKERS["terminal"]),
    }
    
    return features

# -----------------------------
# Multi-pass adaptive retrieval
# -----------------------------
def retrieve_with_constraints(
    backstory: str,
    book_key: str,
    chunks_by_book: Dict,
    chunk_embeddings: Dict,
    model: SentenceTransformer
) -> Dict:
    """
    Multi-pass retrieval strategy:
    1. High-similarity chunks (semantic match)
    2. Constraint-bearing chunks (death, imprisonment, etc.)
    3. Temporal diversity (early/mid/late novel)
    
    IMPORTANT: This provides FEATURES, not decisions.
    """
    
    if book_key not in chunk_embeddings:
        raise ValueError(f"Book key not found: {book_key}")
    
    # Embed backstory
    back_vec = model.encode([backstory], normalize_embeddings=True)
    
    # Compute all similarities
    sims = cosine_similarity(back_vec, chunk_embeddings[book_key])[0]
    
    # ------------------------
    # Pass 1: High similarity
    # ------------------------
    top_sim_indices = set(np.argsort(sims)[-TOP_K_INITIAL:][::-1])
    
    # ------------------------
    # Pass 2: Constraint chunks
    # ------------------------
    # Score chunks by constraint density
    constraint_scores = []
    for idx, chunk in enumerate(chunks_by_book[book_key]):
        features = extract_constraint_features(chunk["text"])
        score = (
            features["terminal_marker_count"] * 2 +
            int(features["has_irreversible_marker"]) * 3 +
            int(features["has_temporal_marker"])
        )
        constraint_scores.append((idx, score))
    
    # Get top constraint chunks that aren't already selected
    constraint_scores.sort(key=lambda x: x[1], reverse=True)
    top_constraint_indices = set()
    for idx, score in constraint_scores:
        if score > 0 and idx not in top_sim_indices:
            top_constraint_indices.add(idx)
            if len(top_constraint_indices) >= TOP_K_CONSTRAINT:
                break
    
    # ------------------------
    # Pass 3: Temporal diversity
    # ------------------------
    # Ensure coverage across novel sections (early/mid/late)
    all_chunks = chunks_by_book[book_key]
    temporal_indices = set()
    
    # Divide into thirds
    early_cutoff = 0.33
    late_cutoff = 0.66
    
    sections = {
        "early": [i for i, c in enumerate(all_chunks) if c["relative_position"] < early_cutoff],
        "mid": [i for i, c in enumerate(all_chunks) if early_cutoff <= c["relative_position"] < late_cutoff],
        "late": [i for i, c in enumerate(all_chunks) if c["relative_position"] >= late_cutoff]
    }
    
    combined_existing = top_sim_indices | top_constraint_indices
    
    for section_name, section_indices in sections.items():
        # Get best similarity chunk from this section not already selected
        section_sims = [(i, sims[i]) for i in section_indices if i not in combined_existing]
        if section_sims:
            section_sims.sort(key=lambda x: x[1], reverse=True)
            for idx, _ in section_sims[:TOP_K_TEMPORAL]:
                temporal_indices.add(idx)
    
    # ------------------------
    # Combine all passes
    # ------------------------
    all_selected = top_sim_indices | top_constraint_indices | temporal_indices
    
    # Build results with metadata
    retrieved_chunks = []
    for idx in all_selected:
        chunk = all_chunks[idx]
        constraint_features = extract_constraint_features(chunk["text"])
        
        # Determine retrieval source
        sources = []
        if idx in top_sim_indices:
            sources.append("similarity")
        if idx in top_constraint_indices:
            sources.append("constraint")
        if idx in temporal_indices:
            sources.append("temporal_diversity")
        
        retrieved_chunks.append({
            "chunk_id": chunk["chunk_id"],
            "similarity": float(sims[idx]),
            "relative_position": chunk["relative_position"],
            "text_preview": chunk["text"][:300],
            "retrieval_sources": sources,
            **constraint_features
        })
    
    # Sort by similarity for consistency
    retrieved_chunks.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Aggregate features
    similarities = [c["similarity"] for c in retrieved_chunks]
    positions = [c["relative_position"] for c in retrieved_chunks]
    
    # Constraint-aware aggregations
    terminal_chunks = [c for c in retrieved_chunks if c["has_terminal_marker"]]
    late_terminal_chunks = [c for c in terminal_chunks if c["relative_position"] > 0.7]
    
    # Temporal coverage
    early_chunks = [c for c in retrieved_chunks if c["relative_position"] < 0.33]
    mid_chunks = [c for c in retrieved_chunks if 0.33 <= c["relative_position"] < 0.66]
    late_chunks = [c for c in retrieved_chunks if c["relative_position"] >= 0.66]
    
    return {
        "retrieved_chunks": retrieved_chunks,
        "retrieval_stats": {
            "total_retrieved": len(retrieved_chunks),
            "from_similarity": len(top_sim_indices),
            "from_constraints": len(top_constraint_indices),
            "from_temporal": len(temporal_indices),
        },
        "features": {
            # Similarity-based
            "max_similarity": float(max(similarities)),
            "mean_similarity": float(np.mean(similarities)),
            "min_similarity": float(min(similarities)),
            
            # Temporal distribution
            "temporal_spread": float(max(positions) - min(positions)),
            "earliest_position": float(min(positions)),
            "latest_position": float(max(positions)),
            "mean_position": float(np.mean(positions)),
            
            # Temporal coverage
            "early_chunk_count": len(early_chunks),
            "mid_chunk_count": len(mid_chunks),
            "late_chunk_count": len(late_chunks),
            "temporal_coverage_score": float(
                (len(early_chunks) > 0) + (len(mid_chunks) > 0) + (len(late_chunks) > 0)
            ) / 3.0,
            
            # Constraint markers
            "terminal_chunk_count": len(terminal_chunks),
            "late_terminal_chunk_count": len(late_terminal_chunks),
            "has_any_terminal": len(terminal_chunks) > 0,
            "has_late_terminal": len(late_terminal_chunks) > 0,
            
            # Temporal markers
            "temporal_marker_chunk_count": sum(
                1 for c in retrieved_chunks if c["has_temporal_marker"]
            ),
            
            # Position bias
            "late_novel_bias": float(np.mean([p > 0.7 for p in positions])),
        }
    }

# -----------------------------
# Main
# -----------------------------
def main():
    print(f"\n{'='*60}")
    print("CONSTRAINT-AWARE RETRIEVAL PIPELINE")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Model:            {MODEL_NAME}")
    print(f"  Chunk size:       {CHUNK_SIZE}")
    print(f"  Overlap:          {OVERLAP}")
    print(f"  Top-K (initial):  {TOP_K_INITIAL}")
    print(f"  Top-K (constr):   {TOP_K_CONSTRAINT}")
    print(f"  Top-K (temporal): {TOP_K_TEMPORAL}")
    print(f"{'='*60}\n")
    
    print("[INFO] Loading and organizing chunks...")
    chunks_by_book = organize_chunks_by_book()

    print("[INFO] Loading test data...")
    test_rows = load_test_rows()

    print("[INFO] Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    print("[INFO] Loading/computing embeddings...")
    chunk_embeddings = load_or_compute_embeddings(chunks_by_book, model)

    # ----------------------------------------
    # Process each test case
    # ----------------------------------------
    print("[INFO] Processing test cases...")
    results = []
    errors = []

    for idx, row in enumerate(test_rows, 1):
        try:
            retrieval_result = retrieve_with_constraints(
                backstory=row["backstory"],
                book_key=row["book_key"],
                chunks_by_book=chunks_by_book,
                chunk_embeddings=chunk_embeddings,
                model=model
            )
            
            results.append({
                "dataset_id": row["id"],
                **retrieval_result
            })
            
            if idx % 10 == 0:
                print(f"[INFO] Processed {idx}/{len(test_rows)}")
        
        except Exception as e:
            errors.append({
                "dataset_id": row["id"],
                "book_name": row["book_name_raw"],
                "error": str(e)
            })
            print(f"[ERROR] Failed on {row['id']}: {e}")

    # ----------------------------------------
    # Save results
    # ----------------------------------------
    out_path = "results/retrieval_features.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if errors:
        error_path = "results/retrieval_errors.json"
        with open(error_path, "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2)
        print(f"\n[WARN] {len(errors)} errors logged to {error_path}")

    # ----------------------------------------
    # Analysis
    # ----------------------------------------
    print(f"\n{'='*60}")
    print(f"RETRIEVAL SUMMARY")
    print(f"{'='*60}")
    print(f"Processed:              {len(results)}")
    print(f"Errors:                 {len(errors)}")
    
    if results:
        features_list = [r["features"] for r in results]
        stats_list = [r["retrieval_stats"] for r in results]
        
        print(f"\nRetrieval Coverage:")
        avg_total = np.mean([s["total_retrieved"] for s in stats_list])
        avg_sim = np.mean([s["from_similarity"] for s in stats_list])
        avg_const = np.mean([s["from_constraints"] for s in stats_list])
        avg_temp = np.mean([s["from_temporal"] for s in stats_list])
        print(f"  Avg total chunks:     {avg_total:.1f}")
        print(f"  Avg from similarity:  {avg_sim:.1f}")
        print(f"  Avg from constraints: {avg_const:.1f}")
        print(f"  Avg from temporal:    {avg_temp:.1f}")
        
        print(f"\nSimilarity Distribution:")
        max_sims = [f["max_similarity"] for f in features_list]
        print(f"  Mean:                 {np.mean(max_sims):.3f}")
        print(f"  Median:               {np.median(max_sims):.3f}")
        print(f"  Std Dev:              {np.std(max_sims):.3f}")
        
        print(f"\nConstraint Detection:")
        has_terminal = sum(f["has_any_terminal"] for f in features_list)
        has_late_terminal = sum(f["has_late_terminal"] for f in features_list)
        print(f"  Terminal markers:     {has_terminal} ({has_terminal/len(results)*100:.1f}%)")
        print(f"  Late terminal:        {has_late_terminal} ({has_late_terminal/len(results)*100:.1f}%)")
        
        print(f"\nTemporal Coverage:")
        coverage = [f["temporal_coverage_score"] for f in features_list]
        print(f"  Mean coverage:        {np.mean(coverage):.3f}")
        print(f"  Full coverage (3/3):  {sum(c == 1.0 for c in coverage)} cases")
    
    print(f"\nOutput: {out_path}")
    print(f"{'='*60}\n")
    
    print("=" * 60)
    print("IMPORTANT USAGE NOTES")
    print("=" * 60)
    print("1. Multi-pass retrieval strategy:")
    print("   • Pass 1: High semantic similarity")
    print("   • Pass 2: Constraint-bearing chunks (death, imprisonment)")
    print("   • Pass 3: Temporal diversity (early/mid/late)")
    print("2. These are FEATURES for reasoning pipelines, NOT decisions")
    print("3. Pass to death_detector.py and timeline_detector.py")
    print("4. Retrieval finds evidence; symbolic reasoning makes decisions")
    print("=" * 60)

if __name__ == "__main__":
    main()