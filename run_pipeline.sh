#!/bin/bash

# ============================================================================
# Complete Pipeline Execution Script
# ============================================================================

echo "======================================================================"
echo "BACKSTORY CONTRADICTION DETECTION PIPELINE"
echo "======================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================================================
# PHASE 1: EMBEDDINGS & RETRIEVAL
# ============================================================================

echo "${GREEN}[PHASE 1]${NC} Embeddings & Retrieval"
echo "----------------------------------------------------------------------"
echo "Running: python pipeline/embeddings.py"
echo ""

python pipeline/embeddings.py

if [ $? -ne 0 ]; then
    echo "${RED}[ERROR]${NC} embeddings.py failed!"
    exit 1
fi

echo ""
echo "${GREEN}✓${NC} Phase 1 complete: retrieval_features.json generated"
echo ""

# ============================================================================
# PHASE 2: DECISION (SIMILARITY-BASED)
# ============================================================================

echo "${GREEN}[PHASE 2]${NC} Similarity-Based Decision"
echo "----------------------------------------------------------------------"
echo "Running: python pipeline/decision.py"
echo ""

python pipeline/decision.py

if [ $? -ne 0 ]; then
    echo "${RED}[ERROR]${NC} decision.py failed!"
    exit 1
fi

echo ""
echo "${GREEN}✓${NC} Phase 2 complete: predictions.csv generated"
echo ""

# ============================================================================
# PHASE 3A: DEATH CONTRADICTION DETECTION
# ============================================================================

echo "${GREEN}[PHASE 3A]${NC} Death Contradiction Detection"
echo "----------------------------------------------------------------------"
echo "Running: python pipeline/death_check.py"
echo ""

python pipeline/death_check.py

if [ $? -ne 0 ]; then
    echo "${RED}[ERROR]${NC} death_check.py failed!"
    exit 1
fi

echo ""
echo "${GREEN}✓${NC} Phase 3A complete: death_contradictions.json generated"
echo ""

# ============================================================================
# PHASE 3B: TIMELINE CONTRADICTION DETECTION
# ============================================================================

echo "${GREEN}[PHASE 3B]${NC} Timeline Contradiction Detection"
echo "----------------------------------------------------------------------"
echo "Running: python pipeline/timeline_reasoning.py"
echo ""

python pipeline/timeline_reasoning.py

if [ $? -ne 0 ]; then
    echo "${RED}[ERROR]${NC} timeline_reasoning.py failed!"
    exit 1
fi

echo ""
echo "${GREEN}✓${NC} Phase 3B complete: timeline_inconsistencies.json generated"
echo ""

# ============================================================================
# PHASE 4: FINAL MERGE
# ============================================================================

echo "${GREEN}[PHASE 4]${NC} Final Merge"
echo "----------------------------------------------------------------------"
echo "Running: python pipeline/final_merge.py"
echo ""

python pipeline/final_merge.py

if [ $? -ne 0 ]; then
    echo "${RED}[ERROR]${NC} final_merge.py failed!"
    exit 1
fi

echo ""
echo "${GREEN}✓${NC} Phase 4 complete: results.csv generated"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================

echo "======================================================================"
echo "${GREEN}PIPELINE COMPLETE${NC}"
echo "======================================================================"
echo ""
echo "Output files generated:"
echo "  1. results/retrieval_features.json       (Phase 1)"
echo "  2. results/predictions.csv               (Phase 2)"
echo "  3. results/death_contradictions.json     (Phase 3A)"
echo "  4. results/timeline_inconsistencies.json (Phase 3B)"
echo "  5. results.csv                           (Phase 4) ${GREEN}← SUBMISSION FILE${NC}"
echo ""

# Check final output
if [ -f "results.csv" ]; then
    # Count total lines (minus header)
    TOTAL=$(($(wc -l < results.csv) - 1))
    
    # Count contradictions (prediction = 1)
    CONTRADICTIONS=$(awk -F',' 'NR>1 && $NF==1 {count++} END {print count}' results.csv)
    CONSISTENT=$((TOTAL - CONTRADICTIONS))
    
    echo "Final Statistics:"
    echo "  Total predictions:  ${TOTAL}"
    echo "  Contradictions:     ${CONTRADICTIONS}"
    echo "  Consistent:         ${CONSISTENT}"
    echo ""
fi

echo "======================================================================"
echo "Next steps:"
echo "  1. Review results.csv for submission"
echo "  2. Check results/death_contradictions.json for details"
echo "  3. Check results/timeline_inconsistencies.json for details"
echo "======================================================================"