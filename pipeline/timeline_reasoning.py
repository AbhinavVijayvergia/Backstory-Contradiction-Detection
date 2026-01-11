"""
Timeline Inconsistency Detector - FINAL COMPLETE VERSION

All fixes applied with stricter thresholds
"""

import os
import csv
import json
import re
from typing import List, Tuple, Optional
from dataclasses import dataclass

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = "data"
NOVELS_DIR = os.path.join(DATA_DIR, "novels")
TRAIN_CSV = os.path.join(DATA_DIR, "test.csv")
PHASE2_PRED = "results/predictions.csv"
OUT_PATH = "results/timeline_inconsistencies.json"

# STRICTER THRESHOLDS
MIN_TERMINAL_CONFIDENCE = 75
MIN_CONTINUATION_CONFIDENCE = 55
MIN_OVERALL_CONFIDENCE = 65

TEMPORAL_MARKERS = [
    "after", "later", "years later", "months later", "decades later",
    "eventually", "subsequently", "afterwards", "following",
    "in later years", "long after", "some time later",
    "from then on", "thereafter", "thenceforth",
    "years passed", "time passed", "much later", "then"
]

STRONG_TERMINAL_MARKERS = [
    "died", "dead", "killed", "executed", "murdered",
    "perished", "slain", "buried", "funeral",
    "imprisoned for life", "exiled forever",
    "vanished forever", "lost forever",
    "never returned", "never heard from again",
    "last seen", "fate sealed"
]

WEAK_TERMINAL_MARKERS = [
    "imprisoned", "exiled", "vanished", "lost",
    "disappeared", "fate unknown"
]

NEGATED_TERMINALS = [
    "believed dead", "thought dead", "rumored dead",
    "presumed dead", "assumed dead", "nearly died",
    "almost died", "escaped death", "survived",
    "not actually dead", "wasn't dead", "lived through",
    "came back", "returned alive", "turned out to be alive"
]

ACTION_VERBS = [
    "returned", "escaped", "traveled", "travelled",
    "worked", "lived", "fought", "helped",
    "formed", "adopted", "decided", "planned",
    "continued", "went", "came", "arrived", "built",
    "married", "founded", "created", "established",
    "spoke", "wrote", "met", "visited", "stayed"
]

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Evidence:
    text: str
    position: int
    confidence: int = 60
    marker_type: str = "unknown"

@dataclass
class TimelineAnalysis:
    has_contradiction: bool
    terminal_evidence: Optional[Evidence] = None
    continuation_evidence: Optional[Evidence] = None
    confidence: int = 0
    reason: str = ""

# ============================================================================
# TEXT PROCESSING
# ============================================================================

def normalize(text: str) -> str:
    return text.lower().replace(".txt", "").replace("the", "").replace(" ", "")

def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    return " ".join(text.split()).strip()

def build_aliases(name: str) -> Tuple[str, ...]:
    parts = name.lower().split()
    aliases = [name.lower()]
    
    if len(parts) > 1 and len(parts[-1]) > 3:
        aliases.append(parts[-1])
    
    if len(parts) > 1 and len(parts[0]) > 3:
        aliases.append(parts[0])
    
    return tuple(set(aliases))

def find_character_introduction(text: str, full_name: str) -> int:
    return text.lower().find(full_name.lower())

def split_sentences_robust(text: str) -> List[str]:
    text = text.replace("Mr.", "Mr").replace("Mrs.", "Mrs")
    text = text.replace("Dr.", "Dr").replace("Prof.", "Prof")
    
    sentences = re.split(r'([.!?])\s+', text)
    
    reconstructed = []
    i = 0
    while i < len(sentences):
        if i + 1 < len(sentences) and sentences[i+1] in '.!?':
            reconstructed.append(sentences[i] + sentences[i+1])
            i += 2
        else:
            reconstructed.append(sentences[i])
            i += 1
    
    return [s for s in reconstructed if len(s.strip()) > 10]

def get_sentence_windows(
    text: str, 
    aliases: Tuple[str, ...], 
    full_name_pos: int,
    window_size: int = 5
) -> List[Tuple[str, int]]:
    sentences = split_sentences_robust(text)
    windows = []
    full_name = aliases[0]
    
    for i in range(len(sentences)):
        current_sentence = sentences[i].lower()
        
        char_in_current = False
        
        for alias in aliases:
            if alias in current_sentence:
                is_partial = alias != full_name
                current_pos = sum(len(s) for s in sentences[:i])
                
                if is_partial and (full_name_pos == -1 or current_pos < full_name_pos):
                    continue
                
                char_in_current = True
                break
        
        if not char_in_current:
            continue
        
        start_idx = max(0, i - 2)
        end_idx = min(len(sentences), i + 3)
        
        window_sentences = sentences[start_idx:end_idx]
        window = " ".join(window_sentences)
        window_pos = sum(len(s) for s in sentences[:start_idx])
        
        if not any(abs(w_pos - window_pos) < 50 for _, w_pos in windows):
            windows.append((clean_text(window), window_pos))
    
    return windows

# ============================================================================
# VICTIM CHECK
# ============================================================================

def character_is_victim_not_perpetrator(
    sentence: str, 
    character_name: str,
    terminal_keyword: str
) -> bool:
    sent_lower = sentence.lower()
    char_lower = character_name.lower()
    keyword = terminal_keyword.lower()
    
    char_pos = sent_lower.find(char_lower)
    keyword_pos = sent_lower.find(keyword)
    
    if char_pos == -1 or keyword_pos == -1:
        return False
    
    # Accusation = perpetrator
    if char_pos < keyword_pos:
        between = sent_lower[char_pos:keyword_pos]
        if any(marker in between for marker in ["—you", "- you", ", you", ": you"]):
            return False
    
    # Passive voice = victim
    if keyword_pos > char_pos:
        between = sent_lower[char_pos:keyword_pos]
        if " was " in between or " were " in between or " had been " in between:
            return True
    
    # Character after keyword = victim
    if keyword_pos < char_pos:
        return True
    
    # No passive = perpetrator
    if keyword_pos > char_pos:
        between = sent_lower[char_pos:keyword_pos]
        if " was " not in between and " were " not in between:
            return False
    
    return True

# ============================================================================
# TERMINAL STATE DETECTION
# ============================================================================

def check_terminal_window(
    window: str,
    aliases: Tuple[str, ...],
    character_full_name: str
) -> Tuple[bool, Optional[Evidence]]:
    w_lower = window.lower()
    
    if not any(alias in w_lower for alias in aliases):
        return (False, None)
    
    # CRITICAL REJECTIONS
    
    # 1. Character is LIVING
    if "the living" in w_lower or "among the living" in w_lower or "to the living" in w_lower:
        for alias in aliases:
            if alias in w_lower:
                living_pos = w_lower.find("living")
                alias_pos = w_lower.find(alias)
                if abs(living_pos - alias_pos) < 50:
                    return (False, None)
    
    # 2. Negations
    for negation in NEGATED_TERMINALS:
        if negation in w_lower:
            return (False, None)
    
    # 3. Active patterns (character doing things)
    active_patterns = [
        "followed by", "accompanied by", "led by", "guided by",
        "along with", "together with", "landed at", "arrived at",
        "went to", "traveled to", "returned to", "came to",
        "speaking", "said", "told", "asked", "replied"
    ]
    
    for pattern in active_patterns:
        if pattern in w_lower:
            pattern_pos = w_lower.find(pattern)
            for alias in aliases:
                alias_pos = w_lower.find(alias)
                if alias_pos != -1 and abs(alias_pos - pattern_pos) < 20:
                    return (False, None)
    
    # 4. Substitution patterns
    if "take the place" in w_lower or "drew the corpse" in w_lower:
        return (False, None)
    
    # Split into sentences
    sentences = []
    for sep in ['. ', '! ', '? ', '." ', '!" ', '?" ']:
        if sep in window:
            window = window.replace(sep, '|||')
    sentences = [s.strip() for s in window.split('|||') if s.strip()]
    
    for sentence in sentences:
        sent_lower = sentence.lower()
        
        # Character in sentence?
        if not any(alias in sent_lower for alias in aliases):
            continue
        
        # CRITICAL: Reject "whose death was wished/intended/sought"
        intended_patterns = [
            "whose death was",
            "death was wished",
            "death was intended",
            "death was sought",
            "death was planned",
            "death was desired"
        ]
        
        if any(pattern in sent_lower for pattern in intended_patterns):
            continue  # Intended death, not actual death
        
        # Reject similes
        simile_markers = [" as the ", " like a ", " like the ", " as a "]
        is_simile = False
        
        for simile in simile_markers:
            if simile in sent_lower:
                simile_pos = sent_lower.find(simile)
                for marker in STRONG_TERMINAL_MARKERS:
                    if marker in sent_lower:
                        marker_pos = sent_lower.find(marker)
                        if 0 < (marker_pos - simile_pos) < 20:
                            is_simile = True
                            break
                if is_simile:
                    break
        
        if is_simile:
            continue
        
        # Reject observer patterns
        if any(pattern in sent_lower for pattern in ["near the bed", "watching", "beside the"]):
            continue
        
        # Reject plural corpses
        if "corpses" in sent_lower:
            continue
        
        # Find terminal marker
        found_terminal_marker = None
        
        for marker in STRONG_TERMINAL_MARKERS:
            if marker in sent_lower:
                found_terminal_marker = marker
                break
        
        if not found_terminal_marker:
            for marker in WEAK_TERMINAL_MARKERS:
                if marker in sent_lower:
                    if any(r in sent_lower for r in ["forever", "permanently", "never"]):
                        found_terminal_marker = marker
                        break
        
        if not found_terminal_marker:
            continue
        
        # Check what comes after marker
        marker_pos = sent_lower.find(found_terminal_marker)
        after_marker = sent_lower[marker_pos + len(found_terminal_marker):][:20]
        
        # Possessive after marker = perpetrator
        if any(poss in after_marker[:10] for poss in [" my ", " his ", " her ", " their "]):
            continue
        
        # Victim check
        if not character_is_victim_not_perpetrator(sentence, character_full_name, found_terminal_marker):
            continue
        
        # Must have character's full name
        if character_full_name.lower() not in sent_lower:
            continue
        
        # Calculate confidence
        confidence = 75
        
        return (True, Evidence(
            text=sentence,
            position=0,
            confidence=confidence,
            marker_type="strong_terminal"
        ))
    
    return (False, None)

def detect_terminal_state(
    novel_text: str,
    aliases: Tuple[str, ...],
    char_intro_pos: int,
    character_full_name: str
) -> Optional[Evidence]:
    windows = get_sentence_windows(novel_text, aliases, char_intro_pos)
    
    terminal_evidence = []
    
    for window, pos in windows:
        is_terminal, evidence = check_terminal_window(window, aliases, character_full_name)
        
        if is_terminal and evidence:
            evidence.position = pos
            terminal_evidence.append(evidence)
    
    if terminal_evidence:
        return max(terminal_evidence, key=lambda e: e.confidence)
    
    return None

# ============================================================================
# CONTINUATION DETECTION
# ============================================================================

def detect_continuation(
    backstory: str,
    aliases: Tuple[str, ...],
    char_intro_pos: int
) -> Optional[Evidence]:
    sentences = split_sentences_robust(backstory)
    
    if len(sentences) <= 2:
        windows = [backstory]
    else:
        windows = []
        for i in range(len(sentences)):
            start_idx = max(0, i - 1)
            end_idx = min(len(sentences), i + 2)
            window = " ".join(sentences[start_idx:end_idx])
            windows.append(window)
    
    continuation_evidence = []
    
    for window in windows:
        w_lower = window.lower()
        
        # Explicit temporal markers
        has_temporal = any(marker in w_lower for marker in TEMPORAL_MARKERS)
        
        if has_temporal:
            temporal_count = sum(1 for m in TEMPORAL_MARKERS if m in w_lower)
            action_count = sum(1 for a in ACTION_VERBS if a in w_lower)
            
            confidence = 40 + (temporal_count + action_count) * 12
            confidence = min(80, confidence)
            
            evidence = Evidence(
                text=window,
                position=0,
                confidence=confidence,
                marker_type="continuation"
            )
            continuation_evidence.append(evidence)
            continue
        
        # Implicit patterns
        implicit_score = 0
        
        if any(v in w_lower for v in ["returned", "escaped", "went", "came back", "fled"]):
            implicit_score += 30
        
        if any(p in w_lower for p in ["onward", "since", "from", "until", "through"]):
            implicit_score += 25
        
        if re.search(r'\b1[5-9]\d{2}\b', w_lower):
            implicit_score += 20
        
        if any(v in w_lower for v in ["lived", "worked", "stayed", "kept", "continued"]):
            implicit_score += 15
        
        if any(v in w_lower for v in ["married", "founded", "established", "built"]):
            implicit_score += 25
        
        if implicit_score >= 25:
            confidence = min(75, 30 + implicit_score)
            evidence = Evidence(
                text=window,
                position=0,
                confidence=confidence,
                marker_type="implicit_continuation"
            )
            continuation_evidence.append(evidence)
    
    if continuation_evidence:
        return max(continuation_evidence, key=lambda e: e.confidence)
    
    return None

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_timeline(
    character: str,
    backstory: str,
    novel_text: str
) -> TimelineAnalysis:
    aliases = build_aliases(character)
    char_intro_pos = find_character_introduction(novel_text, aliases[0])
    
    terminal = detect_terminal_state(novel_text, aliases, char_intro_pos, character)
    
    if not terminal or terminal.confidence < MIN_TERMINAL_CONFIDENCE:
        return TimelineAnalysis(
            has_contradiction=False,
            reason=f"No terminal (conf={terminal.confidence if terminal else 0})"
        )
    
    continuation = detect_continuation(backstory, aliases, char_intro_pos)
    
    if not continuation or continuation.confidence < MIN_CONTINUATION_CONFIDENCE:
        return TimelineAnalysis(
            has_contradiction=False,
            terminal_evidence=terminal,
            reason=f"No continuation (conf={continuation.confidence if continuation else 0})"
        )
    
    confidence = int(0.6 * terminal.confidence + 0.4 * continuation.confidence)
    
    if confidence < MIN_OVERALL_CONFIDENCE:
        return TimelineAnalysis(
            has_contradiction=False,
            terminal_evidence=terminal,
            continuation_evidence=continuation,
            confidence=confidence,
            reason=f"Overall too low ({confidence} < {MIN_OVERALL_CONFIDENCE})"
        )
    
    return TimelineAnalysis(
        has_contradiction=True,
        terminal_evidence=terminal,
        continuation_evidence=continuation,
        confidence=confidence,
        reason="Terminal state contradicts backstory"
    )

# ============================================================================
# I/O
# ============================================================================

def load_novels() -> dict:
    novels = {}
    for filename in os.listdir(NOVELS_DIR):
        if filename.lower().endswith(".txt"):
            path = os.path.join(NOVELS_DIR, filename)
            with open(path, encoding="utf-8", errors="ignore") as f:
                novels[normalize(filename)] = f.read()
    return novels

def load_predictions() -> List[dict]:
    with open(PHASE2_PRED) as f:
        return list(csv.DictReader(f))

def load_train_data() -> dict:
    train = {}
    with open(TRAIN_CSV) as f:
        for row in csv.DictReader(f):
            train[row["id"]] = row
    return train

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("TIMELINE INCONSISTENCY DETECTOR - FINAL")
    print("=" * 70)
    print(f"\nThresholds:")
    print(f"  MIN_TERMINAL_CONFIDENCE:     {MIN_TERMINAL_CONFIDENCE}")
    print(f"  MIN_CONTINUATION_CONFIDENCE: {MIN_CONTINUATION_CONFIDENCE}")
    print(f"  MIN_OVERALL_CONFIDENCE:      {MIN_OVERALL_CONFIDENCE}")
    print()
    
    novels = load_novels()
    predictions = load_predictions()
    train_data = load_train_data()
    
    results = []
    processed = 0
    
    print(f"Processing {len(predictions)} predictions...")
    
    for idx, pred in enumerate(predictions):
        if idx % 50 == 0:
            print(f"[INFO] Processed {idx}/{len(predictions)}...")
        
        dataset_id = pred["id"]
        
        row = train_data.get(dataset_id)
        if not row:
            continue
        
        novel_text = novels.get(normalize(row["book_name"]))
        if not novel_text:
            continue
        
        processed += 1
        
        analysis = analyze_timeline(
            character=row["char"],
            backstory=row["content"],
            novel_text=novel_text
        )
        
        if analysis.has_contradiction:
            results.append({
                "dataset_id": dataset_id,
                "character": row["char"],
                "book": row["book_name"],
                "decision": "contradict",
                "reasoning_type": "timeline_impossibility",
                "confidence": analysis.confidence,
                "terminal_confidence": analysis.terminal_evidence.confidence,
                "continuation_confidence": analysis.continuation_evidence.confidence,
                "novel_evidence": analysis.terminal_evidence.text[:400] + "..." 
                    if len(analysis.terminal_evidence.text) > 400 
                    else analysis.terminal_evidence.text,
                "backstory_evidence": analysis.continuation_evidence.text[:400] + "..."
                    if len(analysis.continuation_evidence.text) > 400
                    else analysis.continuation_evidence.text,
                "terminal_type": analysis.terminal_evidence.marker_type
            })
    
    results.sort(key=lambda x: x["confidence"], reverse=True)
    
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Processed:                {processed}")
    print(f"Timeline contradictions:  {len(results)}")
    print(f"\nSaved to: {OUT_PATH}")
    
    if results:
        print(f"\nAll results:")
        for r in results:
            print(f"  {r['dataset_id']:6s} - {r['character']:20s} ({r['confidence']}%)")
    
    print("=" * 70)

if __name__ == "__main__":
    main()