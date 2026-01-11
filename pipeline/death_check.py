"""
Death Contradiction Detector - BALANCED VERSION

Balanced between catching real contradictions and avoiding false positives
"""

import os
import csv
import json
import re
from typing import Tuple, List, Optional

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = "data"
NOVELS_DIR = os.path.join(DATA_DIR, "novels")
TRAIN_CSV = os.path.join(DATA_DIR, "test.csv")
PHASE2_PRED = "results/predictions.csv"
OUT_PATH = "results/death_contradictions.json"

# LOWER THRESHOLDS - More permissive
MIN_DEATH_CONFIDENCE = 30
MIN_CONTINUATION_CONFIDENCE = 20
MIN_OVERALL_CONFIDENCE = 25

STRONG_DEATH_KEYWORDS = [
    "died", "dead", "killed", "executed", "murdered",
    "perished", "slain", "buried", "funeral", "grave",
    "corpse", "body lay", "lifeless", "expired",
    "met his end", "met her end", "ceased to live",
    "breathed his last", "breathed her last", "passed away"
]

NEGATIONS = [
    "not", "never", "didn't", "did not", "no longer", 
    "barely", "hardly", "wasn't", "weren't"
]

RESURRECTION_MARKERS = [
    "resurrected", "came back to life", "wasn't really dead",
    "thought to be dead", "presumed dead", "faked his death",
    "faked her death", "survived", "turned out to be alive"
]

TEMPORAL_MARKERS = [
    "later", "after", "afterwards", "subsequently", "then",
    "years later", "months later", "days later", "eventually", 
    "continued", "from then on", "thereafter",
    "following", "next", "soon after"
]

ACTION_VERBS = [
    "returned", "went", "came", "traveled", "travelled",
    "worked", "lived", "spoke", "said", "fought", "helped",
    "escaped", "arrived", "departed", "met", "walked", "ran",
    "built", "married", "founded", "created", "established"
]

# ============================================================================
# TEXT PROCESSING
# ============================================================================

def normalize(text: str) -> str:
    return (
        text.lower()
        .replace(".txt", "")
        .replace("the", "")
        .replace(" ", "")
    )

def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    return " ".join(text.split()).strip()

def build_aliases(name: str) -> Tuple[str, ...]:
    parts = name.lower().split()
    aliases = [name.lower()]
    
    if len(parts) > 1:
        last_name = parts[-1]
        if len(last_name) > 3:
            aliases.append(last_name)
    
    if len(parts) > 1:
        first_name = parts[0]
        if len(first_name) > 3:
            aliases.append(first_name)
    
    return tuple(set(aliases))

def find_first_introduction(text: str, full_name: str) -> int:
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
# SIMPLIFIED VICTIM CHECK
# ============================================================================

def character_is_victim_not_perpetrator(
    sentence: str, 
    character_name: str,
    death_keyword: str
) -> bool:
    """Simplified victim check - only reject obvious perpetrators"""
    sent_lower = sentence.lower()
    char_lower = character_name.lower()
    keyword = death_keyword.lower()
    
    char_pos = sent_lower.find(char_lower)
    keyword_pos = sent_lower.find(keyword)
    
    if char_pos == -1 or keyword_pos == -1:
        return False
    
    # ONLY reject clear accusation patterns
    if char_pos < keyword_pos:
        between = sent_lower[char_pos:keyword_pos]
        # Only reject if very clear accusation
        if "—you" in between or "- you" in between or ", you" in between:
            return False
    
    # Accept everything else (be permissive)
    return True

# ============================================================================
# SIMPLIFIED DEATH DETECTION
# ============================================================================

def check_death_window(
    window: str, 
    aliases: Tuple[str, ...],
    character_full_name: str
) -> Tuple[str, int, bool]:
    """Simplified death detection - fewer rejections"""
    w_lower = window.lower()
    
    if not any(alias in w_lower for alias in aliases):
        return ("NO", 0, False)
    
    # Check resurrection
    for marker in RESURRECTION_MARKERS:
        if marker in w_lower:
            return ("NO", 0, True)
    
    # Check negations
    for negation in NEGATIONS:
        if negation in w_lower:
            for keyword in STRONG_DEATH_KEYWORDS:
                if keyword in w_lower:
                    neg_pos = w_lower.find(negation)
                    key_pos = w_lower.find(keyword)
                    if abs(neg_pos - key_pos) < 50:
                        return ("NO", 0, False)
    
    # Count death signals
    strong_death = sum(1 for k in STRONG_DEATH_KEYWORDS if k in w_lower)
    
    if strong_death == 0:
        return ("NO", 0, False)
    
    # Split into sentences
    sentences = []
    for sep in ['. ', '! ', '? ', '." ', '!" ', '?" ']:
        if sep in window:
            window = window.replace(sep, '|||')
    sentences = [s.strip() for s in window.split('|||') if s.strip()]
    
    valid_death_found = False
    
    for sentence in sentences:
        sent_lower = sentence.lower()
        
        # Character in sentence?
        if character_full_name.lower() not in sent_lower:
            continue
        
        # ONLY reject the most obvious false positives:
        
        # 1. "whose death was wished/intended"
        if "whose death was" in sent_lower and ("wished" in sent_lower or "intended" in sent_lower or "sought" in sent_lower):
            continue
        
        # 2. Character explicitly among "the living"
        if "the living" in sent_lower:
            continue
        
        # 3. Obvious accusation: "Character—you killed"
        if "—you" in sent_lower or "- you" in sent_lower:
            char_pos = sent_lower.find(character_full_name.lower())
            you_pos = sent_lower.find("you")
            if char_pos != -1 and you_pos != -1 and 0 < you_pos - char_pos < 10:
                continue
        
        # Find death keyword
        death_in_sentence = None
        for kw in STRONG_DEATH_KEYWORDS:
            if kw in sent_lower:
                death_in_sentence = kw
                break
        
        if not death_in_sentence:
            continue
        
        # Basic victim check (only rejects clear perpetrators)
        if not character_is_victim_not_perpetrator(sentence, character_full_name, death_in_sentence):
            continue
        
        # Found valid death
        valid_death_found = True
        break
    
    if not valid_death_found:
        return ("NO", 0, False)
    
    # Calculate strength
    strength = strong_death * 4
    if strong_death >= 2:
        strength += 2
    
    strength = max(0, min(10, strength))
    
    if strength >= 5:
        return ("YES", strength, False)
    elif strength >= 3:
        return ("UNCLEAR", strength, False)
    else:
        return ("NO", strength, False)

def novel_has_strong_death_evidence(
    novel_text: str, 
    aliases: Tuple[str, ...], 
    full_name_pos: int,
    character: str
) -> Tuple[bool, Optional[str], int, bool]:
    windows = get_sentence_windows(novel_text, aliases, full_name_pos)
    
    votes = []
    evidence_windows = []
    has_fake_death = False
    
    for window, pos in windows:
        vote, strength, is_fake = check_death_window(window, aliases, character)
        
        if is_fake:
            has_fake_death = True
        
        if vote == "YES":
            votes.append(vote)
            evidence_windows.append((window, strength))
    
    if has_fake_death:
        return (False, None, 0, True)
    
    if len(votes) >= 1:
        best_window, best_strength = max(evidence_windows, key=lambda x: x[1])
        total_strength = sum(s for _, s in evidence_windows)
        confidence = min(100, int((total_strength / len(evidence_windows)) * 12))
        
        return (True, best_window, confidence, False)
    
    return (False, None, 0, False)

# ============================================================================
# CONTINUATION DETECTION
# ============================================================================

def backstory_has_continuation(
    backstory: str, 
    aliases: Tuple[str, ...], 
    full_name_pos: int
) -> Tuple[bool, Optional[str], int]:
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
    
    continuation_windows = []
    
    for window in windows:
        w_lower = window.lower()
        
        # Explicit temporal markers
        has_temporal = any(marker in w_lower for marker in TEMPORAL_MARKERS)
        
        if has_temporal:
            temporal_count = sum(1 for m in TEMPORAL_MARKERS if m in w_lower)
            action_count = sum(1 for v in ACTION_VERBS if v in w_lower)
            
            confidence = 40 + (temporal_count + action_count) * 15
            confidence = min(85, confidence)
            
            continuation_windows.append((window, confidence))
            continue
        
        # Implicit patterns
        implicit_score = 0
        
        strong_continuation_verbs = ["returned", "escaped", "went", "came back", "fled", "departed", "left"]
        if any(v in w_lower for v in strong_continuation_verbs):
            implicit_score += 30
        
        temporal_phrases = ["onward", "since", "from", "until", "through", "during", "while", "when"]
        if any(p in w_lower for p in temporal_phrases):
            implicit_score += 25
        
        if re.search(r'\b1[5-9]\d{2}\b', w_lower):
            implicit_score += 20
        
        ongoing_verbs = ["lived", "worked", "stayed", "kept", "continued", "remained", "spent"]
        if any(v in w_lower for v in ongoing_verbs):
            implicit_score += 15
        
        if "first" in w_lower or "second" in w_lower or "third" in w_lower:
            implicit_score += 20
        
        past_tense_markers = ["had", "was", "were", "did", "made", "took", "gave", "saw", "met"]
        if any(v in w_lower for v in past_tense_markers):
            implicit_score += 10
        
        location_verbs = ["arrived", "traveled", "travelled", "moved", "went to", "reached"]
        if any(v in w_lower for v in location_verbs):
            implicit_score += 15
        
        descriptive_action = ["posing", "slipping", "noticing", "learning", "discovering", "finding"]
        if any(v in w_lower for v in descriptive_action):
            implicit_score += 10
        
        if implicit_score >= 20:
            confidence = min(75, 35 + implicit_score)
            continuation_windows.append((window, confidence))
    
    if continuation_windows:
        best, conf = max(continuation_windows, key=lambda x: x[1])
        return (True, best, conf)
    
    return (False, None, 0)

def backstory_acknowledges_death(
    backstory: str, 
    aliases: Tuple[str, ...], 
    full_name_pos: int
) -> bool:
    windows = get_sentence_windows(backstory, aliases, full_name_pos)
    
    if not windows:
        return False
    
    yes_votes = 0
    
    for window, pos in windows:
        vote, _, _ = check_death_window(window, aliases, aliases[0])
        if vote == "YES":
            yes_votes += 1
    
    return yes_votes >= 1

# ============================================================================
# MAIN
# ============================================================================

def load_novels():
    novels = {}
    for fname in os.listdir(NOVELS_DIR):
        if fname.lower().endswith(".txt"):
            key = normalize(fname)
            with open(
                os.path.join(NOVELS_DIR, fname),
                "r",
                encoding="utf-8",
                errors="ignore",
            ) as f:
                novels[key] = f.read()
    return novels

def main():
    print("="*70)
    print("DEATH CONTRADICTION DETECTOR - BALANCED")
    print("="*70)
    print(f"\nThresholds (PERMISSIVE):")
    print(f"  MIN_DEATH_CONFIDENCE:        {MIN_DEATH_CONFIDENCE}")
    print(f"  MIN_CONTINUATION_CONFIDENCE: {MIN_CONTINUATION_CONFIDENCE}")
    print(f"  MIN_OVERALL_CONFIDENCE:      {MIN_OVERALL_CONFIDENCE}")
    print()
    
    novels = load_novels()

    with open(PHASE2_PRED) as f:
        preds = list(csv.DictReader(f))

    train_rows = {}
    with open(TRAIN_CSV) as f:
        for r in csv.DictReader(f):
            train_rows[r["id"]] = r

    results = []
    skipped = {
        "resurrection": 0,
        "no_death": 0,
        "backstory_ack": 0,
        "no_continuation": 0,
        "low_confidence": 0
    }
    
    print(f"[INFO] Processing {len(preds)} predictions...")
    
    for idx, r in enumerate(preds):
        if idx % 50 == 0:
            print(f"[INFO] Processed {idx}/{len(preds)}...")

        dataset_id = r["id"]
        row = train_rows.get(dataset_id)
        if not row:
            continue

        character = row["char"]
        aliases = build_aliases(character)
        backstory = row["content"]
        book_key = normalize(row["book_name"])

        novel_text = novels.get(book_key)
        if not novel_text:
            continue

        full_name_pos = find_first_introduction(novel_text, aliases[0])
        
        has_death, death_evidence, death_conf, has_resurrection = (
            novel_has_strong_death_evidence(novel_text, aliases, full_name_pos, character)
        )
        
        if has_resurrection:
            skipped["resurrection"] += 1
            continue
        
        if not has_death or death_conf < MIN_DEATH_CONFIDENCE:
            skipped["no_death"] += 1
            continue

        if backstory_acknowledges_death(backstory, aliases, full_name_pos):
            skipped["backstory_ack"] += 1
            continue

        has_cont, cont_evidence, cont_conf = backstory_has_continuation(
            backstory, aliases, full_name_pos
        )
        
        if not has_cont or cont_conf < MIN_CONTINUATION_CONFIDENCE:
            skipped["no_continuation"] += 1
            continue

        overall_conf = int(0.7 * death_conf + 0.3 * cont_conf)
        
        if overall_conf < MIN_OVERALL_CONFIDENCE:
            skipped["low_confidence"] += 1
            continue
        
        results.append({
            "dataset_id": dataset_id,
            "character": row["char"],
            "book": row["book_name"],
            "decision": "contradict",
            "confidence": overall_conf,
            "death_confidence": death_conf,
            "continuation_confidence": cont_conf,
            "reason": f"Novel: {character} died (conf: {death_conf}%). Backstory: continuation after death (conf: {cont_conf}%).",
            "novel_evidence": death_evidence[:400] + "..." if len(death_evidence) > 400 else death_evidence,
            "backstory_evidence": cont_evidence[:400] + "..." if len(cont_evidence) > 400 else cont_evidence
        })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Death contradictions found:  {len(results)}")
    print(f"\nSkip breakdown:")
    for reason, count in skipped.items():
        print(f"  {reason:20s}: {count}")
    print(f"\nSaved to: {OUT_PATH}")
    
    if results:
        print(f"\nAll results:")
        for r in results:
            print(f"  {r['dataset_id']:6s} - {r['character']:20s} ({r['confidence']}%)")
    
    print("="*70)

if __name__ == "__main__":
    main()