"""
LLM-Based Contradiction Detection - EXPERIMENTAL/AUXILIARY

================================================================================
⚠️  CRITICAL: THIS MODULE IS NOT USED IN THE PRODUCTION PIPELINE ⚠️
================================================================================

STATUS: Experimental - For exploratory analysis and manual validation only

WHY NOT IN PRODUCTION PIPELINE:
1. Non-deterministic: Same input may produce different outputs across runs
2. No evidence grounding: LLM doesn't cite specific text spans
3. API dependency: Requires external service with inherent risks:
   - API failures (network, service downtime)
   - Rate limits (throttling, quotas)
   - Cost variability (expensive at scale)
4. Epistemic risk: 
   - Can hallucinate facts not present in text
   - Can misinterpret literary nuance (metaphor, irony, etc.)
   - May appear confident even when wrong
5. Uncalibrated confidence: Vote-based scores are heuristic, not probabilistic

APPROPRIATE USES:
✓ Manual spot-checking of edge cases (human-in-the-loop)
✓ Exploratory analysis during development
✓ Research into LLM reasoning capabilities
✓ Optional secondary validation (NOT primary decision)

INAPPROPRIATE USES:
✗ Primary decision maker for final predictions
✗ Production pipeline without human review
✗ Where deterministic/reproducible output is required
✗ Where evidence-based reasoning is mandated

ACCEPTABLE BECAUSE:
• This module is NOT used in decision.py
• Core pipeline uses symbolic reasoning only
• LLM is auxiliary/experimental tool
• Properly documented limitations

llm_auxiliary.py
"""

import hashlib
import os
from typing import Dict, Tuple, Optional, List

# ============================================================================
# LLM IMPLEMENTATION
# ============================================================================

def llm_raw(prompt: str) -> str:
    """
    Call LLM API (implementation placeholder).
    
    MUST BE IMPLEMENTED BEFORE USE.
    
    Implementation options:
    1. OpenAI: openai.ChatCompletion.create()
    2. Anthropic Claude: anthropic.messages.create()
    3. Local model: ollama, llama.cpp, etc.
    
    Risk factors:
    - API may be unavailable (network/service failure)
    - May hit rate limits (throttling)
    - Non-deterministic responses
    - Cost per call (can be expensive at scale)
    
    Returns:
        str: LLM response (should be "YES", "NO", or "UNCLEAR")
    
    Raises:
        NotImplementedError: If API not configured
        Exception: On API call failure (logged, returns "UNCLEAR")
    """
    try:
        # Example implementation (OpenAI)
        # Requires: pip install openai
        # Set env var: OPENAI_API_KEY
        import openai
        
        # Use lowest temperature for determinism (still not fully deterministic)
        response = openai.ChatCompletion.create(
            model="gpt-4",  # or "gpt-3.5-turbo" for lower cost
            messages=[
                {
                    "role": "system", 
                    "content": "You are a precise literary analyzer. Answer only YES, NO, or UNCLEAR. Never explain."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0.0,  # Most deterministic setting (still not guaranteed)
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        
        return response.choices[0].message.content.strip().upper()
    
    except ImportError:
        raise NotImplementedError(
            "LLM API not configured. Install required package:\n"
            "  pip install openai     # for OpenAI\n"
            "  pip install anthropic  # for Claude\n"
            "Set API key in environment variable."
        )
    except Exception as e:
        # Log but don't crash - degrade gracefully
        print(f"[WARNING] LLM call failed: {e}")
        print(f"[WARNING] Returning UNCLEAR - system continues without LLM")
        return "UNCLEAR"

# ============================================================================
# CACHE (Cost/Speed Optimization)
# ============================================================================

_llm_cache: Dict[str, str] = {}

def _cache_key(prompt: str) -> str:
    """
    Generate deterministic cache key from prompt.
    
    Uses MD5 hash to avoid key collisions while keeping cache size manageable.
    """
    return hashlib.md5(prompt.encode('utf-8')).hexdigest()

def llm_raw_cached(prompt: str) -> str:
    """
    Cached LLM call to avoid redundant API requests.
    
    Benefits:
    - Reduces API costs (same prompt → cached result)
    - Improves speed (no network call)
    - Partial determinism within session
    
    Limitations:
    - Cache not persistent across runs
    - Memory usage grows with unique prompts
    - Still non-deterministic for first call
    """
    cache_key = _cache_key(prompt)
    
    if cache_key in _llm_cache:
        return _llm_cache[cache_key]
    
    result = llm_raw(prompt)
    _llm_cache[cache_key] = result
    return result

def clear_cache():
    """Clear LLM cache (useful for testing)"""
    global _llm_cache
    _llm_cache = {}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def build_aliases(full_name: str) -> Tuple[str, ...]:
    """
    Create name variants for matching.
    
    Returns: (full_name, last_name)
    Conservative - only includes reliable aliases.
    """
    parts = full_name.lower().split()
    aliases = [full_name.lower()]
    if len(parts) > 1:
        aliases.append(parts[-1])  # Last name only
    return tuple(aliases)

def valid_alias_in_window(
    window: str,
    aliases: Tuple[str, ...],
    full_name: str,
    full_name_first_pos: int,
    window_start_pos: int
) -> bool:
    """
    Check if character mention in window is valid.
    
    Rules:
    - Full name is always valid
    - Partial alias (last name) only after full name introduction
    
    Prevents false matches like "Smith" before "John Smith" appears.
    """
    w = window.lower()
    
    # Full name always valid
    if full_name in w:
        return True
    
    # Partial alias only after introduction
    if full_name_first_pos == -1:
        return False
    
    if window_start_pos > full_name_first_pos:
        for a in aliases:
            if a != full_name and a in w:
                return True
    
    return False

# ============================================================================
# LLM PROMPTS (Constrained to Reduce Hallucination)
# ============================================================================

def llm_death_anchor(window: str, character: str) -> str:
    """
    Ask LLM if text establishes character death.
    
    WARNING: This is a HEURISTIC, not ground truth.
    
    Prompt design:
    - Constrained to YES/NO/UNCLEAR (reduces hallucination)
    - Explicit rules about certainty
    - No explanation requested (prevents rationalization)
    
    Known failure modes:
    - May hallucinate certainty
    - May miss literary nuance (metaphor, irony)
    - May confuse character with namesake
    """
    prompt = f"""Text:
\"\"\"{window}\"\"\"

Question:
Does this text clearly and unambiguously establish that {character} is dead
in the narrative (not merely believed, reported, or presumed dead)?

Rules:
- Answer ONLY with: YES, NO, or UNCLEAR
- YES means definitively dead in the narrative
- NO means alive or death not established
- UNCLEAR means ambiguous or insufficient information
- Do NOT explain your answer
- Do NOT add information beyond the text
"""
    return llm_raw_cached(prompt)

def llm_continuation(window: str, character: str) -> str:
    """
    Ask LLM if text shows character acting after they shouldn't.
    
    WARNING: This is a HEURISTIC, not ground truth.
    
    Known failure modes:
    - May misinterpret temporal progression
    - May confuse flashbacks with continuation
    - May miss implicit narrative closure
    """
    prompt = f"""Text:
\"\"\"{window}\"\"\"

Question:
Does this text describe {character} performing actions or existing
after the narrative implies they should no longer be present
(e.g., after death, permanent disappearance, or definitive conclusion)?

Rules:
- Answer ONLY with: YES, NO, or UNCLEAR
- YES means character is actively present/acting
- NO means no evidence of continuation
- UNCLEAR means ambiguous or insufficient information
- Do NOT explain your answer
- Do NOT add information beyond the text
"""
    return llm_raw_cached(prompt)

# ============================================================================
# HEURISTICS (Fallback for LLM Failure)
# ============================================================================

TEMPORAL_MARKERS = [
    "later", "after", "continued", "returned",
    "from then on", "eventually", "afterwards",
    "subsequently", "years later", "months later",
    "thereafter", "thenceforth", "from that day",
    "in the sequel", "in time", "by and by"
]

ACTION_VERBS = [
    "lived", "worked", "returned", "continued", "travelled",
    "acted", "went", "came", "arrived", "departed", "stayed",
    "fought", "spoke", "said", "did", "made", "took", "gave"
]

def has_temporal_marker(text: str) -> bool:
    """
    Check if text has temporal progression + action.
    
    Simple heuristic - not as sophisticated as LLM but deterministic.
    """
    text_lower = text.lower()
    has_temporal = any(marker in text_lower for marker in TEMPORAL_MARKERS)
    has_action = any(verb in text_lower for verb in ACTION_VERBS)
    return has_temporal and has_action

# ============================================================================
# VOTING LOGIC (Aggregation Heuristics)
# ============================================================================

def strong_anchor(votes: List[str]) -> bool:
    """
    Require strong consensus: ≥2 YES votes and YES > UNCLEAR.
    
    Conservative threshold to reduce false positives.
    """
    yes = votes.count("YES")
    unclear = votes.count("UNCLEAR")
    return yes >= 2 and yes > unclear

def moderate_signal(votes: List[str]) -> bool:
    """
    Moderate threshold: ≥1 YES, YES > NO, and YES ≥ UNCLEAR.
    """
    yes = votes.count("YES")
    no = votes.count("NO")
    unclear = votes.count("UNCLEAR")
    return yes >= 1 and yes > no and yes >= unclear

def backstory_death_veto(votes: List[str]) -> bool:
    """
    Check if backstory acknowledges death (veto contradiction).
    """
    return strong_anchor(votes)

def compute_confidence(death_votes: List[str], cont_votes: List[str]) -> int:
    """
    Heuristic confidence calculation.
    
    ⚠️  WARNING: This is NOT calibrated probability!
    
    Formula: (death_yes * 2) + cont_yes - cont_no
    
    Interpretation:
    - Use for RANKING only, not absolute certainty
    - Higher score = more votes supporting contradiction
    - NOT a percentage or probability
    
    This is acceptable for:
    ✓ Prioritizing which cases to review manually
    ✓ Relative comparison of contradiction strength
    
    This is NOT acceptable for:
    ✗ Claiming "80% confident" (not calibrated)
    ✗ Statistical inference (not probabilistic)
    """
    death_yes = death_votes.count("YES")
    cont_yes = cont_votes.count("YES")
    cont_no = cont_votes.count("NO")
    score = (death_yes * 2) + cont_yes - cont_no
    return max(0, score)

def find_strongest_evidence(
    windows_and_votes: List[Tuple[str, int, str]],
    target_vote: str = "YES"
) -> Optional[Tuple[str, int, str]]:
    """
    Find longest window with target vote.
    
    Heuristic: Longer windows likely have more context.
    """
    yes_items = [d for d in windows_and_votes if d[2] == target_vote]
    if not yes_items:
        return None
    return max(yes_items, key=lambda x: len(x[0].split()))

# ============================================================================
# MAIN DETECTION (EXPERIMENTAL)
# ============================================================================

def detect_contradiction_llm(
    character: str,
    novel_text: str,
    novel_windows_with_pos: List[Tuple[str, int]],
    back_windows_with_pos: List[Tuple[str, int]]
) -> Optional[dict]:
    """
    Detect contradiction using LLM voting.
    
    ⚠️  EXPERIMENTAL: Use with extreme caution.
    
    This function is:
    - Non-deterministic (LLM variability)
    - Not evidence-grounded (no text span citations)
    - Dependent on external API (failure risk)
    - Heuristic confidence (not calibrated)
    
    Appropriate use: Manual spot-checking only
    
    Returns:
        dict with contradiction details if found, else None
    """
    full_name = character.lower()
    aliases = build_aliases(full_name)
    full_name_first_pos = novel_text.lower().find(full_name)
    
    # Gate 1: Novel death evidence (LLM voting)
    novel_death_data = []
    for w, pos in novel_windows_with_pos:
        if len(w.split()) < 8:  # Skip very short windows
            continue
        if valid_alias_in_window(w, aliases, full_name, full_name_first_pos, pos):
            vote = llm_death_anchor(w, character)
            novel_death_data.append((w, pos, vote))
    
    novel_death_votes = [vote for _, _, vote in novel_death_data]
    
    if not strong_anchor(novel_death_votes):
        return None
    
    # Gate 2: Backstory continuation evidence (LLM voting)
    back_cont_data = []
    for w, pos in back_windows_with_pos:
        if len(w.split()) < 8:
            continue
        if valid_alias_in_window(w, aliases, full_name, full_name_first_pos, pos):
            vote = llm_continuation(w, character)
            back_cont_data.append((w, pos, vote))
    
    back_cont_votes = [vote for _, _, vote in back_cont_data]
    
    if not moderate_signal(back_cont_votes):
        return None
    
    # Gate 3: Temporal marker check (heuristic fallback)
    if not any(
        has_temporal_marker(w) and v == "YES"
        for w, _, v in back_cont_data
    ):
        return None
    
    # Gate 4: Backstory death veto (LLM voting)
    back_death_votes = []
    for w, pos in back_windows_with_pos:
        if len(w.split()) < 8:
            continue
        if valid_alias_in_window(w, aliases, full_name, full_name_first_pos, pos):
            back_death_votes.append(llm_death_anchor(w, character))
    
    if backstory_death_veto(back_death_votes):
        return None
    
    # Found contradiction (according to LLM voting)
    strongest_death = find_strongest_evidence(novel_death_data, "YES")
    strongest_continuation = find_strongest_evidence(back_cont_data, "YES")
    confidence = compute_confidence(novel_death_votes, back_cont_votes)
    
    return {
        "character": character,
        "death_window": strongest_death[0] if strongest_death else None,
        "death_pos": strongest_death[1] if strongest_death else None,
        "continuation_window": strongest_continuation[0] if strongest_continuation else None,
        "continuation_pos": strongest_continuation[1] if strongest_continuation else None,
        "confidence": confidence,  # NOT calibrated probability!
        "death_votes": novel_death_votes,
        "continuation_votes": back_cont_votes,
        "warnings": [
            "LLM-based detection - non-deterministic",
            "Confidence is heuristic, not probability",
            "Use for manual validation only",
            "NOT used in production pipeline"
        ]
    }

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def example_usage():
    """
    Example of how to use this module.
    
    ⚠️  NOT for production pipeline - exploratory use only.
    """
    print("=" * 70)
    print("EXAMPLE: LLM-Based Contradiction Detection (Experimental)")
    print("=" * 70)
    
    # Example windows (in reality, these come from text processing)
    novel_windows = [
        ("John Smith died in 1850 after a long illness.", 1000),
        ("The funeral was attended by hundreds of mourners.", 1500)
    ]
    
    backstory_windows = [
        ("Years later, John Smith traveled to France.", 500),
        ("He worked as a merchant in Paris for many years.", 600)
    ]
    
    # Detect contradiction (EXPERIMENTAL - NOT PRODUCTION)
    try:
        result = detect_contradiction_llm(
            character="John Smith",
            novel_text="... full novel text would go here ...",
            novel_windows_with_pos=novel_windows,
            back_windows_with_pos=backstory_windows
        )
        
        if result:
            print("\n✓ LLM detected potential contradiction:")
            print(f"  Confidence score: {result['confidence']} (heuristic, not probability)")
            print(f"  Death evidence: {result['death_window'][:100]}...")
            print(f"  Continuation: {result['continuation_window'][:100]}...")
            print(f"\n  Warnings:")
            for warning in result['warnings']:
                print(f"    - {warning}")
        else:
            print("\n✗ No contradiction detected by LLM")
            
    except NotImplementedError as e:
        print(f"\n✗ {e}")
        print("  → Implement llm_raw() with your preferred API")
    
    print("=" * 70)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("LLM AUXILIARY MODULE - EXPERIMENTAL")
    print("=" * 70)
    print("\n⚠️  WARNING: This module is NOT part of the core decision pipeline.")
    print("            It is provided for exploratory analysis only.\n")
    
    print("Why NOT in production:")
    print("  1. Non-deterministic (same input → different outputs)")
    print("  2. No evidence grounding (can't verify reasoning)")
    print("  3. API dependency (failures, rate limits, cost)")
    print("  4. Epistemic risk (hallucination, misinterpretation)")
    print("  5. Uncalibrated confidence (not probabilistic)\n")
    
    print("Appropriate uses:")
    print("  ✓ Manual spot-checking of edge cases")
    print("  ✓ Exploratory analysis during development")
    print("  ✓ Research into LLM reasoning")
    print("  ✓ Optional validation (NOT primary decision)\n")
    
    print("To use this module:")
    print("  1. Implement llm_raw() with your LLM API of choice")
    print("  2. Set API key in environment variable")
    print("  3. Run example_usage() to test")
    print("  4. Use ONLY for manual validation, NOT production\n")
    
    print("=" * 70)
    
    # Uncomment to run example:
    # example_usage()