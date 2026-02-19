"""Experiment configurations for factor analysis and prompt testing."""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class FactorAxis:
    """Definition of a single factor axis for factorial analysis."""

    name: str
    values: List[str]
    baseline: Optional[str] = None  # Baseline value (usually generic/empty)


@dataclass
class PromptConfig:
    """Configuration for a single prompt experiment."""

    name: str
    prompt: str
    description: str = ""
    absorber_classes: Optional[List[str]] = None  # For absorber architecture
    target_indices: Optional[List[int]] = None  # Which classes to keep (for absorbers)


# Factor axes from Experiment 7 (original values)
# Note: Original Exp 7 used hierarchical control (taxonomy vs "a flower", others vs "a cowpea flower")
# We use true OFAT: all axes tested against "a flower" baseline
FACTOR_AXES = [
    FactorAxis(
        name="taxonomy",
        values=[
            "cowpea flower",      # species_common (best)
            "bean flower",        # genus_bean
            "pea flower",         # genus_pea
            "legume flower",      # family_common
            "black-eyed pea flower",  # alt_common
            "flower",             # baseline (generic)
            "vigna unguiculata flower",  # scientific (expected to fail)
            "crop flower",        # generic_crop (expected to fail)
        ],
        baseline="flower",
    ),
    FactorAxis(
        name="color",
        values=[
            "yellow",   # best
            "white",
            "cream",
            "",         # baseline (none)
            "purple",   # worst
        ],
        baseline="",
    ),
    FactorAxis(
        name="size",
        values=[
            "",      # baseline (none - best)
            "large",
            "small",
            "tiny",  # worst
        ],
        baseline="",
    ),
    FactorAxis(
        name="phenology",
        values=[
            "bud",           # best
            "open",
            "",              # baseline (none)
            "closed bud",
            "blooming",
            "in bloom",      # worst (expected to fail)
        ],
        baseline="",
    ),
    FactorAxis(
        name="negation",
        values=[
            "not a bud, not the green calyx, not a leaf",  # best (bud+calyx+leaf)
            "not a bud, not the green calyx",              # bud+calyx
            "not a leaf",                                  # leaf_only
            "not a bud",                                   # bud_only
            "not the green calyx",                         # calyx_only
            "",                                            # baseline (none)
        ],
        baseline="",
    ),
    FactorAxis(
        name="anatomy",
        values=[
            "with open petals",        # best (open_petals)
            "with visible petals",     # visible_petals
            "with petals and stamens", # petals+stamens
            "",                        # baseline (none)
            "corolla",                 # worst (expected to fail)
        ],
        baseline="",
    ),
    FactorAxis(
        name="grammar",
        values=[
            "a single",      # best (single)
            "a",             # baseline (article)
            "",              # bare_noun (no article)
            "a photo of a",  # photo_of
            "one",           # one
            "close-up of a", # closeup_of (worst)
        ],
        baseline="a",
    ),
]


# Baseline configurations
BASELINE_CONFIGS = [
    PromptConfig(
        name="C3",
        prompt="a cowpea flower",
        description="Species name only - baseline comparison",
    ),
]


# Combination test configs (from Experiment 8)
COMBINATION_CONFIGS = [
    PromptConfig(
        name="C1",
        prompt="a single yellow cowpea flower with open petals, not a bud, not the green calyx, not a leaf",
        description="Kitchen sink - all components combined",
    ),
    PromptConfig(
        name="C2",
        prompt="a single yellow cowpea flower with open petals",
        description="No negation - best components without text negation",
    ),
]


# Absorber configurations (from Experiment 10)
ABSORBER_CONFIGS = [
    PromptConfig(
        name="H3b",
        prompt="a single yellow cowpea flower with open petals",
        description="Bud + calyx absorbers",
        absorber_classes=[
            "a single yellow cowpea flower with open petals",
            "a cowpea flower bud",
            "a green calyx",
        ],
        target_indices=[0],
    ),
    PromptConfig(
        name="H2a",
        prompt="a single yellow cowpea flower with open petals",
        description="Leaf + stem absorbers",
        absorber_classes=[
            "a single yellow cowpea flower with open petals",
            "a leaf",
            "a stem",
        ],
        target_indices=[0],
    ),
]


def build_prompt_from_components(components: Dict[str, str]) -> str:
    """
    Build a prompt string from component values.
    
    Handles special cases from original Experiment 7:
    - Grammar: "close-up of a" -> "close-up of a"
    - Grammar: "" (bare noun) -> no article
    - Phenology: "bud" -> "cowpea flower bud" (special placement)
    - Phenology: "closed bud" -> "closed cowpea bud" (special placement)
    - Phenology: "open" -> "an open cowpea flower" (article change)
    - Phenology: "blooming" -> "a blooming cowpea flower"
    - Phenology: "in bloom" -> "a cowpea flower in bloom"
    - Anatomy: "corolla" -> "cowpea corolla" (replaces "flower")

    Args:
        components: Dictionary mapping factor names to values

    Returns:
        Constructed prompt string
    """
    grammar = components.get("grammar", "a")
    size = components.get("size", "")
    color = components.get("color", "")
    taxonomy = components.get("taxonomy", "flower")
    anatomy = components.get("anatomy", "")
    phenology = components.get("phenology", "")
    negation = components.get("negation", "")
    
    # Special case: corolla replaces "flower"
    if anatomy == "corolla":
        # "a cowpea corolla" or "a corolla" depending on taxonomy
        if taxonomy == "flower":
            taxonomy = "corolla"
        else:
            taxonomy = taxonomy.replace("flower", "corolla")
        anatomy = ""
    
    # Special case: phenology "bud" or "closed bud" - goes after taxonomy
    if phenology == "bud":
        # "a cowpea flower bud"
        parts = []
        if grammar:
            parts.append(grammar)
        if size:
            parts.append(size)
        if color:
            parts.append(color)
        parts.append(taxonomy)
        parts.append("bud")
        if anatomy:
            parts.append(anatomy)
        if negation:
            parts.append(negation)
        return " ".join(parts).strip()
    elif phenology == "closed bud":
        # "a closed cowpea bud" (replaces "flower" with "bud")
        parts = []
        if grammar:
            parts.append(grammar)
        if size:
            parts.append(size)
        if color:
            parts.append(color)
        # Replace "flower" with "bud"
        taxonomy_bud = taxonomy.replace("flower", "bud")
        parts.append("closed")
        parts.append(taxonomy_bud)
        if anatomy:
            parts.append(anatomy)
        if negation:
            parts.append(negation)
        return " ".join(parts).strip()
    
    # Standard construction
    parts = []
    
    # Grammar (article/framing)
    if grammar:
        if grammar == "close-up of a":
            parts.append("close-up")
            parts.append("of")
            parts.append("a")
        elif grammar == "a photo of a":
            parts.append("a")
            parts.append("photo")
            parts.append("of")
            parts.append("a")
        else:
            parts.append(grammar)
    # Note: if grammar is "", we have bare noun (no article)
    
    # Size (before color/taxonomy)
    if size:
        parts.append(size)
    
    # Phenology "open" or "blooming" goes before taxonomy
    if phenology == "open":
        parts.append("open")
        # Will need to fix article to "an" later
    elif phenology == "blooming":
        parts.append("blooming")
    
    # Color
    if color:
        parts.append(color)
    
    # Taxonomy
    parts.append(taxonomy)
    
    # Phenology "in bloom" goes after taxonomy
    if phenology == "in bloom":
        parts.append("in")
        parts.append("bloom")
    
    # Anatomy
    if anatomy:
        parts.append(anatomy)
    
    # Negation (appended at end)
    if negation:
        parts.append(negation)
    
    result = " ".join(parts).strip()
    
    # Fix "a open" -> "an open"
    result = result.replace("a open", "an open")
    
    return result


def generate_factor_combinations() -> List[Dict[str, str]]:
    """
    Generate all combinations for one-factor-at-a-time (OFAT) analysis.

    Returns:
        List of component dictionaries, one per combination
    """
    combinations = []

    # Baseline: all baselines
    baseline = {axis.name: axis.baseline or "" for axis in FACTOR_AXES}
    combinations.append(baseline)

    # One factor at a time: vary each axis while keeping others at baseline
    for axis in FACTOR_AXES:
        for value in axis.values:
            if value == axis.baseline:
                continue  # Skip baseline (already included)
            combo = baseline.copy()
            combo[axis.name] = value
            combinations.append(combo)

    return combinations
