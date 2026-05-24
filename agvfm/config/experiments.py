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
    FactorAxis(
        name="emoji",
        values=[
            "",         # baseline (none)
            "🌸",      # cherry blossom
            "🌺",      # hibiscus
            "🌻",      # sunflower
            "🌼",      # blossom
            "💐",      # bouquet
            "🌷",      # tulip
        ],
        baseline="",
    ),
]

POD_FACTOR_AXES = [
FactorAxis(
name="taxonomy",
values=[
"cowpea pod",             # species_common (best)
"bean pod",               # genus_bean
"pea pod",                # genus_pea
"legume pod",             # family_common
"black-eyed pea pod",     # alt_common
"pod",                    # baseline (generic)
"vigna unguiculata pod",  # scientific (expected to fail)
"crop fruit",             # generic_crop (expected to fail)
],
baseline="pod",
),
FactorAxis(
name="color",
values=[
"green",   # best
"light green",
"pale green",
"",        # baseline (none)
"purple",  # worst (specific varieties only)
],
baseline="",
),
FactorAxis(
name="size",
values=[
"",        # baseline (none - best)
"long",
"short",
"slender", # worst
],
baseline="",
),
FactorAxis(
name="phenology",
values=[
"immature",      # best
"young",
"",              # baseline (none)
"developing",
"mature",
"dried pod",     # worst (expected to fail)
],
baseline="",
),
FactorAxis(
name="negation",
values=[
"not a leaf, not a stem, not a flower", # best (leaf+stem+flower)
"not a leaf, not a stem",               # leaf+stem
"not a leaf",                           # leaf_only
"not a flower",                         # flower_only
"not a stem",                           # stem_only
"",                                     # baseline (none)
],
baseline="",
),
FactorAxis(
name="anatomy",
values=[
"with visible seeds",     # best (visible_seeds)
"with bumpy surface",     # bumpy_surface
"with a pointed tip",     # pointed_tip
"",                       # baseline (none)
"pericarp",               # worst (expected to fail)
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
        name="comb_species",
        prompt="a cowpea flower",
        description="Species name only - baseline comparison",
    ),
]


# Combination test configs (Phase 2: Systematic combinations)
# Base list — do not import directly; use COMBINATION_CONFIGS which includes emoji variants.
# Naming convention: comb_{components}[_size][_neg_{targets}]
#   Components: clr=color, spp=species/taxonomy, anat=anatomy, gram=grammar
#   Modifiers:  tiny=size:tiny, neg_{x}=text negation excluding x
_BASE_COMBINATION_CONFIGS = [
    # Baseline
    PromptConfig(
        name="comb_species",
        prompt="a cowpea flower",
        description="Species baseline - taxonomy only",
    ),
    # Two-component combinations
    PromptConfig(
        name="comb_clr_spp",
        prompt="a yellow cowpea flower",
        description="Color + Taxonomy",
    ),
    PromptConfig(
        name="comb_clr_anat",
        prompt="a yellow flower with open petals",
        description="Color + Anatomy",
    ),
    PromptConfig(
        name="comb_clr_gram",
        prompt="a single yellow flower",
        description="Color + Grammar",
    ),
    PromptConfig(
        name="comb_spp_anat",
        prompt="a cowpea flower with open petals",
        description="Taxonomy + Anatomy",
    ),
    PromptConfig(
        name="comb_spp_gram",
        prompt="a single cowpea flower",
        description="Taxonomy + Grammar",
    ),
    PromptConfig(
        name="comb_anat_gram",
        prompt="a single flower with open petals",
        description="Anatomy + Grammar",
    ),
    # Three-component combinations
    PromptConfig(
        name="comb_clr_spp_anat",
        prompt="a yellow cowpea flower with open petals",
        description="Color + Taxonomy + Anatomy",
    ),
    PromptConfig(
        name="comb_clr_spp_gram",
        prompt="a single yellow cowpea flower",
        description="Color + Taxonomy + Grammar",
    ),
    PromptConfig(
        name="comb_clr_anat_gram",
        prompt="a single yellow flower with open petals",
        description="Color + Anatomy + Grammar",
    ),
    PromptConfig(
        name="comb_spp_anat_gram",
        prompt="a single cowpea flower with open petals",
        description="Taxonomy + Anatomy + Grammar",
    ),
    # Four-component combination
    PromptConfig(
        name="comb_clr_spp_anat_gram",
        prompt="a single yellow cowpea flower with open petals",
        description="Four components - best without negation",
    ),
    # Kitchen sink (with text negation)
    PromptConfig(
        name="comb_full_neg",
        prompt="a single yellow cowpea flower with open petals, not a bud, not the green calyx, not a leaf",
        description="Kitchen sink - all components with full text negation",
    ),
    # Strategic experiments: Size component integration
    PromptConfig(
        name="comb_clr_spp_anat_gram_tiny",
        prompt="a single tiny yellow cowpea flower with open petals",
        description="Four components + size:tiny",
    ),
    PromptConfig(
        name="comb_full_neg_tiny",
        prompt="a single tiny yellow cowpea flower with open petals, not a bud, not the green calyx, not a leaf",
        description="Full negation + size:tiny",
    ),
    PromptConfig(
        name="comb_clr_spp_gram_tiny",
        prompt="a single tiny yellow cowpea flower",
        description="Color + Taxonomy + Grammar + size:tiny",
    ),
    # Strategic experiments: Partial negation variants
    PromptConfig(
        name="comb_clr_spp_anat_gram_neg_bud",
        prompt="a single yellow cowpea flower with open petals, not a bud",
        description="Four components + negation: bud",
    ),
    PromptConfig(
        name="comb_clr_spp_anat_gram_neg_calyx",
        prompt="a single yellow cowpea flower with open petals, not the green calyx",
        description="Four components + negation: calyx",
    ),
    PromptConfig(
        name="comb_clr_spp_anat_gram_neg_leaf",
        prompt="a single yellow cowpea flower with open petals, not a leaf",
        description="Four components + negation: leaf",
    ),
    PromptConfig(
        name="comb_clr_spp_anat_gram_neg_bud_calyx",
        prompt="a single yellow cowpea flower with open petals, not a bud, not the green calyx",
        description="Four components + negation: bud + calyx",
    ),
    # Strategic experiments: SAM3 size variants (negation-free optimization)
    # Note: comb_clr_spp_anat_gram_tiny is tested for both models (YOLO World and SAM3)
    PromptConfig(
        name="comb_clr_spp_anat_tiny",
        prompt="a tiny yellow cowpea flower with open petals",
        description="Color + Taxonomy + Anatomy + size:tiny (SAM3 optimization)",
    ),
]

# Emoji variants for non-baseline emoji values (from FACTOR_AXES)
_EMOJI_AXIS = next(ax for ax in FACTOR_AXES if ax.name == "emoji")
_EMOJI_LABELS = {
    "🌸": "cherry_blossom",
    "🌺": "hibiscus",
    "🌻": "sunflower",
    "🌼": "blossom",
    "💐": "bouquet",
    "🌷": "tulip",
}


def generate_combination_configs() -> List["PromptConfig"]:
    """
    Return the full set of combination configs: base configs + emoji variants.

    For every base config in ``_BASE_COMBINATION_CONFIGS`` we generate one
    additional ``PromptConfig`` per non-baseline emoji value, appending the
    emoji character to the prompt and tagging the name with the emoji label.
    """
    configs = list(_BASE_COMBINATION_CONFIGS)
    non_baseline_emojis = [v for v in _EMOJI_AXIS.values if v != _EMOJI_AXIS.baseline]
    for base in _BASE_COMBINATION_CONFIGS:
        for emoji in non_baseline_emojis:
            label = _EMOJI_LABELS.get(emoji, emoji)
            configs.append(
                PromptConfig(
                    name=f"{base.name}_{label}",
                    prompt=f"{base.prompt} {emoji}",
                    description=f"{base.description} + {label} emoji",
                )
            )
    return configs


COMBINATION_CONFIGS = generate_combination_configs()


# Absorber configurations (Phase 2: Absorber architecture tests)
# Naming convention: abs_{absorber_targets}[_{base_prompt_variant}]
ABSORBER_CONFIGS = [
    PromptConfig(
        name="abs_bud_calyx",
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
        name="abs_leaf_stem",
        prompt="a single yellow cowpea flower with open petals",
        description="Leaf + stem absorbers",
        absorber_classes=[
            "a single yellow cowpea flower with open petals",
            "a leaf",
            "a stem",
        ],
        target_indices=[0],
    ),
    PromptConfig(
        name="abs_bud_calyx_leaf",
        prompt="a single yellow cowpea flower with open petals",
        description="Bud + calyx + leaf absorbers",
        absorber_classes=[
            "a single yellow cowpea flower with open petals",
            "a cowpea flower bud",
            "a green calyx",
            "a leaf",
        ],
        target_indices=[0],
    ),
    PromptConfig(
        name="abs_all",
        prompt="a single yellow cowpea flower with open petals",
        description="All absorbers (bud, calyx, leaf, stem, soil)",
        absorber_classes=[
            "a single yellow cowpea flower with open petals",
            "a cowpea flower bud",
            "a green calyx",
            "a leaf",
            "a stem",
            "soil",
        ],
        target_indices=[0],
    ),
    # Strategic experiments: Absorber optimization with simpler base prompts
    PromptConfig(
        name="abs_bud_calyx_simple",
        prompt="a yellow cowpea flower",
        description="Bud + calyx absorbers with simpler base (no grammar, no anatomy)",
        absorber_classes=[
            "a yellow cowpea flower",
            "a cowpea flower bud",
            "a green calyx",
        ],
        target_indices=[0],
    ),
    PromptConfig(
        name="abs_bud_calyx_no_anat",
        prompt="a single yellow cowpea flower",
        description="Bud + calyx absorbers without anatomy component",
        absorber_classes=[
            "a single yellow cowpea flower",
            "a cowpea flower bud",
            "a green calyx",
        ],
        target_indices=[0],
    ),
    PromptConfig(
        name="abs_bud_calyx_no_gram",
        prompt="a yellow cowpea flower with open petals",
        description="Bud + calyx absorbers without grammar component",
        absorber_classes=[
            "a yellow cowpea flower with open petals",
            "a cowpea flower bud",
            "a green calyx",
        ],
        target_indices=[0],
    ),
]

# Multi-class test configurations (Phase 2: Multi-class assembly tests)
# Note: These use absorber_classes format but evaluate all non-background classes
# For now, we'll handle multi-class separately in the runner script
# TODO: Add proper multi-class support to Evaluator
MULTICLASS_CONFIGS = [
    PromptConfig(
        name="multiclass_single",
        prompt="a single yellow cowpea flower with open petals",
        description="Single-class baseline",
    ),
    PromptConfig(
        name="multiclass_flower_bud",
        prompt="a single yellow cowpea flower with open petals",
        description="Two-class: Flower + Bud",
        absorber_classes=[
            "a single yellow cowpea flower with open petals",
            "a cowpea flower bud",
        ],
        target_indices=None,  # Keep all non-background classes
    ),
    PromptConfig(
        name="multiclass_yellow_white",
        prompt="a yellow cowpea flower",
        description="Two-class: Yellow + White",
        absorber_classes=[
            "a yellow cowpea flower",
            "a white cowpea flower",
        ],
        target_indices=None,  # Keep all non-background classes
    ),
    PromptConfig(
        name="multiclass_flower_bud_calyx",
        prompt="a single yellow cowpea flower with open petals",
        description="Three-class: Flower + Bud + Calyx",
        absorber_classes=[
            "a single yellow cowpea flower with open petals",
            "a cowpea flower bud",
            "a green calyx",
        ],
        target_indices=None,  # Keep all non-background classes
    ),
]

# Phase 3: Confidence threshold analysis configs (top 3 by mAP@0.5 from Phase 2)
PHASE3_CONFIGS = [
    # YOLO World top 3
    PromptConfig(
        name="comb_clr_spp_anat_gram_neg_bud_calyx",
        prompt="a single yellow cowpea flower with open petals, not a bud, not the green calyx",
        description="YOLO World #1: Partial negation (bud+calyx) - mAP@0.5: 0.4123",
    ),
    PromptConfig(
        name="comb_full_neg",
        prompt="a single yellow cowpea flower with open petals, not a bud, not the green calyx, not a leaf",
        description="YOLO World #2: Full negation - mAP@0.5: 0.3830",
    ),
    PromptConfig(
        name="comb_clr_spp_anat_gram_neg_calyx",
        prompt="a single yellow cowpea flower with open petals, not the green calyx",
        description="YOLO World #3: Negation: calyx - mAP@0.5: 0.3807",
    ),
    # SAM3 top 3
    PromptConfig(
        name="comb_clr_spp_anat_gram_neg_bud",
        prompt="a single yellow cowpea flower with open petals, not a bud",
        description="SAM3 #1: Negation: bud - mAP@0.5: 0.5407",
    ),
    PromptConfig(
        name="comb_clr_spp_anat_tiny",
        prompt="a tiny yellow cowpea flower with open petals",
        description="SAM3 #2: Color + Taxonomy + Anatomy + size:tiny - mAP@0.5: 0.5364",
    ),
    PromptConfig(
        name="comb_clr_spp_anat_gram",
        prompt="a single yellow cowpea flower with open petals",
        description="SAM3 #3: Four components - mAP@0.5: 0.4925",
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
    emoji = components.get("emoji", "")
    
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
        bud_result = " ".join(parts).strip()
        if emoji:
            bud_result = f"{bud_result} {emoji}"
        return bud_result
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
        closed_bud_result = " ".join(parts).strip()
        if emoji:
            closed_bud_result = f"{closed_bud_result} {emoji}"
        return closed_bud_result
    
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
    
    # Append emoji (typically trails text in CLIP training captions)
    if emoji:
        result = f"{result} {emoji}"
    
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
