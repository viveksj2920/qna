"""
Entity post-processor for controlled subtopic extraction.
Handles normalization of drug names, fuzzy matching of plan names,
and basic cleanup of pharmacy/hospital/dental plan names.
"""

import re
import pandas as pd
from rapidfuzz import fuzz, process
from utils.logger_config import logger

# Module-level cache for plan names
_plan_names_cache = None


def normalize_drug_name(raw_name):
    """
    Normalize a drug name to its base/generic form.
    Strips dosages, forms, salt forms, parenthetical content, and lowercases.

    Examples:
        "zolpidem 10mg tablet" -> "zolpidem"
        "ambien (zolpidem)" -> "ambien"
        "zolpidem tartrate 12.5 mg er" -> "zolpidem"
        "Eliquis" -> "eliquis"
    """
    if not raw_name or str(raw_name).lower().strip() in ("null", "none", "n/a", ""):
        return None

    name = raw_name.lower().strip()

    # Remove parenthetical content
    name = re.sub(r'\([^)]*\)', '', name)

    # Remove dosage patterns: digits + unit
    name = re.sub(
        r'\d+\.?\d*\s*'
        r'(mg|mcg|ml|milligram|microgram|gram|g|unit|iu|meq|%)\b',
        '', name
    )

    # Remove standalone numbers and leftover percentage signs
    name = re.sub(r'\b\d+\.?\d*\s*%?', '', name)

    # Remove form/route/salt words
    form_words = [
        'tablet', 'tablets', 'capsule', 'capsules', 'injection', 'solution',
        'sublingual', 'extended-release', 'extended release', 'immediate-release',
        'immediate release', 'delayed-release', 'delayed release',
        'er', 'xr', 'sr', 'cr', 'dr', 'la', 'xl',
        'oral', 'topical', 'cream', 'ointment', 'patch', 'inhaler',
        'suspension', 'syrup', 'elixir', 'spray', 'drops', 'suppository',
        'tartrate', 'succinate', 'hydrochloride', 'hcl', 'mesylate',
        'maleate', 'fumarate', 'besylate', 'sodium', 'potassium',
        'acetate', 'phosphate', 'sulfate', 'citrate', 'carbonate',
        'titrate', 'generic', 'brand',
    ]
    for word in form_words:
        name = re.sub(r'\b' + re.escape(word) + r'\b', '', name)

    # Clean up whitespace and trailing punctuation
    name = re.sub(r'\s+', ' ', name).strip().rstrip('.,;:-')

    return name if name else None


def load_plan_names(csv_path):
    """Load and cache plan names from CSV."""
    global _plan_names_cache
    if _plan_names_cache is None:
        try:
            df = pd.read_csv(csv_path)
            _plan_names_cache = df['plan_name'].dropna().tolist()
            logger.info(f"Loaded {len(_plan_names_cache)} plan names from {csv_path}")
        except Exception as e:
            logger.error(f"Failed to load plan names from {csv_path}: {e}")
            _plan_names_cache = []
    return _plan_names_cache


def fuzzy_match_plan_name(extracted_name, csv_path, score_threshold=75):
    """
    Fuzzy match an extracted plan name against the official plan list.
    Uses token_set_ratio for partial matching (customers often say partial names).
    Returns the best matching official plan name, or the original if no good match.
    """
    if not extracted_name or str(extracted_name).lower().strip() in ("null", "none", "n/a", ""):
        return None

    plan_names = load_plan_names(csv_path)
    if not plan_names:
        return extracted_name.strip()

    result = process.extractOne(
        extracted_name,
        plan_names,
        scorer=fuzz.token_set_ratio,
        score_cutoff=score_threshold
    )

    if result:
        matched_name, score, _ = result
        logger.info(f"Plan name fuzzy match: '{extracted_name}' -> '{matched_name}' (score={score})")
        return matched_name

    logger.info(f"Plan name no match above threshold: '{extracted_name}' (keeping as-is)")
    return extracted_name.strip()


def normalize_entity_name(raw_name):
    """Basic normalization for pharmacy, hospital, dental plan names."""
    if not raw_name or str(raw_name).lower().strip() in ("null", "none", "n/a", ""):
        return None
    return raw_name.strip()


def process_entities(entities_dict, plan_names_csv_path):
    """
    Process all extracted entities and return normalized values.
    Returns a list of entity-based subtopic strings to append to sub_topic list.
    """
    result = []

    if not entities_dict or not isinstance(entities_dict, dict):
        return result

    for entity_type, raw_value in entities_dict.items():
        if raw_value is None or str(raw_value).lower().strip() in ("null", "none", "n/a", ""):
            continue

        if entity_type in ("drug_name", "drug_names"):
            normalized = normalize_drug_name(raw_value)
            if normalized:
                result.append(normalized)

        elif entity_type in ("plan_name", "plan_names"):
            matched = fuzzy_match_plan_name(raw_value, plan_names_csv_path)
            if matched:
                result.append(matched)

        elif entity_type in ("pharmacy_name", "pharmacy_names",
                             "hospital_facility_name", "hospital/facility_names",
                             "dental_plan_name"):
            normalized = normalize_entity_name(raw_value)
            if normalized:
                result.append(normalized)

        else:
            logger.warning(f"Unknown entity type: {entity_type}")
            normalized = normalize_entity_name(raw_value)
            if normalized:
                result.append(normalized)

    return result
