"""
config.py — constants and allowed value definitions for the claim verification system.
"""

# Groq model to use (vision-capable)
# Options: "meta-llama/llama-4-scout-17b-16e-instruct", "llama-3.2-11b-vision-preview"
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

# Output CSV column order (must match problem_statement.md exactly)
OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

# Allowed values per field (from problem_statement.md)
ALLOWED_CLAIM_STATUS = {"supported", "contradicted", "not_enough_information"}

ALLOWED_ISSUE_TYPES = {
    "dent", "scratch", "crack", "glass_shatter", "broken_part",
    "missing_part", "torn_packaging", "crushed_packaging",
    "water_damage", "stain", "none", "unknown",
}

ALLOWED_SEVERITY = {"none", "low", "medium", "high", "unknown"}

ALLOWED_RISK_FLAGS = {
    "none",
    "blurry_image",
    "cropped_or_obstructed",
    "low_light_or_glare",
    "wrong_angle",
    "wrong_object",
    "wrong_object_part",
    "damage_not_visible",
    "claim_mismatch",
    "possible_manipulation",
    "non_original_image",
    "text_instruction_present",
    "user_history_risk",
    "manual_review_required",
}

CAR_OBJECT_PARTS = {
    "front_bumper", "rear_bumper", "door", "hood", "windshield",
    "side_mirror", "headlight", "taillight", "fender",
    "quarter_panel", "body", "unknown",
}

LAPTOP_OBJECT_PARTS = {
    "screen", "keyboard", "trackpad", "hinge", "lid",
    "corner", "port", "base", "body", "unknown",
}

PACKAGE_OBJECT_PARTS = {
    "box", "package_corner", "package_side", "seal",
    "label", "contents", "item", "unknown",
}

OBJECT_PART_MAP = {
    "car": CAR_OBJECT_PARTS,
    "laptop": LAPTOP_OBJECT_PARTS,
    "package": PACKAGE_OBJECT_PARTS,
}

# Rate limiting: seconds between API calls
API_CALL_DELAY = 2.0  # keeps us well under 30 RPM, conserves daily token budget

# Retry settings
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 10  # seconds; 429 errors use longer waits automatically

# Cache file path (relative to code/)
CACHE_FILE = "cache/results_cache.json"
