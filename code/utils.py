"""
utils.py — CSV I/O helpers, image loading, and output validation utilities.
"""

import base64
import csv
import json
import os
from pathlib import Path
from typing import Optional

from config import (
    ALLOWED_CLAIM_STATUS,
    ALLOWED_ISSUE_TYPES,
    ALLOWED_RISK_FLAGS,
    ALLOWED_SEVERITY,
    OBJECT_PART_MAP,
    OUTPUT_COLUMNS,
)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def load_csv(path: str) -> list[dict]:
    """Load a CSV file into a list of dicts."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_csv(path: str, rows: list[dict]) -> None:
    """Write a list of dicts to a CSV file using the required output column order."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


def load_user_history(path: str) -> dict[str, dict]:
    """Load user_history.csv into a dict keyed by user_id."""
    rows = load_csv(path)
    return {row["user_id"]: row for row in rows}


def load_evidence_requirements(path: str) -> list[dict]:
    """Load evidence_requirements.csv into a list of dicts."""
    return load_csv(path)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def load_image_as_base64(image_path: str, dataset_root: str) -> Optional[dict]:
    """
    Loads a single image file and returns a Groq-compatible inline_data dict.
    image_path: path as found in the CSV (e.g. images/test/case_001/img_1.jpg)
                This path is relative to the REPO ROOT (parent of dataset/)
    dataset_root: absolute path to the dataset/ directory
    Returns: {"mime_type": "image/jpeg", "data": "<base64>"} or None if missing.
    """
    # image_paths in CSV (e.g. 'images/test/case_001/img_1.jpg') are relative to dataset_root
    abs_path = os.path.join(dataset_root, image_path.replace("/", os.sep))
    abs_path = os.path.normpath(abs_path)

    if not os.path.exists(abs_path):
        print(f"  [WARN] Image not found: {abs_path}")
        return None

    ext = Path(abs_path).suffix.lower()
    mime = "image/jpeg"
    if ext in (".png",):
        mime = "image/png"
    elif ext in (".webp",):
        mime = "image/webp"
    elif ext in (".gif",):
        mime = "image/gif"

    with open(abs_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    return {"mime_type": mime, "data": data}


def load_images_for_claim(image_paths_str: str, dataset_root: str) -> list[dict]:
    """
    Loads all images for a claim row.
    Returns a list of inline_data dicts for Gemini.
    """
    images = []
    for path in image_paths_str.split(";"):
        path = path.strip()
        if path:
            blob = load_image_as_base64(path, dataset_root)
            if blob:
                images.append(blob)
    return images


def extract_image_ids(image_paths_str: str) -> list[str]:
    """Extract image IDs (filename without extension) from image_paths field."""
    ids = []
    for path in image_paths_str.split(";"):
        path = path.strip()
        if path:
            filename = path.split("/")[-1].split("\\")[-1]
            ids.append(filename.rsplit(".", 1)[0])
    return ids


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------

def validate_and_fix(result: dict, claim_object: str) -> dict:
    """
    Validates the VLM output against allowed value lists and fixes/coerces invalid values.
    Returns a cleaned dict ready for writing to CSV.
    """
    # claim_status
    if result.get("claim_status") not in ALLOWED_CLAIM_STATUS:
        result["claim_status"] = "not_enough_information"

    # issue_type
    if result.get("issue_type") not in ALLOWED_ISSUE_TYPES:
        result["issue_type"] = "unknown"

    # severity
    if result.get("severity") not in ALLOWED_SEVERITY:
        result["severity"] = "unknown"

    # evidence_standard_met — must be "true" or "false" string
    esm = str(result.get("evidence_standard_met", "false")).strip().lower()
    result["evidence_standard_met"] = "true" if esm == "true" else "false"

    # valid_image — must be "true" or "false" string
    vi = str(result.get("valid_image", "true")).strip().lower()
    result["valid_image"] = "true" if vi == "true" else "false"

    # object_part — validate against object-specific list
    allowed_parts = OBJECT_PART_MAP.get(claim_object, set())
    if result.get("object_part") not in allowed_parts:
        result["object_part"] = "unknown"

    # risk_flags — validate each flag
    raw_flags = result.get("risk_flags", "none")
    if not raw_flags or raw_flags.strip() == "":
        result["risk_flags"] = "none"
    else:
        flags = [f.strip() for f in raw_flags.split(";")]
        valid_flags = [f for f in flags if f in ALLOWED_RISK_FLAGS]
        result["risk_flags"] = ";".join(valid_flags) if valid_flags else "none"

    # supporting_image_ids — keep as-is but strip whitespace
    sid = result.get("supporting_image_ids", "none")
    if not sid or sid.strip() == "":
        result["supporting_image_ids"] = "none"
    else:
        result["supporting_image_ids"] = ";".join(
            s.strip() for s in sid.split(";") if s.strip()
        )

    # Truncate / sanitize text fields
    for field in ["evidence_standard_met_reason", "claim_status_justification"]:
        val = result.get(field, "")
        result[field] = str(val).strip()[:500]

    return result


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def load_cache(cache_path: str) -> dict:
    """Load existing cache from JSON file."""
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache_path: str, cache: dict) -> None:
    """Save cache to JSON file."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def make_cache_key(claim_row: dict) -> str:
    """Create a stable cache key from the claim."""
    return f"{claim_row['user_id']}|{claim_row['image_paths']}|{claim_row['claim_object']}"
