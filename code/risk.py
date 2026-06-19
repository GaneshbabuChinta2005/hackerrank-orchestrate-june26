"""
risk.py — Post-VLM risk flag overlay using user history data.
"""

from config import ALLOWED_RISK_FLAGS


def merge_risk_flags(*flag_sets: str) -> str:
    """
    Merge multiple semicolon-separated risk flag strings into one deduplicated string.
    Returns 'none' if the result is empty.
    """
    all_flags = set()
    for flag_str in flag_sets:
        if flag_str and flag_str.strip().lower() != "none":
            for f in flag_str.split(";"):
                f = f.strip()
                if f and f in ALLOWED_RISK_FLAGS:
                    all_flags.add(f)

    # Remove 'none' if we have real flags
    all_flags.discard("none")

    if not all_flags:
        return "none"

    # Consistent ordering: put user_history_risk and manual_review_required last
    priority_last = {"user_history_risk", "manual_review_required"}
    normal = sorted(f for f in all_flags if f not in priority_last)
    last = sorted(f for f in all_flags if f in priority_last)
    return ";".join(normal + last)


def apply_history_risk(result: dict, user_history: dict) -> dict:
    """
    Overlays user history risk flags onto the VLM result's risk_flags field.
    Does NOT override claim_status or any visual determination.

    Args:
        result: The dict produced by the VLM (already validated).
        user_history: The user's history row from user_history.csv.

    Returns:
        The result dict with updated risk_flags.
    """
    if not user_history:
        return result

    history_flags = user_history.get("history_flags", "none")
    vlm_flags = result.get("risk_flags", "none")

    # Merge VLM flags with history flags
    merged = merge_risk_flags(vlm_flags, history_flags)
    result["risk_flags"] = merged

    return result
