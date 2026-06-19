"""
analyzer.py — Groq Vision API wrapper for damage claim analysis.
Uses Groq's vision-capable LLM with base64 image encoding.
"""

import base64
import json
import os
import re
import time
from io import BytesIO
from typing import Optional

from groq import Groq
from dotenv import load_dotenv
from PIL import Image

from config import (
    API_CALL_DELAY,
    INITIAL_RETRY_DELAY,
    MAX_RETRIES,
    MODEL_NAME,
)
from prompts import SYSTEM_PROMPT, build_user_prompt
from utils import load_images_for_claim, validate_and_fix

# Load .env from the code/ directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

API_KEY = os.environ.get("GROQ_API_KEY")
if not API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY environment variable is not set. "
        "Create a .env file in code/ with GROQ_API_KEY=your_key"
    )

_client = Groq(api_key=API_KEY)

# Groq vision: images must be resized — confirmed working at 512px max
MAX_IMAGE_DIMENSION = 512
MAX_IMAGE_BYTES = 3 * 1024 * 1024  # 3MB safety limit


def _resize_image_blob(blob: dict) -> dict:
    """
    ALWAYS resize and re-encode the image as JPEG at 512px max dimension.
    This is required because Groq rejects images that are too large or
    in certain formats (PNG with alpha, etc).
    """
    try:
        raw = base64.b64decode(blob["data"])
        img = Image.open(BytesIO(raw))

        # Convert to RGB first (handles RGBA, P, LA, CMYK, etc.)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Always resize to fit within MAX_IMAGE_DIMENSION
        w, h = img.size
        if w > MAX_IMAGE_DIMENSION or h > MAX_IMAGE_DIMENSION:
            scale = MAX_IMAGE_DIMENSION / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        # Re-encode as JPEG
        quality = 80
        while True:
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            data = buf.getvalue()
            if len(data) <= MAX_IMAGE_BYTES or quality <= 40:
                break
            quality -= 15

        return {
            "mime_type": "image/jpeg",
            "data": base64.b64encode(data).decode("utf-8"),
        }
    except Exception as e:
        print(f"  [WARN] Could not process image: {e}")
        return blob


def _extract_json(text: str) -> dict:
    """
    Extract JSON from the model response. Tries direct parse first,
    then falls back to regex extraction of a JSON object.
    """
    text = text.strip()
    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try regex extraction
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {}


def _default_result(reason: str = "Analysis failed") -> dict:
    """Returns a safe fallback result when the API call fails."""
    return {
        "evidence_standard_met": "false",
        "evidence_standard_met_reason": reason,
        "risk_flags": "none",
        "issue_type": "unknown",
        "object_part": "unknown",
        "claim_status": "not_enough_information",
        "claim_status_justification": reason,
        "supporting_image_ids": "none",
        "valid_image": "false",
        "severity": "unknown",
    }


def _build_content(user_prompt: str, image_blobs: list[dict]) -> list[dict]:
    """
    Build message content: text FIRST, then images.
    Groq vision requires text before images in the content list.
    Images passed as base64 data URIs.
    """
    content = []

    # Text prompt comes first
    content.append({"type": "text", "text": user_prompt})

    # Then images
    for blob in image_blobs:
        resized = _resize_image_blob(blob)
        mime = resized.get("mime_type", "image/jpeg")
        data = resized.get("data", "")
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime};base64,{data}"
            }
        })

    return content


def analyze_claim(
    claim_row: dict,
    user_history: dict,
    evidence_requirements: list[dict],
    dataset_root: str,
) -> dict:
    """
    Sends the claim images + context to Groq vision model and returns structured result.
    """
    image_paths_str = claim_row.get("image_paths", "")
    claim_object = claim_row.get("claim_object", "unknown")

    # Load images
    image_blobs = load_images_for_claim(image_paths_str, dataset_root)
    if not image_blobs:
        print(f"  [WARN] No images loaded for {claim_row.get('user_id')} — text-only mode")

    # Build prompt
    user_prompt = build_user_prompt(claim_row, user_history, evidence_requirements)

    # Build content: text first, then images
    content = _build_content(user_prompt, image_blobs)

    # Call Groq API with retries
    # NOTE: response_format=json_object is NOT used with vision — model is prompted to output JSON
    raw_result = {}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = _client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                temperature=0.0,
                max_tokens=600,
                # No response_format here — not compatible with vision on Groq
            )
            raw_text = response.choices[0].message.content or ""
            raw_result = _extract_json(raw_text)
            if raw_result:
                break
            else:
                print(f"  [WARN] Empty/unparseable JSON on attempt {attempt}. Raw: {raw_text[:300]}")
        except Exception as e:
            err_str = str(e)
            print(f"  [ERROR] API call attempt {attempt} failed: {err_str[:300]}")
            if attempt < MAX_RETRIES:
                # Check if it's a rate limit error (429) — use longer delay
                if "429" in err_str or "rate limit" in err_str.lower():
                    # Try to extract wait time from error message
                    wait_match = re.search(r"try again in (\d+)m", err_str)
                    if wait_match:
                        wait_mins = int(wait_match.group(1))
                        delay = min(wait_mins * 60 + 10, 120)  # cap at 2 minutes
                    else:
                        delay = min(60 * attempt, 120)  # escalating: 60s, 120s, 120s...
                    print(f"  [RATE LIMIT] Waiting {delay}s before retry...")
                else:
                    delay = INITIAL_RETRY_DELAY * (2 ** (attempt - 1))
                    print(f"  Retrying in {delay}s...")
                time.sleep(delay)
            else:
                return _default_result(f"API error after {MAX_RETRIES} attempts: {err_str[:200]}")

    if not raw_result:
        return _default_result("Model returned empty or unparseable JSON")

    # Validate and fix values
    validated = validate_and_fix(raw_result, claim_object)

    # Rate limit delay
    time.sleep(API_CALL_DELAY)

    return validated
