"""
json_parser.py -- Robust JSON parser for Qwen output.

Multi-strategy extraction:
  1. Direct json.loads
  2. Code-fence extraction (```json ... ```)
  3. Brace-balancing (handles escaped quotes, nested braces)
  4. Fallback rfind
"""

import json
import re


def safe_json(text: str) -> dict:
    """Trich xuat JSON tu Qwen response -- robust multi-strategy parser.

    Handles common LLM output patterns:
      - Pure JSON
      - JSON wrapped in ```json ... ``` code fences
      - JSON followed by natural-language explanation
      - JSON with extra whitespace/newlines

    Args:
        text: Raw text output from Qwen.

    Returns:
        Parsed dict.

    Raises:
        ValueError: If no valid JSON can be extracted.
    """
    text = text.strip()

    # ── Strategy 1: Direct parse ──────────────────────────────────
    try:
        return json.loads(text)
    except Exception:
        pass

    # ── Strategy 2: Code fence extraction ─────────────────────────
    for pattern in [
        r"```json\s*([\s\S]+?)\s*```",
        r"```\s*([\s\S]+?)\s*```",
    ]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except Exception:
                pass

    # ── Strategy 3: Brace-balancing ───────────────────────────────
    #   Tim dau '{' roi dem depth, xu ly escape + string literal
    start = text.find("{")
    if start != -1:
        depth = 0
        in_str = False
        escape = False
        for i, ch in enumerate(text[start:], start):
            if escape:
                escape = False
                continue
            if ch == "\\" and in_str:
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except Exception:
                        break  # malformed, try fallback

    # ── Strategy 4: Fallback rfind ────────────────────────────────
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass

    raise ValueError(f"Khong the parse JSON (400 ky tu dau):\n{text[:400]}")
