"""Conservative final-answer extraction, exact GSM8K and symbolic MATH grading."""
from fractions import Fraction
import re

SCORER_VERSION = "thinking_final_answer_v2"


def extract_final_answer(text, require_thinking_end=False):
    # An answer before a completed thinking segment is not the final response.
    if require_thinking_end and "</think>" not in text:
        return None, "unfinished_thinking"
    text = text.rsplit("</think>", 1)[-1].strip()
    text = re.sub(r"<\|(?:im_end|endoftext)\|>", "", text).strip()
    matches = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", text))
    if matches:
        start = matches[-1].end()
        depth = 1
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i].strip(), "boxed"
        return None, "unclosed_box"
    matches = list(re.finditer(r"(?:####|(?:final\s+)?answer\s*(?:is|:|=))\s*([^\n]+)", text, re.I))
    if matches:
        return matches[-1].group(1).strip().rstrip("."), "answer_marker"
    # Accept a bare numeric response, but never guess from the last number in reasoning.
    if re.fullmatch(r"[+-]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:/[+-]?\d+)?", text):
        return text, "bare_number"
    return None, "missing_final_answer"


def exact_number(text):
    if text is None:
        return None
    value = re.sub(r"\s+", "", text).strip("$")
    value = value.replace(r"\,", "").replace(r"\left", "").replace(r"\right", "")
    frac = re.fullmatch(r"([+-]?)\\(?:d?frac)\{([^{}]+)\}\{([^{}]+)\}", value)
    if frac:
        numerator, denominator = exact_number(frac[2]), exact_number(frac[3])
        if numerator is None or denominator in (None, 0):
            return None
        return (-1 if frac[1] == "-" else 1) * numerator / denominator
    if "," in value:
        # Thousands separators must not turn a list such as "1,2" into 12.
        if not re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d*)?(?:[eE][+-]?\d+)?", value):
            return None
        value = value.replace(",", "")
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+|/[+-]?\d+)?", value):
        return None
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None


def grade_answer(dataset, generated_text, gold, *, timeout_seconds=5, require_thinking_end=False):
    answer, method = extract_final_answer(generated_text, require_thinking_end=require_thinking_end)
    base = {"extracted_answer": answer, "extraction_method": method,
            "scorer": SCORER_VERSION, "parse_success": False, "correct": False}
    if dataset == "gsm8k":
        target = exact_number(gold.rsplit("####", 1)[-1].strip())
        if target is None:
            raise ValueError("Invalid numeric GSM8K reference answer")
        prediction = exact_number(answer)
        return base | {"parse_success": prediction is not None,
                       "correct": prediction is not None and prediction == target}
    if dataset != "math500":
        raise ValueError(f"Unsupported dataset: {dataset}")
    try:
        from math_verify import LatexExtractionConfig, parse, verify
    except ImportError as error:
        raise RuntimeError("MATH-500 grading requires requirements-bench.txt") from error
    config = [LatexExtractionConfig(boxed_match_priority=0)]
    # Restrict parsing to the extracted final answer; never match a reasoning substring.
    target = parse(r"\boxed{" + gold + "}", extraction_config=config,
                   fallback_mode="no_fallback", parsing_timeout=timeout_seconds)
    if not target:
        raise ValueError(f"MATH reference answer could not be parsed: {gold!r}")
    if answer is None:
        return base
    prediction = parse(r"\boxed{" + answer + "}", extraction_config=config,
                       fallback_mode="no_fallback", parsing_timeout=timeout_seconds)
    return base | {"parse_success": bool(prediction),
                   "correct": bool(prediction) and bool(verify(target, prediction, timeout_seconds=timeout_seconds))}


def grade_token_ids(dataset, tokenizer, token_ids, gold, *, timeout_seconds=5, require_thinking_end=True):
    # Thinking's tokenizer marks </think> special. Never strip it before grading.
    raw = tokenizer.decode(token_ids, skip_special_tokens=False)
    display = tokenizer.decode(token_ids, skip_special_tokens=True)
    return {"generated_text": display, "generated_text_raw": raw,
            **grade_answer(dataset, raw, gold, timeout_seconds=timeout_seconds,
                           require_thinking_end=require_thinking_end)}
