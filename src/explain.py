"""Plain-language explanation layer.

This is the ONLY place in the pipeline that calls an LLM, and it is
deliberately last in the chain: it receives the already-computed Verdict and
turns it into a short, readable paragraph for a non-technical user. It never
sees raw pixels and never decides safe/unsafe — that decision was already
made by src/verdict.py before this module is called. If this call fails or
is skipped (no API key), the system still returns a complete, correct
verdict — this module only adds readability on top of it.
"""

from __future__ import annotations

import os

from src.models import Verdict

SYSTEM_PROMPT = (
    "You write short, plain-language inspection summaries for a fastener "
    "safety-check tool. You are given a final verdict and a list of "
    "structured findings that a rule engine already computed — you do not "
    "re-judge safety, you only explain the findings that were given to you "
    "in clear language a non-technical person can follow. Keep it to one "
    "short paragraph. State the overall verdict first, then the specific "
    "findings. Do not add caveats, disclaimers, or safety advice beyond "
    "what the findings state — that policy language belongs elsewhere in "
    "the product, not in this summary."
)


class ExplanationUnavailableError(RuntimeError):
    """Raised when no API key is configured or the API call fails/refuses."""


def explain(verdict: Verdict) -> str:
    """Call claude-opus-5 to render `verdict` as a plain-language paragraph.

    Raises ExplanationUnavailableError on any failure (missing key, refusal,
    API error) so callers can decide whether to fall back to a templated
    string instead of crashing the whole pipeline over an optional feature.
    """
    if not os.environ.get("ANTHROPIC_API_KEY") and not _has_oauth_profile():
        raise ExplanationUnavailableError(
            "No ANTHROPIC_API_KEY set and no `ant auth login` profile found; "
            "skipping the plain-language explanation. Set the key or run "
            "`ant auth login` to enable this feature."
        )

    import anthropic

    client = anthropic.Anthropic()

    findings_text = "\n".join(
        f"- {f.label}: severity={f.severity.value} — {f.reason}" for f in verdict.findings
    ) or "- no defects found"

    user_content = (
        f"Overall verdict: {verdict.overall.value}\n"
        f"Worst severity found: {verdict.worst_severity.value}\n"
        f"Findings:\n{findings_text}\n"
        f"Low-confidence detections present: {verdict.low_confidence_flagged}"
    )

    response = client.messages.create(
        model="claude-opus-5",
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )

    if response.stop_reason == "refusal":
        raise ExplanationUnavailableError(
            "The explanation request was declined by the model's safety "
            "classifiers; falling back to the structured findings only."
        )

    text_blocks = [b.text for b in response.content if b.type == "text"]
    if not text_blocks:
        raise ExplanationUnavailableError("Model returned no text content.")

    return "".join(text_blocks).strip()


def explain_or_fallback(verdict: Verdict) -> str:
    """Best-effort wrapper: returns the LLM explanation, or a deterministic
    templated fallback if the LLM call is unavailable for any reason. Used
    by the pipeline and the GUI so a missing API key never breaks the demo.
    """
    try:
        return explain(verdict)
    except ExplanationUnavailableError:
        return _templated_fallback(verdict)


def _templated_fallback(verdict: Verdict) -> str:
    if not verdict.findings:
        return f"Verdict: {verdict.overall.value}. No defects were found."
    lines = [f"Verdict: {verdict.overall.value}."]
    for f in verdict.findings:
        lines.append(f"- {f.label}: {f.reason}")
    return "\n".join(lines)


def _has_oauth_profile() -> bool:
    """Best-effort check for an `ant auth login` profile without importing
    the SDK's internal credential resolution — this just avoids raising a
    misleading "no key" error for a user authenticated via profile.
    """
    config_dir = os.environ.get(
        "ANTHROPIC_CONFIG_DIR",
        os.path.expanduser("~/.config/anthropic"),
    )
    return os.path.isdir(os.path.join(config_dir, "credentials"))
