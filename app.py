"""Streamlit GUI for the fastener defect & assembly-safety inspector.

Upload a photo of a bolted assembly, see each expected fastener position
boxed and labeled, and get an overall safety verdict with a plain-language
explanation. This calls the exact same src.pipeline.inspect() function the
CLI uses, so the demo can never drift out of sync with the command line.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from src.detect import FastenerDetector
from src.match_template import load_template
from src.models import VerdictLabel
from src.pipeline import DEFAULT_TEMPLATE_PATH, InspectionResult, inspect

PROJECT_ROOT = Path(__file__).resolve().parent
TEMPLATES_DIR = PROJECT_ROOT / "templates"

VERDICT_COLOR = {
    VerdictLabel.SAFE: "#2e7d32",
    VerdictLabel.NEEDS_INSPECTION: "#f9a825",
    VerdictLabel.UNSAFE: "#c62828",
}

STATUS_BOX_COLOR = {
    # matched, no defect
    "ok": (46, 125, 50),
    # matched, defect found
    "defect": (198, 40, 40),
    # unmatched — missing entirely
    "missing": (255, 152, 0),
}


def _list_templates() -> dict[str, Path]:
    return {p.stem: p for p in sorted(TEMPLATES_DIR.glob("*.json"))}


def _annotate_image(result: InspectionResult) -> Image.Image:
    """Draw a colored box + label for every expected position: green for
    ok, red for a detected defect, orange dashed-style marker for missing.
    """
    img = Image.open(result.image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:
        font = ImageFont.load_default()

    for status in result.statuses:
        label = status.position.label
        if status.is_missing:
            # No bounding box exists for a missing fastener — mark the
            # expected position with a circle instead of a box.
            r = 30
            cx, cy = status.position.x, status.position.y
            draw.ellipse(
                [cx - r, cy - r, cx + r, cy + r],
                outline=STATUS_BOX_COLOR["missing"],
                width=4,
            )
            draw.text(
                (cx - r, cy - r - 22),
                f"{label}: MISSING",
                fill=STATUS_BOX_COLOR["missing"],
                font=font,
            )
            continue

        det = status.detection
        box = det.bbox
        color = (
            STATUS_BOX_COLOR["ok"]
            if det.detection_class.value == "bolt_ok"
            else STATUS_BOX_COLOR["defect"]
        )
        draw.rectangle(
            [box.x, box.y, box.x + box.width, box.y + box.height],
            outline=color,
            width=4,
        )
        draw.text(
            (box.x, max(0, box.y - 22)),
            f"{label}: {det.detection_class.value} ({det.confidence:.2f})",
            fill=color,
            font=font,
        )

    return img


def main() -> None:
    st.set_page_config(page_title="Fastener Safety Inspector", page_icon="🔩", layout="wide")
    st.title("🔩 Fastener defect & assembly-safety inspector")
    st.caption(
        "Upload a photo of a bolted assembly to check whether every fastener "
        "is present and undamaged, and get a safety verdict."
    )

    templates = _list_templates()
    if not templates:
        st.error(f"No templates found in {TEMPLATES_DIR}")
        return

    with st.sidebar:
        st.header("Settings")
        template_name = st.selectbox("Assembly template", list(templates.keys()))
        include_explanation = st.checkbox(
            "Generate plain-language explanation (calls claude-opus-5)",
            value=True,
        )
        st.divider()
        detector = FastenerDetector()
        if detector.is_fine_tuned:
            st.success("Using a fine-tuned fastener detection model.")
        else:
            st.warning(
                "⚠️ No fine-tuned model found — running on stock/demo weights. "
                "Detections are NOT reliable fastener classifications. "
                "See train/README.md to train a real model."
            )

    uploaded_file = st.file_uploader("Upload an assembly photo", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        st.info("Upload a photo to run an inspection.")
        return

    with tempfile.NamedTemporaryFile(suffix=Path(uploaded_file.name).suffix, delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = Path(tmp.name)

    with st.spinner("Running inspection…"):
        result = inspect(
            tmp_path,
            template_path=templates[template_name],
            detector=detector,
            include_explanation=include_explanation,
        )

    col_image, col_verdict = st.columns([2, 1])

    with col_image:
        st.subheader("Annotated image")
        st.image(_annotate_image(result), use_container_width=True)

    with col_verdict:
        st.subheader("Verdict")
        color = VERDICT_COLOR[result.verdict.overall]
        st.markdown(
            f"<h2 style='color:{color}'>{result.verdict.overall.value.upper()}</h2>",
            unsafe_allow_html=True,
        )
        st.caption(f"Worst severity found: {result.verdict.worst_severity.value}")

        if not result.model_is_fine_tuned:
            st.warning("This verdict is not trustworthy — see the sidebar warning.")

        if result.verdict.findings:
            st.subheader("Findings")
            for finding in result.verdict.findings:
                st.markdown(f"**[{finding.severity.value}]** {finding.reason}")
        else:
            st.success("No defects found.")

        if include_explanation:
            st.subheader("Explanation")
            st.write(result.explanation)

    with st.expander("Raw detections and matched status"):
        for status in result.statuses:
            if status.is_missing:
                st.write(f"❌ **{status.position.label}**: missing")
            else:
                det = status.detection
                st.write(
                    f"✅ **{status.position.label}**: {det.detection_class.value} "
                    f"(confidence {det.confidence:.2f}, distance {status.distance_px:.1f}px)"
                )


if __name__ == "__main__":
    main()
