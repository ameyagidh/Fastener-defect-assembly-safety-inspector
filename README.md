# Fastener defect & assembly-safety inspector

A computer-vision system that inspects a photo of a bolted car-part assembly,
detects fastener defects (missing, corroded, misaligned, damaged bolts), and
returns a transparent **safe / needs inspection / unsafe** verdict — through
a GUI, backed by an auditable rule engine rather than a black-box classifier.

---

## Table of contents

- [The problem](#the-problem)
- [Why this needs CV *and* rules, not either alone](#why-this-needs-cv-and-rules-not-either-alone)
- [Architecture](#architecture)
- [Project layout](#project-layout)
- [Setup](#setup)
- [Usage](#usage)
- [How the safety verdict is decided](#how-the-safety-verdict-is-decided)
- [Data sources & provenance](#data-sources--provenance)
- [Evaluation methodology](#evaluation-methodology)
- [Current status & limitations](#current-status--limitations-read-this)
- [Design decisions and why](#design-decisions-and-why)
- [Testing](#testing)
- [Scope guardrails / what this v1 deliberately does not do](#scope-guardrails--what-this-v1-deliberately-does-not-do)

---

## The problem

On an assembly line, or when inspecting a used part before resale or
installation, a human currently eyeballs whether every fastener is present,
undamaged, and properly seated. This is repetitive, fatigue-prone, and
inconsistent between inspectors — exactly the kind of defect that gets
caught late (after the part ships), rather than at the point of assembly.

**Who this is for:** QA line inspectors, small auto shops doing used-parts
resale, and consumers checking a part before installing it.

---

## Why this needs CV *and* rules, not either alone

"Is a bolt present and undamaged?" cannot be answered with a simple pixel
threshold — it requires a model that has learned what a bolt looks like,
damaged or not, under varying lighting and angle. But *whether a given
defect makes the assembly unsafe* is a domain judgment, and a domain
judgment embedded inside an opaque model is not auditable, not editable by a
non-ML domain expert, and not something you can point to and say "here is
exactly why it failed."

So the system is deliberately split in two:

1. A **learned vision model** (YOLOv8, fine-tuned) does the part that
   genuinely requires learning from examples: recognizing a bolt and its
   visible condition in a photo.
2. A **plain-data rule engine** (`rules.yaml` + `src/verdict.py`) does the
   part that must be transparent: mapping detected conditions to a
   pass/fail verdict. It contains zero model calls and is fully
   unit-tested without a GPU, an image, or a network connection.

An LLM (`claude-opus-5`) sits *after* the verdict has already been decided,
and only turns it into a plain-language paragraph — it never participates in
the safety decision itself.

---

## Architecture

```
photo.jpg
   │
   ▼
[1] detect   ──────────► list[Detection]
   YOLOv8 model             {class: bolt_ok | bolt_corroded | bolt_misaligned |
   (src/detect.py)           bolt_damaged, bbox, confidence}
   │
   ▼
[2] match to template ─► list[FastenerStatus]
   templates/*.json          expected position → matched detection, or
   (src/match_template.py)   "missing" if nothing matched within tolerance
   │
   ▼
[3] severity rules ────► Verdict
   rules.yaml                overall: safe | needs_inspection | unsafe
   (src/verdict.py)          per-defect: severity + human-readable reason
   │
   ▼
[4] explain (optional) ─► one-paragraph plain-language report
   claude-opus-5              "3 of 4 bolts present and undamaged; the
   (src/explain.py,           top-right bolt shows moderate corrosion..."
    reporting only —
    never decision-making)
```

**Why the template-diff step (2) exists and can't be skipped:** a pure
object detector can report "I see 3 bolts, all look fine" — it has no way to
know a 4th bolt is *missing* unless something tells it 4 were expected.
`templates/*.json` supplies that domain knowledge explicitly, per part type,
rather than trying to infer "how many bolts should be here" from a single
image, which isn't information the image itself contains.

`src/pipeline.py` wires all four stages together and is the single entry
point both `app.py` (GUI) and the CLI usage below call into, so the two can
never drift out of sync with each other.

---

## Project layout

```
AIProject/
├── README.md                   # this file
├── requirements.txt
├── .gitignore
├── rules.yaml                  # the entire safety-critical decision surface — plain data, no code
├── templates/
│   └── bracket_4bolt.json      # expected fastener positions for a 4-bolt bracket
├── data/
│   ├── raw/                    # (gitignored) downloaded public datasets — see train/README.md
│   ├── self_collected/         # (gitignored images; README tracked) your own labeled photos
│   └── merged/                 # (generated) train/prepare_dataset.py output
├── models/
│   └── best.pt                 # (gitignored, not shipped) fine-tuned YOLOv8 weights — see below
├── src/
│   ├── models.py                # shared dataclasses: Detection, FastenerStatus, Verdict, ...
│   ├── detect.py                 # YOLOv8 inference wrapper
│   ├── match_template.py         # detection ↔ template diffing (nearest-match)
│   ├── verdict.py                 # rule engine — the safety-critical logic
│   ├── explain.py                 # claude-opus-5 plain-language explanation (reporting only)
│   └── pipeline.py                # wires detect → match → verdict → explain
├── train/
│   ├── README.md                  # honest account of what training requires and why it isn't done here
│   ├── prepare_dataset.py         # merge/split labeled sources into data/merged/
│   └── train.py                    # fine-tune YOLOv8n, writes models/best.pt
├── app.py                          # Streamlit GUI
└── tests/
    ├── test_verdict.py              # rule engine — pure logic, always runs
    ├── test_match_template.py       # template diffing — pure logic, always runs
    ├── test_pipeline.py              # wiring test with an injected fake detector
    └── test_eval_holdout.py          # full-pipeline eval against a trained model (skips if none exists)
```

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To enable the plain-language explanation layer, set an Anthropic API key
(or authenticate via `ant auth login` — see the Anthropic CLI docs):

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

This is optional. Without it, the system still returns a complete, correct
verdict — `src/explain.py` falls back to a deterministic templated summary
(see `explain_or_fallback`).

---

## Usage

### Command line

```bash
python -m src.pipeline path/to/photo.jpg --template templates/bracket_4bolt.json
```

Prints every expected fastener position with its matched detection (or
"missing"), the overall verdict, the per-defect findings with severity, and
the plain-language explanation.

### GUI

```bash
streamlit run app.py
```

Upload a photo, pick the assembly template in the sidebar, and see the
annotated image (green boxes = ok, red boxes = detected defect, orange
circles = expected-but-missing) alongside the verdict panel.

---

## How the safety verdict is decided

Everything here is in `rules.yaml`, in plain text, editable without touching
any code:

| Detected condition | Severity | Rationale |
|---|---|---|
| `bolt_ok` | none | — |
| `bolt_corroded` | major | Corrosion compromises fastener strength — not cosmetic |
| `bolt_misaligned` | major | Cross-threading / improper seating risks failure under load |
| `bolt_damaged` | critical | Visibly deformed/sheared head or shank |
| *(fastener expected, not detected)* | critical | A missing fastener is always the most severe finding |

The **overall verdict** is the *worst* severity found across every fastener
position in the assembly:

| Worst severity found | Overall verdict |
|---|---|
| none | `safe` |
| minor | `needs_inspection` |
| major | `unsafe` |
| critical | `unsafe` |

**Low-confidence guard:** if any single detection falls below
`min_confidence` (default 0.5), the verdict is capped at
`needs_inspection` regardless of what the severity table would otherwise
say — the system never asserts an unqualified "safe" on weak evidence. This
is deliberately a floor, not a ceiling: a low-confidence detection can never
*downgrade* an already-`unsafe` verdict back to something milder.

This asymmetry is intentional: a false "safe" on an actually-unsafe assembly
is the costly failure mode (see [Evaluation methodology](#evaluation-methodology)); a false
"needs_inspection" on an actually-safe one just costs a human a second look.

---

## Data sources & provenance

There is no single off-the-shelf "car bolt assembly" dataset with
present/missing/corroded/misaligned labels. The intended sourcing strategy
(see `train/README.md` for the full workflow):

- **Public data:** Roboflow Universe hosts community datasets for
  bolt/fastener detection and rust/corrosion classification under various
  open licenses. Each must be individually reviewed for license
  compatibility (prefer CC BY / public domain) before use.
- **Self-collected data:** classes public data doesn't adequately cover
  (most likely `bolt_misaligned`) are filled in with a small
  self-photographed, self-labeled set (20-30 images of a bolted bracket in
  good/bad states), tracked with explicit provenance notes in
  `data/self_collected/README.md`.
- **Assembly templates** (`templates/*.json`) are hand-authored per part
  type — this is supplied domain knowledge, not learned from images.

**Any model trained from this repo's scripts must state which classes came
from which source** in this README's evaluation table, so accuracy numbers
aren't misattributed to a class with too few real examples to mean anything.

---

## Evaluation methodology

Two different things are measured, deliberately kept separate rather than
collapsed into one accuracy number:

### 1. Detection quality — is the model seeing bolts correctly?

Standard object-detection metrics on a held-out split (`data/merged/images/holdout/`,
produced by `train/prepare_dataset.py` and never used in training):
mAP@0.5, and per-class precision/recall reported separately for
`bolt_ok` / `bolt_corroded` / `bolt_misaligned` / `bolt_damaged` — corrosion
detection is expected to be the hardest class and the one needing the most
self-collected examples.

Run via: `pytest tests/test_eval_holdout.py::TestDetectionQuality -v`

### 2. Verdict quality — does the end-to-end system make the right call?

This is the number that matters for the product, not mAP. A held-out set of
*whole assembly photos* (not individual bolts), each with a human-assigned
ground-truth verdict, is scored end to end through the full pipeline.

**False negatives — calling an actually-unsafe assembly "safe" — are the
headline metric**, reported separately from overall accuracy, not buried
inside it.

Run via: `pytest tests/test_eval_holdout.py::TestVerdictAccuracy -v`
(requires `data/holdout_verdicts.jsonl` — a human-labeled ground-truth file;
see the test's docstring)

### Results

| Metric | Value |
|---|---|
| mAP@0.5 (holdout) | *not yet measured — no fine-tuned model exists in this repo; see [Current status](#current-status--limitations-read-this)* |
| Per-class precision/recall | *pending real training data* |
| Verdict accuracy | *pending* |
| Verdict false-negative rate | *pending* |
| Holdout sample size | *pending* |

This table is intentionally unfilled rather than filled with placeholder
numbers — see the next section for why, and what filling it in requires.

---

## Current status & limitations (read this)

**Fully built, tested, and working right now, with no external
dependencies beyond `pip install`:**

- The rule engine (`src/verdict.py`) — 15 unit tests, 100% passing, pure
  logic, zero model calls.
- The template-matching engine (`src/match_template.py`) — 8 unit tests,
  100% passing, pure logic.
- The full pipeline wiring (`src/pipeline.py`) — 3 tests using an injected
  fake detector, validating detect → match → verdict → explain end to end,
  including the explanation fallback path when no API key is configured.
- The GUI (`app.py`) and CLI (`python -m src.pipeline`) — both runnable
  today, using a stock COCO-pretrained YOLOv8n checkpoint as a placeholder
  detector (see below).

**Not yet done, and honestly explained rather than faked:**

- **No fine-tuned fastener-detection model ships in this repo.** Training
  one requires a labeled image dataset this environment does not have —
  see `train/README.md` for exactly what's needed and the scripts
  (`train/prepare_dataset.py`, `train/train.py`) that are ready to run
  against it the moment real data is sourced. Fabricating a "trained"
  model or invented accuracy numbers here would be worse than admitting
  the gap: it would report confidence the system hasn't earned, on exactly
  the kind of safety judgment where that matters most.
- **Until that model exists**, `FastenerDetector` falls back to a stock
  `yolov8n.pt` (COCO classes, not fastener-specific). To keep this from
  silently producing a false "safe" verdict, every detection from the
  fallback model is forced to near-zero confidence, which triggers the
  `min_confidence` guard in `rules.yaml` and caps the verdict at
  `needs_inspection`. Both the CLI output and the GUI sidebar surface
  `InspectionResult.model_is_fine_tuned` prominently so this is never
  ambiguous to whoever is looking at a result.
- **The evaluation table above is unfilled** for the same reason — it will
  be filled in once a real model is trained against sourced data, following
  the exact procedure in `train/README.md` and the Evaluation Methodology
  section above.

**Path to closing this gap:** source 2-3 public fastener datasets, take
20-30 self-collected photos for the misalignment class, run
`train/prepare_dataset.py` then `train/train.py`, then
`pytest tests/test_eval_holdout.py`. Every piece of that pipeline is written
and ready; only the data-sourcing step (which requires a human to select and
license-check datasets, and to physically photograph some brackets) remains.

---

## Design decisions and why

- **Rule-based verdict, not an end-to-end "safe/unsafe" classifier.** A
  classifier trained directly on safe/unsafe labels would be a black box on
  exactly the decision that most needs to be explainable and editable by a
  non-ML domain expert. Splitting detection (learned) from severity mapping
  (rules) makes the safety logic auditable in a YAML diff, not a retraining
  run.
- **Template diffing as a separate stage**, not folded into the detector.
  "How many bolts should be here" is domain knowledge, not something
  learnable from a single photo — see [Architecture](#architecture).
- **Greedy nearest-match, not optimal bipartite assignment**, in
  `src/match_template.py::match`. With 4-8 well-separated fastener
  positions per assembly, greedy and optimal assignment agree in practice,
  and greedy is trivial to reason about and unit-test. If templates ever
  need tightly-clustered fasteners, swap in
  `scipy.optimize.linear_sum_assignment` — the function signature would not
  need to change (this is called out in the code as well).
- **The LLM only explains, never decides.** `src/explain.py` receives an
  already-computed `Verdict` and is explicitly instructed not to re-judge
  safety or add its own caveats — see `SYSTEM_PROMPT`. This also means a
  missing API key, a refusal, or an API error degrades gracefully to a
  templated fallback rather than breaking the pipeline (see
  `explain_or_fallback`).
- **Model: `claude-opus-5`.** No `temperature`/`top_p`/`top_k` are passed
  (they 400 on this model); adaptive thinking is left on by default rather
  than disabled, since this is a short, low-effort completion where the
  default behavior is appropriate.
- **The low-confidence guard only ever raises the floor, never lowers an
  already-worse verdict** (`src/verdict.py::evaluate`) — see the asymmetry
  discussed in [How the safety verdict is decided](#how-the-safety-verdict-is-decided).
  This is directly unit-tested (`test_low_confidence_does_not_downgrade_an_already_worse_verdict`).

---

## Testing

```bash
pytest tests/ -v
```

29 tests, all passing on a fresh checkout with no trained model and no API
key configured:

- `test_verdict.py` (15 tests) — rule engine correctness, including the
  low-confidence asymmetry and a malformed-`rules.yaml` rejection path.
- `test_match_template.py` (8 tests) — template loading/validation,
  coordinate rescaling, and the nearest-match diffing logic including
  contested-detection and no-detections-at-all cases.
- `test_pipeline.py` (3 tests) — full wiring with an injected fake
  detector: all-present-and-ok, one-missing-bolt, and the explanation
  fallback when no API key is set.
- `test_eval_holdout.py` — skips cleanly (not a failure) until a real
  fine-tuned model and holdout dataset exist; see
  [Current status & limitations](#current-status--limitations-read-this).

---

## Scope guardrails / what this v1 deliberately does not do

Not built, and not needed to make the core pipeline convincing:
multi-part-type auto-detection (v1 ships one template — a 4-bolt bracket),
real-time video, a mobile app, cloud deployment, or a defect class without
at least ~15-20 real labeled examples behind it. Each is a reasonable
follow-on; none of them would make the demo more convincing than a working,
honestly-evaluated single-image pipeline.
