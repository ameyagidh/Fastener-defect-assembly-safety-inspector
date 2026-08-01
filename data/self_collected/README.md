# Self-collected training images

Put your own labeled fastener photos here in YOLO format:

```
data/self_collected/
    images/*.jpg
    labels/*.txt   # one per image, same basename, YOLO box format
```

These are the images you take yourself (e.g. of a bolted bracket in good and
bad states) to cover defect classes that public datasets don't adequately
represent — see `train/README.md` for the full rationale and workflow.

Actual image files are gitignored (`.jpg`/`.png` under this directory) so the
repo doesn't bloat with binary training data; only this README is tracked.
Record provenance for any images you add here (where/when taken, what
class(es) they represent) so the dataset's composition stays auditable.
