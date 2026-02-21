# Manuscripts Directory

This directory contains all manuscript versions and revision tracking.

## Current Manuscript

- **Active Version:** `manuscript.md` / `manuscript.pdf`
- **Revision Tracking:** See [manuscript_index.md](manuscript_index.md)

## Structure

```
manuscripts/
├── manuscript.md          # Current active version (edit this)
├── manuscript.pdf         # Current PDF (auto-generated)
├── manuscript_index.md    # Revision tracking
├── README.md              # This file
└── versions/              # Archived versions (created as needed)
    └── vX.Y/
        ├── manuscript.md
        └── manuscript.pdf
```

## Working with Manuscripts

### Editing the Manuscript

1. **Edit:** `manuscript.md` (the active version)
2. **Generate PDF:** Run the PDF generator script
3. **Review:** Check PDF for formatting issues
4. **Archive:** Before major changes, archive current version (see below)

### Generating PDF

```bash
# From project root
python scripts/generate_plan_pdf.py .cursor/plans/manuscripts/manuscript.md \
    -t "Zero-Shot Agricultural Object Detection: A Systematic Comparison of Vision Foundation Models"
```

Or use the shorter path:
```bash
cd .cursor/plans/manuscripts
python ../../../scripts/generate_plan_pdf.py manuscript.md \
    -t "Zero-Shot Agricultural Object Detection: A Systematic Comparison of Vision Foundation Models"
```

### Archiving Versions

Before making major changes, archive the current version:

```bash
# Create version directory
mkdir -p .cursor/plans/manuscripts/versions/v1.0

# Copy current files
cp .cursor/plans/manuscripts/manuscript.md .cursor/plans/manuscripts/versions/v1.0/
cp .cursor/plans/manuscripts/manuscript.pdf .cursor/plans/manuscripts/versions/v1.0/

# Update manuscript_index.md with new version entry
```

### Version Naming

- **v1.0, v1.1, v1.2...**: Minor revisions
- **v2.0, v2.1, v2.2...**: Major revisions
- **v3.0, v3.1, v3.2...**: Structural changes

## Integration with Reviews

Manuscript revisions are tracked in relation to external reviews:

- **Review Tracking:** [../external_review/](../external_review/)
- **Review Index:** [../external_review/review_index.md](../external_review/review_index.md)

When addressing review feedback:
1. Update `manuscript_index.md` with planned changes
2. Make revisions to `manuscript.md`
3. Update `manuscript_index.md` with completed changes
4. Update review response tracking in external_review directory

## Figures

Figures are referenced with relative paths from the manuscript location:
- `experiments/results/phase1_factor_analysis/plots/...`
- `experiments/results/phase2_combinations/plots/...`
- `experiments/results/phase3_confidence_sweeps/plots/...`

The PDF generator resolves these paths relative to the project root, so figures should be accessible from the manuscript location.

## Notes

- Always generate PDF after making changes to verify formatting
- Archive versions before major revisions
- Document all changes in `manuscript_index.md`
- Keep manuscript and PDF in sync
