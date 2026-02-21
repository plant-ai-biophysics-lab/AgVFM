# Manuscript Revision Tracking

> **Purpose:** Track all versions and revisions of the manuscript

## Current Manuscript

**Active Version:** v1.0 (Initial Complete Draft)  
**Location:** `manuscripts/manuscript.md`  
**PDF:** `manuscripts/manuscript.pdf`  
**Last Updated:** 2025-02-19  
**Status:** Complete (Phases 1-3), Pending Review Response

## Revision History

| Version | Date | Author | Changes | Status | Notes |
|---------|------|--------|---------|--------|-------|
| v1.0 | 2025-02-19 | - | Initial complete draft with all phases | Complete | Includes all 9 figures, all sections through Conclusion |
| - | - | - | - | - | - |

## Version Naming Convention

- **v1.0, v1.1, v1.2...**: Minor revisions (typos, clarifications, figure updates)
- **v2.0, v2.1, v2.2...**: Major revisions (new sections, significant content changes)
- **v3.0, v3.1, v3.2...**: Major structural changes (reorganization, new experiments)

## File Organization

### Current Structure
```
manuscripts/
├── manuscript.md          # Current active version
├── manuscript.pdf         # Current PDF
├── manuscript_index.md    # This file (revision tracking)
└── versions/              # Archived versions (created as needed)
    ├── v1.0/
    │   ├── manuscript.md
    │   └── manuscript.pdf
    └── ...
```

### Archiving Versions

When creating a new version:
1. Copy current `manuscript.md` to `versions/vX.Y/manuscript.md`
2. Copy current `manuscript.pdf` to `versions/vX.Y/manuscript.pdf`
3. Update this index with the new version entry
4. Make changes to `manuscript.md` in the root manuscripts directory

## Review Integration

**External Review:** [review_001_20250219.md](../external_review/review_001_20250219.md)  
**Review Status:** Pending Response  
**Planned Revisions Based on Review:**
- [ ] Add references section
- [ ] Add statistical analysis (confidence intervals)
- [ ] Clarify dev set usage
- [ ] Strengthen generalizability claims
- [ ] Expand threshold sensitivity analysis
- [ ] Add comparison tables
- [ ] Deepen discussion section

## Change Log Template

When making revisions, document changes here:

### Version X.Y (YYYY-MM-DD)
**Author:** [Name]  
**Changes:**
- [ ] Section X: [Description of change]
- [ ] Section Y: [Description of change]
- [ ] Figures: [Description of change]

**Review Response:**
- Addressed review item #X: [Description]
- Deferred review item #Y: [Reason]

**Status:** [Draft | In Review | Complete]

---

## Quick Reference

**Generate PDF:**
```bash
python scripts/generate_plan_pdf.py .cursor/plans/manuscripts/manuscript.md -t "Zero-Shot Agricultural Object Detection: A Systematic Comparison of Vision Foundation Models"
```

**Archive Current Version:**
```bash
# Before making changes, archive current version
mkdir -p .cursor/plans/manuscripts/versions/v1.0
cp .cursor/plans/manuscripts/manuscript.md .cursor/plans/manuscripts/versions/v1.0/
cp .cursor/plans/manuscripts/manuscript.pdf .cursor/plans/manuscripts/versions/v1.0/
```
