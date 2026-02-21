#!/usr/bin/env python3
"""
Generate PDF from a markdown plan file.

Usage:
    python scripts/generate_plan_pdf.py .cursor/plans/manuscripts/manuscript.md
    python scripts/generate_plan_pdf.py .cursor/plans/manuscripts/manuscript.md -o output.pdf
    python scripts/generate_plan_pdf.py .cursor/plans/manuscripts/manuscript.md -t "Custom Title"
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from agvfm.utils.pdf_generator import markdown_to_pdf


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate PDF from markdown plan file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "markdown_file",
        type=Path,
        help="Path to markdown file",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output PDF path (default: same name as input with .pdf extension)",
    )
    parser.add_argument(
        "-t", "--title",
        type=str,
        default=None,
        help="PDF document title (default: filename without extension)",
    )
    
    args = parser.parse_args()
    
    # Resolve input file
    markdown_file = args.markdown_file.resolve()
    if not markdown_file.exists():
        print(f"❌ Error: Markdown file not found: {markdown_file}", file=sys.stderr)
        sys.exit(1)
    
    # Set default title
    if args.title is None:
        args.title = markdown_file.stem.replace("_", " ").title()
    
    try:
        pdf_path = markdown_to_pdf(
            markdown_file,
            args.output,
            args.title,
        )
        print(f"✅ PDF generated: {pdf_path}")
        print(f"   Size: {pdf_path.stat().st_size / 1024:.1f} KB")
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}", file=sys.stderr)
        print("   Install with: pip install markdown weasyprint", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error generating PDF: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
