"""Utility functions for generating PDFs from markdown plan files."""

import sys
from pathlib import Path
from typing import Optional

try:
    import markdown
    from markdown.extensions import codehilite, fenced_code, tables
except ImportError:
    markdown = None

try:
    from weasyprint import HTML, CSS
    from weasyprint.text.fonts import FontConfiguration
except ImportError:
    HTML = None
    CSS = None
    FontConfiguration = None


def markdown_to_pdf(
    markdown_file: Path,
    output_pdf: Optional[Path] = None,
    title: Optional[str] = None,
    css_style: Optional[str] = None,
) -> Path:
    """
    Convert a markdown file to PDF.
    
    Args:
        markdown_file: Path to input markdown file
        output_pdf: Path to output PDF file (default: same name as input with .pdf extension)
        title: Optional title for the PDF document
        css_style: Optional custom CSS string for styling
    
    Returns:
        Path to generated PDF file
    
    Raises:
        ImportError: If required dependencies (markdown, weasyprint) are not installed
        FileNotFoundError: If markdown file doesn't exist
    """
    if markdown is None:
        raise ImportError(
            "markdown package required. Install with: pip install markdown"
        )
    
    if HTML is None:
        raise ImportError(
            "weasyprint package required. Install with: pip install weasyprint"
        )
    
    if not markdown_file.exists():
        raise FileNotFoundError(f"Markdown file not found: {markdown_file}")
    
    # Set output path
    if output_pdf is None:
        output_pdf = markdown_file.with_suffix(".pdf")
    
    # Read markdown content
    with open(markdown_file, "r", encoding="utf-8") as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    md_extensions = [
        "codehilite",
        "fenced_code",
        "tables",
        "toc",
        "nl2br",
    ]
    
    html_body = markdown.markdown(
        md_content,
        extensions=md_extensions,
    )
    
    # Convert relative image paths to absolute paths for PDF generation
    # Markdown images like ![alt](path) need to be converted to absolute paths
    import re
    
    # Try to find project root (look for common markers like .git, pyproject.toml, etc.)
    # Strategy: Start from markdown file and walk up, but also check current working directory
    import os
    
    def find_project_root(start_path):
        """Find project root by walking up from start_path."""
        current = Path(start_path).resolve()
        for _ in range(10):  # Look up to 10 levels up
            # Prioritize agvfm directory (most specific marker for this project)
            if (current / 'agvfm').exists():
                return current
            # Then check for other project markers (but require agvfm or multiple markers)
            markers_found = sum(1 for marker in ['.git', 'pyproject.toml', 'setup.py'] 
                               if (current / marker).exists())
            # Require at least 2 markers (to avoid false positives from README.md in subdirectories)
            if markers_found >= 2:
                return current
            if current == current.parent:  # Reached filesystem root
                break
            current = current.parent
        return None
    
    # Try from markdown file location first
    project_root = find_project_root(markdown_file.parent)
    
    # If not found, try from current working directory
    if project_root is None:
        cwd = Path(os.getcwd()).resolve()
        project_root = find_project_root(cwd)
    
    # If still not found, default to markdown file's parent (fallback)
    if project_root is None:
        project_root = markdown_file.parent
    
    def convert_image_path(match):
        alt_text = match.group(1)
        img_path = match.group(2)
        # If relative path, try multiple resolution strategies
        if not Path(img_path).is_absolute():
            # Strategy 1: Resolve relative to markdown file's directory
            abs_path = (markdown_file.parent / img_path).resolve()
            if abs_path.exists():
                # Use relative path from project root for weasyprint base_url
                try:
                    rel_path = abs_path.relative_to(project_root)
                    return f'<img src="{rel_path}" alt="{alt_text}" style="max-width: 100%; height: auto;" />'
                except ValueError:
                    # If not relative to project root, use absolute path
                    return f'<img src="file://{abs_path}" alt="{alt_text}" style="max-width: 100%; height: auto;" />'
            
            # Strategy 2: Resolve relative to project root
            abs_path = (project_root / img_path).resolve()
            if abs_path.exists():
                # Use relative path from project root
                try:
                    rel_path = abs_path.relative_to(project_root)
                    return f'<img src="{rel_path}" alt="{alt_text}" style="max-width: 100%; height: auto;" />'
                except ValueError:
                    return f'<img src="file://{abs_path}" alt="{alt_text}" style="max-width: 100%; height: auto;" />'
            
            # Strategy 3: Try from current working directory
            abs_path = Path(img_path).resolve()
            if abs_path.exists():
                try:
                    rel_path = abs_path.relative_to(project_root)
                    return f'<img src="{rel_path}" alt="{alt_text}" style="max-width: 100%; height: auto;" />'
                except ValueError:
                    return f'<img src="file://{abs_path}" alt="{alt_text}" style="max-width: 100%; height: auto;" />'
        
        # If absolute path, try to make it relative to project root
        if Path(img_path).is_absolute():
            try:
                rel_path = Path(img_path).relative_to(project_root)
                return f'<img src="{rel_path}" alt="{alt_text}" style="max-width: 100%; height: auto;" />'
            except ValueError:
                return f'<img src="file://{img_path}" alt="{alt_text}" style="max-width: 100%; height: auto;" />'
        
        return match.group(0)
    
    # Replace markdown image syntax with HTML img tags with absolute paths
    html_body = re.sub(
        r'!\[([^\]]*)\]\(([^)]+)\)',
        convert_image_path,
        html_body
    )
    
    # Default CSS style
    if css_style is None:
        css_style = """
        @page {
            size: letter;
            margin: 1in;
        }
        body {
            font-family: "Times New Roman", serif;
            font-size: 11pt;
            line-height: 1.5;
            color: #000;
        }
        h1 {
            font-size: 18pt;
            font-weight: bold;
            margin-top: 24pt;
            margin-bottom: 12pt;
            page-break-after: avoid;
        }
        h2 {
            font-size: 14pt;
            font-weight: bold;
            margin-top: 18pt;
            margin-bottom: 10pt;
            page-break-after: avoid;
        }
        h3 {
            font-size: 12pt;
            font-weight: bold;
            margin-top: 14pt;
            margin-bottom: 8pt;
            page-break-after: avoid;
        }
        h4 {
            font-size: 11pt;
            font-weight: bold;
            margin-top: 12pt;
            margin-bottom: 6pt;
        }
        p {
            margin-top: 6pt;
            margin-bottom: 6pt;
            text-align: justify;
        }
        ul, ol {
            margin-top: 6pt;
            margin-bottom: 6pt;
            padding-left: 24pt;
        }
        li {
            margin-top: 3pt;
            margin-bottom: 3pt;
        }
        code {
            font-family: "Courier New", monospace;
            font-size: 10pt;
            background-color: #f5f5f5;
            padding: 2pt 4pt;
            border-radius: 3pt;
        }
        pre {
            font-family: "Courier New", monospace;
            font-size: 9pt;
            background-color: #f5f5f5;
            padding: 8pt;
            border-radius: 4pt;
            overflow-x: auto;
            page-break-inside: avoid;
        }
        blockquote {
            margin-left: 24pt;
            margin-right: 24pt;
            padding-left: 12pt;
            border-left: 3pt solid #ccc;
            font-style: italic;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin-top: 12pt;
            margin-bottom: 12pt;
            page-break-inside: avoid;
        }
        th, td {
            border: 1pt solid #ddd;
            padding: 6pt;
            text-align: left;
        }
        th {
            background-color: #f0f0f0;
            font-weight: bold;
        }
        hr {
            border: none;
            border-top: 1pt solid #ccc;
            margin: 18pt 0;
        }
        img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 12pt auto;
            page-break-inside: avoid;
        }
        figure {
            margin: 18pt 0;
            page-break-inside: avoid;
            text-align: center;
        }
        figcaption {
            font-size: 9pt;
            font-style: italic;
            margin-top: 6pt;
            text-align: center;
        }
        """
    
    # Create full HTML document
    html_doc = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title or markdown_file.stem}</title>
    <style>
        {css_style}
    </style>
</head>
<body>
    {html_body}
</body>
</html>"""
    
    # Generate PDF
    font_config = FontConfiguration()
    # Use project_root as base_url so relative image paths resolve correctly
    html_obj = HTML(string=html_doc, base_url=str(project_root))
    
    html_obj.write_pdf(
        output_pdf,
        font_config=font_config,
    )
    
    return output_pdf


def generate_plan_pdf(
    plan_file: Path,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Generate PDF from a plan markdown file in .cursor/plans/.
    
    Args:
        plan_file: Path to plan markdown file (can be relative or absolute)
        output_dir: Directory to save PDF (default: same directory as markdown file)
    
    Returns:
        Path to generated PDF file
    """
    plan_file = Path(plan_file).resolve()
    
    if output_dir is None:
        output_dir = plan_file.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    output_pdf = output_dir / f"{plan_file.stem}.pdf"
    
    return markdown_to_pdf(plan_file, output_pdf, title=plan_file.stem.replace("_", " ").title())


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate PDF from markdown plan file")
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
        help="PDF document title",
    )
    
    args = parser.parse_args()
    
    try:
        pdf_path = markdown_to_pdf(
            args.markdown_file,
            args.output,
            args.title,
        )
        print(f"✅ PDF generated: {pdf_path}")
    except Exception as e:
        print(f"❌ Error generating PDF: {e}", file=sys.stderr)
        sys.exit(1)
