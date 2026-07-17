"""
Convert report/REPORT.md to a styled PDF.

Method: markdown -> HTML (styled) -> PDF via Microsoft Edge headless printing.
Run from the project folder with:  py report/make_pdf.py
"""

import subprocess
from pathlib import Path

import markdown

REPORT_DIR = Path(__file__).resolve().parent
MD_PATH = REPORT_DIR / "REPORT.md"
HTML_PATH = REPORT_DIR / "REPORT.html"
PDF_PATH = REPORT_DIR / "REPORT.pdf"

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

CSS = """
body { font-family: 'Segoe UI', Calibri, Arial, sans-serif; font-size: 11pt;
       line-height: 1.5; color: #222; max-width: 800px; margin: 0 auto; padding: 24px; }
h1 { font-size: 20pt; color: #1a365d; border-bottom: 3px solid #1a365d; padding-bottom: 6px; }
h2 { font-size: 15pt; color: #1a365d; border-bottom: 1px solid #cbd5e0; padding-bottom: 3px; margin-top: 26px; }
h3 { font-size: 12.5pt; color: #2c5282; margin-top: 18px; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 10pt; }
th, td { border: 1px solid #a0aec0; padding: 5px 9px; text-align: left; }
th { background: #edf2f7; }
tr:nth-child(even) { background: #f7fafc; }
code { background: #edf2f7; padding: 1px 4px; border-radius: 3px;
       font-family: Consolas, monospace; font-size: 9.5pt; }
pre { background: #f7fafc; border: 1px solid #cbd5e0; border-radius: 4px;
      padding: 10px; font-size: 9pt; overflow-x: hidden; }
pre code { background: none; }
hr { border: none; border-top: 1px solid #cbd5e0; margin: 20px 0; }
a { color: #2b6cb0; }
h2 { page-break-after: avoid; }
table, pre { page-break-inside: avoid; }
"""


def main():
    md_text = MD_PATH.read_text(encoding="utf-8")
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    html = f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"
    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"HTML written: {HTML_PATH}")

    subprocess.run(
        [EDGE, "--headless", "--disable-gpu", f"--print-to-pdf={PDF_PATH}",
         "--no-pdf-header-footer", str(HTML_PATH)],
        check=True, timeout=120,
    )
    print(f"PDF written:  {PDF_PATH} ({PDF_PATH.stat().st_size/1000:.0f} KB)")


if __name__ == "__main__":
    main()
