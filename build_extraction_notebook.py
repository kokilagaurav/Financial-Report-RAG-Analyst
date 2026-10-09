import json
from pathlib import Path

cells = []


def markdown(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.strip().splitlines(True)})


def code(text):
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                  'outputs': [], 'source': text.strip().splitlines(True)})


markdown('''
# Extract financial PDFs to Markdown

Run the cells in order with the **myenv** Python kernel. This notebook only extracts
data: it does not chunk documents, create embeddings, or build an index.

PyMuPDF 1.28.2 was used for this run. If needed, install it from a terminal:
```powershell
conda run -n myenv python -m pip install PyMuPDF==1.28.2
```
Only PyMuPDF and Python's standard library are used by the extraction code.
See the [PyMuPDF table documentation](https://pymupdf.readthedocs.io/en/latest/page.html#Page.find_tables).

**How to interpret the output:** detected tables become Markdown; uncertain table
regions remain aligned text in fenced blocks marked **NEEDS REVIEW**. Blank cells
stay blank, and figures remain strings (including signs, commas, and percentages).
Original header rows are retained as data rows beneath an empty Markdown header,
so a financial data row is never silently treated as a column heading.

Page numbers refer to the PDF's physical, 1-based pages. Reading order is approximated
top-to-bottom, then left-to-right. Complex columns, merged headers, charts, and scanned
pages require source review. No OCR or image interpretation is performed. Review
warnings are conservative checks, not a guarantee that every layout issue is detected.
''')

markdown('''## 1. Import libraries
Use the existing `myenv` environment. `fitz` below is an alias for PyMuPDF's current import name.''')
code('''
from pathlib import Path
from collections import Counter
from html import escape
import re
import sys
import pymupdf as fitz

assert Path(sys.prefix).name.lower() == "myenv", "Select the myenv notebook kernel."
print("Python:", sys.executable)
print("PyMuPDF:", fitz.VersionBind)
''')

markdown('''## 2. Define paths and find PDFs
Works when Jupyter starts in the project root or its `notebook/` folder.
Relative paths preserve every company/year subfolder and the original filename.''')
code('''
project_root = Path.cwd()
if not (project_root / "data" / "raw").is_dir():
    project_root = project_root.parent

input_dir = project_root / "data" / "raw"
output_dir = project_root / "data" / "interim"
pdf_paths = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() == ".pdf")
assert pdf_paths, f"No PDFs found in {input_dir}"
print(f"Found {len(pdf_paths)} PDFs in {input_dir}")
''')

markdown('''## 3. Arrange extracted words into readable lines
Group nearby word baselines, then read each line from left to right. For uncertain
tables, spaces approximate the original horizontal positions instead of flattening columns.''')
code('''
def word_lines(words):
    lines = []
    for word in sorted(words, key=lambda w: (w[3], w[0])):
        if not lines or abs(word[3] - lines[-1][0][3]) > 3:
            lines.append([])
        lines[-1].append(word)
    return [sorted(line, key=lambda w: w[0]) for line in lines]


def aligned_text(lines):
    left = min(w[0] for line in lines for w in line)
    widths = [(w[2] - w[0]) / len(w[4]) for line in lines for w in line if w[4]]
    space_width = sorted(widths)[len(widths) // 2] or 4
    result = []
    for line in lines:
        text = ""
        for word in line:
            column = round((word[0] - left) / space_width)
            text += " " * max(1 if text else 0, column - len(text)) + word[4]
        result.append(text)
    return "\\n".join(result)


def inside(word, box):
    return fitz.Point((word[0] + word[2]) / 2, (word[1] + word[3]) / 2) in box
''')

markdown('''## 4. Format table cells as Markdown
Keep all source rows, including multi-row headers. Escape Markdown/HTML characters
and retain line breaks within cells. Join adjacent currency-symbol/value columns and
omit completely empty spacer columns. Never fill missing cells with guessed values.''')
code('''
def markdown_table(rows):
    def cell_text(value):
        text = escape(value or "", quote=False)
        return text.replace("\\\\", "&#92;").replace("|", "&#124;").replace("\\n", "<br>")

    rows = [list(row) for row in rows]
    column = 0
    while column < len(rows[0]) - 1:
        values = [row[column] for row in rows]
        next_values = [row[column + 1] for row in rows if row[column + 1]]
        currency = any(value in ("$", "€", "£") for value in values)
        numeric = next_values and all(re.fullmatch(r"[\\d,().%+−-]+", v) for v in next_values)
        if currency and numeric:
            for row in rows:
                row[column:column + 2] = [" ".join(v for v in row[column:column + 2] if v)]
        column += 1
    keep = [i for i in range(len(rows[0])) if any(row[i] for row in rows)]
    rows = [[row[i] for i in keep] for row in rows]
    columns = len(keep)
    lines = ["| " + " | ".join([""] * columns) + " |",
             "| " + " | ".join(["---"] * columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(cell_text(value) for value in row) + " |")
    return "\\n".join(lines)
''')

markdown('''## 5. Extract tables and check text coverage
Use PyMuPDF's line-based table detector. Include external header cells when available.
Before removing table words from prose, compare **all non-whitespace characters** in
the source region with the extracted cells. This catches dropped figures and punctuation,
but does not prove that cell relationships are correct. Merged cells are flagged.
If the comparison fails, retain the source words for the review fallback.''')
code('''
def extract_tables(page, words):
    items, used, warnings = [], set(), []
    try:
        tables = page.find_tables().tables
    except Exception as error:
        return items, used, [f"Table detection failed ({type(error).__name__}); review source layout."]

    for table in tables:
        if table.col_count < 2:
            continue
        rows = table.extract()
        boxes = [fitz.Rect(table.bbox)]
        if table.header.external:
            rows = [table.header.names] + rows
            boxes += [fitz.Rect(b) for b in table.header.cells if b is not None]

        selected = {i for i, word in enumerate(words) if any(inside(word, b) for b in boxes)}
        source = "".join(words[i][4] for i in sorted(selected))
        extracted = "".join(value or "" for row in rows for value in row)
        source_chars = Counter(re.sub(r"\\s", "", source))
        table_chars = Counter(re.sub(r"\\s", "", extracted))
        if not rows or not selected or selected & used or source_chars != table_chars:
            warnings.append("A detected table failed text coverage/overlap checks; its source text is retained.")
            continue

        note = ""
        if any(value is None for row in rows for value in row):
            note = "> NEEDS REVIEW: merged/missing cells; verify column and header relationships.\\n\\n"
        top = min(b.y0 for b in boxes)
        items.append((top, table.bbox[0], note + markdown_table(rows)))
        used.update(selected)
    return items, used, warnings
''')

markdown('''## 6. Preserve uncertain table regions
Repeated rows with separated numeric columns suggest a borderless or missed table.
Keep these rows and their intervening text together in an aligned review block.
This intentionally avoids the text detector's tendency to split ordinary row labels.
The heuristic can miss small/non-numeric tables; check the source before using the data.''')
code('''
def extract_remaining_text(words):
    lines = word_lines(words)
    candidates = []
    for i, line in enumerate(lines):
        numbers = [w for w in line if re.fullmatch(r"[($€£+−-]*\\d[\\d,./%()−-]*", w[4])]
        if any(b[0] - a[2] > 15 for a, b in zip(numbers, numbers[1:])):
            candidates.append(i)

    groups = []
    for i in candidates:
        if not groups or lines[i][0][1] - lines[groups[-1][-1]][0][1] > 45:
            groups.append([])
        groups[-1].append(i)
    bands = {g[0]: g[-1] for g in groups if len(g) >= 2}

    items = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if i in bands:
            end = bands[i] + 1
            text = aligned_text(lines[i:end])
            fence = "`" * max(3, max((len(s) + 1 for s in re.findall(r"`+", text)), default=3))
            text = "> NEEDS REVIEW: possible borderless/incomplete table; source alignment retained.\\n\\n" + fence + "text\\n" + text + "\\n" + fence
            i = end
        else:
            text = escape(" ".join(w[4] for w in line), quote=False)
            text = re.sub(r"([\\\\`*_\\[\\]|])", r"\\\\\\1", text)
            text = re.sub(r"^(#{1,6}|>|[-+])(?= )", r"\\\\\\1", text)
            i += 1
        items.append((min(w[1] for w in line), line[0][0], text))
    return items
''')

markdown('''## 7. Extract one PDF with page boundaries
Combine tables and remaining text by page position. Words assigned to a table are
excluded from prose, preventing duplication. Keep headings, page furniture, and
source wording; flag unreadable characters and pages without extractable text.''')
code('''
def extract_pdf(pdf_path):
    relative = pdf_path.relative_to(input_dir).as_posix()
    sections = [f"# {relative}\\n\\nSource: data/raw/{relative}",
                "> Extraction note: empty Markdown headers are structural placeholders. "
                "Source header rows are retained below them. Check all NEEDS REVIEW notices."]
    table_count = 0
    with fitz.open(pdf_path) as document:
        for page in document:
            words = page.get_text("words", sort=True)
            items, used, warnings = extract_tables(page, words)
            table_count += len(items)
            remaining = [w for i, w in enumerate(words) if i not in used]
            items += extract_remaining_text(remaining)
            if not words:
                warnings.append("No extractable text; inspect this page for scanned text/images. OCR was not run.")
            if any("�" in w[4] for w in words):
                warnings.append("The PDF text contains replacement characters; compare with the visible page.")
            body = "\\n\\n".join(text for _, _, text in sorted(items, key=lambda item: (item[0], item[1])))
            notices = "\\n\\n".join("> NEEDS REVIEW: " + message for message in warnings)
            sections.append(f"## Page {page.number + 1}\\n\\n" + notices + "\\n\\n" + body)
        page_count = len(document)
    return "\\n\\n---\\n\\n".join(sections) + "\\n", page_count, table_count
''')

markdown('''## 8. Process all PDFs
Store this small collection's extracted documents in memory. Print progress after
each report; a page may contain multiple review notices.''')
code('''
extracted_reports = []
for pdf_path in pdf_paths:
    text, page_count, table_count = extract_pdf(pdf_path)
    extracted_reports.append((pdf_path, text, page_count, table_count))
    print(f"{pdf_path.relative_to(input_dir)}: {page_count} pages, "
          f"{table_count} Markdown tables, {text.count('> NEEDS REVIEW:')} review notices", flush=True)
''')

markdown('''## 9. Save Markdown files
Create matching subfolders and save UTF-8 text. Re-running replaces only the Markdown
file corresponding to each input PDF; it does not modify source PDFs.''')
code('''
for pdf_path, text, _, _ in extracted_reports:
    output_path = output_dir / pdf_path.relative_to(input_dir).with_suffix(".md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
print(f"Saved {len(extracted_reports)} Markdown files to {output_dir}")
''')

markdown('''## 10. Verify files and page boundaries
Check saved content, matching paths, and every page marker. These checks verify file
integrity, not semantic accuracy. Inspect flagged pages in the PDF before chunking.
The review summary stays in the notebook, keeping `data/interim/` limited to report files.''')
code('''
for pdf_path, text, page_count, table_count in extracted_reports:
    output_path = output_dir / pdf_path.relative_to(input_dir).with_suffix(".md")
    saved = output_path.read_text(encoding="utf-8")
    assert saved == text, f"Saved content differs: {output_path}"
    markers = re.findall(r"^## Page (\\d+)$", saved, flags=re.MULTILINE)
    assert markers == [str(i) for i in range(1, page_count + 1)]
    review_pages = [i for i, section in enumerate(saved.split("## Page ")[1:], 1)
                    if "> NEEDS REVIEW:" in section]
    print(f"{output_path.relative_to(output_dir)}: verified; review pages {review_pages}")

print("\\nPreview:\\n", extracted_reports[0][1][:1800])
''')

notebook = {
    'cells': cells,
    'metadata': {
        'kernelspec': {'display_name': 'Python (myenv)', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
    },
    'nbformat': 4,
    'nbformat_minor': 5,
}
for i, cell in enumerate(cells):
    cell['id'] = f'extraction-{i:02d}'
Path('notebook/data_extracting.ipynb').write_text(json.dumps(notebook, indent=1) + '\n', encoding='utf-8')
