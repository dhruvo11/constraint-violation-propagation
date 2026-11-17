"""Build the final project report (Word) in the RV University project-report template.

The formatting (page size and margins, styles, list numbering, footers with page numbers, the
university logo, and the exact cover, certificate and declaration layout) is taken from a
template report; the text comes from report/final_report/content/*.txt, written in a small
markup (see parse_content). Uses the Python standard library only.

  build_report.py --template some_report.docx    first run: extract the template's formatting
  build_report.py                                  rebuild report/final_report/CVP_Final_Report.docx
  build_report.py --pages pages.json               fill the contents page numbers (see report_pages.py)
"""
import argparse
import json
import os
import re
import struct
import sys
import zipfile
from xml.sax.saxutils import escape

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(BASE, "report", "final_report")
PARTS = os.path.join(ROOT, "template_parts")
CONTENT = os.path.join(ROOT, "content")
OUT = os.path.join(ROOT, "CVP_Final_Report.docx")

TITLE = "CONSTRAINT VIOLATION PROPAGATION: ATTRIBUTING DOCUMENT FORGERY WHEN VISUAL EVIDENCE FAILS"
STUDENTS = [("HARSHITH RUDRESH", "Harshith Rudresh", "1RUA24CSE0157"), ("DHRUV S P", "Dhruv S P", "1RUA24CSE0133")]
GUIDE, GUIDE_TITLE = "[Guide Name]", "[Designation]"
TEXT_W = 8306  # twips between the margins (A4, 1.5 in left, 1 in right)
FONT = "Times New Roman"
COPY = ["word/styles.xml", "word/numbering.xml", "word/settings.xml", "word/fontTable.xml",
        "word/_rels/fontTable.xml.rels", "word/theme/theme1.xml", "word/footer1.xml", "word/footer2.xml",
        "word/footer3.xml", "[Content_Types].xml", "_rels/.rels"]


# ------------------------------------------------------------------ template extraction
def top_level(body):
    """Top-level elements (paragraphs, tables, section properties) of a document body, as strings."""
    tok = re.compile(r"<(/?)w:(p|tbl|sectPr)\b[^>]*?(/?)>")
    out, depth, start = [], 0, 0
    for m in tok.finditer(body):
        close, selfc = m.group(1), m.group(3)
        if not close and not selfc:
            if depth == 0:
                start = m.start()
            depth += 1
        elif not close and selfc:
            if depth == 0:
                out.append(body[m.start():m.end()])
        else:
            depth -= 1
            if depth == 0:
                out.append(body[start:m.end()])
    return out


def text_of(xml):
    return re.sub(r"<[^>]+>", "", xml)


def set_text(xml, new):
    """Replace all the text of a paragraph by `new`, kept in its first text run."""
    first = [True]

    def sub(m):
        if first[0]:
            first[0] = False
            return m.group(1) + escape(new) + "</w:t>"
        return m.group(1) + "</w:t>"
    return re.sub(r"(<w:t(?: [^>]*)?>)[^<]*</w:t>", sub, xml)


def rows_of(tbl):
    return re.findall(r"<w:tr[ >].*?</w:tr>", tbl, re.S)


def extract(template):
    z = zipfile.ZipFile(template)
    os.makedirs(os.path.join(PARTS, "word", "media"), exist_ok=True)
    for name in COPY + [n for n in z.namelist() if n.startswith("word/fonts/")]:
        dest = os.path.join(PARTS, name)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as fh:
            fh.write(z.read(name))
    doc = z.read("word/document.xml").decode("utf8")
    root_tag = re.search(r"<w:document[^>]*>", doc).group(0)
    body = doc[doc.index("<w:body>") + 8: doc.rindex("</w:body>")]
    els = top_level(body)
    end = next(i for i, e in enumerate(els) if "<w:sectPr" in e)  # end of the cover section
    front = els[:end + 1]
    # cover: our title, names and guide
    for i, e in enumerate(front):
        t = text_of(e).strip()
        if t.startswith("DYSLEXI ADAPT"):
            front[i] = set_text(e, TITLE)
        elif t.startswith("JEETHU"):
            front[i] = set_text(e, GUIDE.upper())
        elif t.startswith("Assistant Professor, School"):
            front[i] = set_text(e, f"{GUIDE_TITLE}, School of Computer Science and Engineering")
    names = [i for i, e in enumerate(front) if re.match(r"[A-Z ]+\(1RUA24CSE\d{4}\)$", text_of(e).strip())]
    for i, (upper, _, usn) in zip(names, STUDENTS):
        front[i] = set_text(front[i], f"{upper} ({usn})")
    for i in reversed(names[len(STUDENTS):]):
        del front[i]
    # certificate and declaration text, rebuilt in the template's paragraph formats
    for i, e in enumerate(front):
        t = text_of(e)
        if t.startswith("This is to certify"):
            who = ", ".join(f"{name} (USN: {usn})" for _, name, usn in STUDENTS[:-1]) + \
                f" and {STUDENTS[-1][1]} (USN: {STUDENTS[-1][2]})"
            front[i] = para([run("This is to certify that the project work titled "), run(f"“{TITLE}”", bold=True),
                             run(f" is performed by {who}, bonafide students of Bachelor of Technology (Honours) at the "
                                 "School of Computer Science and Engineering, RV University, Bengaluru, in partial "
                                 "fulfillment for the award of the degree Bachelor of Technology (Honours) in Computer "
                                 "Science & Engineering, during the Academic year 2026–2027.")],
                            after=360, line=360)
        elif t.startswith("We hereby declare"):
            front[i] = para([run("We hereby declare that the thesis entitled "), run(f"“{TITLE}”", bold=True),
                             run(" submitted by us, for the award of the degree Bachelor of Technology (Honours) in "
                                 "Computer Science and Engineering at RV University, Bengaluru, is a record of bonafide "
                                 f"work carried out by us under the supervision and guidance of {GUIDE}, {GUIDE_TITLE}, "
                                 "School of Computer Science and Engineering, RV University.")],
                            after=240, line=360)
        elif e.startswith("<w:tbl") and "Jeethu V Devasia" in t:
            front[i] = e.replace("Jeethu V Devasia", GUIDE).replace(">Assistant Professor<", f">{GUIDE_TITLE}<")
        elif e.startswith("<w:tbl") and "Name of the Candidate" in t:
            rows = rows_of(e)
            keep = rows[:1] + [set_text(rows[1 + k], f"{upper} ({usn})") for k, (upper, _, usn) in enumerate(STUDENTS)]
            front[i] = e[:e.index(rows[0])] + "".join(keep) + e[e.index(rows[-1]) + len(rows[-1]):]
    ack_sig = next(e for e in els if e.startswith("<w:tbl") and "Place: Bangalore" in text_of(e))
    cells = re.findall(r"<w:tc>.*?</w:tc>", ack_sig, re.S)
    right = cells[1]
    paras = re.findall(r"<w:p[ >].*?</w:p>", right, re.S)
    new_right = right.replace("".join(paras), "".join(set_text(paras[0], name) for _, name, _ in STUDENTS))
    ack_sig = ack_sig.replace(right, new_right)
    rels = z.read("word/_rels/document.xml.rels").decode("utf8")
    used = set(re.findall(r'r:(?:embed|id)="(rId\d+)"', "".join(front)))
    keep_rels = {}
    for rid, typ, target in re.findall(r'Id="(rId\d+)" Type="([^"]+)" Target="([^"]+)"', rels):
        if rid in used or typ.endswith("/footer"):
            keep_rels[rid] = (typ, target)
            if typ.endswith("/image"):
                with open(os.path.join(PARTS, "word", target), "wb") as fh:
                    fh.write(z.read("word/" + target))
    footers = {t: r for r, (ty, t) in keep_rels.items() if ty.endswith("/footer")}
    json.dump({"root_tag": root_tag, "front": front, "ack_sig": ack_sig, "rels": keep_rels, "footers": footers},
              open(os.path.join(PARTS, "front.json"), "w"), indent=0)
    print(f"template formatting extracted to {PARTS}")


# ------------------------------------------------------------------ XML building blocks
def rpr(bold=False, italic=False, size=12, font=FONT, underline=False, color=None):
    s = f'<w:rFonts w:ascii="{font}" w:cs="{font}" w:eastAsia="{font}" w:hAnsi="{font}"/>'
    s += '<w:b w:val="1"/><w:bCs w:val="1"/>' if bold else '<w:b w:val="0"/><w:bCs w:val="0"/>'
    s += '<w:i w:val="1"/><w:iCs w:val="1"/>' if italic else '<w:i w:val="0"/><w:iCs w:val="0"/>'
    if color:
        s += f'<w:color w:val="{color}"/>'
    s += f'<w:sz w:val="{int(size * 2)}"/><w:szCs w:val="{int(size * 2)}"/>'
    if underline:
        s += '<w:u w:val="single"/>'
    return f"<w:rPr>{s}</w:rPr>"


def run(text, **kw):
    parts = text.split("\t")
    inner = "<w:tab/>".join(f'<w:t xml:space="preserve">{escape(p)}</w:t>' for p in parts)
    return f"<w:r>{rpr(**kw)}{inner}</w:r>"


def inline(text, size=12, bold=False, italic=False):
    """Runs for text with **bold**, *italic* and `code` markup."""
    out = []
    for tok in re.split(r"(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`)", text):
        if not tok:
            continue
        if tok.startswith("**"):
            out.append(run(tok[2:-2], bold=True, italic=italic, size=size))
        elif tok.startswith("*"):
            out.append(run(tok[1:-1], bold=bold, italic=True, size=size))
        elif tok.startswith("`"):
            out.append(run(tok[1:-1], bold=bold, italic=italic, size=size - 1, font="Courier New"))
        else:
            out.append(run(tok, bold=bold, italic=italic, size=size))
    return out


def para(runs, jc="both", after=120, before=0, line=360, ind=None, keep_next=False, page_break=False,
         style=None, num=None, tabs=None, keep_lines=False, sect=None, outline=None, leader="none"):
    p = ""
    if style:
        p += f'<w:pStyle w:val="{style}"/>'
    if keep_next:
        p += '<w:keepNext w:val="1"/>'
    if keep_lines:
        p += '<w:keepLines w:val="1"/>'
    if page_break:
        p += '<w:pageBreakBefore w:val="1"/>'
    p += '<w:widowControl w:val="1"/>'
    if num:
        p += f'<w:numPr><w:ilvl w:val="0"/><w:numId w:val="{num}"/></w:numPr>'
    if tabs:
        p += "<w:tabs>" + "".join(f'<w:tab w:val="right" w:leader="{leader}" w:pos="{t}"/>' for t in tabs) + "</w:tabs>"
    p += f'<w:spacing w:after="{after}" w:before="{before}" w:line="{line}" w:lineRule="auto"/>'
    if ind:
        p += "<w:ind " + " ".join(f'w:{k}="{v}"' for k, v in ind.items()) + "/>"
    p += f'<w:jc w:val="{jc}"/>'
    if outline is not None:
        p += f'<w:outlineLvl w:val="{outline}"/>'
    if sect:
        p += sect
    return f"<w:p><w:pPr>{p}</w:pPr>{''.join(runs)}</w:p>"


def page_break():
    return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


def title(text, page_break_before=True):
    """Front-matter title: 14 pt bold, underlined, centred (ABSTRACT, ACKNOWLEDGEMENT, ...)."""
    return para([run(text, bold=True, size=14, underline=True)], jc="center", after=240, line=360,
                keep_next=True, page_break=page_break_before, outline=0)


def border(val="single", sz=4):
    return "".join(f'<w:{s} w:color="000000" w:space="0" w:sz="{sz}" w:val="{val}"/>'
                   for s in ("top", "left", "bottom", "right"))


def table(rows, widths, header=True, size=10, first_bold=False, center_cols=()):
    """A bordered table as in the template: header row bold and centred, 10 pt text."""
    total = sum(widths)
    w = [round(TEXT_W * x / total) for x in widths]
    w[-1] += TEXT_W - sum(w)
    out = ['<w:tbl><w:tblPr><w:tblW w:w="%d" w:type="dxa"/><w:jc w:val="center"/><w:tblBorders>%s'
           '<w:insideH w:color="000000" w:space="0" w:sz="4" w:val="single"/>'
           '<w:insideV w:color="000000" w:space="0" w:sz="4" w:val="single"/></w:tblBorders>'
           '<w:tblLayout w:type="fixed"/><w:tblLook w:val="0000"/></w:tblPr>' % (TEXT_W, border())]
    out.append("<w:tblGrid>" + "".join(f'<w:gridCol w:w="{x}"/>' for x in w) + "</w:tblGrid>")
    for r, cells in enumerate(rows):
        head = header and r == 0
        out.append("<w:tr><w:trPr><w:cantSplit w:val=\"1\"/>" + ('<w:tblHeader w:val="1"/>' if head else "") + "</w:trPr>")
        for c, cell in enumerate(cells):
            jc = "center" if head or c in center_cols else "left"
            bold = head or (first_bold and c == 0)
            out.append(f'<w:tc><w:tcPr><w:tcW w:w="{w[c]}" w:type="dxa"/><w:tcBorders>{border()}</w:tcBorders>'
                       '<w:tcMar><w:top w:w="50" w:type="dxa"/><w:left w:w="90" w:type="dxa"/>'
                       '<w:bottom w:w="50" w:type="dxa"/><w:right w:w="90" w:type="dxa"/></w:tcMar>'
                       '<w:vAlign w:val="center"/></w:tcPr>')
            for line in (cell.split("<br>") or [""]):
                out.append(para(inline(line.strip(), size=size, bold=bold), jc=jc, after=0, line=252))
            out.append("</w:tc>")
        out.append("</w:tr>")
    out.append("</w:tbl>")
    return "".join(out)


def caption(text):
    return para(inline(text, size=11, bold=True), jc="center", after=100, before=160, line=276, keep_next=True,
                style="TableCaption")


def png_size(path):
    with open(path, "rb") as fh:
        head = fh.read(24)
    return struct.unpack(">II", head[16:24])


class Images:
    def __init__(self):
        self.items = []

    def add(self, path):
        rid = f"rIdImg{len(self.items) + 1}"
        self.items.append((rid, path))
        return rid


def figure(path, width_in, cap, images, n):
    w, h = png_size(path)
    cx = int(width_in * 914400)
    cy = int(cx * h / w)
    rid = images.add(path)
    drawing = (f'<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>'
               f'<wp:effectExtent l="0" t="0" r="0" b="0"/><wp:docPr id="{100 + n}" name="Figure {n}" descr="{escape(cap)}"/>'
               '<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>'
               '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
               f'<pic:pic><pic:nvPicPr><pic:cNvPr id="{100 + n}" name="Figure {n}"/><pic:cNvPicPr/></pic:nvPicPr>'
               f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
               f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect">'
               '<a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>')
    return (para([drawing], jc="center", after=60, before=120, line=240, keep_next=True) +
            para(inline(cap, size=11, bold=True), jc="center", after=200, line=276, style="FigureCaption"))


# ------------------------------------------------------------------ content markup
class Doc:
    def __init__(self):
        self.xml, self.toc, self.tables, self.figures = [], [], [], []
        self.images, self.lists, self.nfig = Images(), [], 0

    def heading(self, level, text):
        size = {1: 16, 2: 14, 3: 12}[level]
        kw = dict(style=f"Heading{level}", keep_next=True, outline=level - 1)
        if level == 1:
            self.xml.append(para([run(text, bold=True, size=size)], jc="center", after=240, line=360,
                                 page_break=True, **kw))
        else:
            self.xml.append(para([run(text, bold=True, size=size)], jc="left", before=280 if level == 2 else 200,
                                 after=140 if level == 2 else 100, line=360, **kw))
        if level <= 2:
            self.toc.append((level, text))

    def new_list(self):
        self.lists.append(200 + len(self.lists))
        return self.lists[-1]


def parse_content(doc, text):
    """Markup: '# ' chapter, '## ' section, '### ' subsection; blank lines separate paragraphs;
    '- ' bullets; '1. ' numbered items; **bold**, *italic*, `code`;
    '!table <caption>' / '!widths a,b,c' / rows 'x | y | z' / '!end';
    '!figure <png> | <width in> | <caption>'; '!eq <equation>'; '!code <pt>' ... '!end';
    '!refs' ... '!end' (one reference per line); '!pagebreak'."""
    lines = text.split("\n")
    i, buf, list_num = 0, [], None

    def flush():
        if buf:
            doc.xml.append(para(inline(" ".join(buf))))
            buf.clear()
    while i < len(lines):
        ln = lines[i].rstrip()
        s = ln.strip()
        if s.startswith("#"):
            flush()
            level = len(s) - len(s.lstrip("#"))
            doc.heading(level, s[level:].strip())
            list_num = None
        elif s.startswith("!table"):
            flush()
            cap = s[len("!table"):].strip()
            widths, rows, opts = None, [], set()
            i += 1
            while lines[i].strip() != "!end":
                t = lines[i].strip()
                if t.startswith("!widths"):
                    widths = [float(x) for x in t.split()[1].split(",")]
                elif t.startswith("!opts"):
                    opts = set(t.split()[1:])
                elif t:
                    rows.append([c.strip() for c in lines[i].rstrip().split("|")])
                i += 1
            widths = widths or [1] * len(rows[0])
            centre = tuple(range(1, len(rows[0]))) if "center" in opts else ()
            doc.xml.append(caption(cap))
            doc.xml.append(table(rows, widths, first_bold="firstbold" in opts, center_cols=centre))
            doc.xml.append(para([], after=120, line=240))
            doc.tables.append(cap)
        elif s.startswith("!figure"):
            flush()
            path, width, cap = [x.strip() for x in s[len("!figure"):].split("|", 2)]
            doc.nfig += 1
            doc.xml.append(figure(os.path.join(BASE, path), float(width), cap, doc.images, doc.nfig))
            doc.figures.append(cap)
        elif s.startswith("!eq"):
            flush()
            doc.xml.append(para([run(s[3:].strip(), italic=False)], jc="center", before=60, after=120, line=300))
        elif s.startswith("!code"):
            flush()
            pt = float(s.split()[1]) if len(s.split()) > 1 else 9
            i += 1
            while lines[i].strip() != "!end":
                doc.xml.append(para([run(lines[i].rstrip("\n"), size=pt, font="Courier New")], jc="left", after=0,
                                    line=240, ind={"left": 240}))
                i += 1
            doc.xml.append(para([], after=120, line=240))
        elif s.startswith("!refs"):
            flush()
            i += 1
            while lines[i].strip() != "!end":
                if lines[i].strip():
                    doc.xml.append(para(inline(lines[i].strip()), jc="left", after=140, line=300,
                                        ind={"left": 720, "hanging": 720}))
                i += 1
        elif s == "!pagebreak":
            flush()
            doc.xml.append(page_break())
        elif s.startswith("- "):
            flush()
            doc.xml.append(para(inline(s[2:]), after=60, line=360, num=11, ind={"left": 720, "hanging": 360}))
        elif re.match(r"\d+\. ", s):
            flush()
            if list_num is None:
                list_num = doc.new_list()
            doc.xml.append(para(inline(s[s.index(" ") + 1:]), after=60, line=360, num=list_num,
                                ind={"left": 720, "hanging": 360}))
        elif not s:
            flush()
            list_num = None if (i + 1 < len(lines) and not re.match(r"\s*\d+\. ", lines[i + 1])) else list_num
        else:
            buf.append(s)
        i += 1
    flush()


# ------------------------------------------------------------------ front matter
def front_matter(f, doc, pages, abstract, ack, abbrev):
    out = list(f["front"])
    out.append(title("ABSTRACT", page_break_before=False))
    for p in abstract:
        if p.startswith("Keywords:"):
            out.append(para([run("Keywords:", bold=True), run(" "), run(p[len("Keywords:"):].strip(), italic=True)],
                            after=80, line=288))
        else:
            out.append(para(inline(p), after=80, line=288))
    out.append(title("ACKNOWLEDGEMENT"))
    for p in ack:
        out.append(para(inline(p), after=120, line=360))
    out.append(para([], after=240, line=240))
    out.append(f["ack_sig"])
    # table of contents and the lists of tables and figures: plain, editable text, as in the template.
    # Page numbers come from a rendering pass (scripts/report_pages.py); each sits on its entry's line at
    # a right tab on the margin, with a dot leader.
    pg = lambda key: str(pages.get(key, ""))  # noqa: E731

    def toc_line(text, page, level):
        return para([run(f"{text}\t{page}", bold=(level == 1), size=11)], jc="left", line=264,
                    after=40 if level == 1 else 30, before=100 if level == 1 else 0, tabs=[TEXT_W], leader="dot",
                    ind={"left": 480 if level == 2 else 0, "right": 0, "firstLine": 0}, keep_lines=True)
    out.append(title("TABLE OF CONTENTS"))
    fixed = ["ABSTRACT", "ACKNOWLEDGEMENT", "TABLE OF CONTENTS", "LIST OF TABLES", "LIST OF FIGURES",
             "LIST OF SYMBOLS, ABBREVIATIONS AND NOMENCLATURE"]
    for name in fixed:
        out.append(toc_line(name, pg(name), 1))
    for level, text in doc.toc:
        out.append(toc_line(text, pg(text), level))
    for heading, items, word in [("LIST OF TABLES", doc.tables, "Table"), ("LIST OF FIGURES", doc.figures, "Figure")]:
        out.append(title(heading))
        rows = [[f"{word} No.", f"{word} Caption", "Page No."]]
        for cap in items:
            num, rest = cap.split(":", 1)
            rows.append([num.replace(word + " ", ""), rest.strip(), pg(num)])
        out.append(table(rows, [1300, 5906, 1100], size=11, center_cols=(0, 2)))
    out.append(title("LIST OF SYMBOLS, ABBREVIATIONS AND NOMENCLATURE"))
    out.append(table([["Abbreviation / Symbol", "Expansion and Meaning"]] + abbrev, [2400, 5906], size=11))
    sect2 = (f'<w:sectPr><w:footerReference r:id="{f["footers"]["footer2.xml"]}" w:type="default"/><w:type w:val="nextPage"/>'
             '<w:pgSz w:h="16838" w:w="11906" w:orient="portrait"/><w:pgMar w:bottom="1440" w:top="1440" w:left="2160" '
             'w:right="1440" w:header="720" w:footer="720"/><w:pgNumType w:start="1"/></w:sectPr>')
    out.append(para([], sect=sect2, after=0, line=240))
    return out


def read_blocks(name):
    """Paragraph blocks of a front-matter content file (blank-line separated)."""
    text = open(os.path.join(CONTENT, name)).read()
    return [" ".join(b.split()) for b in re.split(r"\n\s*\n", text) if b.strip()]


def build(pages):
    f = json.load(open(os.path.join(PARTS, "front.json")))
    doc = Doc()
    for name in sorted(os.listdir(CONTENT)):
        if re.match(r"\d\d_.*\.txt$", name):
            parse_content(doc, open(os.path.join(CONTENT, name)).read())
    abbrev = [[c.strip() for c in ln.split(" | ")] for ln in open(os.path.join(CONTENT, "abbreviations.txt"))
              if ln.strip()]
    body = front_matter(f, doc, pages, read_blocks("abstract.txt"), read_blocks("acknowledgement.txt"), abbrev)
    body += doc.xml
    final = (f'<w:sectPr><w:footerReference r:id="{f["footers"]["footer3.xml"]}" w:type="default"/><w:type w:val="nextPage"/>'
             '<w:pgSz w:h="16838" w:w="11906" w:orient="portrait"/><w:pgMar w:bottom="1440" w:top="1440" w:left="2160" '
             'w:right="1440" w:header="720" w:footer="720"/></w:sectPr>')
    document = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' + f["root_tag"] + "<w:body>" +
                "".join(body) + final + "</w:body></w:document>")
    # relationships: template parts, the logo, the figures
    rels = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">']
    fixed = {"rIdTheme": ("theme", "theme/theme1.xml"), "rIdSettings": ("settings", "settings.xml"),
             "rIdFonts": ("fontTable", "fontTable.xml"), "rIdNum": ("numbering", "numbering.xml"),
             "rIdStyles": ("styles", "styles.xml")}
    ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/"
    for rid, (typ, target) in fixed.items():
        rels.append(f'<Relationship Id="{rid}" Type="{ns}{typ}" Target="{target}"/>')
    for rid, (typ, target) in f["rels"].items():
        rels.append(f'<Relationship Id="{rid}" Type="{typ}" Target="{target}"/>')
    media = {}
    for k, (rid, path) in enumerate(doc.images.items):
        target = f"media/fig{k + 1}.png"
        media[target] = path
        rels.append(f'<Relationship Id="{rid}" Type="{ns}image" Target="{target}"/>')
    rels.append("</Relationships>")
    styles = open(os.path.join(PARTS, "word", "styles.xml")).read()
    tnr = f'<w:rFonts w:ascii="{FONT}" w:cs="{FONT}" w:eastAsia="{FONT}" w:hAnsi="{FONT}"/>'
    extra_styles = ""
    for sid, name, bold, ind, leader in [("TOC1", "toc 1", True, 0, "none"), ("TOC2", "toc 2", False, 480, "none"),
                                         ("TOC3", "toc 3", False, 0, "dot")]:
        extra_styles += (f'<w:style w:type="paragraph" w:styleId="{sid}"><w:name w:val="{name}"/><w:basedOn w:val="Normal"/>'
                         '<w:next w:val="Normal"/><w:uiPriority w:val="39"/><w:pPr><w:tabs>'
                         f'<w:tab w:val="right" w:leader="{leader}" w:pos="{TEXT_W}"/></w:tabs>'
                         f'<w:spacing w:after="30" w:before="{100 if bold else 0}" w:line="264" w:lineRule="auto"/>'
                         f'<w:ind w:left="{ind}" w:right="700"/></w:pPr><w:rPr>{tnr}'
                         + ('<w:b w:val="1"/><w:bCs w:val="1"/>' if bold else "") +
                         '<w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr></w:style>')
    for sid, name in [("TableCaption", "Table Caption"), ("FigureCaption", "Figure Caption")]:
        extra_styles += (f'<w:style w:type="paragraph" w:customStyle="1" w:styleId="{sid}"><w:name w:val="{name}"/>'
                         '<w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:jc w:val="center"/></w:pPr>'
                         f'<w:rPr>{tnr}<w:b w:val="1"/><w:bCs w:val="1"/><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr></w:style>')
    styles = styles.replace("</w:styles>", extra_styles + "</w:styles>")
    settings = open(os.path.join(PARTS, "word", "settings.xml")).read()
    numbering = open(os.path.join(PARTS, "word", "numbering.xml")).read()
    extra = "".join(f'<w:num w:numId="{n}"><w:abstractNumId w:val="1"/><w:lvlOverride w:ilvl="0">'
                    '<w:startOverride w:val="1"/></w:lvlOverride></w:num>' for n in doc.lists)
    numbering = numbering.replace("</w:numbering>", extra + "</w:numbering>")
    if os.path.exists(OUT):
        os.remove(OUT)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(os.path.join(PARTS, "[Content_Types].xml"), "[Content_Types].xml")
        z.write(os.path.join(PARTS, "_rels", ".rels"), "_rels/.rels")
        z.writestr("word/document.xml", document)
        z.writestr("word/_rels/document.xml.rels", "".join(rels))
        z.writestr("word/numbering.xml", numbering)
        z.writestr("word/styles.xml", styles)
        z.writestr("word/settings.xml", settings)
        for root, _, files in os.walk(os.path.join(PARTS, "word")):
            for name in files:
                full = os.path.join(root, name)
                arc = os.path.relpath(full, PARTS)
                if arc not in ("word/numbering.xml", "word/styles.xml", "word/settings.xml"):
                    z.write(full, arc)
        for target, path in media.items():
            z.write(path, "word/" + target)
    print(f"wrote {OUT}: {len(doc.toc)} headings, {len(doc.tables)} tables, {len(doc.figures)} figures")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", help="a report .docx in the target template (first run)")
    ap.add_argument("--pages", help="JSON {heading or caption number: page} for the contents pages")
    a = ap.parse_args()
    if a.template:
        extract(a.template)
    if not os.path.exists(os.path.join(PARTS, "front.json")):
        sys.exit("run once with --template <report.docx> first")
    build(json.load(open(a.pages)) if a.pages else {})


if __name__ == "__main__":
    main()
