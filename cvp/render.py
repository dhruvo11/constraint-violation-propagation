"""Parameterised transcript template and renderer.

Every field's bounding box is recorded at render time, so each sample comes with
per-field localisation ground truth and no manual annotation.
"""
import datetime as dt
import io
import os
from dataclasses import dataclass, field, asdict

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .schema import DEPTS, EARLIEST, FIELDS, MARK_FIELDS, TODAY, coherent_values, fmt

W, H = 640, 832
_FONT_DIR = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
FONT_SANS = os.path.join(_FONT_DIR, "DejaVuSans.ttf")
FONT_BOLD = os.path.join(_FONT_DIR, "DejaVuSans-Bold.ttf")
FONT_SERIF = os.path.join(_FONT_DIR, "DejaVuSerif.ttf")
FONT_MONO = os.path.join(_FONT_DIR, "DejaVuSansMono.ttf")
_font_cache = {}


def font(path, size):
    key = (path, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(path, size)
    return _font_cache[key]


VALUE_FONT = (FONT_SANS, 16)
ROLL_FONT = (FONT_SANS, 15)
ROW_Y = [312 + 38 * k for k in range(5)]
# Template anchors (x, y) of each field before the global page offset.
ANCHORS = {**{m: (530, y) for m, y in zip(MARK_FIELDS, ROW_Y)},
           "total": (530, 512), "pct": (180, 572), "grade": (440, 572),
           "date": (180, 612), "roll": (210, 178)}
SLOT_WIDTH = {**{m: 70 for m in MARK_FIELDS}, "total": 70, "pct": 90, "grade": 60,
              "date": 120, "roll": 170}
HEADER_RULE = (40, 84)  # left end of the header rule, used by OCR to register the page

COURSES = [("Data Structures", "CS"), ("Operating Systems", "CS"), ("Database Systems", "CS"),
           ("Computer Networks", "CS"), ("Theory of Computation", "CS"),
           ("Discrete Mathematics", "MA"), ("Linear Algebra", "MA"), ("Probability and Statistics", "MA"),
           ("Natural Language Processing", "CS"), ("Machine Learning", "CS"),
           ("Computer Architecture", "CS"), ("Software Engineering", "CS"),
           ("Digital Logic Design", "EC"), ("Signals and Systems", "EC"), ("Compiler Design", "CS")]
FIRST = ["Aarav", "Diya", "Ishaan", "Meera", "Rohan", "Ananya", "Kabir", "Sneha", "Vikram", "Priya",
         "Arjun", "Kavya", "Nikhil", "Riya", "Siddharth", "Tara", "Aditya", "Neha", "Rahul", "Pooja"]
LAST = ["Sharma", "Iyer", "Reddy", "Nair", "Rao", "Gupta", "Menon", "Kulkarni", "Patil", "Joshi",
        "Hegde", "Shetty", "Kumar", "Bhat", "Verma", "Das", "Pillai", "Gowda", "Naidu", "Mehta"]
SESSIONS = [(2022, 5), (2022, 12), (2023, 5), (2023, 12), (2024, 5), (2024, 12),
            (2025, 5), (2025, 12), (2026, 5)]


@dataclass
class Style:
    ox: int
    oy: int
    jitter: dict
    tint: tuple
    ink: tuple
    sigma: float
    blur: float
    q1: int
    wm_angle: float
    wm_shade: int
    noise_seed: int


@dataclass
class Content:
    name: str
    courses: list
    session: str
    semester: str


@dataclass
class Document:
    """A rendered document: the image plus everything needed to re-render or score it."""
    image: np.ndarray
    values: dict
    boxes: dict
    content: Content
    style: Style
    meta: dict = field(default_factory=dict)


def sample_roll(rng, year=None):
    year = year if year is not None else int(rng.integers(20, 26))
    return f"1RUA{year:02d}{DEPTS[rng.integers(len(DEPTS))]}{int(rng.integers(1, 10000)):04d}"


def sample_document_spec(rng):
    marks = np.clip(np.round(rng.normal(66, 14, 5)), 18, 100).astype(int)
    sy, sm = SESSIONS[rng.integers(len(SESSIONS))]
    start = dt.date(sy, sm, 28)
    latest = min(TODAY, start + dt.timedelta(days=150))
    date = start + dt.timedelta(days=int(rng.integers(10, max(11, (latest - start).days))))
    roll = sample_roll(rng, year=int(rng.integers(max(20, sy % 100 - 4), sy % 100 + 1)))
    values = coherent_values(marks.tolist(), roll, date)
    idx = rng.choice(len(COURSES), 5, replace=False)
    courses = [(f"{COURSES[i][1]}{int(rng.integers(21, 49))}{int(rng.integers(0, 10))}", COURSES[i][0])
               for i in idx]
    content = Content(name=f"{FIRST[rng.integers(len(FIRST))]} {LAST[rng.integers(len(LAST))]}",
                      courses=courses, session=f"{['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][sm]} {sy}",
                      semester=["II", "IV", "VI", "VIII"][rng.integers(4)])
    base = int(rng.integers(244, 253))
    tint = (base + int(rng.integers(0, 3)), base + int(rng.integers(-2, 2)), base - int(rng.integers(2, 12)))
    g = int(rng.integers(15, 55))
    style = Style(ox=int(rng.integers(-8, 9)), oy=int(rng.integers(-8, 9)),
                  jitter={f: [int(rng.integers(-2, 3)), int(rng.integers(-2, 3))] for f in FIELDS},
                  tint=tint, ink=(g, g, g + int(rng.integers(0, 15))),
                  sigma=float(rng.uniform(2.0, 5.0)), blur=float(rng.uniform(0.3, 0.7)),
                  q1=int(rng.integers(82, 96)), wm_angle=float(rng.uniform(25, 40)),
                  wm_shade=int(rng.integers(10, 18)), noise_seed=int(rng.integers(1 << 30)))
    return values, content, style


def anchor(field_name, style):
    ax, ay = ANCHORS[field_name]
    dx, dy = style.jitter[field_name]
    return ax + style.ox + dx, ay + style.oy + dy


def field_font(field_name):
    return font(*(ROLL_FONT if field_name == "roll" else VALUE_FONT))


def render_clean(values, content, style):
    """Noise-free rendering. Returns (PIL image, {field: tight text box})."""
    img = Image.new("RGB", (W, H), style.tint)
    # Faint diagonal watermark: gives the background texture that a flat fill destroys.
    wm = Image.new("L", (W, H), 0)
    wd = ImageDraw.Draw(wm)
    for k in range(-2, 6):
        wd.text((-60 + 40 * k, 180 + 150 * k), "INSTITUTE OF ADVANCED TECHNOLOGY", fill=255,
                font=font(FONT_BOLD, 22))
    wm = wm.rotate(style.wm_angle, resample=Image.BILINEAR, center=(W // 2, H // 2))
    shade = Image.new("RGB", (W, H), tuple(max(0, c - style.wm_shade) for c in style.tint))
    img = Image.composite(shade, img, wm)

    d = ImageDraw.Draw(img)
    ox, oy, ink = style.ox, style.oy, style.ink

    def text(xy, s, f, center=False):
        x, y = xy
        if center:
            x = (W - d.textlength(s, font=f)) / 2
        d.text((x + ox, y + oy), s, fill=ink, font=f)

    text((0, 28), "INSTITUTE OF ADVANCED TECHNOLOGY", font(FONT_BOLD, 22), center=True)
    text((0, 60), "Office of the Controller of Examinations", font(FONT_SANS, 13), center=True)
    d.rectangle([HEADER_RULE[0] + ox, HEADER_RULE[1] + oy, 600 + ox, HEADER_RULE[1] + 1 + oy], fill=ink)
    text((0, 96), "STATEMENT OF MARKS", font(FONT_BOLD, 16), center=True)
    text((0, 118), f"Semester End Examination - Semester {content.semester}", font(FONT_SANS, 12), center=True)

    lab = font(FONT_SANS, 13)
    for y, label in [(150, "Name of Candidate"), (178, "Roll Number"), (206, "Programme"), (234, "Examination")]:
        text((48, y + 1), label, lab)
        text((190, y + 1), ":", lab)
    text((210, 150), content.name, font(FONT_SANS, 14))
    text((210, 206), "Bachelor of Technology", font(FONT_SANS, 14))
    text((210, 234), content.session, font(FONT_SANS, 14))

    def rule(y):
        d.rectangle([40 + ox, y + oy, 600 + ox, y + oy], fill=ink)

    rule(270)
    hb = font(FONT_BOLD, 13)
    for x, s in [(48, "Code"), (130, "Course Title"), (430, "Max"), (520, "Marks")]:
        text((x, 280), s, hb)
    rule(300)
    for (code, title), y in zip(content.courses, ROW_Y):
        text((48, y + 2), code, lab)
        text((130, y + 2), title, lab)
        text((430, y + 1), "100", font(FONT_SANS, 14))
    rule(500)
    text((130, 514), "Total", hb)
    text((430, 513), "500", font(FONT_SANS, 14))
    rule(544)
    text((48, 574), "Percentage", lab)
    text((360, 574), "Grade", lab)
    text((48, 614), "Date of Issue", lab)

    d.ellipse([85 + ox, 680 + oy, 175 + ox, 770 + oy], outline=ink, width=2)
    text((108, 717), "SEAL", font(FONT_BOLD, 14))
    pts = [(420 + ox + 6 * i, 725 + oy + (4 if i % 2 else -4)) for i in range(25)]
    d.line(pts, fill=ink, width=2)
    text((420, 742), "Controller of Examinations", font(FONT_SANS, 12))
    text((0, 800), "Computer-generated statement. Any alteration renders this document invalid.",
         font(FONT_SANS, 10), center=True)

    boxes = {}
    for f in FIELDS:
        xy = anchor(f, style)
        s = fmt(f, values[f])
        ff = field_font(f)
        d.text(xy, s, fill=ink, font=ff)
        boxes[f] = list(d.textbbox(xy, s, font=ff))
    return img, boxes


def jpeg_bytes(arr, q):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=int(q))
    return buf.getvalue()


def decode(data):
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def jpeg(arr, q):
    return decode(jpeg_bytes(arr, q))


def finish(img, style, sigma=None, blur=None, seed=None):
    """Acquisition model: sensor noise, optical blur, first JPEG compression."""
    sigma = style.sigma if sigma is None else sigma
    blur = style.blur if blur is None else blur
    rng = np.random.default_rng(style.noise_seed if seed is None else seed)
    img = img.filter(ImageFilter.GaussianBlur(blur))
    a = np.asarray(img).astype(np.float32)
    lum = rng.normal(0, sigma, (H, W, 1))
    chroma = rng.normal(0, sigma * 0.3, (H, W, 3))
    a = np.clip(a + lum + chroma, 0, 255).astype(np.uint8)
    return jpeg(a, style.q1)


def render(values, content, style):
    img, boxes = render_clean(values, content, style)
    return Document(image=finish(img, style), values=dict(values), boxes=boxes, content=content, style=style)


def generate(rng):
    values, content, style = sample_document_spec(rng)
    return render(values, content, style)


def to_json(values):
    return {k: (v.isoformat() if isinstance(v, dt.date) else v) for k, v in values.items()}


def from_json(values):
    out = dict(values)
    if isinstance(out.get("date"), str):
        out["date"] = dt.date.fromisoformat(out["date"])
    return out


def style_dict(style):
    return asdict(style)
