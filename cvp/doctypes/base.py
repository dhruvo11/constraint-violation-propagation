"""What a document type has to provide to plug into CVP."""
import datetime as dt

import numpy as np

BLANK = ""  # value of an optional field that is not printed (an unused invoice line, say)


class DocType:
    name = ""
    title = ""
    schema = None            # cvp.core.schema.Schema
    page = (640, 832)        # (width, height) of rendered pages
    slots = {}               # field -> (x, y, width[, up, down]): OCR crop in template coordinates
    wide = frozenset()       # fields long enough to need the wide OCR crop (MRZ lines, amounts in words)
    header_rule = (40, 84)   # left end of the rule OCR uses to register the page
    rule_min = 300           # dark pixels a row needs to count as that rule
    field_probs = None       # distribution over fields for single-field edits (or use `weights`)
    weights = None           # field -> relative chance of being the target of a single-field edit
    date_fields = ()
    class_labels = {}        # display names per equivalence class
    unary_labels = ("unary rule kept", "unary rule broken")

    # ------------------------------------------------ synthetic template
    def sample_spec(self, rng):
        """Draw coherent field values, static content and a rendering style."""
        raise NotImplementedError

    def render_clean(self, values, content, style):
        """Noise-free rendering. Returns (PIL image, {field: tight text box})."""
        raise NotImplementedError

    def anchor(self, field, style):
        """Where `field`'s text is drawn on this page."""
        raise NotImplementedError

    def place(self, field, text, fnt, style):
        """Drawing origin of `text` in `fnt` for `field` (alignment-aware; default: left)."""
        return self.anchor(field, style)

    def font_scale(self, field):
        """Size of `field`'s own font relative to 16 px, so crude edits use a comparable size."""
        return 1.0

    def alt_fonts(self, field):
        """Fonts a crude edit of `field` may use (None: any of the taxonomy's mismatched fonts)."""
        return None

    # ------------------------------------------------ forgery value models
    def pick_field(self, values, rng):
        """Target of a single-field edit. Fields without a rule (a name nothing repeats) are
        never picked: editing them breaks no constraint, so they belong to coherent edits."""
        names = self.schema.field_names
        if self.weights is None:
            return names[rng.choice(len(names), p=self.field_probs)]
        cand = [f for f in names if self.weights.get(f, 0) > 0 and values.get(f) != BLANK]
        p = np.array([self.weights[f] for f in cand], float)
        return cand[rng.choice(len(cand), p=p / p.sum())]

    def violating_value(self, field, values, rng):
        """A new value for a single-field edit; the edit must break at least one constraint."""
        raise NotImplementedError

    def coherent_edit(self, values, rng):
        """A multi-field edit that keeps every constraint satisfied: (values, edited, primary)."""
        raise NotImplementedError

    def random_values(self, rng, values):
        """Values beyond the valid range, so the OCR learns no prior that would 'correct' a forgery."""
        raise NotImplementedError

    # ------------------------------------------------ serialisation
    def from_json(self, values):
        out = dict(values)
        for f in self.date_fields:
            if isinstance(out.get(f), str) and out[f]:
                out[f] = dt.date.fromisoformat(out[f])
        return out

    def uniform_field_probs(self):
        n = len(self.schema.field_names)
        return np.full(n, 1.0 / n)


class Layout:
    """Helper for fixed-layout templates: each field has an anchor, an alignment, a font and
    an OCR slot derived from them, so the rendering, the forger and the reader agree."""

    def __init__(self, spec):
        # spec: field -> (x, y, align, font_path, size, slot_width[, up, down])
        self.spec = spec

    def slots(self):
        """OCR crop per field: `width` px of text room, a 10 px margin on each side."""
        out = {}
        for f, (x, y, align, _, size, width, *ud) in self.spec.items():
            x0 = x if align == "l" else x - width + 10 if align == "r" else x - width // 2 + 5
            up, down = ud if ud else (round(size * 0.5), round(size * 1.875))
            out[f] = (x0, y, width, up, down)
        return out

    def wide(self, crop_h=32, crop_w=160):
        """Fields whose crop would be squeezed in a normal-width OCR input."""
        return frozenset(f for f, (_, _, w, up, down) in self.slots().items() if (w + 10) * crop_h / (up + down) > crop_w)

    def font(self, f):
        from ..core.render import font
        _, _, _, path, size, *_ = self.spec[f]
        return font(path, size)

    def scale(self, f):
        return self.spec[f][4] / 16.0

    def origin(self, f, text, fnt, style):
        x, y, align, *_ = self.spec[f]
        dx, dy = style.jitter.get(f, (0, 0))
        w = fnt.getlength(text) if text else 0
        x0 = x if align == "l" else x - w if align == "r" else x - w / 2
        return round(x0) + style.ox + dx, y + style.oy + dy
