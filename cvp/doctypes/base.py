"""What a document type has to provide to plug into CVP."""
import datetime as dt

import numpy as np


class DocType:
    name = ""
    title = ""
    schema = None            # cvp.core.schema.Schema
    page = (640, 832)        # (width, height) of rendered pages
    slots = {}               # field -> (x, y, width): OCR slot anchor in template coordinates
    header_rule = (40, 84)   # left end of the rule OCR uses to register the page
    field_probs = None       # distribution over fields for single-field edits
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

    # ------------------------------------------------ forgery value models
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
            if isinstance(out.get(f), str):
                out[f] = dt.date.fromisoformat(out[f])
        return out

    def uniform_field_probs(self):
        n = len(self.schema.field_names)
        return np.full(n, 1.0 / n)
