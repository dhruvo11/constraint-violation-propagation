"""Schema machinery shared by every document type.

A document type declares its fields and its constraint system C. Everything the method
derives from C is computed here rather than written by hand per type: the violation and
residual vectors, the constraint-equivalence classes of Proposition 1, the necessary and
possible signature of each class, the candidate set nominated by a violation pattern and
the search-space reduction factor (Eq. 1).
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Field:
    name: str
    label: str
    charset: str            # characters the OCR may emit for this field
    fmt: Callable = str     # value -> printed text
    parse: Callable = None  # printed text -> value, or None when unreadable


@dataclass(frozen=True)
class Constraint:
    """A predicate over a scope of fields, evaluated on printed values.

    `check(values)` returns (violated, residual) and is only called when every field in
    the scope was read; otherwise the constraint reports `on_missing`. `sensitive` lists
    the fields any change of which necessarily violates the predicate (every term of a
    sum, say); `exchangeable` lists groups of fields the predicate sees only through a
    symmetric combination (their sum, or the product qty x rate), so that any edit of one
    is matched by an edit of the other with the same (w, r): this is what makes them
    indistinguishable (Proposition 1). `family` groups per-field copies of one rule (c4
    for every mark) for reporting.
    """
    name: str
    kind: str
    scope: tuple
    check: Callable
    text: str = ""
    sensitive: tuple = ()
    exchangeable: tuple = ()
    on_missing: tuple = (1.0, 0.0)
    family: str = ""

    @property
    def unary(self):
        return len(self.scope) == 1


class Schema:
    def __init__(self, name, fields, constraints, groups=None):
        self.name = name
        self.fields = list(fields)
        self.constraints = list(constraints)
        self.field_names = [f.name for f in self.fields]
        self.field_index = {f: i for i, f in enumerate(self.field_names)}
        self.by_name = {f.name: f for f in self.fields}
        self.constraint_names = [c.name for c in self.constraints]
        self.c_index = {c: i for i, c in enumerate(self.constraint_names)}
        self.families = list(dict.fromkeys(c.family or c.name for c in self.constraints))

        self.classes, self.class_fields = self._equivalence_classes(groups or {})
        self.class_names = ["coherent"] + self.classes
        self.class_index = {k: i for i, k in enumerate(self.class_names)}
        self.field_class = {f: k for k, fs in self.class_fields.items() for f in fs}
        self.class_fields["coherent"] = list(self.field_names)
        self.possible = {k: {c.name for c in self.constraints if set(c.scope) & set(fs)}
                         for k, fs in self.class_fields.items() if k != "coherent"}
        self.necessary = {k: {c.name for c in self.constraints if fs[0] in c.sensitive}
                          for k, fs in self.class_fields.items() if k != "coherent"}
        self._unary_of = {f: [self.c_index[c.name] for c in self.constraints if c.unary and c.scope[0] == f]
                          for f in self.field_names}

    # ------------------------------------------------------------ Proposition 1
    def _equivalence_classes(self, groups):
        """Fields are constraint-equivalent when every non-unary constraint contains both
        or neither, and treats them exchangeably whenever it contains both."""
        non_unary = [c for c in self.constraints if not c.unary]

        def equivalent(a, b):
            shared = False
            for c in non_unary:
                ina, inb = a in c.scope, b in c.scope
                if ina != inb:
                    return False
                if ina:
                    if not any(a in g and b in g for g in c.exchangeable):
                        return False
                    shared = True
            return shared

        names, members_of, seen = [], {}, set()
        for f in self.field_names:
            if f in seen:
                continue
            members = [g for g in self.field_names if g not in seen and (g == f or equivalent(f, g))]
            seen.update(members)
            if len(members) == 1:
                name = members[0]
            else:
                name = next((n for n, fs in groups.items() if set(fs) == set(members)), "+".join(members))
            names.append(name)
            members_of[name] = members
        return names, members_of

    def class_sizes(self):
        return [len(self.class_fields[k]) for k in self.classes]

    def reduction_factor(self):
        """rho = 1 - E_f[|K(f)|] / |F| for a uniformly chosen tampered field (Eq. 1)."""
        n = len(self.field_names)
        expected = sum(s * s for s in self.class_sizes()) / n
        return 1.0 - expected / n, expected

    # ------------------------------------------------------------ evaluation
    def evaluate(self, values):
        """Symbolic evaluation of C: binary violation vector w and residual vector r.
        A constraint whose scope contains an unreadable field reports `on_missing`."""
        m = len(self.constraints)
        w = np.zeros(m, np.float32)
        r = np.zeros(m, np.float32)
        for j, c in enumerate(self.constraints):
            if any(values.get(f) is None for f in c.scope):
                w[j], r[j] = c.on_missing
                continue
            violated, residual = c.check(values)
            w[j], r[j] = float(violated), float(residual)
        return w, r

    def violated(self, w):
        return {c for c in self.constraint_names if w[self.c_index[c]] > 0}

    def unary_violated(self, field, w):
        return any(w[j] > 0 for j in self._unary_of[field])

    def signature_class(self, w):
        """Deterministic signature lookup: the class whose necessary signature is contained
        in, and whose possible signature contains, the observed violations. Patterns no
        single-field edit explains (OCR noise, multi-field edits) go to the best overlap."""
        V = self.violated(w)
        if not V:
            return "coherent"
        fits = [k for k in self.classes if self.necessary[k] <= V <= self.possible[k]]
        if fits:
            return max(fits, key=lambda k: len(self.necessary[k]))
        return max(self.classes, key=lambda k: (len(V & self.possible[k]) - len(self.necessary[k] - V),
                                                len(self.necessary[k])))

    def candidate_fields(self, cls, w):
        """Fields of class K, narrowed to those a violated unary constraint points at."""
        fields = self.class_fields[cls]
        if cls == "coherent":
            return list(fields)
        hit = [f for f in fields if self.unary_violated(f, w)]
        return hit or list(fields)

    def consistent_fields(self, w):
        """Every field a single-field edit could have produced this violation pattern from:
        the candidates of each class whose signature interval contains it. Proposition 1's
        classes are one source of ambiguity; the other is two classes whose intervals
        overlap (a printed date and its machine-readable copy both violate only their
        equality, say). Patterns no single edit explains fall back to the best overlap."""
        V = self.violated(w)
        if not V:
            return list(self.field_names)
        fits = [k for k in self.classes if self.necessary[k] <= V <= self.possible[k]] or [self.signature_class(w)]
        out = []
        for k in fits:
            out += [f for f in self.candidate_fields(k, w) if f not in out]
        return out

    def constrained(self, field):
        """Whether any constraint reads `field` (a field no rule reads can only be checked visually)."""
        return bool(self.possible[self.field_class[field]])
