"""Train g_phi and the CVP heads on cached features, run every baseline, write metrics,
LaTeX tables and the raw numbers behind the figures."""
import json
import os
import pickle
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvp import doctypes, paths, pipeline  # noqa: E402

CATS = ["genuine", "T1", "T2", "T3", "T4"]
METHODS = ["ELA", "Visual-only", "Constraint-only", "Concat fusion", "CVP (rule)", "CVP"]
DT = doctypes.get("transcript")  # the document type under evaluation; set from --doctype in main()
S = DT.schema
TAB = os.path.join(paths.REPORT, "tables", DT.name)
# Design of the learned attribution head. Field prior: "class" spreads each class's posterior
# over its candidates; "shared" also pools classes that share the observed signature
# (pipeline.field_prior). Heat: how the pooled heatmap enters (pipeline.attribution_features).
SHARED = False
HEAT = "prob"


def set_doctype(name):
    global DT, S, TAB
    DT = doctypes.get(name)
    S = DT.schema
    TAB = os.path.join(paths.REPORT, "tables", name)


def nanmean(x):
    """Mean that is NaN (not a warning) on an empty selection: a type may have no
    ambiguous edits of some kind at all."""
    x = np.asarray(x, float)
    return float(x.mean()) if x.size else float("nan")


def load(split, cond="clean", tag=""):
    return dict(np.load(os.path.join(paths.feature_dir(DT.name), f"{split}_{cond}{tag}.npz")))


def subset(d, mask):
    return {k: v[mask] for k, v in d.items()}


def without_blended(d):
    """The blended categories (T2/T4) stand in for an unseen generative attack: nothing
    that is trained or tuned in the main setting ever sees them."""
    return subset(d, ~np.isin(d["category"], [2, 4]))


def class_labels(d):
    y = np.zeros(len(d["category"]), int)
    for i, (c, p) in enumerate(zip(d["category"], d["primary"])):
        if c in (1, 2):
            y[i] = S.class_index[S.field_class[S.field_names[p]]]
    return y


def unary_hit(d):
    """Single-field edits whose own unary constraint broke (e.g. an out-of-range mark)."""
    return np.array([p >= 0 and S.unary_violated(S.field_names[p], w) for p, w in zip(d["primary"], d["w_true"])])


def consistent_counts(d):
    """|C(V)|: how many fields the true violation pattern is consistent with."""
    return np.array([len(S.consistent_fields(w)) for w in d["w_true"]])


def ambiguous(d):
    """The ambiguity split: T1/T2 edits whose violation pattern is consistent with more
    than one field, so constraint reasoning alone cannot beat 1/|C(V)|. This covers the
    edits inside a Proposition-1 class that no unary rule points at (five marks, say) and
    edits whose signature another class shares (a printed date and its MRZ copy)."""
    return np.isin(d["category"], [1, 2]) & (consistent_counts(d) > 1)


def chance(d):
    """Expected accuracy of a uniform guess among the consistent fields, on the ambiguity split."""
    amb = ambiguous(d)
    return nanmean(1.0 / consistent_counts(d)[amb]) if amb.any() else float("nan")


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max(1, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


class Heads:
    def fit(self, tr, va):
        self.scaler = StandardScaler().fit(tr["cfeat"])
        self.g = MLPClassifier((64, 64), max_iter=800, random_state=0).fit(
            self.scaler.transform(tr["cfeat"]), class_labels(tr))
        self.cscaler = StandardScaler().fit(np.hstack([tr["cfeat"], tr["emb"]]))
        ycat = np.where(tr["category"] > 0, tr["primary"] + 1, 0)
        self.concat = MLPClassifier((256,), max_iter=300, early_stopping=True, random_state=0).fit(
            self.cscaler.transform(np.hstack([tr["cfeat"], tr["emb"]])), ycat)
        p = self.p_cls(va)
        A = self.attr_features(va, p)
        t = va["category"] > 0
        self.attr_w = fit_softmax(A[t], va["primary"][t])
        self.det = LogisticRegression(C=1.0, max_iter=1000).fit(self.vis_features(va), (va["category"] > 0).astype(int))
        return self

    def p_cls(self, d):
        return pipeline.class_posterior(self.g, self.scaler.transform(d["cfeat"]), len(S.class_names))

    @staticmethod
    def attr_features(d, p):
        return np.stack([pipeline.attribution_features(p[i], d["w"][i], d["pooled"][i], S, SHARED, HEAT)
                         for i in range(len(p))])

    @staticmethod
    def vis_features(d):
        return np.stack([pipeline.visual_detection_features(x) for x in d["pooled"]])

    def predict(self, d):
        """Returns {method: (detection score [n], field scores [n,10] or expected-acc fn)}."""
        n = len(d["category"])
        p = self.p_cls(d)
        A = self.attr_features(d, p)
        out = {}
        out["ELA"] = (d["ela"][:, :, 1].max(1), d["ela"][:, :, 1])
        out["Visual-only"] = (d["pooled"][:, :, 0].max(1), d["pooled"][:, :, 0])
        # Constraint-only: every field the violation pattern is consistent with, ties broken
        # uniformly at random (scored by expectation).
        cand = np.zeros((n, len(S.field_names)))
        for i in range(n):
            c = S.consistent_fields(d["w"][i])
            cand[i, [S.field_index[f] for f in c]] = 1.0 / len(c)
        out["Constraint-only"] = (d["w"].sum(1), cand)
        pc = pipeline.class_posterior(self.concat, self.cscaler.transform(np.hstack([d["cfeat"], d["emb"]])),
                                      len(S.field_names) + 1)
        out["Concat fusion"] = (1 - pc[:, 0], pc[:, 1:])
        det = pipeline.noisy_or(1 - p[:, 0], self.det.predict_proba(self.vis_features(d))[:, 1])
        rule = np.zeros((n, len(S.field_names)))
        for i in range(n):
            rule[i, S.field_index[pipeline.cvp_rule(p[i], d["w"][i], d["pooled"][i], S)]] = 1
        out["CVP (rule)"] = (det, rule)
        out["CVP"] = (det, A @ self.attr_w)
        self.last_p = p
        return out


def fit_softmax(A, y, l2=1e-3):
    """Attribution head: score(f) = theta . phi(f); softmax over fields, trained on frozen
    outputs. theta is constrained non-negative (softplus) so that the score is monotone in
    the evidence: more heat or a higher constraint posterior can never make a field less
    suspicious, and the head cannot learn to reward an absence of evidence."""
    A, y = torch.tensor(A, dtype=torch.float64), torch.tensor(y)
    init = [0.5, 0.5] + [-2.0] * (A.shape[-1] - 2)  # prior and max heat on; extra statistics start near off
    raw = torch.tensor(init, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([raw], max_iter=300, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        theta = torch.nn.functional.softplus(raw)
        loss = torch.nn.functional.cross_entropy(A @ theta, y) + l2 * (theta ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return torch.nn.functional.softplus(raw).detach().numpy()


def attribution_correct(scores, d):
    """Per-document probability that the attributed field is one of the edited fields.
    Exact ties (constraint-only's uniform guess among the consistent fields) are scored by
    their expectation. Near-equal scores are not ties: a saturated heatmap (sigmoid ~1e-7)
    still ranks fields by its logits, every method -- and the deployed system -- takes the
    arg-max, and an arg-max that follows noise is right only at the chance rate."""
    acc = np.zeros(len(scores))
    for i, s in enumerate(scores):
        top = s == s.max()
        acc[i] = (top & d["edited"][i]).sum() / top.sum()
    return acc


def metrics(heads, d, thresholds=None):
    preds = heads.predict(d)
    y = (d["category"] > 0).astype(int)
    t = y == 1
    amb = ambiguous(d)
    single = np.isin(d["category"], [1, 2])
    res = {}
    for m, (s, sc) in preds.items():
        r = {"AUC": roc_auc_score(y, s)}
        for k in range(1, 5):
            sel = (d["category"] == 0) | (d["category"] == k)
            r[f"AUC_{CATS[k]}"] = roc_auc_score(y[sel], s[sel])
        if thresholds is not None:
            r["F1"] = f1_score(y, s >= thresholds[m])
            for k in range(5):
                r[f"rate_{CATS[k]}"] = float((s[d["category"] == k] >= thresholds[m]).mean())
        acc = attribution_correct(sc, d)
        r["attr_all"] = nanmean(acc[t])
        for k in range(1, 5):
            r[f"attr_{CATS[k]}"] = nanmean(acc[d["category"] == k])
        r["attr_amb"] = nanmean(acc[amb])
        r["attr_amb_T1"] = nanmean(acc[amb & (d["category"] == 1)])
        r["attr_amb_T2"] = nanmean(acc[amb & (d["category"] == 2)])
        r["attr_unamb"] = nanmean(acc[single & ~amb])
        r["attr_single"] = nanmean(acc[single])
        ious = []
        for i in np.nonzero(single)[0]:
            f = int(np.argmax(sc[i]))
            ious.append(box_iou(d["boxes"][i][f], d["gt_box"][i]))
        ious = np.array(ious)
        r["box_iou50"] = float((ious >= 0.5).mean())
        res[m] = {k: float(v) for k, v in r.items()}
    vi = d["vis_iou"][single]
    res["Visual-only"]["pixel_iou50"] = float((vi >= 0.5).mean())
    res["Visual-only"]["pixel_iou"] = float(np.nanmean(vi))
    res["Visual-only"]["pixel_f1"] = float(np.nanmean(d["vis_f1"][single]))
    for k in (1, 2):
        sel = d["category"] == k
        res["Visual-only"][f"pixel_f1_{CATS[k]}"] = float(np.nanmean(d["vis_f1"][sel]))
    return res, preds


def best_thresholds(heads, va):
    preds = heads.predict(va)
    y = (va["category"] > 0).astype(int)
    th = {}
    for m, (s, _) in preds.items():
        cands = np.unique(np.quantile(s, np.linspace(0, 1, 400)))
        th[m] = float(max(cands, key=lambda c: f1_score(y, s >= c)))
    th["Constraint-only"] = 0.5
    return th


def split_by_unary(k):
    """A multi-field class whose fields carry unary rules gets one signature row for edits
    that keep them and one for edits that break them."""
    return len(S.class_fields[k]) > 1 and any(c.unary and c.scope[0] in S.class_fields[k] for c in S.constraints)


def signature_rows():
    """Row labels of the empirical signature table, in class order."""
    rows = []
    for k in S.classes:
        label = DT.class_labels.get(k, k)
        if split_by_unary(k):
            rows += [(k, False, f"{label} ({DT.unary_labels[0]})"), (k, True, f"{label} ({DT.unary_labels[1]})")]
        else:
            rows.append((k, None, label))
    return rows


def signature_table(d):
    """Empirical violation frequency per edited-field type (ground-truth values), one column
    per constraint family. Per-field copies are collapsed per document before averaging, so
    an out-of-range mark counts its own range rule rather than a fifth of the five."""
    rows = {}
    single = np.isin(d["category"], [1, 2])
    hit = unary_hit(d)
    labels = {(k, u): lab for k, u, lab in signature_rows()}
    for i in np.nonzero(single)[0]:
        k = S.field_class[S.field_names[d["primary"][i]]]
        key = labels[(k, bool(hit[i]) if split_by_unary(k) else None)]
        rows.setdefault(key, []).append(family_columns(d["w_true"][i], len(S.class_fields[k]) > 1))
    return {k: np.mean(v, 0) for k, v in rows.items()}, {k: len(v) for k, v in rows.items()}


def family_columns(v, multi):
    """Collapse per-field copies of a rule (c4 for each mark) into one column: the copy
    that fired for a multi-field class, the total otherwise."""
    cols = []
    for fam in S.families:
        idx = [S.c_index[c.name] for c in S.constraints if (c.family or c.name) == fam]
        cols.append(v[idx].max() if multi else v[idx].sum())
    return cols


def tex_family(fam):
    members = sum((c.family or c.name) == fam for c in S.constraints)
    num = fam[1:] if fam[:1] == "c" and fam[1:].isdigit() else None
    base = f"c_{{{num}}}" if num else f"\\mathrm{{{fam}}}"
    return f"${base}^{{(\\cdot)}}$" if members > 1 else f"${base}$"


def fmt_pct(x):
    return "--" if np.isnan(x) else f"{100 * x:.1f}"


def write_tables(res, rob, sig, sig_n, ocr_acc, rho_emp, cand_size, held=None, chance_amb=float("nan")):
    os.makedirs(TAB, exist_ok=True)
    best = {k: np.nanmax([res[m][k] for m in METHODS]) if not all(np.isnan(res[m][k]) for m in METHODS)
            else float("nan") for k in res["CVP"]}

    def cell(m, k):
        v = fmt_pct(res[m][k])
        return f"\\textbf{{{v}}}" if np.isclose(res[m][k], best[k]) else v

    with open(os.path.join(TAB, "main.tex"), "w") as fh:
        fh.write("\\begin{tabular}{lcccccc}\n\\toprule\n"
                 " & \\multicolumn{2}{c}{Detection} & \\multicolumn{3}{c}{Attribution (top-1, \\%)} & Loc.\\\\\n"
                 "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}\\cmidrule(lr){7-7}\n"
                 "Method & AUC & F1 & All & Unamb. & Ambig. & IoU$_{.5}$\\\\\n\\midrule\n")
        for m in METHODS:
            fh.write(f"{m} & {res[m]['AUC']:.3f} & {res[m]['F1']:.3f} & {cell(m, 'attr_all')} & "
                     f"{cell(m, 'attr_unamb')} & {cell(m, 'attr_amb')} & {cell(m, 'box_iou50')}\\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")

    with open(os.path.join(TAB, "categories.tex"), "w") as fh:
        fh.write("\\begin{tabular}{lcccc|cccc}\n\\toprule\n"
                 " & \\multicolumn{4}{c|}{Detection AUC} & \\multicolumn{4}{c}{Attribution (\\%)}\\\\\n"
                 "Method & T1 & T2 & T3 & T4 & T1 & T2 & T3 & T4\\\\\n\\midrule\n")
        for m in METHODS:
            a = " & ".join(f"{res[m][f'AUC_T{k}']:.3f}" for k in range(1, 5))
            b = " & ".join(fmt_pct(res[m][f"attr_T{k}"]) for k in range(1, 5))
            fh.write(f"{m} & {a} & {b}\\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")

    with open(os.path.join(TAB, "ambiguity.tex"), "w") as fh:
        fh.write("\\begin{tabular}{lccc}\n\\toprule\n"
                 "Method & Ambig.\\ (T1) & Ambig.\\ (T2) & Ambig.\\ (all)\\\\\n\\midrule\n")
        for m in METHODS:
            fh.write(f"{m} & {fmt_pct(res[m]['attr_amb_T1'])} & {fmt_pct(res[m]['attr_amb_T2'])} & "
                     f"{fmt_pct(res[m]['attr_amb'])}\\\\\n")
        c = fmt_pct(chance_amb)
        fh.write(f"\\midrule\nChance ($1/|C(V)|$) & {c} & {c} & {c}\\\\\n\\bottomrule\n\\end{{tabular}}\n")

    with open(os.path.join(TAB, "signatures.tex"), "w") as fh:
        fh.write("\\begin{tabular}{lr" + "c" * len(S.families) + "}\n\\toprule\nField edited & $n$ & "
                 + " & ".join(tex_family(f) for f in S.families) + "\\\\\n\\midrule\n")
        for k, u, label in signature_rows():
            if label not in sig:
                continue
            vals = sig[label]
            fh.write(f"{label} & {sig_n[label]} & " + " & ".join(f"{x:.2f}" for x in vals) + "\\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")

    with open(os.path.join(TAB, "robustness.tex"), "w") as fh:
        conds = list(rob)
        fh.write("\\begin{tabular}{l" + "ccc" * len(conds) + "}\n\\toprule\n")
        fh.write(" & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{c}}}" for c in conds) + "\\\\\n")
        fh.write("Method & " + " & ".join(["AUC & All & Amb."] * len(conds)) + "\\\\\n\\midrule\n")
        for m in METHODS:
            fh.write(m + " & " + " & ".join(f"{rob[c][m]['AUC']:.3f} & {fmt_pct(rob[c][m]['attr_all'])} & "
                                            f"{fmt_pct(rob[c][m]['attr_amb'])}" for c in conds) + "\\\\\n")
        fh.write("\\midrule\nOCR field acc. & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{fmt_pct(ocr_acc[c])}}}"
                                                          for c in conds) + "\\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")

    if held is not None:
        names = {"blind": "Crude only (zero-shot)", "main": "Crude + older generator (main)",
                 "oracle": "Evaluated generator (oracle)"}
        with open(os.path.join(TAB, "regimes.tex"), "w") as fh:
            fh.write("\\begin{tabular}{llcccc}\n\\toprule\nVisual branch trained on & Method & AUC$_{T2}$ & "
                     "Attr.\\ T2 & Ambig.\\ T2 & Attr.\\ all\\\\\n\\midrule\n")
            for tag in ["blind", "main", "oracle"]:
                if tag not in held:
                    continue
                for j, m in enumerate(["Visual-only", "CVP"]):
                    r = held[tag][m]
                    lead = f"\\multirow{{2}}{{*}}{{{names[tag]}}}" if j == 0 else ""
                    fh.write(f"{lead} & {m} & {r['AUC_T2']:.3f} & {fmt_pct(r['attr_T2'])} & "
                             f"{fmt_pct(r['attr_amb_T2'])} & {fmt_pct(r['attr_all'])}\\\\\n")
                fh.write("\\midrule\n")
            r = held["main"]["Constraint-only"]
            fh.write(f"(no visual branch) & Constraint-only & {r['AUC_T2']:.3f} & {fmt_pct(r['attr_T2'])} & "
                     f"{fmt_pct(r['attr_amb_T2'])} & {fmt_pct(r['attr_all'])}\\\\\n")
            fh.write("\\bottomrule\n\\end{tabular}\n")


def write_numbers(summary):
    """LaTeX macros for every number quoted in the report text (report/numbers/<type>.tex);
    macros of a type other than the transcript carry its name as a prefix."""
    words = dict(zip("0123456789", ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]))

    prefix = "" if DT.name == "transcript" else DT.name.capitalize()

    def name(s):
        return prefix + "".join(words.get(c, c) for c in s if c.isalnum())

    keys = {"ELA": "Ela", "Visual-only": "Vis", "Constraint-only": "Con", "Concat fusion": "Cat",
            "CVP (rule)": "Rule", "CVP": "Cvp"}
    out = []
    for m, k in keys.items():
        for metric, v in summary["results"][m].items():
            val = "--" if np.isnan(v) else f"{v:.3f}" if metric.startswith("AUC") or metric == "F1" else f"{100 * v:.1f}"
            out.append(f"\\newcommand{{\\{name(k + metric.replace('_', ''))}}}{{{val}}}")
        for c, r in summary["robustness"].items():
            tag = {"Clean": "Clean"}.get(c, "Jpeg" if "JPEG" in c else "Resize")
            out.append(f"\\newcommand{{\\{name(k + tag + 'AUC')}}}{{{r[m]['AUC']:.3f}}}")
            out.append(f"\\newcommand{{\\{name(k + tag + 'Amb')}}}{{{100 * r[m]['attr_amb']:.1f}}}")
            out.append(f"\\newcommand{{\\{name(k + tag + 'All')}}}{{{100 * r[m]['attr_all']:.1f}}}")
    for c, v in summary["ocr_acc"].items():
        tag = {"Clean": "Clean"}.get(c, "Jpeg" if "JPEG" in c else "Resize")
        out.append(f"\\newcommand{{\\{name('Ocr' + tag)}}}{{{100 * v:.1f}}}")
    out.append(f"\\newcommand{{\\{name('GphiAcc')}}}{{{100 * summary['g_phi_acc']:.1f}}}")
    out.append(f"\\newcommand{{\\{name('CandSize')}}}{{{summary['cand_size']:.2f}}}")
    out.append(f"\\newcommand{{\\{name('RhoEmp')}}}{{{summary['rho_empirical']:.2f}}}")
    out.append(f"\\newcommand{{\\{name('NAmb')}}}{{{summary['n_ambiguous']}}}")
    for c, n in summary["n_test"].items():
        out.append(f"\\newcommand{{\\{name('NTest' + c.capitalize())}}}{{{n}}}")
    for tag, pre in [("blind", "Zs"), ("oracle", "Orc")]:
        h = (summary.get("heldout") or {}).get(tag)
        if not h:
            continue
        for m, k in [("Visual-only", "Vis"), ("CVP", "Cvp"), ("Constraint-only", "Con")]:
            out.append(f"\\newcommand{{\\{name(pre + k + 'AUCTtwo')}}}{{{h[m]['AUC_T2']:.3f}}}")
            out.append(f"\\newcommand{{\\{name(pre + k + 'AUC')}}}{{{h[m]['AUC']:.3f}}}")
            out.append(f"\\newcommand{{\\{name(pre + k + 'AttrTtwo')}}}{{{100 * h[m]['attr_T2']:.1f}}}")
            out.append(f"\\newcommand{{\\{name(pre + k + 'AmbTtwo')}}}{{{100 * h[m]['attr_amb_T2']:.1f}}}")
            out.append(f"\\newcommand{{\\{name(pre + k + 'All')}}}{{{100 * h[m]['attr_all']:.1f}}}")
    os.makedirs(os.path.join(paths.REPORT, "numbers"), exist_ok=True)
    with open(os.path.join(paths.REPORT, "numbers", f"{DT.name}.tex"), "w") as fh:
        fh.write("% Generated by scripts/evaluate.py -- do not edit by hand.\n" + "\n".join(out) + "\n")


def main():
    global SHARED, HEAT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--doctype", default="transcript", choices=doctypes.names())
    ap.add_argument("--prior", default="shared" if SHARED else "class", choices=["class", "shared"],
                    help="field prior of the attribution head (see pipeline.field_prior)")
    ap.add_argument("--heat", default=HEAT, choices=pipeline.HEAT,
                    help="how the heatmap enters the attribution head (see pipeline.attribution_features)")
    a = ap.parse_args()
    set_doctype(a.doctype)
    SHARED, HEAT = a.prior == "shared", a.heat
    tr, va, te = without_blended(load("train")), without_blended(load("val")), load("test")
    heads = Heads().fit(tr, va)
    th = best_thresholds(heads, va)
    res, preds = metrics(heads, te, th)

    p = heads.last_p
    yk = class_labels(te)
    g_acc = float((p.argmax(1) == yk).mean())
    single = np.isin(te["category"], [1, 2])
    cs = []
    for i in np.nonzero(single)[0]:
        k = S.class_names[int(p[i].argmax())]
        cs.append(len(S.candidate_fields(k, te["w"][i])))
    cand_size = float(np.mean(cs))
    rho_emp = 1 - cand_size / len(S.field_names)
    rho_th, e_th = S.reduction_factor()

    rob, ocr_acc = {}, {}
    n_rob = len(load("test", "jpeg60")["category"])  # degraded copies cover the first n_rob test documents
    for cond, name in [("clean", "Clean"), ("jpeg60", "JPEG $q{=}60$"), ("resize60", "Resize $0.6\\times$")]:
        d = load("test", cond)
        d = subset(d, np.arange(len(d["category"])) < n_rob)
        rob[name] = metrics(heads, d, th)[0]
        ocr_acc[name] = float(d["ocr_ok"].mean())

    # Visual-training regimes: zero-shot (crude edits only), main (crude + older generator G1),
    # oracle (the evaluated generator G2 itself). Heads are refitted per regime on validation
    # features; only the oracle may see blended documents while fitting.
    tags = [t for t in ("blind", "oracle") if os.path.exists(os.path.join(paths.feature_dir(DT.name), f"test_clean_{t}.npz"))]
    n_reg = min([len(load("test", tag=f"_{t}")["category"]) for t in tags] + [len(te["category"])])

    def head(d):
        return subset(d, np.arange(len(d["category"])) < n_reg)

    regimes = {"main": metrics(heads, head(te), th)[0]}
    for tag in tags:
        vr, tr_ = load("val", tag=f"_{tag}"), load("train")
        if tag == "blind":
            vr, tr_ = without_blended(vr), without_blended(tr_)
        hr = Heads().fit(tr_, vr)
        regimes[tag] = metrics(hr, head(load("test", tag=f"_{tag}")), best_thresholds(hr, vr))[0]
    held = regimes

    sig, sig_n = signature_table(te)
    chance_amb = chance(te)
    write_tables(res, rob, sig, sig_n, ocr_acc, rho_emp, cand_size, held, chance_amb)
    summary = dict(results=res, robustness=rob, ocr_acc=ocr_acc, g_phi_acc=g_acc,
                   ocr_clean_by_field={f: float(te["ocr_ok"][:, i].mean()) for i, f in enumerate(S.field_names)},
                   cand_size=cand_size, rho_empirical=rho_emp, rho_theory=rho_th, expected_K=e_th,
                   attr_w=heads.attr_w.tolist(), det_threshold=th["CVP"], thresholds=th, heldout=held,
                   signatures={k: v.tolist() for k, v in sig.items()}, signature_n=sig_n,
                   n_test={c: int((te["category"] == i).sum()) for i, c in enumerate(CATS)},
                   n_ambiguous=int(ambiguous(te).sum()), chance_ambiguous=chance_amb, n_regimes=int(n_reg),
                   prior="shared" if SHARED else "class", heat=HEAT)
    os.makedirs(paths.results_dir(DT.name), exist_ok=True)
    with open(os.path.join(paths.results_dir(DT.name), "results.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    write_numbers(summary)
    with open(pipeline.heads_path(DT.name), "wb") as fh:
        pickle.dump(dict(g_phi=heads.g, g_scaler=heads.scaler, det=heads.det, attr_w=heads.attr_w,
                         det_threshold=th["CVP"], shared_prior=SHARED, heat=HEAT), fh)
    for m in METHODS:
        r = res[m]
        print(f"{m:16s} AUC {r['AUC']:.3f} F1 {r['F1']:.3f} | attr all {r['attr_all']:.3f} unamb {r['attr_unamb']:.3f} "
              f"amb {r['attr_amb']:.3f} (T1 {r['attr_amb_T1']:.3f}, T2 {r['attr_amb_T2']:.3f}) | "
              f"AUC T1..T4 {[round(r[f'AUC_T{k}'], 3) for k in range(1, 5)]} | box IoU50 {r['box_iou50']:.3f}")
    print("g_phi acc", g_acc, "cand size", cand_size, "rho emp", rho_emp, "rho th", rho_th, "ocr", ocr_acc)


if __name__ == "__main__":
    main()
