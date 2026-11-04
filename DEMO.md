# Demo guide (5 minutes)

## Start it

```bash
cd cvp
.venv/bin/python app.py
```

Open http://127.0.0.1:7860. Startup takes ~10 s (models load); each analysis takes under 1 s.

## What is on screen

- **Left:** document seed, forgery type, field to edit, and a list of ready-made scenarios.
  Click a scenario row, then **Forge and analyse**.
- **Middle:** the transcript as received, and the CVP evidence: red heat = visual tamper
  evidence, orange boxes = the candidate fields the constraint branch nominated, dark-red
  box = the field CVP blames, green boxes = ground truth.
- **Right / below:** the verdict, the constraint table (which arithmetic or logical rule broke,
  with residuals), what the OCR read, and the class posterior p(K).

## Suggested order

1. **Seed 2, T2 blended, Subject mark 3** (the headline case). An inpainting-style edit that
   looks clean. Only `c1` (sum of marks = Total) breaks, so constraint reasoning knows *a mark*
   was changed but not *which* (all five marks are orange: Proposition 1). The heatmap ranks the
   five candidates and picks mark 3. Point out that neither branch alone can do this.
2. **Seed 3, T1 crude, Subject mark 2.** A conventional edit with a different font and a flat
   fill. The visual branch lights it up directly; constraints agree.
3. **Seed 5, T2 blended, Total.** The total enters `c1` and `c2`, a signature no other field
   has, so the constraint branch attributes it alone (singleton class).
4. **Seed 4, T3 coherent.** The forger repaired every dependent field, so every constraint holds
   and the decision rests on the visual branch alone.
5. **Seed 8, T4 complete.** Coherent *and* blended: the hardest case, which the paper says is
   outside either branch's guarantee. Say plainly that a T4 edit is caught only when the visual
   branch happens to see it.
6. **Seed 9, genuine.** All constraints hold, there is no heat, and the verdict is GENUINE.
7. **Results tab:** the numbers and charts from the full test-set evaluation.

## One-line pitch

Generative inpainting removes the pixel traces forensic detectors look for, but it does not
know the document's arithmetic. CVP uses the violated constraints to shrink the suspect set,
then uses even weak visual evidence to pick the edited field within it.
