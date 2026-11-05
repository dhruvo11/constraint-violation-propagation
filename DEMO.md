# Demo guide (10 minutes)

## Start it

Double-click **Start CVP.command** in the project folder. A Terminal window opens, the
models load (about 15 s) and the app opens in your browser by itself. Keep that window open
while presenting; close it to stop. Each analysis takes under a second.

## What is on screen

- **Left:** the document type, document seed, forgery type, field to edit, and a list of
  ready-made scenarios. Click a scenario row, then **Forge and analyse**.
- **Middle:** the document as received, and the CVP evidence:
  - red heat: visual tamper evidence;
  - orange boxes: the candidate fields the constraint branch nominated;
  - thin orange boxes: other fields that would break exactly the same rules;
  - dark-red box: the field CVP blames;
  - green boxes: ground truth.
- **Right / below:** the verdict, the constraint table (which arithmetic or logical rule broke,
  with residuals), what the OCR read, and the class posterior p(K).

The scenarios were picked to show one mechanism each; the **Results** tab has the measured
rates over 1,500 test documents per type.

## Suggested order

### Transcript: the core idea

1. **Seed 2, T2 blended, Subject mark 3** (the headline case). An inpainting-style edit that
   looks clean. Only `c1` (sum of marks = Total) breaks, so constraint reasoning knows *a mark*
   was changed but not *which* (all five marks are orange: Proposition 1). The heatmap ranks the
   five candidates and picks mark 3. Point out that neither branch alone can do this.
2. **Seed 3, T1 crude, Subject mark 2.** A conventional edit with a different font and a flat
   fill. The visual branch lights it up directly; constraints agree.
3. **Seed 5, T2 blended, Total.** The total enters `c1` and `c2`, a signature no other field
   has, so the constraint branch attributes it alone (singleton class).
4. **Seed 4, T3 coherent.** The forger changed the date of issue to another valid date, so
   every constraint holds and the decision rests on the visual branch alone. It catches the
   crude rendering and blames the date.
5. **Seed 8, T4 complete.** Coherent *and* blended: the hardest case, outside either
   branch's guarantee. Here CVP misses it (P = 0.94, just under the 0.95 threshold), although
   the heatmap's top field is the edited date. Say plainly that a T4 edit is caught only
   when the visual branch sees it (45% of them on the test set).
6. **Seed 9, genuine.** All constraints hold, there is no heat, and the verdict is GENUINE.

### GST invoice: line items, taxes and the amount in words

7. **Seed 3, T2 blended, Quantity, line 2.** Only `c_ln2` (quantity x rate = amount, line 2)
   breaks. Quantity and rate enter that rule symmetrically, so the rules cannot say which of
   the two changed (both orange: a Proposition 1 class of two); the heatmap picks the quantity.
8. **Seed 4, T2 blended, Total.** The total no longer equals taxable value + taxes (`c_tot`)
   or the amount in words (`c_wd`). No other field breaks that pair, so the rules alone name
   the total.
9. **Seed 5, T3 coherent.** The forger raised the rate on line 2 and recomputed the amount,
   taxable value, IGST, total and amount in words: all 19 rules hold. The crude re-rendering
   gives it away; CVP flags it and blames the line-2 amount, one of the six edited fields.

### Shop receipt: cash, change and a shared signature

10. **Seed 4, T2 blended, Change.** Only `c_chg` (change = cash - total) breaks. Editing the
    change or the cash leaves exactly this pattern, so the rules cannot separate the two (the
    cash line gets a thin orange box: a shared signature, not a Proposition 1 class). The
    heatmap decides for the change.
11. **Seed 5, T1 crude, Total.** The total breaks `c_tot` (subtotal + taxes) and `c_chg`
    (cash - total = change): a pattern only the total has.

### Certificate: grade bands, and a name no rule reads

12. **Seed 6, T2 blended, Grade.** Only `c_bnd` (the grade matches the score's band) breaks.
    A grade edit and a score edit within the valid range leave the same pattern. Here the
    constraint branch even leans to the score (orange box), but the heatmap finds the edit on
    the grade (thin orange, then red): visual evidence overrules a guess the rules cannot
    support.
13. **Seed 8, T4 complete.** The recipient's name was replaced. No rule reads a name, so
    every rule holds; the visual branch still catches the blended edit and names the field.
    Say plainly that unconstrained fields such as names rest on the visual branch alone.

### Passport: the printed page against its machine-readable zone

14. **Seed 6, T2 blended, Date of birth.** Only `c_dob` breaks: the printed date of birth no
    longer matches its copy in the MRZ. A forger who changed the MRZ copy instead (and
    recomputed its check digits) would leave exactly the same pattern, so both are suspects
    (MRZ line 2 gets a thin orange box); the heatmap finds the edit in the printed date.
15. **Seed 7, T1 crude, MRZ line 2.** This time the sex letter inside the MRZ was changed:
    `c_sex` breaks (printed sex against the MRZ). The same two-sided ambiguity, and the visual
    branch places the edit in the MRZ, not the printed field.

### Student ID card

16. **Seed 7, T2 blended, Valid upto.** Only `c_vld` breaks: validity must be 31 July of the
    batch's final year. The batch enters the same rule, and the signature analysis cannot rule
    out a batch edit that breaks it alone, so the batch stays a suspect (thin orange box); the
    heatmap picks the validity date.
17. **Seed 9, genuine.** Every rule holds and there is no heat: GENUINE.

### Results tab

Pick a document type to see its numbers and charts; the last two charts compare every type
and show one page of each.

## One-line pitch

Generative inpainting removes the pixel traces forensic detectors look for, but it does not
know the document's arithmetic. CVP uses the violated constraints to shrink the suspect set,
then uses even weak visual evidence to pick the edited field within it.
