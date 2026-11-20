/* CVP web app: plain JavaScript, no build step, works offline. */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const md = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");
const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html !== undefined) e.innerHTML = html; return e; };

async function api(url, body, isForm) {
  const opt = body === undefined ? {} : { method: "POST", body: isForm ? body : JSON.stringify(body),
    headers: isForm ? {} : { "Content-Type": "application/json" } };
  const res = await fetch(url, opt);
  let data = null;
  try { data = await res.json(); } catch (e) { /* not JSON */ }
  if (!res.ok) throw new Error((data && data.detail) || `${res.status} ${res.statusText}`);
  return data;
}

const state = { catalog: null, tour: [], tourAt: 0, ready: false };
// ?theme=light forces the light theme (screenshots for print); otherwise the system's choice
if (new URLSearchParams(location.search).get("theme") === "light") document.documentElement.dataset.theme = "light";

/* ---------------------------------------------------------------- views */
function route() {
  // #analyse, #tour, #how, #results; direct links: #tour/6 (a tour step), #example/ai-905 (an example)
  const [v, arg] = (location.hash || "#analyse").slice(1).split("/");
  const view = v === "example" ? "analyse" : ["analyse", "tour", "how", "results"].includes(v) ? v : "analyse";
  $$(".view").forEach((s) => (s.hidden = s.id !== `view-${view}`));
  $$(".views a").forEach((a) => a.classList.toggle("on", a.dataset.view === view));
  if (view === "tour" && state.catalog) {
    if (arg) openTourStep(+arg - 1);
    else if (!$("#tour-card").childElementCount) openTourStep(state.tourAt);
  }
  if (v === "example" && arg && state.catalog) {
    $$(".example").forEach((x) => x.classList.toggle("on", x.dataset.id === arg));
    runInto($("#result"), () => api(`/api/sample/${arg}`, {}));
  }
  if (view === "results" && !$("#tiles").childElementCount) renderResultsPage();
  window.scrollTo({ top: 0 });
}
window.addEventListener("hashchange", route);

/* ---------------------------------------------------------------- status */
async function pollStatus() {
  try {
    const s = await api("/api/status");
    const box = $("#status");
    state.ready = s.ready;
    box.classList.toggle("ready", s.ready && !s.error);
    box.classList.toggle("error", !!s.error);
    $(".txt", box).textContent = s.error ? "Some models failed to load" : s.ready ? "Ready" : `Loading: ${s.step}…`;
    box.title = s.error || "";
    if (!s.ready) setTimeout(pollStatus, 1200);
  } catch (e) {
    $(".txt", $("#status")).textContent = "Server not reachable";
    $("#status").classList.add("error");
    setTimeout(pollStatus, 3000);
  }
}

/* ---------------------------------------------------------------- examples */
const KIND_ICON = { synthetic: ["DOC", "k-synthetic"], cord: ["RCP", "k-cord"], aiforge: ["AI", "k-aiforge"],
  "receipt-cord": ["EXP", "k-receipt"], "receipt-gst": ["EXP", "k-receipt"] };
const GROUP_NOTE = { cord: "Needs the CORD dataset", aiforge: "Needs AIForge-Doc (gated; on this laptop only)",
  receipt: "Needs docTR" };

function renderExamples(groups) {
  const box = $("#examples");
  box.innerHTML = "";
  for (const g of groups) {
    const sec = el("div", "group");
    sec.appendChild(el("h4", "", esc(g.title)));
    g.items.forEach((it, i) => {
      const b = el("button", "example");
      b.type = "button";
      b.dataset.id = it.id;
      const [ic, cls] = g.id === "tour" ? [String(i + 1), "k-tour"] : KIND_ICON[it.kind] || ["?", ""];
      b.innerHTML = `<span class="ic ${cls}">${ic}</span><span><span class="t">${esc(it.title)}</span>` +
        (it.available ? "" : `<small>${esc(GROUP_NOTE[g.id] || "Not available here")}</small>`) + "</span>";
      b.disabled = !it.available;
      b.addEventListener("click", () => {
        if (g.id === "tour") { state.tourAt = i; location.hash = "#tour"; openTourStep(i); return; }
        $$(".example").forEach((x) => x.classList.toggle("on", x === b));
        runInto($("#result"), () => api(`/api/sample/${it.id}`, {}));
      });
      sec.appendChild(b);
    });
    box.appendChild(sec);
  }
}

/* ---------------------------------------------------------------- running and loading */
const LOADING = ["Reading the document…", "Checking the rules…", "Looking at the pixels…", "Weighing the evidence…"];
async function runInto(container, fetcher) {
  container.innerHTML = "";
  const box = el("div", "loading", `<div class="spinner"></div><div class="lt">${LOADING[0]}</div>` +
    (state.ready ? "" : `<p class="hint">The models are still loading, so this first one takes a little longer.</p>`));
  container.appendChild(box);
  let i = 0;
  const timer = setInterval(() => { i = (i + 1) % LOADING.length; $(".lt", box).textContent = LOADING[i]; }, 1100);
  try {
    const r = await fetcher();
    clearInterval(timer);
    renderResult(container, r);
    return r;
  } catch (e) {
    clearInterval(timer);
    container.innerHTML = "";
    container.appendChild(el("div", "error-box", esc(e.message)));
  }
}

/* ---------------------------------------------------------------- the result */
const ICON = {
  tampered: '<svg viewBox="0 0 24 24"><path d="M12 3l9.5 17h-19z"/><path d="M12 10v4.5M12 17.5v.5"/></svg>',
  genuine: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M7.5 12.5l3 3 6-6.5"/></svg>',
  warning: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 7.5v6M12 16.5v.5"/></svg>',
  unknown: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.7.3-1 .9-1 1.7M12 16.5v.5"/></svg>',
};
const ROLE_TEXT = { blamed: "Blamed", suspect: "Suspect", "suspect-weak": "Suspect", none: "" };
const RULE_ICON = { ok: "✓", broken: "✗", warning: "!", unread: "?" };

function renderResult(container, r) {
  container.innerHTML = "";
  container.appendChild(el("p", "source", `<b>${esc(r.title)}</b> · ${esc(r.source)}`));
  container.appendChild(verdictCard(r));
  const grid = el("div", "rgrid");
  const left = el("div", "rcol"), right = el("div", "rcol");
  left.appendChild(viewer(r));
  right.appendChild(stepsCard(r));
  right.appendChild(rulesCard(r));
  grid.append(left, right);
  container.appendChild(grid);
  container.appendChild(fieldsCard(r, container));
  if (r.rows) container.appendChild(rowsCard(r));
  container.appendChild(techCard(r));
}

function verdictCard(r) {
  const v = r.verdict;
  const card = el("div", `verdict s-${v.status}`);
  card.appendChild(el("div", "v-badge", ICON[v.status] || ICON.unknown));
  const head = el("div", "v-head", `<h2>${esc(v.headline)}</h2>` +
    (v.blamed ? `<span class="v-blamed">Edited field: <b>${esc(v.blamed)}</b></span>` : ""));
  card.appendChild(head);
  card.appendChild(el("p", "v-reason", md(v.reason)));
  const broken = r.rules.filter((x) => x.status === "broken").length;
  const meta = el("div", "v-meta", `<span>${r.rules.length} rules checked, ${broken} broken</span>` +
    `<span>${r.fields.length} fields read</span><span>${esc(r.kind)}</span>`);
  card.appendChild(meta);
  if (r.corrected) {
    const a = el("div", "answer v-partial", `<span class="ak">✎ Your corrections</span><span>Re-checked with ` +
      r.corrected.map((c) => `<b>${esc(c.label)}</b> = <span class="chg">${esc(c.value)}</span>`).join(", ") +
      ". The verdict above is for the corrected readings.</span>");
    a.style.gridColumn = "1 / -1";
    card.appendChild(a);
  }
  if (r.truth) {
    const t = r.truth;
    const chg = (t.changes || []).filter((c) => c.old !== undefined && c.new !== undefined)
      .map((c) => `${esc(c.label)}: ${esc(c.old)} → ${esc(c.new)}`).join(" · ");
    const mark = { right: "✓", partial: "◐", wrong: "✗" }[t.verdict] || "";
    const a = el("div", `answer v-${t.verdict}`, `<span class="ak">${mark} Answer key</span><span>${md(t.text)}` +
      (chg ? `<br><span class="chg">${chg}</span>` : "") + "</span>");
    a.style.gridColumn = "1 / -1";
    card.appendChild(a);
  }
  return card;
}

function viewer(r) {
  const card = el("div", "card viewer");
  const bar = el("div", "vbar");
  const layers = [["heat", "Heatmap", "linear-gradient(90deg,#f7a0a0,#e02f2f)", true],
    ["sus", "Suspects", "var(--suspect)", true], ["blame", "Blamed", "var(--blamed)", true]];
  if (r.truth && (r.truth.rects || []).length) layers.push(["truth", "Answer key", "var(--truth)", true]);
  layers.push(["all", "All fields", "rgba(80,90,110,.6)", false]);
  const stage = el("div", "stage");
  for (const [key, label, color, on] of layers) {
    const c = el("label", "chip", `<input type="checkbox" ${on ? "checked" : ""}><span class="sw" style="background:${color}"></span>${label}`);
    const inp = $("input", c);
    const apply = () => stage.classList.toggle(key === "all" ? "show-all" : `no-${key}`, key === "all" ? inp.checked : !inp.checked);
    inp.addEventListener("change", apply);
    apply();
    bar.appendChild(c);
  }
  bar.appendChild(el("span", "spacer"));
  const dl = el("a", "chip", "⤓ Download");
  dl.href = r.image.src;
  dl.download = (r.title || "document").replace(/[^\w#-]+/g, "_") + (r.image.src.startsWith("data:image/png") ? ".png" : ".jpg");
  dl.title = "Save this document, for example to upload it again";
  bar.appendChild(dl);
  card.appendChild(bar);

  const img = el("img");
  img.src = r.image.src;
  img.alt = r.title;
  stage.appendChild(img);
  if (r.heat) { const h = el("img", "heat"); h.src = r.heat.src; h.alt = ""; stage.appendChild(h); }
  stage.appendChild(boxesSvg(r));
  stage.addEventListener("click", (e) => { if (!e.target.closest(".bx")) openLightbox(stage); });
  card.appendChild(stage);
  card.appendChild(el("div", "legend",
    '<span><i class="heat-key"></i>edit traces (heat)</span><span><i style="border-color:var(--suspect)"></i>fields the broken rules fit</span>' +
    '<span><i style="border-color:var(--blamed)"></i>blamed field</span>' +
    (r.truth && (r.truth.rects || []).length ? '<span><i style="border-color:var(--truth);border-top-style:dashed"></i>what was really edited</span>' : "") +
    '<span>Click the image to enlarge; hover a box for its reading.</span>'));
  return card;
}

function boxesSvg(r) {
  const W = r.image.width, H = r.image.height;
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("preserveAspectRatio", "none");
  const order = { none: 0, "suspect-weak": 1, suspect: 2, blamed: 3 };
  const fields = [...r.fields].sort((a, b) => order[a.role] - order[b.role]);
  const pad = Math.max(2, Math.round(W / 300));
  for (const f of fields) {
    const boxes = f.boxes && f.boxes.length ? f.boxes : f.box ? [f.box] : [];
    for (const b of boxes) {
      const rect = document.createElementNS(NS, "rect");
      const p = f.role === "blamed" ? pad * 2.2 : f.role === "none" ? pad * 0.5 : pad * 1.4;
      rect.setAttribute("x", b[0] - p); rect.setAttribute("y", b[1] - p);
      rect.setAttribute("width", Math.max(2, b[2] - b[0] + 2 * p)); rect.setAttribute("height", Math.max(2, b[3] - b[1] + 2 * p));
      rect.setAttribute("rx", 3);
      rect.setAttribute("class", `bx ${f.role}`);
      rect.addEventListener("mousemove", (e) => tip(e, f));
      rect.addEventListener("mouseleave", hideTip);
      svg.appendChild(rect);
    }
    if (f.role === "blamed" && boxes.length) {
      const b = boxes[0];
      const t = document.createElementNS(NS, "text");
      const right = (b[0] + b[2]) / 2 > W / 2;  // label towards the middle so it never runs off the page
      t.setAttribute("text-anchor", right ? "end" : "start");
      t.setAttribute("x", right ? Math.min(W - 2, b[2] + pad * 2) : Math.max(2, b[0] - pad * 2));
      t.setAttribute("y", Math.max(14, b[1] - pad * 3.4));
      t.setAttribute("class", "tag"); t.setAttribute("fill", "#e03131");
      t.setAttribute("font-size", Math.max(12, Math.round(W / 45)));
      t.textContent = "Blamed: " + f.label;
      svg.appendChild(t);
    }
  }
  for (const b of (r.truth && r.truth.rects) || []) {
    if (!b) continue;
    const rect = document.createElementNS(NS, "rect");
    rect.setAttribute("x", b[0] - pad * 3); rect.setAttribute("y", b[1] - pad * 3);
    rect.setAttribute("width", b[2] - b[0] + pad * 6); rect.setAttribute("height", b[3] - b[1] + pad * 6);
    rect.setAttribute("rx", 4); rect.setAttribute("class", "bx truth");
    svg.appendChild(rect);
  }
  return svg;
}

function tip(e, f) {
  const t = $("#tooltip");
  t.hidden = false;
  t.innerHTML = `<b>${esc(f.label)}</b>Read: <span class="num">${esc(f.read)}</span>` +
    (f.printed && f.printed !== f.read ? `<br>Printed: <span class="num">${esc(f.printed)}</span>` : "") +
    `<br>Heat: <span class="num">${(f.heat || 0).toFixed(2)}</span>` + (ROLE_TEXT[f.role] ? ` · ${ROLE_TEXT[f.role]}` : "");
  const x = Math.min(window.innerWidth - t.offsetWidth - 12, e.clientX + 14);
  const y = Math.min(window.innerHeight - t.offsetHeight - 12, e.clientY + 14);
  t.style.left = `${x}px`; t.style.top = `${y}px`;
}
function hideTip() { $("#tooltip").hidden = true; }

function openLightbox(stageOrImg) {
  const lb = $("#lightbox"), box = $("#lb-stage");
  box.innerHTML = "";
  const c = stageOrImg.cloneNode(true);
  if (c.tagName === "IMG") c.classList.add("single");
  box.appendChild(c);
  $$(".bx", c).forEach((rect) => rect.replaceWith(rect.cloneNode(true)));
  lb.hidden = false;
}
$("#lightbox").addEventListener("click", (e) => { if (e.target.id === "lightbox" || e.target.closest(".close") || e.target.closest(".stage") || e.target.classList.contains("single")) $("#lightbox").hidden = true; });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("#lightbox").hidden = true; });

function stepsCard(r) {
  const card = el("div", "card steps", "<h3>How CVP decided</h3>");
  const ol = el("ol");
  r.steps.forEach((s, i) => {
    const li = el("li", i === r.steps.length - 1 ? "decide" : "", `<span><span class="st">${esc(s.title)}</span><span class="sx">${md(s.text)}</span></span>`);
    ol.appendChild(li);
  });
  card.appendChild(ol);
  return card;
}

function rulesCard(r) {
  const card = el("div", "card rules");
  const bad = r.rules.filter((x) => x.status !== "ok");
  card.innerHTML = `<h3>The document's rules <span class="count">${r.rules.length - bad.length} hold · ${bad.length} not</span></h3>`;
  const ordered = [...bad, ...r.rules.filter((x) => x.status === "ok")];
  const SHOW = 6;
  ordered.forEach((x, i) => {
    const row = el("div", `rule r-${x.status}`, `<span class="ri">${RULE_ICON[x.status]}</span><span><span class="rt">${esc(x.text)}` +
      (x.level ? `<span class="lvl">${x.level === "alarm" ? "alarm" : "warning only"}</span>` : "") +
      `</span><br><span class="rd">${esc(x.detail)}</span></span>`);
    if (i >= Math.max(SHOW, bad.length)) row.hidden = true;
    card.appendChild(row);
  });
  if (ordered.length > Math.max(SHOW, bad.length)) {
    const more = el("button", "ghost more", `Show all ${ordered.length} rules`);
    more.addEventListener("click", () => { $$(".rule", card).forEach((x) => (x.hidden = false)); more.remove(); });
    card.appendChild(more);
  }
  if (!r.rules.length) card.appendChild(el("p", "hint", "No rule could be checked on what was read."));
  return card;
}

function fieldsCard(r, container) {
  const editable = r.mode === "receipt" && r.id;
  const card = el("div", "card full");
  card.innerHTML = `<h3>What CVP read <span class="count">${r.fields.length} fields</span></h3>`;
  const edited = new Set((r.truth && r.truth.edited) || []);
  const tbl = el("table", "fields");
  tbl.innerHTML = `<thead><tr><th>Field</th><th>Read</th>${r.fields.some((f) => f.printed) ? "<th>Printed</th>" : ""}` +
    `<th>Edit traces</th><th></th>${editable ? "<th>Correct a misreading</th>" : ""}</tr></thead>`;
  const tb = el("tbody");
  const printedCol = r.fields.some((f) => f.printed);
  for (const f of r.fields) {
    const tr = el("tr", f.role !== "none" ? "hl" : "");
    const roles = [];
    if (f.role !== "none") roles.push(`<span class="role ${f.role}">${ROLE_TEXT[f.role]}</span>`);
    if (edited.has(f.name)) roles.push('<span class="role edited">Really edited</span>');
    tr.innerHTML = `<td>${esc(f.label)}</td><td class="num">${esc(f.read)}</td>` +
      (printedCol ? `<td class="num">${esc(f.printed ?? "")}</td>` : "") +
      `<td><span class="bar"><span style="width:${Math.round(100 * Math.min(1, f.heat || 0))}%"></span></span><span class="num">${(f.heat || 0).toFixed(2)}</span></td>` +
      `<td>${roles.join(" ")}</td>` +
      (editable ? `<td><input class="fix" data-field="${esc(f.name)}" value="${esc(f.read === "unreadable" ? "" : f.read)}" aria-label="Correct ${esc(f.label)}"></td>` : "");
    tb.appendChild(tr);
  }
  tbl.appendChild(tb);
  card.appendChild(tbl);
  if (editable) {
    const orig = Object.fromEntries(r.fields.map((f) => [f.name, f.read === "unreadable" ? "" : f.read]));
    $$("input.fix", card).forEach((inp) => inp.addEventListener("input", () => inp.classList.toggle("changed", inp.value !== orig[inp.dataset.field])));
    const bar = el("div", "recheck");
    const go = el("button", "primary", "Re-check with my corrections");
    go.addEventListener("click", async () => {
      const values = {};
      $$("input.fix", card).forEach((inp) => { if (inp.value !== orig[inp.dataset.field]) values[inp.dataset.field] = inp.value.replace(/,/g, ""); });
      if (!Object.keys(values).length) { go.textContent = "Change a value first"; setTimeout(() => (go.textContent = "Re-check with my corrections"), 1500); return; }
      go.disabled = true;
      try { renderResult(container, await api("/api/recheck", { id: r.id, values })); }
      catch (e) { go.disabled = false; alert(e.message); }
    });
    bar.appendChild(go);
    bar.appendChild(el("span", "hint", "OCR misread a value? Type what the photo shows and re-check the rules."));
    card.appendChild(bar);
  }
  if (r.notes && r.notes.length) card.appendChild(el("ul", "notes", r.notes.map((n) => `<li>${esc(n)}</li>`).join("")));
  return card;
}

function rowsCard(r) {
  const card = el("div", "card full");
  card.innerHTML = '<h3>Every printed row, as read <span class="count">items in blue, bill lines in green</span></h3>';
  const box = el("div", "rowsread");
  for (const row of r.rows) {
    const k = row.kind === "item" ? "item" : row.kind === "other" ? "other" : "sum";
    box.appendChild(el("div", "", `<span class="k k-${k}">${esc(row.kind === "other" ? "—" : row.kind)}</span><span>${esc(row.text)}</span><span>${esc(k === "other" ? "" : row.value)}</span>`));
  }
  card.appendChild(box);
  return card;
}

function techCard(r) {
  const d = el("details", "card tech full", "<summary>Technical details</summary>");
  d.appendChild(el("dl", "", r.tech.map((t) => `<dt>${esc(t.label)}</dt><dd>${esc(t.value)}</dd>`).join("")));
  return d;
}

/* ---------------------------------------------------------------- forge */
function setupForge(cat) {
  const sel = $("#f-doctype");
  sel.innerHTML = cat.doctypes.map((d) => `<option value="${d.name}" ${d.available ? "" : "disabled"}>${esc(d.title)}</option>`).join("") +
    (cat.cord.available ? `<option value="cord">Real receipt photo (CORD)</option>` : "");
  const cats = $("#f-cats");
  const DESC = { genuine: "Leave it alone: a control.", T1: "Paste a new value in a different font.",
    T2: "Redraw a value smoothly, like an AI editor.", T3: "Obvious edit, and fix every total that depends on it.",
    T4: "Smooth edit, and fix every total: keeps every rule." };
  cats.innerHTML = Object.entries(cat.categories).map(([k, v], i) =>
    `<label class="cat"><input type="radio" name="cat" value="${k}" ${k === "T2" ? "checked" : ""}><span><b>${esc(v)}</b><small>${DESC[k]}</small></span></label>`).join("");
  const fieldSel = $("#f-field");
  async function fillFields() {
    const dt = sel.value;
    let fields = [];
    if (dt === "cord") {
      try { fields = await api(`/api/cord_fields/${+$("#f-seed").value || 0}`); } catch (e) { fields = []; }
    } else {
      fields = (cat.doctypes.find((d) => d.name === dt) || { fields: [] }).fields;
    }
    fieldSel.innerHTML = `<option value="">Any field (random)</option>` + fields.map((f) => `<option value="${f.name}">${esc(f.label)}</option>`).join("");
    $("#f-seed-label").textContent = dt === "cord" ? `Receipt number (0–${cat.cord.receipts - 1})` : "Document number";
    $("#f-seed").max = dt === "cord" ? cat.cord.receipts - 1 : 99999;
  }
  function single() {
    const c = $("input[name=cat]:checked").value;
    $("#f-field-wrap").hidden = !(c === "T1" || c === "T2");
  }
  sel.addEventListener("change", fillFields);
  $("#f-seed").addEventListener("change", () => { if (sel.value === "cord") fillFields(); });
  cats.addEventListener("change", single);
  $("#f-dice").addEventListener("click", () => {
    $("#f-seed").value = sel.value === "cord" ? Math.floor(Math.random() * cat.cord.receipts) : Math.floor(Math.random() * 10000);
    if (sel.value === "cord") fillFields();
  });
  fillFields(); single();
  $("#forge-form").addEventListener("submit", (e) => {
    e.preventDefault();
    $$(".example").forEach((x) => x.classList.remove("on"));
    const body = { doctype: sel.value, seed: +$("#f-seed").value || 0, category: $("input[name=cat]:checked").value,
      field: $("#f-field-wrap").hidden ? null : fieldSel.value || null };
    runInto($("#result"), () => api("/api/forge", body));
  });
}

/* ---------------------------------------------------------------- upload */
function setupUpload(cat) {
  const file = $("#u-file"), drop = $("#drop"), go = $("#u-go");
  $("#u-type").innerHTML = cat.doctypes.filter((d) => d.available).map((d) => `<option value="${d.name}">${esc(d.title)}</option>`).join("");
  if (!cat.receipt) {
    $("#u-kind input[value=receipt]").disabled = true;
    $("#u-kind input[value=template]").checked = true;
  }
  const kindChanged = () => { $("#u-type-wrap").hidden = $("input[name=kind]:checked").value !== "template"; };
  $("#u-kind").addEventListener("change", kindChanged); kindChanged();
  const take = (f) => {
    if (!f) return;
    const dt = new DataTransfer(); dt.items.add(f); file.files = dt.files;
    const prev = $("#drop-preview");
    prev.src = URL.createObjectURL(f); prev.hidden = false; $("#drop-in").hidden = true; go.disabled = false;
  };
  file.addEventListener("change", () => take(file.files[0]));
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => take(e.dataTransfer.files[0]));
  $("#upload-form").addEventListener("submit", (e) => {
    e.preventDefault();
    if (!file.files[0]) return;
    $$(".example").forEach((x) => x.classList.remove("on"));
    const fd = new FormData();
    fd.append("file", file.files[0]);
    const k = $("input[name=kind]:checked").value;
    fd.append("kind", k === "template" ? $("#u-type").value : "receipt");
    runInto($("#result"), () => api("/api/upload", fd, true));
  });
}

/* ---------------------------------------------------------------- tour */
function renderStepper() {
  const ol = $("#stepper");
  ol.innerHTML = "";
  state.tour.forEach((s, i) => {
    const li = el("li");
    const b = el("button", (i === state.tourAt ? "on " : "") + (s.seen ? "seen" : ""), `<span class="n">${i + 1}</span><span class="lbl">${esc(s.title)}</span>`);
    b.disabled = !s.available;
    b.addEventListener("click", () => openTourStep(i));
    li.appendChild(b);
    ol.appendChild(li);
  });
}

function openTourStep(i) {
  if (!state.tour.length) return;
  i = Math.max(0, Math.min(state.tour.length - 1, i));
  state.tourAt = i;
  const s = state.tour[i];
  s.seen = true;
  renderStepper();
  const card = $("#tour-card");
  card.innerHTML = `<div><div class="kicker">Step ${i + 1} of ${state.tour.length}</div><h2>${esc(s.title)}</h2><p>${esc(s.intro || "")}</p></div>` +
    `<div class="notice"><b>What to notice</b>${esc(s.notice || "")}</div>`;
  const nav = el("div", "tour-nav");
  const prev = el("button", "ghost", "← Back"), next = el("button", "primary", i === state.tour.length - 1 ? "Finish" : "Next →");
  prev.disabled = i === 0;
  prev.addEventListener("click", () => openTourStep(i - 1));
  next.addEventListener("click", () => { if (i === state.tour.length - 1) { location.hash = "#analyse"; } else openTourStep(i + 1); });
  nav.append(next, prev);
  card.appendChild(nav);
  if (!s.available) { $("#tour-result").innerHTML = '<div class="error-box">This step needs data that is not on this computer.</div>'; return; }
  runInto($("#tour-result"), () => api(`/api/sample/${s.id}`, {}));
}

/* ---------------------------------------------------------------- results page */
const FIGS = [
  ["architecture.png", "How CVP is built: the OCR and the rules shortlist the fields; the visual heatmap picks one."],
  ["cross_type.png", "Six synthetic document types: detection and attribution of CVP against the visual model and the rules alone."],
  ["cross_flags.png", "What each method flags: genuine documents, smooth edits, and edits that keep every rule."],
  ["cord_regimes.png", "Real receipt photos (CORD) with three versions of the visual model."],
  ["aiforge_results.png", "Receipts edited by commercial AI editors (AIForge-Doc): the image model is at chance, the rules are not."],
  ["aiforge_example.png", "A GPT-Image-2 forgery: an empty heatmap, two broken rules, one field that fits both."],
  ["findit_rules.png", "Real forgeries made by people (Find it again!): forged receipts break rules twice as often."],
  ["cord_rules.png", "Real receipts follow the rules: each rule holds on 96.6–99.7% of genuine CORD receipts."],
];

async function renderResultsPage() {
  const tiles = $("#tiles");
  let R = {};
  try { R = await api("/api/results"); } catch (e) { /* show figures anyway */ }
  const pct = (x) => `${(100 * x).toFixed(1)}%`;
  const range = (xs, f) => xs.length ? (Math.min(...xs) === Math.max(...xs) ? f(xs[0]) : `${f(Math.min(...xs))}–${f(Math.max(...xs))}`) : "–";
  const items = [];
  if (R.synthetic) {
    const types = Object.values(R.synthetic);
    items.push(["Six synthetic document types", range(types.map((t) => t.CVP.attr), pct), "edited field named by CVP",
      `The visual model alone: ${range(types.map((t) => t["Visual-only"].attr), pct)}; the rules alone: ${range(types.map((t) => t["Constraint-only"].attr), pct)}.`]);
  }
  if (R.cord && R.cord["zero-shot"]) {
    const z = R.cord["zero-shot"];
    items.push(["Real receipt photos (CORD)", `${z["Visual-only"].AUC.toFixed(3)} → ${z.CVP.AUC.toFixed(3)}`, "detection AUC",
      `With a visual model that never saw a real photo, CVP lifts attribution from ${pct(z["Visual-only"].attr)} to ${pct(z.CVP.attr)}.`]);
  }
  if (R.aiforge) {
    const g = R.aiforge["GPT-Image-2"];
    if (g) items.push(["Real AI forgeries (GPT-Image-2)", `${g["Constraint-only"].AUC.toFixed(3)}`, "AUC of the rules",
      `The image model: ${g["Visual-only"].AUC.toFixed(3)}, a coin flip. People in the dataset's own study: 0.501. CVP names the field for ${pct(g.CVP.attr)}.`]);
  }
  if (R.experimental) {
    const x = R.experimental;
    items.push(["Experimental reader (any photo)", pct(x.false_alarm), "false alarms on genuine photos",
      `On ${x.genuine} genuine CORD photos read without labels; ${pct(x.warning_genuine)} more get a warning. It flags ${pct(x.caught_or_warned)} of smooth single-field edits.`]);
  }
  tiles.innerHTML = items.map(([k, big, unit, p]) => `<div class="tile"><div class="k">${esc(k)}</div><div class="big">${esc(big)} <small>${esc(unit)}</small></div><p>${esc(p)}</p></div>`).join("");
  $("#figs").innerHTML = FIGS.map(([f, c]) => `<figure class="fig"><img src="/figures/${f}" alt="${esc(c)}"><figcaption>${esc(c)}</figcaption></figure>`).join("");
  $$("#figs img").forEach((img) => img.addEventListener("click", () => openLightbox(img)));
}

/* ---------------------------------------------------------------- start */
$$(".tab").forEach((t) => t.addEventListener("click", () => {
  $$(".tab").forEach((x) => x.classList.toggle("on", x === t));
  $$(".tabpane").forEach((p) => p.classList.toggle("on", p.id === `tab-${t.dataset.tab}`));
}));
$("#start-tour").addEventListener("click", () => { location.hash = "#tour"; });

(async function start() {
  pollStatus();
  try {
    const cat = await api("/api/catalog");
    state.catalog = cat;
    state.tour = cat.groups.find((g) => g.id === "tour").items;
    renderExamples(cat.groups);
    setupForge(cat);
    setupUpload(cat);
  } catch (e) {
    $("#result").innerHTML = `<div class="error-box">Could not reach the CVP server: ${esc(e.message)}</div>`;
  }
  route();
})();
