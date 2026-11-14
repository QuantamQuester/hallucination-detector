/* Hallucination Detector UI. Plain JavaScript, no dependencies. Every number shown comes from the API. */
(function () {
"use strict";
const $ = (s, r = document) => r.querySelector(s), $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (p, d = 0) => (p * 100).toFixed(d) + "%";
const prob = (p, d = 0) => (p > 0.999 ? "over 99.9%" : p < 0.001 ? "under 0.1%" : pct(p, d));   // a calibrated score is never a certainty
const LIM = { question: 1000, knowledge: 6000, answer: 2000 };
const store = { get(k) { try { return JSON.parse(localStorage.getItem(k)); } catch (e) { return null; } }, set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) {} } };

async function api(path, body) {
  const t0 = performance.now();
  const r = await fetch(path, body === undefined ? {} : { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  let data = null; const text = await r.text();
  try { data = JSON.parse(text); } catch (e) { data = text; }
  return { status: r.status, ok: r.ok, body: data, ms: Math.round(performance.now() - t0) };
}
function busy(btn, on) { btn.classList.toggle("busy", on); btn.disabled = on; }
let toastTimer;
function toast(msg) { const t = $("#toast"); t.textContent = msg; t.hidden = false; clearTimeout(toastTimer); toastTimer = setTimeout(() => (t.hidden = true), 2600); }
const band = p => (p >= 0.8 ? "high" : p >= 0.5 ? "medium" : p >= 0.2 ? "lowmed" : "low");
const BAND_NAME = { low: "low", lowmed: "low to medium", medium: "medium", high: "high" };
const issues = body => (body && Array.isArray(body.detail) ? body.detail.map(d => [d.loc.slice(1).join(" › ") || "body", d.msg]) : [["request", "rejected"]]);

/* ---------- tooltip (shared by all charts) ---------- */
const tipEl = $("#tooltip");
document.addEventListener("mouseover", e => { const t = e.target.closest("[data-tip]"); if (!t) { tipEl.hidden = true; return; } tipEl.innerHTML = t.dataset.tip; tipEl.hidden = false; });
document.addEventListener("mousemove", e => { if (tipEl.hidden) return; const w = tipEl.offsetWidth, x = Math.min(window.innerWidth - w - 8, e.clientX + 14); tipEl.style.left = x + "px"; tipEl.style.top = (e.clientY + 16) + "px"; });

/* ---------- theme ---------- */
let theme = store.get("hd_theme") || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
function applyTheme() { document.documentElement.dataset.theme = theme; $("#themeTxt").textContent = theme === "dark" ? "Light theme" : "Dark theme"; }
applyTheme();
$("#theme").onclick = () => { theme = theme === "dark" ? "light" : "dark"; store.set("hd_theme", theme); applyTheme(); };

/* ---------- router ---------- */
const VIEWS = ["check", "batch", "monitor", "types", "model", "system"];
let monitorTimer = null;
function route() {
  const v = VIEWS.includes(location.hash.slice(1)) ? location.hash.slice(1) : "check";
  VIEWS.forEach(n => { $("#v-" + n).hidden = n !== v; });
  $$("#nav a").forEach(a => a.dataset.view === v ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current"));
  clearInterval(monitorTimer);
  if (v === "monitor") { loadMonitor(); monitorTimer = setInterval(loadMonitor, 10000); }
  if (v === "model") loadModel();
  if (v === "types") loadTypes();
  if (v === "system") loadSystem();
  window.scrollTo(0, 0);
}
addEventListener("hashchange", route);

/* ---------- health ---------- */
const ALGO = { svm: "Linear SVM", logreg: "Logistic regression", rf: "Random forest", gb: "Gradient boosting", nb: "Naive Bayes", knn: "k-nearest neighbours" };
async function loadHealth() {
  try {
    const h = (await api("/health")).body;
    $("#lamp").className = "lamp up"; $("#statusTxt").textContent = "Service is up";
    $("#railMeta").innerHTML = esc(ALGO[h.algorithm] || h.algorithm || "model") + "<br>version " + esc(h.model_version);
  } catch (e) { $("#lamp").className = "lamp down"; $("#statusTxt").textContent = "Service unreachable"; $("#railMeta").textContent = ""; }
}

/* =====================================================================  CHECK  */
const F = { q: $("#question"), k: $("#knowledge"), a: $("#answer") };
const EX = {
  ex1: { question: "Which magazine was started first, Arthur's Magazine or First for Women?",
    knowledge: "Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century. First for Women is a woman's magazine published by Bauer Media Group in the USA. It was started in 1989.",
    answer: "Arthur's Magazine" },
  ex2: { question: "Which magazine was started first, Arthur's Magazine or First for Women?",
    knowledge: "Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century. First for Women is a woman's magazine published by Bauer Media Group in the USA. It was started in 1989.",
    answer: "First for Women was started first, in 1824, by a New York publisher." },
  ex3: { question: "Who wrote the novel Pride and Prejudice and in which year was it published?", knowledge: "",
    answer: "Pride and Prejudice was written by Charlotte Bronte and published in 1855." }
};
let truth = null, examples = null, lastId = null;
function counters() {
  [["q", "question", "cq"], ["k", "knowledge", "ck"], ["a", "answer", "ca"]].forEach(([k, name, c]) => {
    const n = F[k].value.length, el = $("#" + c);
    el.textContent = n ? n.toLocaleString() + " / " + LIM[name].toLocaleString() : ""; el.classList.toggle("over", n > LIM[name]); F[k].classList.remove("invalid");
  });
}
Object.values(F).forEach(el => el.addEventListener("input", () => { truth = null; counters(); }));
function setForm(o) { F.q.value = o.question || ""; F.k.value = o.knowledge || ""; F.a.value = o.answer || ""; truth = o.truth || null; counters(); }
["ex1", "ex2", "ex3"].forEach(id => ($("#" + id).onclick = () => { setForm(EX[id]); analyze(); }));
async function getExamples() { if (!examples) examples = (await api("/api/examples")).body.examples || []; return examples; }
$("#exRand").onclick = async () => {
  const ex = await getExamples(); if (!ex.length) return toast("No held-out examples are bundled with this model");
  setForm(ex[Math.floor(Math.random() * ex.length)]); analyze();
};
const QGROUP = t => (t === "why" ? "WHY" : t === "polar" ? "POLAR" : "OTHER");
[["exWhy", "WHY"], ["exPolar", "POLAR"]].forEach(([id, g]) => ($("#" + id).onclick = async () => {
  const ex = (await getExamples()).filter(e => QGROUP(e.question_type) === g); if (!ex.length) return toast("No held-out " + g + " example is bundled with this model");
  setForm(ex[Math.floor(Math.random() * ex.length)]); analyze();
}));
$("#clear").onclick = () => { setForm({}); show("empty"); F.q.focus(); };

function show(which) { $("#proofEmpty").hidden = which !== "empty"; $("#proofErr").hidden = which !== "err"; $("#proof").hidden = which !== "proof"; }
function showError(title, items, note) {
  $("#proofErr").innerHTML = `<div class="err" role="alert"><h2>${esc(title)}</h2><ul>${items.map(([f, m]) => `<li><code>${esc(f)}</code> ${esc(m)}</li>`).join("")}</ul>${note ? `<p>${note}</p>` : ""}</div>`;
  show("err");
}
function localProblems() {
  const p = [], q = F.q.value.trim(), a = F.a.value.trim();
  if (q.length < 5) p.push(["question", "needs at least 5 characters", F.q]);
  if (q.length > LIM.question) p.push(["question", "can have at most 1,000 characters", F.q]);
  if (!a.length) p.push(["answer", "is empty", F.a]);
  if (a.length > LIM.answer) p.push(["answer", "can have at most 2,000 characters", F.a]);
  if (F.k.value.length > LIM.knowledge) p.push(["knowledge", "can have at most 6,000 characters", F.k]);
  return p;
}
const formPayload = () => ({ question: F.q.value.trim(), knowledge: F.k.value.trim() || null, answer: F.a.value.trim() });
const marks = pairs => { let i = 0; return pairs.map(([t, m]) => (m ? `<mark class="m-${m}" style="--i:${i++}">${esc(t)}</mark>` : esc(t))).join(""); };

function whyBlock(exp) {
  if (!exp || !exp.groups.length) return "";
  const max = Math.max(...exp.groups.map(g => Math.abs(g.contribution))) || 1;
  const rows = exp.groups.map((g, i) => {
    const w = (Math.abs(g.contribution) / max * 50).toFixed(1), pos = g.contribution >= 0;
    const det = g.details.map(d => `<span>${esc(d.label)}${d.value === null ? "" : " = " + esc(typeof d.value === "number" ? +d.value.toFixed(3) : d.value)}: ${d.contribution >= 0 ? "+" : "−"}${Math.abs(d.contribution).toFixed(2)}</span>`).join("");
    return `<button type="button" class="why-row" aria-expanded="false" aria-controls="wd${i}"><span class="why-name">${esc(g.label)}</span><span class="why-track"><span class="why-bar ${pos ? "pos" : "neg"}" style="width:${w}%"></span></span><span class="why-val">${pos ? "+" : "−"}${Math.abs(g.contribution).toFixed(2)}</span></button><div class="why-detail" id="wd${i}" hidden>${det}</div>`;
  }).join("");
  const ph = exp.top_phrases.length ? `<p class="phrases">Wording that moved the score most: ${exp.top_phrases.map(p => `<span class="chip ${p.contribution >= 0 ? "pos" : "neg"}">${esc(p.phrase)}<small>${p.contribution >= 0 ? "+" : "−"}${Math.abs(p.contribution).toFixed(2)}</small></span>`).join("")}</p>` : "";
  return `<div class="sec"><h3>Why the model gave this score</h3>
    <div class="why-axis"><span></span><div><span><i class="key" style="background:var(--blue)"></i>towards faithful</span><span>towards hallucinated<i class="key" style="background:var(--verm);margin:0 0 0 5px"></i></span></div><span></span></div>
    <div class="why">${rows}</div>${ph}
    <p class="muted" style="margin-top:8px">Contribution of each group of inputs, measured as ${esc(exp.method)}. Select a row for the individual inputs.</p></div>`;
}

function renderProof(payload, res) {
  const j = res.body, p = j.hallucination_probability, b = band(p), s = j.signals, hall = j.label === "hallucinated";
  lastId = j.id;
  const bands = ["low", "lowmed", "medium", "high"];
  const truthLine = truth ? `<p class="truth">This is a held-out test case. Its true label is <b>${esc(truth)}</b>, so the model is <b class="${truth === j.label ? "yes" : "no"}">${truth === j.label ? "right" : "wrong"}</b> here.</p>` : "";
  const ev = j.marked.evidence.length ? `<details class="proof-ev"${j.marked.evidence.length < 120 ? " open" : ""}><summary>Evidence, with the words the answer reuses underlined</summary><p>${marks(j.marked.evidence)}</p></details>`
    : `<p class="truth">No evidence was given, so no word could be checked against a source. The score rests on the answer's length and wording alone.</p>`;
  const req = JSON.stringify(payload, null, 2);
  $("#proof").innerHTML = `
    <div class="sheet proof-sheet">
      <div class="stamp ${b}"><b>${p > 0.995 ? "99%+" : pct(p)}</b>${hall ? "Likely hallucinated" : "Likely faithful"}</div>
      <span class="qpill ${esc((j.question_type || "").toLowerCase())}" data-tip="Detected from the wording of the question. WHY and yes/no questions get their own checks.">${j.question_type === "POLAR" ? "Yes / no question" : esc(j.question_type || "") + " question"}</span>
      <p class="proof-q">${esc(payload.question)}</p>
      <p class="proof-a">${marks(j.marked.answer)}</p>
      ${truthLine}${ev}
    </div>
    <div class="ruler" role="img" aria-label="Probability of hallucination ${prob(p, 1)}, ${BAND_NAME[b]} risk">
      <div class="ruler-bar">${bands.map(x => `<i class="${x === b ? "on" : ""}"></i>`).join("")}<span class="ruler-pin" id="pin"></span></div>
      <div class="ruler-scale">${[["low", "Low, under 20%"], ["lowmed", "Low to medium"], ["medium", "Medium, from 50%"], ["high", "High, from 80%"]].map(([x, t]) => `<span class="${x === b ? "on" : ""}">${t}</span>`).join("")}</div>
    </div>
    <p class="verdict-line">The model puts the chance that this answer is hallucinated at <b class="num">${prob(p, 1)}</b>. ${hall ? "Treat it as unverified until the marked words are checked." : "Nothing in it contradicts the sources at word level."}</p>
    <div class="sec"><h3>What to check</h3><ul class="notes">${j.tips.map(t => `<li>${esc(t)}</li>`).join("")}</ul></div>
    ${whyBlock(j.explanation)}
    <div class="sec"><dl class="facts">
      <div><dt>Answer words found in evidence</dt><dd>${s.grounding_in_evidence === null ? "n/a" : pct(s.grounding_in_evidence)}</dd></div>
      <div><dt>Answer length</dt><dd>${s.answer_words} <span style="font-size:13px;font-weight:500">words</span></dd></div>
      <div><dt>Numbers not in sources</dt><dd>${s.unsupported_numbers === null ? "n/a" : s.unsupported_numbers}</dd></div>
      <div><dt>Names not in evidence</dt><dd>${s.new_named_terms === null ? "n/a" : s.new_named_terms}</dd></div>
      <div><dt>Found in one evidence sentence</dt><dd>${s.best_sentence_support == null ? "n/a" : pct(s.best_sentence_support)}</dd></div>
      <div><dt>Longest copied run</dt><dd>${s.copied_span == null ? "n/a" : pct(s.copied_span)} <span style="font-size:13px;font-weight:500">of the answer</span></dd></div>
    </dl></div>
    <div class="fb" id="fb"><span>Was this verdict right?</span><button class="btn" type="button" data-fb="true">Yes, it was right</button><button class="btn" type="button" data-fb="false">No, it was wrong</button></div>
    <details class="dev"><summary>Request and response</summary>
      <div class="code-head"><span>POST /predict</span><button class="link" type="button" data-copy="devReq">Copy</button></div><pre class="code" id="devReq">${esc(req)}</pre>
      <div class="code-head"><span>HTTP ${res.status}, ${res.ms} ms, model ${esc(j.model_version)}</span><button class="link" type="button" data-copy="devRes">Copy</button></div><pre class="code" id="devRes">${esc(JSON.stringify(j, null, 2))}</pre>
    </details>`;
  show("proof");
  if (window.innerWidth < 1080) $("#proof").scrollIntoView({ behavior: "smooth", block: "start" });
  requestAnimationFrame(() => requestAnimationFrame(() => { $("#pin").style.left = Math.min(99.6, Math.max(0.4, p * 100)) + "%"; }));
}
document.addEventListener("click", async e => {
  const row = e.target.closest(".why-row");
  if (row) { const d = $("#" + row.getAttribute("aria-controls")), open = row.getAttribute("aria-expanded") === "true"; row.setAttribute("aria-expanded", String(!open)); d.hidden = open; return; }
  const fb = e.target.closest("[data-fb]");
  if (fb && lastId) {
    const correct = fb.dataset.fb === "true", r = await api("/feedback", { prediction_id: lastId, correct });
    $("#fb").innerHTML = r.ok ? `<span>Recorded as ${correct ? "right" : "wrong"}. It counts towards the agreement rate in <a href="#monitor">Monitoring</a>.</span>` : `<span>The feedback could not be saved (HTTP ${r.status}).</span>`;
    return;
  }
  const cp = e.target.closest("[data-copy]");
  if (cp) { try { await navigator.clipboard.writeText($("#" + cp.dataset.copy).textContent); toast("Copied"); } catch (err) { toast("Copy is blocked in this browser"); } }
});

async function analyze() {
  const probs = localProblems();
  if (probs.length) { probs.forEach(p => p[2].classList.add("invalid")); showError("The form is not ready to send", probs.map(p => [p[0], p[1]])); probs[0][2].focus(); return; }
  const payload = formPayload(), btn = $("#go"); busy(btn, true);
  try {
    const res = await api("/predict", payload);
    if (res.ok) { renderProof(payload, res); pushHist(payload, res.body); }
    else if (res.status === 422) showError("The API rejected the request (HTTP 422)", issues(res.body), "Validation runs before the model, so a malformed request never causes a server error.");
    else showError("The server answered HTTP " + res.status, [["server", "unexpected response"]]);
  } catch (e) { showError("The service cannot be reached", [["network", "check that the API is running, then try again"]]); }
  finally { busy(btn, false); }
}
$("#form").addEventListener("submit", e => { e.preventDefault(); analyze(); });
document.addEventListener("keydown", e => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter" && !$("#v-check").hidden) { e.preventDefault(); analyze(); } });
$("#bad").onclick = async () => {
  const payload = { question: "Hi", answer: "", temperature: 0.7 };
  try {
    const res = await api("/predict", payload);
    showError("The API rejected the request (HTTP " + res.status + ")", issues(res.body),
      `Sent on purpose: <code>${esc(JSON.stringify(payload))}</code>. The question is too short, the answer is empty and <code>temperature</code> is not part of the contract.`);
  } catch (e) { showError("The service cannot be reached", [["network", "check that the API is running"]]); }
};

/* history (this browser only) */
const hist = () => store.get("hd_hist") || [];
function pushHist(payload, j) { const h = hist(); h.unshift({ payload, p: j.hallucination_probability, label: j.label }); store.set("hd_hist", h.slice(0, 8)); drawHist(); }
function drawHist() {
  const h = hist(); $("#histEmpty").hidden = h.length > 0; $("#clrHist").hidden = !h.length;
  $("#hist").innerHTML = h.map((e, i) => `<button class="hrow" type="button" data-i="${i}"><span class="tag ${e.label === "faithful" ? "f" : "h"}">${e.label === "faithful" ? "Faithful" : "Hallucinated"}</span><span class="ha">${esc(e.payload.answer)}<small>${esc(e.payload.question)}</small></span><span class="hp">${e.p > 0.995 ? "99%+" : pct(e.p)}</span></button>`).join("");
  $$(".hrow").forEach(b => (b.onclick = () => { setForm(h[+b.dataset.i].payload); analyze(); window.scrollTo({ top: 0, behavior: "smooth" }); }));
}
$("#clrHist").onclick = () => { store.set("hd_hist", []); drawHist(); };

/* =====================================================================  BATCH  */
$$(".seg button").forEach(b => (b.onclick = () => {
  $$(".seg button").forEach(x => x.setAttribute("aria-selected", String(x === b)));
  $("#b-rank").hidden = b.dataset.mode !== "rank"; $("#b-file").hidden = b.dataset.mode !== "file"; $("#batchOut").innerHTML = "";
}));
$("#rankFill").onclick = () => { $("#bq").value = F.q.value; $("#bk").value = F.k.value; if (F.a.value && !$("#bcands").value) $("#bcands").value = F.a.value; };
async function scoreItems(items) {            // chunks of 50 = the API's batch limit
  const out = [];
  for (let i = 0; i < items.length; i += 50) {
    const r = await api("/predict/batch", { items: items.slice(i, i + 50).map(({ question, knowledge, answer }) => ({ question, knowledge: knowledge || null, answer })) });
    if (!r.ok) throw r;
    out.push(...r.body.results);
  }
  return out;
}
function batchError(r) {
  const items = r && r.status === 422 ? issues(r.body).map(([f, m]) => [f.replace(/^items › (\d+)/, (_, n) => "row " + (+n + 1)), m]) : [["network", "the service cannot be reached"]];
  $("#batchOut").innerHTML = `<div class="err" role="alert"><h2>${r && r.status ? "The API rejected the batch (HTTP " + r.status + ")" : "Nothing was scored"}</h2><ul>${items.slice(0, 12).map(([f, m]) => `<li><code>${esc(f)}</code> ${esc(m)}</li>`).join("")}</ul><p>A batch is accepted or rejected as a whole, so fix the listed rows and send it again.</p></div>`;
}
const pcell = p => `<span class="pbar"><i style="width:${p * 100}%"></i></span>${pct(p, 1)}`;
const vtag = l => `<span class="tag ${l === "faithful" ? "f" : "h"}">${l === "faithful" ? "Faithful" : "Hallucinated"}</span>`;
$("#rankGo").onclick = async () => {
  const q = $("#bq").value.trim(), k = $("#bk").value.trim(), c = $("#bcands").value.split("\n").map(s => s.trim()).filter(Boolean);
  if (q.length < 5 || !c.length) { $("#batchOut").innerHTML = `<div class="err"><h2>Add a question and at least one answer</h2><p>The question needs 5 characters or more; put each candidate answer on its own line.</p></div>`; return; }
  const btn = $("#rankGo"); busy(btn, true);
  try {
    const res = await scoreItems(c.map(a => ({ question: q, knowledge: k, answer: a })));
    const rows = res.map((r, i) => ({ a: c[i], r })).sort((x, y) => x.r.hallucination_probability - y.r.hallucination_probability);
    $("#batchOut").innerHTML = `<div class="summary"><span><b>${rows.length}</b>answers ranked, most faithful first</span></div><div class="table-wrap"><table><thead><tr><th class="n">Rank</th><th>Answer</th><th>Probability of hallucination</th><th>Verdict</th><th class="n">In evidence</th></tr></thead><tbody>${rows.map((x, i) =>
      `<tr class="${i === 0 ? "best" : ""}"><td class="n">${i + 1}</td><td class="txt">${esc(x.a)}</td><td class="n" style="text-align:left">${pcell(x.r.hallucination_probability)}</td><td>${vtag(x.r.label)}</td><td class="n">${x.r.signals.grounding_in_evidence === null ? "n/a" : pct(x.r.signals.grounding_in_evidence)}</td></tr>`).join("")}</tbody></table></div>`;
  } catch (r) { batchError(r); } finally { busy(btn, false); }
};
function parseCSV(text) {
  const rows = []; let row = [], cur = "", q = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (q) { if (ch === '"') { if (text[i + 1] === '"') { cur += '"'; i++; } else q = false; } else cur += ch; }
    else if (ch === '"') q = true;
    else if (ch === ",") { row.push(cur); cur = ""; }
    else if (ch === "\n" || ch === "\r") { if (ch === "\r" && text[i + 1] === "\n") i++; row.push(cur); cur = ""; if (row.some(c => c !== "")) rows.push(row); row = []; }
    else cur += ch;
  }
  row.push(cur); if (row.some(c => c !== "")) rows.push(row);
  if (!rows.length) return [];
  const head = rows[0].map(h => h.trim().toLowerCase());
  return rows.slice(1).map(r => Object.fromEntries(head.map((h, i) => [h, r[i] === undefined ? "" : r[i]])));
}
function parseRows(text) {
  text = text.trim(); if (!text) return [];
  const rows = text[0] === "[" ? JSON.parse(text) : parseCSV(text);
  return rows.map(r => ({ question: String(r.question ?? ""), knowledge: r.knowledge ? String(r.knowledge) : "", answer: String(r.answer ?? ""), truth: r.truth ? String(r.truth).trim().toLowerCase() : "" }));
}
const csvCell = v => '"' + String(v ?? "").replace(/"/g, '""') + '"';
$("#rows").addEventListener("input", () => { try { const n = parseRows($("#rows").value).length; $("#rowsCount").textContent = n ? n + " rows" : ""; } catch (e) { $("#rowsCount").textContent = "not valid yet"; } });
function setRows(text) { $("#rows").value = text; $("#rows").dispatchEvent(new Event("input")); }
$("#file").onchange = async e => { const f = e.target.files[0]; if (f) setRows(await f.text()); };
const drop = $("#drop");
["dragover", "dragenter"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", async e => { const f = e.dataTransfer.files[0]; if (f) setRows(await f.text()); });
$("#loadHeld").onclick = async () => {
  const ex = await getExamples();
  setRows("question,knowledge,answer,truth\n" + ex.map(x => [x.question, x.knowledge || "", x.answer, x.truth].map(csvCell).join(",")).join("\n"));
  toast(ex.length + " held-out test cases loaded");
};
$("#fileGo").onclick = async () => {
  let rows; try { rows = parseRows($("#rows").value); } catch (e) { rows = null; }
  if (!rows || !rows.length) { $("#batchOut").innerHTML = `<div class="err"><h2>No rows to score</h2><p>Paste CSV with a header line (question, knowledge, answer) or a JSON array, or choose a file.</p></div>`; return; }
  const btn = $("#fileGo"); busy(btn, true);
  try {
    const res = await scoreItems(rows), hasTruth = rows.some(r => r.truth);
    const nH = res.filter(r => r.label === "hallucinated").length, right = rows.filter((r, i) => r.truth && r.truth === res[i].label).length, nT = rows.filter(r => r.truth).length;
    const csv = "question,knowledge,answer,hallucination_probability,label,risk_level" + (hasTruth ? ",truth,correct" : "") + "\n" + rows.map((r, i) => [r.question, r.knowledge, r.answer, res[i].hallucination_probability, res[i].label, res[i].risk_level].concat(hasTruth ? [r.truth, r.truth ? r.truth === res[i].label : ""] : []).map(csvCell).join(",")).join("\n");
    $("#batchOut").innerHTML = `<div class="summary"><span><b>${rows.length}</b>rows scored</span><span><b>${nH}</b>flagged as hallucinated</span>${hasTruth ? `<span><b>${pct(right / nT, 1)}</b>agree with the truth column (${right} of ${nT})</span>` : ""}<a class="btn" id="dl" download="hallucination_scores.csv">Download results as CSV</a></div>
      <div class="table-wrap"><table><thead><tr><th class="n">Row</th><th>Answer and question</th><th>Probability of hallucination</th><th>Verdict</th>${hasTruth ? "<th>Truth</th><th>Match</th>" : ""}</tr></thead><tbody>${rows.map((r, i) =>
      `<tr><td class="n">${i + 1}</td><td class="txt">${esc(r.answer)}<small>${esc(r.question)}</small></td><td class="n" style="text-align:left">${pcell(res[i].hallucination_probability)}</td><td>${vtag(res[i].label)}</td>${hasTruth ? `<td>${esc(r.truth || "")}</td><td>${r.truth ? (r.truth === res[i].label ? "✓ yes" : "✗ no") : ""}</td>` : ""}</tr>`).join("")}</tbody></table></div>`;
    $("#dl").href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  } catch (r) { batchError(r); } finally { busy(btn, false); }
};

/* =====================================================================  MONITORING  */
const tile = (label, value, small) => `<div class="tile"><dt>${label}</dt><dd>${value}</dd>${small ? `<small>${small}</small>` : ""}</div>`;
const FEATURE_NAME = { ans_wc: "Answer length (words)", ans_chars: "Answer length (characters)", avg_wlen: "Average word length", q_wc: "Question length", ctx_wc: "Evidence length", len_ratio: "Answer to evidence length ratio", ctx_support: "Answer words found in evidence", jaccard_ak: "Answer–evidence overlap", novel_content: "Words in neither source", ans_in_q: "Answer words from the question", n_numbers: "Numbers in the answer", unsup_numbers: "Numbers not in sources", novel_caps: "Names not in evidence", negations: "Negation words", has_number: "Answer has a number", kn_missing: "Evidence missing", causal_markers: "Causal words in the answer", hedges: "Hedging words", ttr: "Share of distinct words", soft_overlap: "Fuzzy match with best evidence sentence", q_ctx_overlap: "Question words found in evidence", best_sent_support: "Answer words in one evidence sentence", span_support: "Longest run copied from evidence", evid_causal: "Supporting sentence states a cause", neg_mismatch: "Negation disagrees with evidence" };
function histogram(h) {
  const total = h.reduce((a, b) => a + b, 0); if (!total) return `<div class="empty">No predictions yet. Check an answer, or replay held-out requests below.</div>`;
  const W = 520, H = 190, L = 34, B = 26, T = 10, max = Math.max(...h), bw = (W - L - 8) / 10, y = v => T + (H - T - B) * (1 - v / max);
  const ticks = [0, Math.ceil(max / 2), max].filter((v, i, a) => a.indexOf(v) === i);
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Histogram of predicted probabilities, ${total} predictions">
    ${ticks.map(t => `<line class="grid" x1="${L}" x2="${W - 8}" y1="${y(t)}" y2="${y(t)}"/><text x="${L - 6}" y="${y(t) + 4}" text-anchor="end">${t}</text>`).join("")}
    ${h.map((v, i) => { const x = L + i * bw + 1, yy = y(v), hh = H - B - yy; return `<path class="mk" d="M${x},${H - B} V${yy + Math.min(4, hh)} Q${x},${yy} ${x + Math.min(4, hh)},${yy} H${x + bw - 2 - Math.min(4, hh)} Q${x + bw - 2},${yy} ${x + bw - 2},${yy + Math.min(4, hh)} V${H - B} Z" fill="var(--blue)" ${v ? "" : 'opacity="0"'}/><rect class="hit" x="${x - 1}" y="${T}" width="${bw}" height="${H - T - B}" data-tip="<b>${i * 10}–${i * 10 + 10}%</b><br>${v} prediction${v === 1 ? "" : "s"} (${pct(v / total)})"/>`; }).join("")}
    <line class="axis" x1="${L}" x2="${W - 8}" y1="${H - B}" y2="${H - B}"/>
    <line x1="${L + 5 * bw}" x2="${L + 5 * bw}" y1="${T}" y2="${H - B}" stroke="var(--ink-2)" stroke-dasharray="3 3"/><text class="v" x="${L + 5 * bw + 5}" y="${T + 9}">decision threshold</text>
    ${[0, 20, 40, 50, 60, 80, 100].map(v => `<text x="${L + v / 10 * bw}" y="${H - 8}" text-anchor="middle">${v}%</text>`).join("")}</svg>`;
}
function timeline(tl) {
  if (!tl.length) return `<div class="empty">No predictions yet.</div>`;
  const W = 520, H = 190, L = 34, B = 26, T = 10, n = tl.length, x = i => L + 8 + (W - L - 24) * (n === 1 ? 0.5 : i / (n - 1)), y = p => T + (H - T - B) * (1 - p);
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Probability of the last ${n} predictions">
    ${[0, 0.5, 1].map(t => `<line class="grid" x1="${L}" x2="${W - 8}" y1="${y(t)}" y2="${y(t)}" ${t === 0.5 ? 'stroke-dasharray="3 3" style="stroke:var(--ink-2)"' : ""}/><text x="${L - 6}" y="${y(t) + 4}" text-anchor="end">${t * 100}%</text>`).join("")}
    ${tl.map((d, i) => `<circle class="mk" cx="${x(i)}" cy="${y(d.p)}" r="4" fill="var(--blue)" stroke="var(--sheet)" stroke-width="1.5" data-tip="<b>${pct(d.p, 1)}</b> ${d.p >= 0.5 ? "hallucinated" : "faithful"}<br>${esc(new Date(d.ts).toLocaleTimeString())}, ${d.ms} ms"/>`).join("")}
    <text x="${L + 8}" y="${H - 8}">oldest</text><text x="${W - 16}" y="${H - 8}" text-anchor="end">newest</text></svg>`;
}
async function loadMonitor() {
  let s, d;
  try { [s, d] = (await Promise.all([api("/api/stats"), api("/api/drift")])).map(r => r.body); }
  catch (e) { $("#tiles").innerHTML = `<div class="empty">The monitoring endpoints cannot be reached.</div>`; return; }
  const n = s.window, fb = s.feedback;
  $("#tiles").innerHTML = tile("Predictions logged", n.toLocaleString(), "most recent " + s.window_max.toLocaleString() + " kept in memory")
    + tile("Flagged as hallucinated", n ? pct(s.labels.hallucinated / n) : "–", n ? s.labels.hallucinated + " of " + n : "")
    + tile("Median model time per answer", s.latency_ms.p50 === null ? "–" : s.latency_ms.p50 + " ms", s.latency_ms.p95 === null ? "" : "95th percentile " + s.latency_ms.p95 + " ms")
    + tile("Requests rejected", s.rejected_422, "HTTP 422 since start-up")
    + tile("Verdicts users agreed with", fb.agreement_rate === null ? "–" : pct(fb.agreement_rate), fb.count ? fb.agree + " of " + fb.count + " rated" : "no feedback yet")
    + (s.question_types ? tile("WHY and yes/no questions", s.question_types.WHY.count + " / " + s.question_types.POLAR.count, "flagged: " + s.question_types.WHY.flagged + " WHY, " + s.question_types.POLAR.flagged + " yes/no") : "");
  $("#histo").innerHTML = histogram(s.histogram); $("#timeline").innerHTML = timeline(s.timeline);
  const flagged = d.features.filter(f => f.status === "DRIFT");
  $("#driftBadge").innerHTML = d.status === "ok" ? `<span class="tag f">No drift in ${d.samples} requests</span>` : d.status === "drift" ? `<span class="tag h">Drift in ${flagged.length} of ${d.features.length} inputs</span>` : `<span class="tag warn">Needs ${d.min_samples} requests, has ${d.samples}</span>`;
  $("#driftNote").textContent = d.status === "insufficient_data" || d.status === "unavailable" ? "The check compares the average and spread of each logged input with the training data. It starts once enough requests have been logged."
    : "An input is flagged when its average moved by more than half a training standard deviation, or its spread at least doubled or halved. Flagged inputs are a reason to inspect recent traffic and consider retraining.";
  $("#driftTable").innerHTML = d.features.length ? `<thead><tr><th>Input</th><th class="n">Training average</th><th class="n">Live average</th><th class="n">Shift, in training SDs</th><th class="n">Spread ratio</th><th>Status</th></tr></thead><tbody>${d.features.map(f =>
    `<tr><td>${esc(FEATURE_NAME[f.feature] || f.feature)}</td><td class="n">${f.train_mean}</td><td class="n">${f.live_mean}</td><td class="n">${f.mean_shift_in_sd.toFixed(2)}</td><td class="n">${f.std_ratio.toFixed(2)}</td><td>${f.status === "DRIFT" ? '<span class="tag h">Drift</span>' : '<span class="tag f">Stable</span>'}</td></tr>`).join("")}</tbody>` : "";
}
$("#simNormal").onclick = async e => {
  const btn = e.currentTarget; busy(btn, true);
  try { const ex = await getExamples(); await scoreItems(ex.concat(ex).slice(0, 48)); toast("48 held-out requests replayed"); await loadMonitor(); } catch (r) { toast("The replay failed"); } finally { busy(btn, false); }
};
$("#simDrift").onclick = async e => {
  const btn = e.currentTarget; busy(btn, true);
  const long = "The magazine in question was originally established well before its competitor by a group of regional publishers and editors who later moved the company to a different city where it grew considerably during the following decade. ";
  try { const ex = await getExamples(); await scoreItems(Array.from({ length: 40 }, (_, i) => ({ question: ex[i % ex.length].question, knowledge: null, answer: (long + long).trim() }))); toast("40 long, evidence-free requests sent"); await loadMonitor(); } catch (r) { toast("The requests failed"); } finally { busy(btn, false); }
};

/* =====================================================================  MODEL  */
let modelLoaded = false;
const HP_SHORT = { n_estimators: "trees", max_depth: "depth", learning_rate: "rate", max_leaf_nodes: "leaves", n_neighbors: "k", var_smoothing: "smoothing", C: "C" };
const shortHp = params => Object.entries(params).filter(([k]) => k !== "min_samples_leaf").map(([k, v]) => (HP_SHORT[k] || k) + " " + (+v)).join(", ");
const weightName = n => FEATURE_NAME[n] || (n.startsWith("support_bin_") ? "Grounding class: " + n.slice(12) : n.startsWith("qtype_") ? "Question type: " + n.slice(6) : n.startsWith("ans_len_cat") ? "Answer length class" : n.replace(/_/g, " "));
function expChart(runs, chosen) {
  if (!runs.length) return `<div class="empty">No experiment table is bundled with this model.</div>`;
  const rows = [...runs].sort((a, b) => b.cv_f1 - a.cv_f1), W = 560, rh = 25, L = 214, R = 18, T = 8, H = T + rows.length * rh + 30;
  const lo = Math.min(0.90, Math.floor(Math.min(...rows.map(r => Math.min(r.cv_f1, r.test_f1))) * 100) / 100), hi = 1, x = v => L + (W - L - R) * (v - lo) / (hi - lo);
  const ticks = []; for (let t = lo; t <= hi + 1e-9; t += 0.02) ticks.push(+t.toFixed(2));
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="F1 of each experiment on cross-validation and on the test set">
    ${ticks.map(t => `<line class="grid" x1="${x(t)}" x2="${x(t)}" y1="${T}" y2="${H - 26}"/><text x="${x(t)}" y="${H - 10}" text-anchor="middle">${t.toFixed(2)}</text>`).join("")}
    ${rows.map((r, i) => { const y = T + i * rh + rh / 2, name = (ALGO[r.algorithm] || r.algorithm), hp = Object.entries(r.params).map(([k, v]) => k + "=" + (+v)).join(", ");
      return `<text class="${r.run === chosen ? "v" : ""}" x="${L - 10}" y="${y + 4}" text-anchor="end">${esc(name + ", " + shortHp(r.params))}${r.run === chosen ? " ●" : ""}</text>
      <line x1="${x(r.cv_f1)}" x2="${x(r.test_f1)}" y1="${y}" y2="${y}" stroke="var(--rule-2)" stroke-width="2"/>
      <circle class="mk" cx="${x(r.cv_f1)}" cy="${y}" r="5" fill="var(--blue)" stroke="var(--sheet)" stroke-width="2" data-tip="<b>${esc(name)}</b> (${esc(hp)})<br>Cross-validation F1 ${r.cv_f1.toFixed(4)}"/>
      <rect class="mk" x="${x(r.test_f1) - 4.5}" y="${y - 4.5}" width="9" height="9" transform="rotate(45 ${x(r.test_f1)} ${y})" fill="var(--verm)" stroke="var(--sheet)" stroke-width="2" data-tip="<b>${esc(name)}</b> (${esc(hp)})<br>Test F1 ${r.test_f1.toFixed(4)}"/>`; }).join("")}
    </svg><div class="lg"><span><svg width="12" height="12" style="display:inline;width:12px;vertical-align:-1px"><circle cx="6" cy="6" r="5" fill="var(--blue)"/></svg> cross-validation F1</span><span><svg width="12" height="12" style="display:inline;width:12px;vertical-align:-1px"><rect x="2" y="2" width="8" height="8" transform="rotate(45 6 6)" fill="var(--verm)"/></svg> test F1</span><span>● the saved model</span></div>`;
}
function weightRows(items, key) {
  const max = Math.max(...items.map(i => Math.abs(i.weight))) || 1;
  return items.map(i => { const pos = i.weight >= 0, w = (Math.abs(i.weight) / max * 50).toFixed(1);
    return `<div class="gw-row"><span>${esc(weightName(i[key]))}</span><span class="why-track"><span class="why-bar ${pos ? "pos" : "neg"}" style="width:${w}%"></span></span><span class="why-val">${pos ? "+" : "−"}${Math.abs(i.weight).toFixed(2)}</span></div>`; }).join("");
}
async function loadModel() {
  if (modelLoaded) return;
  let m; try { m = (await api("/api/model")).body; } catch (e) { $("#modelLead").textContent = "The model card cannot be loaded."; return; }
  modelLoaded = true;
  const mt = m.metrics || {}, hp = Object.entries(m.params || {}).map(([k, v]) => k + " = " + v).join(", ");
  $("#modelLead").textContent = `${ALGO[m.algorithm] || m.algorithm}${hp ? " (" + hp + ")" : ""}, trained on ${(m.n_train_rows || 0).toLocaleString()} answers${m.trained_at ? " on " + new Date(m.trained_at).toLocaleDateString(undefined, { year: "numeric", month: "long", day: "numeric" }) : ""}. It was saved because it had the best cross-validated F1 of the ${m.experiments.length} tracked runs; the test set played no part in the choice.`;
  $("#mTiles").innerHTML = tile("Test accuracy", pct(mt.test_accuracy, 1)) + tile("Test F1", mt.test_f1.toFixed(3), "precision " + mt.test_precision.toFixed(3) + ", recall " + mt.test_recall.toFixed(3))
    + tile("ROC-AUC", mt.test_roc_auc.toFixed(3)) + tile("Cross-validated F1", mt.cv_f1.toFixed(3), "± " + mt.cv_f1_std.toFixed(3) + " across grouped folds") + tile("Decision threshold", pct(m.decision_threshold), "scores at or above count as hallucinated");
  $("#expChart").innerHTML = expChart(m.experiments, m.run_name);
  $("#cmCap").textContent = `${(m.n_test_rows || 0).toLocaleString()} answers the model never saw during training`;
  if (m.by_question_type) $("#mTiles").innerHTML += Object.entries(m.by_question_type).map(([q, v]) => tile((q === "POLAR" ? "Yes/no" : q === "OTHER" ? "Other" : q) + " questions", pct(v.accuracy, 1), v.n.toLocaleString() + " test answers")).join("");
  const cm = m.confusion_matrix;
  $("#confusion").innerHTML = cm ? `<div class="cm"><div></div><div class="h">Predicted faithful</div><div class="h">Predicted hallucinated</div>
    <div class="h">Actually faithful</div><div class="c good"><b>${cm[0][0].toLocaleString()}</b>correct</div><div class="c miss"><b>${cm[0][1]}</b>false alarms</div>
    <div class="h">Actually hallucinated</div><div class="c miss"><b>${cm[1][0]}</b>missed</div><div class="c good"><b>${cm[1][1].toLocaleString()}</b>caught</div></div>
    <p class="muted" style="text-align:center">${cm[1][0]} hallucinations slipped through and ${cm[0][1]} faithful answers were flagged, out of ${(m.n_test_rows || 0).toLocaleString()}.</p>` : `<div class="empty">Not recorded for this model.</div>`;
  $("#expTable").innerHTML = `<thead><tr><th>Run</th><th>Algorithm</th><th>Hyper-parameters</th><th class="n">CV F1</th><th class="n">Test accuracy</th><th class="n">Test F1</th><th class="n">ROC-AUC</th><th class="n">Train time</th></tr></thead><tbody>${m.experiments.map(r =>
    `<tr class="${r.run === m.run_name ? "chosen" : ""}"><td>${esc(r.run)}${r.run === m.run_name ? " <b>(saved)</b>" : ""}</td><td>${esc(ALGO[r.algorithm] || r.algorithm)}</td><td>${esc(Object.entries(r.params).map(([k, v]) => k + "=" + (+v)).join(", "))}</td><td class="n">${r.cv_f1.toFixed(4)}</td><td class="n">${r.test_accuracy.toFixed(4)}</td><td class="n">${r.test_f1.toFixed(4)}</td><td class="n">${r.test_roc_auc.toFixed(4)}</td><td class="n">${r.train_seconds.toFixed(1)} s</td></tr>`).join("")}</tbody>`;
  const g = m.global_weights;
  $("#gwKind").textContent = g.features.length ? "Largest " + g.kind + ". Bars to the right push a score towards hallucinated, bars to the left towards faithful." : "Not available for this model type.";
  $("#gwFeat").innerHTML = weightRows(g.features, "name");
  $("#gwPhr").innerHTML = g.phrases_hallucinated.length ? `<p class="phrases" style="margin-top:0"><b>Words that raise the score</b><br>${g.phrases_hallucinated.map(p => `<span class="chip pos">${esc(p.phrase)}<small>+${p.weight.toFixed(2)}</small></span>`).join("")}</p><p class="phrases"><b>Words that lower it</b><br>${g.phrases_faithful.map(p => `<span class="chip neg">${esc(p.phrase)}<small>−${Math.abs(p.weight).toFixed(2)}</small></span>`).join("")}</p><p class="muted" style="margin-top:10px">“was” and “is” top the list because the benchmark's hallucinated answers are written as full sentences. This is the length shortcut described under Known limits.</p>` : "";
  const props = m.request_schema.properties, req = m.request_schema.required || [];
  const rule = p => { const s = p.anyOf ? p.anyOf.find(x => x.type === "string") : p; return [s.minLength !== undefined ? "at least " + s.minLength : "", s.maxLength !== undefined ? "at most " + s.maxLength.toLocaleString() + " characters" : ""].filter(Boolean).join(", "); };
  $("#contract").innerHTML = `<thead><tr><th>Field</th><th>Type</th><th>Required</th><th>Allowed</th><th>What it is</th></tr></thead><tbody>${Object.entries(props).map(([k, p]) =>
    `<tr><td><code>${esc(k)}</code></td><td>${p.anyOf ? "string or null" : esc(p.type)}</td><td>${req.includes(k) ? "yes" : "no"}</td><td>${esc(rule(p))}${req.includes(k) ? ", not blank" : ""}</td><td>${esc(p.description || "")}</td></tr>`).join("")}
    ${m.request_schema.additionalProperties === false ? `<tr><td>any other field</td><td>–</td><td>–</td><td>rejected</td><td>Unknown fields are refused, so a misspelt field name cannot be ignored silently.</td></tr>` : ""}</tbody>`;
}

/* =====================================================================  QUESTION TYPES  */
let typesLoaded = false;
const QNAME = { WHY: "WHY questions", POLAR: "Yes / no questions", OTHER: "All other questions" };
const STRAT = { chatgpt: "written by ChatGPT (HaluEval)", foreign_cause: "invented cause (from another topic)", evidence_distractor: "wrong clause of the same passage", altered_fact: "true cause with one fact changed" };
const hbar = (name, v, on, tip) => `<div class="hbar ${on ? "on" : ""}"${tip ? ` data-tip="${esc(tip)}"` : ""}><span class="nm">${esc(name)}</span><span class="trk"><i style="width:${(v * 100).toFixed(1)}%"></i></span><b>${pct(v, 1)}</b></div>`;
async function loadTypes() {
  if (typesLoaded) return;
  let d; try { d = (await api("/api/research")).body; } catch (e) { d = null; }
  if (!d || !d.available) { $("#typesLead").textContent = "The question-type study has not been generated for this model. Run python qtype_study.py."; return; }
  typesLoaded = true;
  const pipes = d.pipelines.map(p => ({ key: p.key, name: p.key + ". " + p.name, sub: p.features + ", " + p.train_data, q: p.by_question_type, s: p.recall_by_strategy, n: p.overall.n }));
  if (d.deployed && d.deployed.by_question_type) pipes.push({ key: "D", name: "Deployed model (" + (ALGO[d.deployed.algorithm] || d.deployed.algorithm) + ")", sub: "the model behind this service, new features and data", q: d.deployed.by_question_type, s: d.deployed.recall_by_strategy || {}, on: true });
  if (d.bert) pipes.push({ key: "BERT", name: "BERT transformer", sub: `${d.bert.model}, ${d.bert.n_train_rows.toLocaleString()} training rows`, q: d.bert.by_question_type, s: d.bert.recall_by_strategy || {} });
  const A = d.pipelines[0].by_question_type, dep = (d.deployed && d.deployed.by_question_type) || d.pipelines[2].by_question_type;
  const audit = Object.fromEntries(d.audit.rows.map(r => [r.type, r]));
  $("#tTiles").innerHTML = tile("WHY questions in HaluEval-QA", audit.why.v2 + " of " + d.audit.n_questions.toLocaleString(), "too few to study, hence the extension")
    + tile("WHY / POLAR dataset", d.dataset.rows.toLocaleString() + " rows", d.dataset.questions.toLocaleString() + " questions, two answers each")
    + tile("WHY accuracy", pct(A.WHY.accuracy, 0) + " → " + pct(dep.WHY.accuracy, 0), "first deployment → model now served")
    + tile("Yes/no accuracy", pct(dep.POLAR.accuracy, 1), "driven by answer length, see below")
    + tile("Typed “other” before", pct(audit.other.v1 / d.audit.n_questions, 0) + " → " + pct(audit.other.v2 / d.audit.n_questions, 0), "share of questions the rule could not type");
  $("#pipeNote").textContent = `${d.dataset.split.test_rows.toLocaleString()} held-out answers (${Object.entries(d.dataset.split.test_groups).map(([k, v]) => v.toLocaleString() + " " + k).join(", ")}). A, B and C use the same algorithm (${d.pipeline_algorithm.split(" (")[0]}), so the differences come from the data and the features.${d.bert ? ` BERT was scored on ${d.bert.n_test_rows.toLocaleString()} of these rows.` : ""} The dashed line marks 50%, the accuracy of guessing.`;
  $("#pipeCharts").innerHTML = ["WHY", "POLAR", "OTHER"].map(g => `<div class="qpanel"><h3>${QNAME[g]}</h3><span class="muted">${(d.dataset.split.test_groups[g] || 0).toLocaleString()} test answers</span>${pipes.filter(p => p.q[g]).map(p => hbar(p.name, p.q[g].accuracy, p.on, `<b>${p.name}</b><br>${p.sub}<br>accuracy ${pct(p.q[g].accuracy, 1)}, F1 ${p.q[g].f1.toFixed(3)}, AUC ${p.q[g].roc_auc.toFixed(3)}<br>${p.q[g].n} answers`)).join("")}</div>`).join("");
  const keys = [...new Set(pipes.flatMap(p => Object.keys(p.s)))].sort((a, b) => (a.startsWith("WHY") ? 0 : 1) - (b.startsWith("WHY") ? 0 : 1) || a.localeCompare(b));
  $("#stratTable").innerHTML = `<thead><tr><th>Question type</th><th>How the hallucinated answer was made</th>${pipes.map(p => `<th class="n" data-tip="${esc(p.name + "<br>" + p.sub)}">${p.key === "D" ? "Deployed" : p.key}</th>`).join("")}</tr></thead><tbody>${keys.map(k => { const [g, st] = k.split("/");
    return `<tr><td>${QNAME[g].replace(" questions", "")}</td><td>${esc(STRAT[st] || st)}</td>${pipes.map(p => `<td class="n">${p.s[k] ? pcell(p.s[k].recall) : "–"}</td>`).join("")}</tr>`; }).join("")}</tbody>`;
  const fams = ["lexical", "semantic", "contextual", "all three"];
  $("#famChart").innerHTML = `<div class="table-wrap"><table><thead><tr><th>Question type</th>${fams.map(f => `<th class="n">${f[0].toUpperCase() + f.slice(1)}</th>`).join("")}</tr></thead><tbody>${["WHY", "POLAR", "OTHER"].map(g => { const row = d.family_accuracy[g], best = Math.max(...fams.slice(0, 3).map(f => row[f]));
    return `<tr><td>${QNAME[g]}</td>${fams.map(f => `<td class="n">${row[f] === best && f !== "all three" ? "<b>" + pct(row[f], 1) + "</b>" : pct(row[f], 1)}</td>`).join("")}</tr>`; }).join("")}</tbody></table></div>
    <p class="muted" style="margin-top:10px"><b>Lexical</b>: the answer's own surface (length, numbers, negations, causal and hedging words). <b>Semantic</b>: fuzzy and set overlap between answer, question and evidence. <b>Contextual</b>: whether the evidence supports the answer (per sentence, copied spans, new names and numbers, causal support). The best single family per row is in bold.</p>`;
  const amax = Math.max(...d.audit.rows.map(r => Math.max(r.v1, r.v2)));
  $("#auditChart").innerHTML = `<div class="table-wrap"><table><thead><tr><th>Type</th><th class="n">Old rule</th><th class="n">New rule</th><th></th></tr></thead><tbody>${d.audit.rows.map(r => `<tr><td>${r.type === "polar" ? "yes / no" : r.type}</td><td class="n">${r.v1.toLocaleString()}</td><td class="n"><b>${r.v2.toLocaleString()}</b></td><td><span class="pbar" style="width:110px"><i style="width:${r.v2 / amax * 100}%"></i></span></td></tr>`).join("")}</tbody></table></div>
    <p class="muted" style="margin-top:10px">The old rule looked for a question word in the first four words only. Many questions end with it (“… was born in which year?”), so the new rule also takes the last question word.</p>`;
  $("#contrastCols").innerHTML = ["WHY", "POLAR", "OTHER"].map(g => `<div class="qpanel"><h3>${QNAME[g]}</h3><span class="muted">${d.contrasts[g].n_rows.toLocaleString()} training answers</span>${d.contrasts[g].features.slice(0, 6).map(f =>
    `<div class="crow"><span>${esc(FEATURE_NAME[f.feature] || f.feature)}<span class="fam">${f.family}</span></span><b>AUC ${f.auc.toFixed(2)}</b><small>${f.cohen_d > 0 ? "↑ higher in hallucinated" : "↑ higher in faithful"} answers (${f.faithful} faithful, ${f.hallucinated} hallucinated)</small></div>`).join("")}</div>`).join("");
  $("#dsNote").innerHTML = `<code>data/why_polar_dataset.csv</code> holds the question, its type, the evidence, the answer, the label and how the answer was produced. <code>data/why_polar_features.csv</code> adds every lexical, semantic and contextual feature per row. Both are rebuilt by <code>build_why_polar.py</code> and are in the <a href="https://github.com/QuantamQuester/hallucination-detector/tree/main/data">repository</a>.`;
  $("#dsTable").innerHTML = `<thead><tr><th>Question type</th><th>Source</th><th>Answer</th><th class="n">Rows</th></tr></thead><tbody>${d.dataset.composition.map(r => `<tr><td>${r.qtype === "POLAR" ? "Yes / no" : r.qtype}</td><td>${esc(r.source)}</td><td>${esc({ gold_answer: "correct answer", gold_cause: "annotated cause (faithful)" }[r.strategy] || "hallucinated: " + (STRAT[r.strategy] || r.strategy))}</td><td class="n">${r.rows.toLocaleString()}</td></tr>`).join("")}</tbody>`;
  if (d.bert) { const b = d.bert, m = b.metrics;
    $("#bertBlock").hidden = false;
    $("#bertNote").textContent = `${b.model} reads the question, the answer and the evidence as one sequence (up to ${b.max_len} word pieces) and is fine-tuned to output faithful or hallucinated. This run trained ${b.frozen_layers ? "the top " + (12 - b.frozen_layers) + " of 12 layers" : "all layers"} for ${b.epochs_done} epochs on ${b.n_train_rows.toLocaleString()} of the ${b.n_train_rows_available.toLocaleString()} training rows (${b.device.toUpperCase()}, ${b.train_minutes} minutes). It is an offline experiment: the 420 MB model is not part of this service.`;
    $("#bertTiles").innerHTML = tile("Test accuracy", pct(m.test_accuracy, 1), b.n_test_rows.toLocaleString() + " test answers") + tile("F1", m.test_f1.toFixed(3), "precision " + m.test_precision.toFixed(3) + ", recall " + m.test_recall.toFixed(3)) + tile("ROC-AUC", m.test_roc_auc.toFixed(3))
      + Object.entries(b.by_question_type).map(([q, v]) => tile(QNAME[q], pct(v.accuracy, 1), "AUC " + v.roc_auc.toFixed(3))).join(""); }
}

/* =====================================================================  SYSTEM  */
const PIPE = [["Data", "12,000 question–answer pairs from HaluEval-QA plus a WHY / yes-no extension, half faithful and half hallucinated.", "data/"],
  ["Features", "Lexical, semantic and contextual inputs with a question-type rule. One module serves both training and the API.", "features.py"],
  ["Training", "Six algorithms on a split grouped by question, chosen by grouped cross-validation.", "train.py"],
  ["Experiment tracking", "MLflow records each run's settings, metrics and model, and marks the champion.", "mlflow.db"],
  ["Model file", "The best pipeline is saved once and loaded once when the service starts.", "model/model.joblib"],
  ["API and this interface", "FastAPI validates every request, scores it, explains it and logs it.", "app.py"],
  ["Container and CI", "A slim Docker image; GitHub Actions trains, tests and builds on every push.", "Dockerfile"],
  ["Deployment and monitoring", "Hosted with a health check and restarts; logs, metrics and a drift check watch it.", "render.yaml"]];
const ENDPOINTS = [["GET", "/health", "Liveness probe used by the host", true], ["POST", "/predict", "Score one answer, with reasons and word marks", false], ["POST", "/predict/batch", "Score up to 50 answers in one call", false],
  ["POST", "/feedback", "Record whether a verdict was right", false], ["GET", "/api/stats", "Recent traffic for the Monitoring page", true], ["GET", "/api/drift", "Input drift against the training data", true],
  ["GET", "/api/model", "Model card, experiments and request schema", true], ["GET", "/api/examples", "Held-out test cases", true], ["GET", "/api/research", "Question-type study and BERT results", true], ["GET", "/metrics", "Prometheus metrics", true], ["GET", "/docs", "Interactive API documentation", true], ["GET", "/openapi.json", "Machine-readable API description", true]];
async function ping() {
  const btn = $("#pingAll"); busy(btn, true);
  await Promise.all(ENDPOINTS.map(async ([m, path, , can], i) => {
    const cell = $("#ep" + i); if (!can) return;
    try { const t0 = performance.now(), r = await fetch(path), ms = Math.round(performance.now() - t0); cell.innerHTML = `<span class="dotc ${r.ok ? "up" : "down"}"></span>HTTP ${r.status}, ${ms} ms`; }
    catch (e) { cell.innerHTML = `<span class="dotc down"></span>unreachable`; }
  }));
  busy(btn, false);
}
let systemLoaded = false;
function loadSystem() {
  if (!systemLoaded) {
    systemLoaded = true;
    $("#pipeline").innerHTML = PIPE.map(([t, d, f]) => `<li><b>${t}</b><p>${d}</p><code>${f}</code></li>`).join("");
    $("#endpoints").innerHTML = `<thead><tr><th>Method</th><th>Path</th><th>What it does</th><th>Right now</th></tr></thead><tbody>${ENDPOINTS.map(([m, p, d, can], i) => `<tr><td>${m}</td><td>${can ? `<a href="${p}">${p}</a>` : `<code>${p}</code>`}</td><td>${d}</td><td id="ep${i}">${can ? "…" : `<span class="muted">used by this interface</span>`}</td></tr>`).join("")}</tbody>`;
    const body = { question: "Who wrote Pride and Prejudice?", knowledge: "Pride and Prejudice is an 1813 novel by Jane Austen.", answer: "Charlotte Bronte wrote it in 1847." };
    $("#snCurl").textContent = `curl -s -X POST ${location.origin}/predict \\\n  -H 'content-type: application/json' \\\n  -d '${JSON.stringify(body)}'`;
    $("#snPy").textContent = `import requests\n\nr = requests.post("${location.origin}/predict", json=${JSON.stringify(body, null, 4)})\nr.raise_for_status()          # 422 if the request breaks the contract\nprint(r.json()["hallucination_probability"], r.json()["tips"])`;
    $("#pingAll").onclick = ping;
  }
  ping();
}

/* ---------- start ---------- */
counters(); drawHist(); loadHealth(); setInterval(loadHealth, 30000); route();
})();
