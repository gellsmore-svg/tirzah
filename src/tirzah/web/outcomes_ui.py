"""Single-file authoring UI for the outcomes-validation loop (phase 3).

Served at ``GET /process/outcomes`` by the web app. Mirrors the family's
single-file composer pattern (Cairn/Ed): author outcomes + loop config for a
process template, preview how a sample answer would score (drift), and save the
loop onto the template (a new version). Talks only to the existing
``/api/process/*`` JSON routes.
"""

from __future__ import annotations

OUTCOMES_COMPOSER_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tirzah — outcomes loop</title>
<style>
:root { --bg:#f6f7f9; --surface:#fff; --ink:#1c2128; --muted:#667085; --line:#e3e6ea;
  --line-soft:#edf0f3; --accent:#2f5fd0; --good:#14803c; --bad:#c22736; }
@media (prefers-color-scheme: dark) { :root { --bg:#12151b; --surface:#1a1e26; --ink:#e5e8ee;
  --muted:#97a0af; --line:#2a3039; --line-soft:#232833; --accent:#7c9cff; --good:#7edc9f; --bad:#ff9aa4; } }
* { box-sizing:border-box; }
body { font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; margin:0; color:var(--ink);
  background:var(--bg); font-size:14px; line-height:1.5; }
header { position:sticky; top:0; background:var(--surface); border-bottom:1px solid var(--line); }
header nav { display:flex; align-items:center; gap:.6em; padding:.55em 1em; }
.brand { font-weight:700; } header .muted { color:var(--muted); }
.status { margin-left:auto; color:var(--muted); font-size:.85em; font-family:ui-monospace,Menlo,monospace; }
.grid { display:grid; grid-template-columns: 1fr 1fr; gap:1px; background:var(--line);
  min-height:calc(100vh - 47px); }
.pane { background:var(--bg); overflow:auto; padding:1em; }
.pane h2 { font-size:.78em; text-transform:uppercase; letter-spacing:.05em; color:var(--muted);
  font-weight:600; margin:0 0 .7em; }
label.f { display:block; font-size:.82em; color:var(--muted); margin:.6em 0 .2em; }
input[type=text], input[type=number], select, textarea { width:100%; padding:.4em .5em;
  background:var(--surface); color:var(--ink); border:1px solid var(--line); border-radius:6px;
  font-size:13px; font-family:inherit; }
textarea { resize:vertical; font-family:ui-monospace,Menlo,monospace; font-size:12.5px; }
.row { display:flex; gap:.6em; } .row > * { flex:1; }
button { cursor:pointer; border:1px solid var(--line); background:var(--surface); color:var(--ink);
  border-radius:6px; padding:.45em .7em; font-size:13px; } button:hover { border-color:var(--accent); }
button.primary { background:var(--accent); border-color:var(--accent); color:#fff; font-weight:600; }
.outcome { border:1px solid var(--line); border-radius:8px; padding:.6em; margin-bottom:.5em; background:var(--surface); }
.outcome .row { align-items:center; }
.outcome button.del { flex:0 0 auto; padding:.2em .5em; }
.per { border:1px solid var(--line); border-radius:8px; padding:.5em .7em; margin:.3em 0; background:var(--surface); }
.met { color:var(--good); } .unmet { color:var(--bad); } .partial { color:var(--muted); }
.score { font-size:1.4em; font-weight:700; } .drift { color:var(--bad); } .ok { color:var(--good); }
hr { border:none; border-top:1px solid var(--line); margin:1em 0; }
small.hint { color:var(--muted); } code { background:var(--line-soft); padding:.05em .35em; border-radius:4px; }
</style></head>
<body>
<header><nav><span class="brand">Tirzah</span>
  <span class="muted">outcomes-validation loop</span>
  <span class="status" id="status"></span></nav></header>
<div class="grid">
  <section class="pane">
    <h2>Author</h2>
    <label class="f">Template</label>
    <div class="row"><select id="template"></select><button id="load" style="flex:0 0 auto">Load</button></div>
    <label class="f">Outcomes</label>
    <div id="outcomes"></div>
    <button id="addOutcome">+ Add outcome</button>
    <hr>
    <div class="row">
      <div><label class="f">On drift</label><select id="on_drift">
        <option>reanchor_then_gate</option><option>reanchor</option><option>gate</option><option>log</option>
      </select></div>
      <div><label class="f">Cadence</label><select id="cadence">
        <option>every_revision</option><option>on_complete</option><option>every_n_calls</option>
      </select></div>
    </div>
    <div class="row">
      <div><label class="f">Drift threshold (0–1)</label>
        <input type="number" id="drift_threshold" min="0" max="1" step="0.01" value="0.34"></div>
      <div><label class="f">Judge</label><select id="judge">
        <option>deterministic</option><option>llm</option>
      </select></div>
    </div>
    <hr>
    <div class="row"><button class="primary" id="save">Save loop to template</button></div>
    <small class="hint">Saves a new template version with these outcomes + loop.</small>
  </section>

  <section class="pane">
    <h2>Preview</h2>
    <label class="f">Sample answer / work</label>
    <textarea id="work" rows="6" placeholder="Paste a sample answer to score against the outcomes…"></textarea>
    <button id="preview" style="margin-top:.5em">Preview drift</button>
    <div id="result" style="margin-top:.8em"><small class="hint">Author outcomes, paste a sample answer, and preview.</small></div>
  </section>
</div>
<script>
const $ = (id) => document.getElementById(id);
const setStatus = (t) => { $("status").textContent = t; };
function esc(s){
  return String(s==null?"":s)
    .replace(/&/g,"&amp;")
    .replace(/</g,"&lt;")
    .replace(/>/g,"&gt;")
    .replace(/"/g,"&quot;")
    .replace(/'/g,"&#39;");
}

function outcomeRow(o){
  const div = document.createElement("div");
  div.className = "outcome";
  const row = document.createElement("div");
  row.className = "row";
  const id = document.createElement("input");
  id.type = "text";
  id.className = "o-id";
  id.placeholder = "id (O1)";
  id.style.flex = "0 0 5em";
  id.value = o && o.id ? String(o.id) : "";
  const stmt = document.createElement("input");
  stmt.type = "text";
  stmt.className = "o-stmt";
  stmt.placeholder = "outcome statement";
  stmt.value = o && o.statement ? String(o.statement) : "";
  const del = document.createElement("button");
  del.type = "button";
  del.className = "del";
  del.textContent = "✕";
  del.onclick = () => div.remove();
  row.append(id, stmt, del);
  const check = document.createElement("input");
  check.type = "text";
  check.className = "o-check";
  check.placeholder = "optional check (keywords that must appear)";
  check.style.marginTop = ".4em";
  check.value = o && o.check ? String(o.check) : "";
  div.append(row, check);
  return div;
}
function addOutcome(o){ $("outcomes").appendChild(outcomeRow(o||{})); }

function readOutcomes(){
  return [...document.querySelectorAll("#outcomes .outcome")].map(d => {
    const out = { statement: d.querySelector(".o-stmt").value.trim() };
    const id = d.querySelector(".o-id").value.trim(); if (id) out.id = id;
    const check = d.querySelector(".o-check").value.trim(); if (check) out.check = check;
    return out;
  }).filter(o => o.statement);
}
function readLoop(){
  return { on_drift: $("on_drift").value, cadence: $("cadence").value,
    drift_threshold: Number($("drift_threshold").value), judge: $("judge").value };
}

async function loadTemplates(){
  const data = await (await fetch("/api/process/templates")).json();
  const sel = $("template"); sel.innerHTML = "";
  for (const t of (data.templates||[])){
    const o = document.createElement("option"); o.value = t.template_id;
    o.textContent = t.name + " (" + t.template_id + ")"; sel.appendChild(o);
  }
}
async function loadOutcomes(){
  const id = $("template").value; if (!id) return;
  const data = await (await fetch("/api/process/templates/"+encodeURIComponent(id)+"/outcomes")).json();
  $("outcomes").innerHTML = "";
  (data.outcomes||[]).forEach(addOutcome);
  if (!(data.outcomes||[]).length) addOutcome();
  const loop = data.outcomes_loop || {};
  if (loop.on_drift) $("on_drift").value = loop.on_drift;
  if (loop.cadence) $("cadence").value = loop.cadence;
  if (loop.drift_threshold != null) $("drift_threshold").value = loop.drift_threshold;
  if (loop.judge) $("judge").value = loop.judge;
  setStatus("loaded "+id);
}
async function save(){
  const id = $("template").value; if (!id) return;
  const res = await fetch("/api/process/templates/"+encodeURIComponent(id)+"/outcomes", {
    method:"PUT", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({ outcomes: readOutcomes(), outcomes_loop: readLoop() }),
  });
  const data = await res.json();
  if (res.ok && data.ok) setStatus("saved v"+data.template.version);
  else setStatus("save failed: "+(data.detail||"error"));
}
async function preview(){
  setStatus("scoring…");
  const res = await fetch("/api/process/outcomes/preview", {
    method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({ outcomes: readOutcomes(), outcomes_loop: readLoop(), work: { answer: $("work").value } }),
  });
  const data = await res.json();
  if (!res.ok || !data.ok){ $("result").innerHTML = "<span class='unmet'>"+esc(data.detail||"error")+"</span>"; return; }
  const v = data.validation;
  if (!v.ready){ $("result").innerHTML = "<small class='hint'>Add at least one outcome to score.</small>"; return; }
  const rows = v.per_outcome.map(o =>
    "<div class='per'><b>"+esc(o.id)+"</b> <span class='"+o.status+"'>"+o.status+"</span>"
    + " <small class='hint'>(coverage "+o.coverage+")</small><br><small>"+esc(o.statement)+"</small></div>").join("");
  $("result").innerHTML =
    "<div class='score "+(v.drifting?"drift":"ok")+"'>drift "+(v.drift_score*100).toFixed(0)+"%"
    + (v.drifting?" — DRIFTING":" — on track")+"</div>"
    + "<small class='hint'>threshold "+v.threshold+(v.model_used?" · model-judged":" · deterministic")+"</small>"
    + rows;
  setStatus("scored");
}

$("addOutcome").onclick = () => addOutcome();
$("load").onclick = loadOutcomes;
$("save").onclick = save;
$("preview").onclick = preview;
(async () => { await loadTemplates(); await loadOutcomes(); })();
</script></body></html>
"""
