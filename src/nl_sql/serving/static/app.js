"use strict";

const $ = (id) => document.getElementById(id);
const el = (tag, text, className) => {
  const node = document.createElement(tag);
  if (text !== undefined && text !== null) node.textContent = String(text);
  if (className) node.className = className;
  return node;
};
const store = {
  get(key) { try { return localStorage.getItem(key); } catch { return null; } },
  set(key, value) { try { localStorage.setItem(key, value); } catch {} },
};

const REQUEST_TIMEOUT_MS = 150_000;
const STEP_LABELS = {
  list_tables: "Listed the tables",
  inspect_schema: "Read every table's columns",
  inspect_table: "Read one table's columns",
  query: "Ran a read-only query",
  lookup_column_meaning: "Looked up what a column means",
  search_knowledge: "Checked the business definitions",
  submit_answer: "Wrote the answer",
};
// Shown while waiting; the server answers in one response, so these describe the usual path.
const WAITING_STEPS = ["Reading the table layout", "Writing a query", "Running it and checking the rows", "Writing the answer"];
const SQL_KEYWORDS = new Set("select from where join left right inner outer on group by order limit as and or not in is null sum count avg min max distinct having case when then else end with desc asc union all like between exists json_extract coalesce cast round".split(" "));

/* Theme */
const themeButton = $("theme-toggle");
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  themeButton.setAttribute("aria-label", theme === "dark" ? "Switch to light theme" : "Switch to dark theme");
}
applyTheme(document.documentElement.dataset.theme || "dark");
themeButton.addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  applyTheme(next);
  store.set("nlsql-theme", next);
});

/* Tabs, with arrow-key navigation and a shareable #about link */
const tabs = [...document.querySelectorAll(".tab")];
function selectTab(tab, focus = false) {
  for (const other of tabs) {
    const selected = other === tab;
    other.setAttribute("aria-selected", String(selected));
    other.tabIndex = selected ? 0 : -1;
    $(other.getAttribute("aria-controls")).hidden = !selected;
  }
  if (focus) tab.focus();
  const wanted = tab.id === "tab-about" ? "#about" : "";
  if (location.hash !== wanted) history.replaceState(null, "", wanted || location.pathname);
  if (tab.id === "tab-about") loadResults();
}
tabs.forEach((tab, index) => {
  tab.addEventListener("click", () => selectTab(tab));
  tab.addEventListener("keydown", (event) => {
    if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
    const step = event.key === "ArrowRight" ? 1 : -1;
    selectTab(tabs[(index + step + tabs.length) % tabs.length], true);
  });
});
const tabFromHash = () => selectTab($(location.hash === "#about" ? "tab-about" : "tab-playground"));
window.addEventListener("hashchange", tabFromHash);

/* Playground setup */
const form = $("ask-form");
const question = $("question");
const askButton = $("ask-button");
const result = $("result");
const modelSelect = $("model");

question.addEventListener("input", () => { $("char-count").textContent = `${question.value.length} / 500`; });
question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); form.requestSubmit(); }
});

fetch("/api/info").then((response) => response.json()).then((info) => {
  const examples = $("examples");
  for (const text of info.examples || []) {
    const button = el("button", text, "example");
    button.type = "button";
    button.addEventListener("click", () => { question.value = text; question.dispatchEvent(new Event("input")); form.requestSubmit(); });
    const item = el("li"); item.append(button); examples.append(item);
  }
  const saved = store.get("nlsql-model");
  for (const name of info.models || [info.model]) {
    const option = el("option", name); option.value = name; modelSelect.append(option);
  }
  if (saved && (info.models || []).includes(saved)) modelSelect.value = saved;
  if (info.limits) {
    $("model-hint").textContent = `Both run on Groq's free tier, so answers can take up to a minute when it's busy. Each visitor gets ${info.limits.per_minute} questions a minute and ${info.limits.per_day} a day.`;
  }
}).catch(() => {
  $("examples").append(el("li", "Examples couldn't load. Type a question instead.", "hint"));
});
modelSelect.addEventListener("change", () => store.set("nlsql-model", modelSelect.value));

/* Asking */
let waitingTimer = null;
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = question.value.trim();
  if (!text) { question.focus(); return; }
  showLoading();
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: text, model: modelSelect.value || undefined }),
      signal: controller.signal,
    });
    const data = await response.json().catch(() => ({}));
    if (response.ok) showAnswer(text, data);
    else showError(data.error || `The server answered with an error (HTTP ${response.status}).`, response.headers.get("Retry-After"));
  } catch (error) {
    showError(error.name === "AbortError"
      ? "The model took too long to answer. Try again, or pick a shorter question."
      : "Couldn't reach the server. Check your connection and try again.");
  } finally {
    clearTimeout(timeout);
    clearInterval(waitingTimer);
    askButton.disabled = false;
    askButton.textContent = "Ask";
  }
});

function showLoading() {
  askButton.disabled = true;
  askButton.textContent = "Asking…";
  const started = Date.now();
  const line = el("span", WAITING_STEPS[0]);
  const box = el("div", null, "loading");
  box.setAttribute("role", "status");
  box.append(el("span", null, "pulse"), line);
  result.replaceChildren(box);
  clearInterval(waitingTimer);
  waitingTimer = setInterval(() => {
    const seconds = Math.round((Date.now() - started) / 1000);
    const step = WAITING_STEPS[Math.min(WAITING_STEPS.length - 1, Math.floor(seconds / 4))];
    line.textContent = seconds >= 20 ? `${step} (${seconds}s; the free model tier is busy)` : step;
  }, 1000);
}

function showError(message, retryAfter) {
  const block = el("div", null, "answer-block");
  block.append(statusLine("err", "Didn't get an answer"), el("p", message, "answer is-error"));
  if (retryAfter) block.append(el("p", `You can ask again in about ${formatWait(Number(retryAfter))}.`, "hint"));
  result.replaceChildren(block);
}

function showAnswer(text, data) {
  const block = el("div", null, "answer-block");
  if (data.clarification) {
    block.append(statusLine("warn", "Needs more detail"), el("p", data.clarification, "answer"));
    block.append(el("p", "It asked instead of guessing. Add the missing detail to your question and ask again.", "hint"));
  } else if (data.incomplete || !data.answer) {
    block.append(statusLine("warn", "Stopped before answering"), el("p", "It ran out of steps before it was confident. Try rephrasing the question more specifically.", "answer is-error"));
  } else {
    block.append(statusLine("ok", "Answered"));
    const answer = el("p", null, "answer");
    answer.append(el("mark", data.answer));
    block.append(answer);
  }
  const meta = el("p", null, "meta");
  meta.append(el("span", `${data.seconds}s`), el("span", `model ${data.model}`));
  if (data.input_tokens != null) meta.append(el("span", `${(data.input_tokens + (data.output_tokens || 0)).toLocaleString()} tokens`));
  block.append(meta);

  const working = el("div", null, "working");
  working.append(trail(data.steps || []));
  if (data.sql) working.append(sqlBlock(data.sql));
  if (data.sql) working.append(rowsBlock(data.rows || []));
  result.replaceChildren(block, working);
}

function statusLine(kind, label) {
  const line = el("p", null, "status");
  line.append(el("strong", label, kind));
  return line;
}

function trail(steps) {
  const section = el("section");
  section.append(el("h3", "How it got there"));
  const list = el("ol", null, "trail");
  for (const step of steps) {
    const item = el("li");
    item.append(el("span", STEP_LABELS[step.tool] || step.tool));
    item.append(el("span", step.outcome === "error" ? "failed, retried" : "", step.outcome === "error" ? "bad" : "good"));
    list.append(item);
  }
  if (!steps.length) list.append(el("li", "No database steps were needed."));
  section.append(list);
  return section;
}

function sqlBlock(sql) {
  const section = el("section");
  const head = el("div", null, "sql-head");
  head.append(el("h3", "The query the AI wrote"));
  const copy = el("button", "Copy", "ghost");
  copy.type = "button";
  copy.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(sql); copy.textContent = "Copied"; }
    catch { copy.textContent = "Copy failed"; }
    setTimeout(() => { copy.textContent = "Copy"; }, 1600);
  });
  head.append(copy);
  const pre = el("pre", null, "sql");
  for (const part of sql.split(/(\b[A-Za-z_]+\b)/)) {
    if (SQL_KEYWORDS.has(part.toLowerCase())) pre.append(el("span", part, "kw"));
    else pre.append(document.createTextNode(part));
  }
  section.append(head, pre, el("p", "Checked as read-only before it ran. It can't change your data.", "hint"));
  return section;
}

function rowsBlock(rows) {
  const section = el("section");
  section.append(el("h3", rows.length === 1 ? "The row it returned" : `The ${rows.length} rows it returned`));
  section.append(rows.length ? dataTable(Object.keys(rows[0]), rows) : el("p", "The query returned no rows.", "hint"));
  return section;
}

function dataTable(columns, rows) {
  const wrap = el("div", null, "table-wrap");
  const table = el("table");
  // Numbers read best right-aligned, so their headers must sit over them.
  const numeric = new Set(columns.filter((column) => rows.some((row) => typeof row[column] === "number")));
  const headRow = el("tr");
  for (const column of columns) headRow.append(el("th", column, numeric.has(column) ? "num" : ""));
  const head = el("thead"); head.append(headRow);
  const body = el("tbody");
  for (const row of rows) {
    const tr = el("tr");
    for (const column of columns) {
      const value = row[column];
      tr.append(el("td", value === null ? "null" : value, numeric.has(column) ? "num" : ""));
    }
    body.append(tr);
  }
  table.append(head, body);
  wrap.append(table);
  return wrap;
}

function formatWait(seconds) {
  if (!Number.isFinite(seconds) || seconds < 60) return `${Math.max(1, Math.round(seconds || 1))} seconds`;
  const minutes = Math.ceil(seconds / 60);
  return minutes === 1 ? "a minute" : `${minutes} minutes`;
}

/* Data browser */
fetch("/api/data").then((response) => response.json()).then((data) => {
  const box = $("tables");
  if (data.error) { box.append(el("p", data.error, "hint")); return; }
  for (const table of data.tables) {
    const details = el("details", null, "table-detail");
    const summary = el("summary");
    summary.append(el("strong", table.name), el("span", `${table.rows.length} rows`));
    const columns = table.columns.map((column) => column.name ?? column.column_name ?? column);
    details.append(summary, dataTable(columns, table.rows));
    box.append(details);
  }
}).catch(() => $("tables").append(el("p", "The data list couldn't load.", "hint")));

/* About: measured results from the committed evaluation reports */
let resultsLoaded = false;
function loadResults() {
  if (resultsLoaded) return;
  resultsLoaded = true;
  fetch("/api/results").then((response) => response.json()).then(({ reports }) => {
    const box = $("results");
    if (!reports || !reports.length) {
      box.replaceChildren(el("p", "No evaluation has been run yet.", "hint"));
      return;
    }
    box.replaceChildren(...reports.flatMap(reportView));
  }).catch(() => $("results").replaceChildren(el("p", "The results couldn't load.", "hint")));
}

function reportView(report) {
  const s = report.summary;
  const date = new Date(report.run_timestamp_utc).toLocaleDateString(undefined, { year: "numeric", month: "long", day: "numeric" });
  const intro = el("p", `We asked the AI ${s.questions} fixed questions about the demo shop and checked each answer by running its query and comparing the rows with a hand-written correct query. Run on ${date} with ${report.model}.`);
  const score = el("div", null, "score");
  const big = el("p", null, "score-big");
  big.append(document.createTextNode(`${s.passed} of ${s.scored}`), el("small", " correct"));
  score.append(big, el("p", `Typical answer time ${s.median_seconds}s, about ${Number(s.median_tokens).toLocaleString()} tokens per question.`, "hint"));
  const parts = [intro, score];
  if (s.errors) parts.push(el("p", `${s.errors} question(s) hit a model error, such as the free tier's limit, and aren't counted either way.`, "hint"));

  const wrap = el("div", null, "table-wrap");
  const table = el("table", null, "result-table");
  const headRow = el("tr");
  for (const label of ["Question", "Kind of question", "Result", "Time"]) headRow.append(el("th", label));
  const head = el("thead"); head.append(headRow);
  const body = el("tbody");
  for (const row of report.results) {
    const tr = el("tr");
    const verdict = row.status === "passed" ? ["Correct", "pass"] : row.status === "failed" ? ["Wrong", "fail"] : ["Model error", ""];
    tr.append(el("td", row.question), el("td", row.skill), el("td", verdict[0], `status-cell ${verdict[1]}`), el("td", `${row.seconds}s`, "num"));
    body.append(tr);
  }
  table.append(head, body);
  wrap.append(table);
  parts.push(wrap);
  return parts;
}

/* Open the About tab last, once everything it uses (like loadResults) is defined. */
if (location.hash === "#about") tabFromHash();
