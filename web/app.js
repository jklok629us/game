"use strict";

/* =========================================================
 * 猜詞作戰 — 前端
 *  - 載入 int8 詞向量，在瀏覽器計算語意相似度
 *  - 相關度：按「在詞庫中排第幾接近」換算成 0–100%
 *  - 模式：測試模式 / 隱藏詞對戰（參考《魔鬼的詭計：死亡重啟》第 5 日主賽「히든 워드」）
 * ========================================================= */

const STOP_PCT = 95;            // 提示達到此相關度即停止提示
const MAX_HINTS = 10;           // 每階段最多提示數；用完仍未到 95% → AI 公開最相似的詞
const MAX_PHASES = 3;           // 最多幾個提示／作答階段
const GUESSER_POINT = 1;        // 估中者得分（有人估中即結束該出題者的回合）
const SETTER_POINTS = [1, 2, 3];// 第 n 階段無人估中 → 出題者得分
const ENVELOPE = [0, 0, 1, 1, 2];// 每人派 5 個答案：2 易、2 中、1 難（人人一樣）
const TIER_NAME = ["易", "中", "難"];

// ---------------- 語意引擎 ----------------
const Engine = {
  words: [],
  index: new Map(),
  aliases: {},
  dim: 0,
  vecs: null,
  answers: [],     // 答案 index
  tiers: new Map(),// 答案 index -> 難度 0/1/2
  pos: "",         // 每個詞的詞性：n 一般名詞 / p 專有名詞 / o 其他

  async load(onProgress) {
    const [vocab, answers] = await Promise.all([
      fetch("data/vocab.json").then((r) => r.json()),
      fetch("data/answers.json").then((r) => r.json()),
    ]);
    const raw = await fetchWithProgress("data/vectors.bin", onProgress);
    this.words = vocab.words;
    this.aliases = vocab.aliases;
    this.dim = vocab.dim;
    this.answers = answers.map((a) => a[0]);
    this.tiers = new Map(answers);
    this.pos = vocab.pos || "";
    this.words.forEach((w, i) => this.index.set(w, i));

    const q = new Int8Array(raw);
    const n = this.words.length, d = this.dim;
    if (q.length !== n * d) throw new Error("詞向量檔案大小不符");
    const v = new Float32Array(n * d);
    for (let i = 0; i < n; i++) {
      let s = 0;
      const o = i * d;
      for (let k = 0; k < d; k++) s += q[o + k] * q[o + k];
      const inv = 1 / Math.sqrt(s || 1);
      for (let k = 0; k < d; k++) v[o + k] = q[o + k] * inv;
    }
    this.vecs = v;
  },

  isNoun(i) {
    return !this.pos || this.pos[i] !== "o";
  },

  lookup(raw) {
    const w = raw.replace(/\s+/g, "");
    if (!w) return -1;
    if (this.index.has(w)) return this.index.get(w);
    if (w in this.aliases) return this.aliases[w];
    return -1;
  },

  /** 為一個題目預先計算所有詞的排名 */
  topic(target) {
    const n = this.words.length, d = this.dim, v = this.vecs;
    const sims = new Float32Array(n);
    const t = target * d;
    for (let i = 0; i < n; i++) {
      let s = 0;
      const o = i * d;
      for (let k = 0; k < d; k++) s += v[o + k] * v[t + k];
      sims[i] = s;
    }
    sims[target] = 2;
    const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => sims[b] - sims[a]);
    const rank = new Int32Array(n);
    order.forEach((idx, r) => (rank[idx] = r)); // 題目本身 = 0，最接近 = 1
    return { target, order, rank, pct: (i) => pctFromRank(rank[i]) };
  },
};

/**
 * 排名 → 相關度百分比。
 * 用 log10(排名) 做分段線性插值：
 *   第 1 名 99%、第 10 名 96%、第 100 名 90%、第 1000 名 78%、
 *   第 5000 名 55%、第 15000 名 25%、第 30000 名 0%。
 * 約最接近的 13 個詞會 ≥ 95%（節目中例如「樹／樹根」96.5%）。
 */
const PCT_ANCHORS = [[0, 99], [1, 96], [2, 90], [3, 78], [3.7, 55], [4.18, 25], [4.48, 0]];
function pctFromRank(r) {
  if (r === 0) return 100;
  const x = Math.log10(r);
  for (let i = 1; i < PCT_ANCHORS.length; i++) {
    const [x0, y0] = PCT_ANCHORS[i - 1], [x1, y1] = PCT_ANCHORS[i];
    if (x <= x1) return Math.max(0, y0 + ((x - x0) / (x1 - x0)) * (y1 - y0));
  }
  return 0;
}
const fmtPct = (p) => (p >= 100 ? "100%" : `${(Math.floor(p * 10) / 10).toFixed(1)}%`);

async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body || !total) return res.arrayBuffer();
  // 伺服器可能用 gzip 傳送（例如 GitHub Pages），content-length 是壓縮後大小，
  // 所以不能用它預先分配緩衝區。
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onProgress(Math.min(got / total, 1));
  }
  const buf = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.length; }
  return buf.buffer;
}

// ---------------- 工具 ----------------
const $ = (id) => document.getElementById(id);
function show(id) {
  for (const s of document.querySelectorAll("main > section")) s.classList.toggle("hidden", s.id !== id);
  const input = document.querySelector(`#${id} input:not([type=checkbox])`);
  if (input) setTimeout(() => input.focus(), 50);
  window.scrollTo(0, 0);
}
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}
function heat(p) {
  if (p >= 100) return "win";
  if (p >= STOP_PCT) return "hot";
  if (p >= 90) return "warm";
  if (p >= 78) return "near";
  return "cold";
}
/** 一行「詞 + 相關度條」 */
function scoreItem(word, p, extra) {
  const li = el("li", heat(p));
  const top = el("div", "line");
  top.append(el("span", "w", word));
  if (extra) top.append(el("span", "tag", extra));
  top.append(el("span", "p", fmtPct(p)));
  const bar = el("div", "meter");
  const fill = el("div");
  fill.style.width = `${p}%`;
  bar.append(fill);
  li.append(top, bar);
  return li;
}
const randomAnswer = () => Engine.answers[Math.floor(Math.random() * Engine.answers.length)];
function shuffle(a) {
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}
function bindReveal(checkbox, input) {
  checkbox.addEventListener("change", () => (input.type = checkbox.checked ? "text" : "password"));
}

// =========================================================
// 測試模式
// =========================================================
const SB = { topic: null, tested: [] };

$("sb-random").addEventListener("click", () => {
  $("sb-topic").value = Engine.words[randomAnswer()];
});
$("sb-hide").addEventListener("change", () => ($("sb-topic").type = $("sb-hide").checked ? "password" : "text"));

$("sb-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const idx = Engine.lookup($("sb-topic").value);
  if (idx < 0) {
    $("sb-setup-msg").textContent = "呢個詞唔喺詞庫入面，試另一個（或者撳「隨機」）。";
    return;
  }
  $("sb-setup-msg").textContent = "";
  SB.topic = Engine.topic(idx);
  SB.tested = [];
  SB.hidden = $("sb-hide").checked;
  renderSandbox();
  show("sb-play");
});

function renderSandbox(latest) {
  const word = Engine.words[SB.topic.target];
  $("sb-topic-label").textContent = SB.hidden ? "●●●（已隱藏）" : word;
  $("sb-reveal").classList.toggle("hidden", !SB.hidden);
  $("sb-last").replaceChildren();
  if (latest != null) {
    const ul = el("ul", "scores latest");
    ul.append(scoreItem(Engine.words[latest], SB.topic.pct(latest), "最新"));
    $("sb-last").append(ul);
  }
  const sorted = SB.tested.slice().sort((a, b) => SB.topic.rank[a] - SB.topic.rank[b]);
  $("sb-list").replaceChildren(...sorted.map((i) => scoreItem(Engine.words[i], SB.topic.pct(i))));
  $("sb-near").classList.add("hidden");
  $("sb-word").value = "";
}

$("sb-test").addEventListener("submit", (e) => {
  e.preventDefault();
  const raw = $("sb-word").value;
  const idx = Engine.lookup(raw);
  if (idx < 0) {
    $("sb-msg").textContent = raw.trim() ? `「${raw.trim()}」唔喺詞庫入面。` : "";
    return;
  }
  $("sb-msg").textContent = idx === SB.topic.target ? "🎯 100%！就係題目本身。" : "";
  if (!SB.tested.includes(idx)) SB.tested.push(idx);
  if (idx === SB.topic.target) SB.hidden = false;
  renderSandbox(idx);
  $("sb-word").focus();
});
$("sb-reveal").addEventListener("click", () => {
  SB.hidden = false;
  renderSandbox();
});
$("sb-clear").addEventListener("click", () => {
  SB.tested = [];
  renderSandbox();
});
$("sb-nearest").addEventListener("click", () => {
  const ol = $("sb-near");
  ol.replaceChildren(
    ...SB.topic.order.slice(1, 11).map((i) => el("li", null, `${Engine.words[i]}　${fmtPct(SB.topic.pct(i))}`)),
  );
  ol.classList.toggle("hidden");
});

// =========================================================
// 隱藏詞對戰（跟《魔鬼的詭計：死亡重啟》第 5 日主賽「히든 워드」）
// =========================================================
let HW = null;

function renderNameInputs() {
  const n = Number($("hw-count").value);
  const box = $("hw-names");
  const old = [...box.querySelectorAll("input")].map((i) => i.value);
  box.replaceChildren();
  for (let i = 0; i < n; i++) {
    const input = el("input");
    input.placeholder = `玩家 ${i + 1}`;
    input.value = old[i] || "";
    input.maxLength = 10;
    box.append(input);
  }
}
$("hw-count").addEventListener("change", renderNameInputs);

/** 派信封：每人 5 個一般名詞，難度組合人人一樣（2 易、2 中、1 難），全場不重複 */
function dealEnvelopes(n) {
  const byTier = [[], [], []];
  for (const [i, t] of Engine.tiers) byTier[t].push(i);
  byTier.forEach(shuffle);
  return Array.from({ length: n }, () => ENVELOPE.map((t) => byTier[t].pop()));
}

$("hw-start").addEventListener("click", () => {
  const names = [...$("hw-names").querySelectorAll("input")].map((i, k) => i.value.trim() || `玩家 ${k + 1}`);
  const rounds = Number($("hw-rounds").value);
  const turns = [];
  for (let r = 0; r < rounds; r++) {
    for (const s of shuffle(names.map((_, i) => i))) turns.push({ round: r, setter: s });
  }
  HW = {
    names, rounds, turns, turn: -1,
    scores: names.map(() => 0),
    envelopes: dealEnvelopes(names.length),
    used: names.map(() => new Set()),
  };
  nextTurn();
});

function nextTurn() {
  HW.turn++;
  if (HW.turn >= HW.turns.length) return finalScreen();
  const { round, setter } = HW.turns[HW.turn];
  HW.cur = { setter, topic: null, phase: 0, hints: [], phaseHints: 0, stopped: false, guessQueue: [], guesses: new Map() };
  HW.pick = -1;
  $("hw-secret-round").textContent = `第 ${round + 1} / ${HW.rounds} 回合 · 第 ${HW.turn + 1} / ${HW.turns.length} 題`;
  $("hw-secret-title").textContent = `${HW.names[setter]} 出題`;
  renderEnvelope();
  show("hw-secret");
}

function renderEnvelope() {
  const s = HW.cur.setter;
  const box = $("hw-envelope");
  box.replaceChildren(...HW.envelopes[s].map((idx) => {
    const used = HW.used[s].has(idx);
    const b = el("button", `card t${Engine.tiers.get(idx)}${HW.pick === idx ? " picked" : ""}`);
    b.type = "button";
    b.disabled = used;
    b.append(el("b", null, Engine.words[idx]), el("small", null, used ? "已用" : TIER_NAME[Engine.tiers.get(idx)]));
    b.addEventListener("click", () => { HW.pick = idx; renderEnvelope(); });
    return b;
  }));
  $("hw-secret-ok").disabled = HW.pick < 0;
}

$("hw-secret-ok").addEventListener("click", () => {
  if (HW.pick < 0) return;
  HW.used[HW.cur.setter].add(HW.pick);
  HW.cur.topic = Engine.topic(HW.pick);
  startHintPhase();
});

function startHintPhase() {
  const c = HW.cur;
  c.phase++;
  c.phaseHints = 0;
  c.stopped = false;
  $("hw-hint-msg").textContent = "";
  renderHint();
  show("hw-hint");
}

function renderHint() {
  const c = HW.cur;
  $("hw-hint-title").textContent = `第 ${c.phase} / ${MAX_PHASES} 階段 · 提示`;
  $("hw-hint-sub").textContent =
    `出題者：${HW.names[c.setter]}　· 今階段無人估中，出題者得 ${SETTER_POINTS[c.phase - 1]} 分`;
  $("hw-hint-count").textContent = `${c.phaseHints} / ${MAX_HINTS}`;
  $("hw-hints").replaceChildren(
    ...c.hints.map((h) => scoreItem(Engine.words[h.idx], c.topic.pct(h.idx), h.ai ? "🤖 AI 公開" : `階段 ${h.phase}`)),
  );
  $("hw-hint-input").disabled = c.stopped;
  $("hw-hint-form").querySelector("button").disabled = c.stopped;
  $("hw-to-guess").classList.toggle("hidden", !c.stopped);
  renderBoard($("hw-board-hint"));
}

const sharesChar = (a, b) => [...b].some((ch) => a.includes(ch));

/** 10 個提示都未到 95%：AI 公開一個同隱藏詞最相似、未出現過的名詞 */
function aiReveal() {
  const c = HW.cur;
  const answer = Engine.words[c.topic.target];
  const shown = new Set(c.hints.map((h) => h.idx));
  for (const i of c.topic.order.slice(1)) {
    if (!shown.has(i) && Engine.isNoun(i) && !sharesChar(Engine.words[i], answer)) return i;
  }
  return -1;
}

$("hw-hint-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const c = HW.cur;
  if (c.stopped) return;
  const raw = $("hw-hint-input").value;
  const idx = Engine.lookup(raw);
  const msg = $("hw-hint-msg");
  if (idx < 0) {
    msg.textContent = raw.trim() ? `「${raw.trim()}」唔喺詞庫入面。` : "";
    return;
  }
  const word = Engine.words[idx], answer = Engine.words[c.topic.target];
  if (!Engine.isNoun(idx)) {
    msg.textContent = `「${word}」唔係名詞。提示只可以用名詞（專有名詞都得）。`;
    return;
  }
  if (idx === c.topic.target || sharesChar(word, answer)) {
    msg.textContent = "提示唔可以包含隱藏詞入面的字。";
    return;
  }
  if (c.hints.some((h) => h.idx === idx)) {
    msg.textContent = `「${word}」已經出現過。`;
    return;
  }
  c.hints.push({ idx, phase: c.phase });
  c.phaseHints++;
  $("hw-hint-input").value = "";
  const p = c.topic.pct(idx);
  if (p >= STOP_PCT) {
    c.stopped = true;
    msg.textContent = `「${word}」相關度 ${fmtPct(p)}，超過 ${STOP_PCT}%！提示即時停止，進入作答。`;
  } else if (c.phaseHints >= MAX_HINTS) {
    c.stopped = true;
    const ai = aiReveal();
    if (ai >= 0) c.hints.push({ idx: ai, phase: c.phase, ai: true });
    msg.textContent = `${MAX_HINTS} 個提示都未超過 ${STOP_PCT}%。` +
      (ai >= 0 ? `AI 公開同隱藏詞最相似的詞：「${Engine.words[ai]}」（${fmtPct(c.topic.pct(ai))}）。` : "");
  } else msg.textContent = "";
  renderHint();
});

$("hw-to-guess").addEventListener("click", startGuessPhase);

function startGuessPhase() {
  const c = HW.cur;
  c.guessQueue = HW.names.map((_, i) => i).filter((i) => i !== c.setter);
  c.guesses = new Map();
  nextGuesser();
}

function nextGuesser() {
  const c = HW.cur;
  if (!c.guessQueue.length) return revealGuesses();
  const p = c.guessQueue[0];
  $("hw-guess-title").textContent = `輪到 ${HW.names[p]} 作答`;
  const best = c.hints.slice().sort((a, b) => c.topic.rank[a.idx] - c.topic.rank[b.idx]);
  $("hw-guess-hints").replaceChildren(
    ...best.map((h) => scoreItem(Engine.words[h.idx], c.topic.pct(h.idx), h.ai ? "🤖" : null)),
  );
  $("hw-guess-input").value = "";
  $("hw-guess-input").type = "password";
  $("hw-guess-show").checked = false;
  $("hw-guess-msg").textContent = "";
  show("hw-guess");
}

bindReveal($("hw-guess-show"), $("hw-guess-input"));
$("hw-guess-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const c = HW.cur;
  const raw = $("hw-guess-input").value.trim();
  if (!raw) return;
  const idx = Engine.lookup(raw);
  if (idx < 0) {
    $("hw-guess-msg").textContent = "呢個詞唔喺詞庫入面，換一個寫法或者另一個詞。";
    return;
  }
  c.guesses.set(c.guessQueue.shift(), idx);
  nextGuesser();
});

function revealGuesses() {
  const c = HW.cur;
  const winners = [];
  const rows = [];
  for (const [player, idx] of c.guesses) {
    const correct = idx === c.topic.target;
    if (correct) {
      HW.scores[player] += GUESSER_POINT;
      winners.push(HW.names[player]);
    }
    const tr = el("tr", heat(c.topic.pct(idx)));
    tr.append(el("td", null, HW.names[player]), el("td", "w", Engine.words[idx] + (correct ? " 🎯" : "")),
      el("td", "p", fmtPct(c.topic.pct(idx))));
    rows.push(tr);
  }
  $("hw-reveal-list").replaceChildren(...rows);
  $("hw-reveal-title").textContent = `第 ${c.phase} 階段 · 公開答案`;

  let msg;
  if (winners.length) {
    msg = `${winners.join("、")} 估中！各得 ${GUESSER_POINT} 分。${HW.names[c.setter]} 的出題結束。`;
  } else {
    const pts = SETTER_POINTS[c.phase - 1];
    HW.scores[c.setter] += pts;
    msg = `無人估中！出題者 ${HW.names[c.setter]} 得 ${pts} 分。`;
  }
  const over = winners.length > 0 || c.phase >= MAX_PHASES;
  if (over) msg += `\n隱藏詞係「${Engine.words[c.topic.target]}」（難度：${TIER_NAME[Engine.tiers.get(c.topic.target)]}）。`;
  $("hw-reveal-msg").textContent = msg;

  const near = $("hw-reveal-near");
  near.classList.toggle("hidden", !over);
  if (over) {
    near.replaceChildren(
      ...c.topic.order.slice(1, 11).map((i) => el("li", null, `${Engine.words[i]}　${fmtPct(c.topic.pct(i))}`)),
    );
  }
  const btn = $("hw-reveal-next");
  btn.textContent = over ? (HW.turn + 1 >= HW.turns.length ? "睇最終結果" : "下一位出題") : `進入第 ${c.phase + 1} 階段提示`;
  btn.onclick = over ? nextTurn : startHintPhase;
  renderBoard($("hw-board-reveal"));
  show("hw-reveal");
}

function renderBoard(box) {
  const order = HW.names.map((_, i) => i).sort((a, b) => HW.scores[b] - HW.scores[a]);
  const c = HW.cur;
  box.replaceChildren(el("h3", null, "分數"), ...order.map((i) => {
    const row = el("div", "brow");
    row.append(el("span", null, HW.names[i]), el("span", "tag", c && i === c.setter ? "出題" : ""),
      el("b", null, String(HW.scores[i])));
    return row;
  }));
}

function finalScreen() {
  HW.cur = null;
  const max = Math.max(...HW.scores);
  const winners = HW.names.filter((_, i) => HW.scores[i] === max);
  $("hw-final-title").textContent = winners.length > 1 ? `平手：${winners.join("、")}` : `🏆 ${winners[0]} 勝出！`;
  renderBoard($("hw-board-final"));
  show("hw-final");
}

// ---------------- 導航 ----------------
for (const b of document.querySelectorAll("[data-go]")) {
  b.addEventListener("click", () => {
    if (b.dataset.go === "hw-setup") renderNameInputs();
    show(b.dataset.go);
  });
}

Engine.load((p) => ($("load-bar").style.width = `${Math.round(p * 100)}%`))
  .then(() => show("menu"))
  .catch((err) => {
    $("loading").replaceChildren(el("p", "msg", `載入失敗：${err.message}。請重新整理頁面再試。`));
  });
