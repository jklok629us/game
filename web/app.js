"use strict";

/* =========================================================
 * 猜詞作戰 — 前端
 *  - 載入 int8 詞向量，在瀏覽器計算 cosine similarity
 *  - 遊戲模式：每日一題 / 隨機練習 / 雙人搶答 / 出題對決
 * ========================================================= */

const TOP_N = 1000;          // 顯示排名的範圍
const DUEL_HINT_PENALTY = 3; // 出題對決：每次提示加幾次猜測
const DAY0 = Date.UTC(2026, 0, 1);

// ---------------- 語意引擎 ----------------
const Engine = {
  words: [],
  index: new Map(),
  aliases: {},
  dim: 0,
  vecs: null,     // Float32Array，已正規化
  answers: [],

  async load(onProgress) {
    const [vocab, answers] = await Promise.all([
      fetch("data/vocab.json").then((r) => r.json()),
      fetch("data/answers.json").then((r) => r.json()),
    ]);
    const raw = await fetchWithProgress("data/vectors.bin", onProgress);
    this.words = vocab.words;
    this.aliases = vocab.aliases;
    this.dim = vocab.dim;
    this.answers = answers;
    this.words.forEach((w, i) => this.index.set(w, i));

    const q = new Int8Array(raw);
    const n = this.words.length, d = this.dim;
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

  lookup(raw) {
    const w = raw.replace(/\s+/g, "");
    if (!w) return -1;
    if (this.index.has(w)) return this.index.get(w);
    if (w in this.aliases) return this.aliases[w];
    return -1;
  },

  /** 為一個答案預先計算所有詞的相似度及排名 */
  puzzle(target) {
    const n = this.words.length, d = this.dim, v = this.vecs;
    const sims = new Float32Array(n);
    const t = target * d;
    for (let i = 0; i < n; i++) {
      let s = 0;
      const o = i * d;
      for (let k = 0; k < d; k++) s += v[o + k] * v[t + k];
      sims[i] = s;
    }
    sims[target] = 1;
    const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => sims[b] - sims[a]);
    const rank = new Int32Array(n);
    order.forEach((idx, r) => (rank[idx] = r)); // 答案 = 0，最接近 = 1
    return { target, sims, order, rank };
  },
};

async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body || !total) return res.arrayBuffer();
  // 注意：伺服器可能用 gzip 傳送（例如 GitHub Pages），content-length 是壓縮後大小，
  // 而讀到的是解壓後資料，所以不能用 content-length 預先分配緩衝區。
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
const show = (id) => {
  for (const s of document.querySelectorAll("main > section")) s.classList.toggle("hidden", s.id !== id);
};
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}
function mulberry32(a) {
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
/** 固定種子洗牌，令每日題目順序不按分類排列 */
function dailyOrder() {
  const a = Engine.answers.slice();
  const rnd = mulberry32(20260101);
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}
function todayNumber() {
  const now = new Date();
  return Math.floor((Date.UTC(now.getFullYear(), now.getMonth(), now.getDate()) - DAY0) / 86400000);
}
function heat(rank) {
  if (rank === 0) return "win";
  if (rank <= 10) return "hot";
  if (rank <= 100) return "warm";
  if (rank <= TOP_N) return "near";
  return "cold";
}
const store = {
  get(k) { try { return JSON.parse(localStorage.getItem(k)); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* 無痕模式等 */ } },
};

// ---------------- 遊戲狀態 ----------------
let game = null;   // 目前這一局
let duel = null;   // 出題對決的整體進度

function newGame({ mode, target, title, sub, players, saveKey }) {
  game = {
    mode, title, sub, saveKey,
    puzzle: Engine.puzzle(target),
    players: players.map((name) => ({ name, guesses: 0, hints: 0 })),
    current: 0,
    list: [],            // {idx, player, n, hint}
    guessed: new Set(),
    over: false,
    winner: -1,
  };
  if (saveKey) {
    const saved = store.get(saveKey);
    if (saved && saved.target === target) {
      for (const g of saved.list) addGuess(g.idx, g.player, g.hint, true);
      if (saved.over) game.over = true;
    }
  }
  renderGame();
  show("game");
  if (game.over) finish(false);
  else $("guess-input").focus();
}

function save() {
  if (!game.saveKey) return;
  store.set(game.saveKey, {
    target: game.puzzle.target,
    list: game.list.map(({ idx, player, hint }) => ({ idx, player, hint })),
    over: game.over,
  });
}

function addGuess(idx, player, hint = false, silent = false) {
  const p = game.players[player];
  if (hint) p.hints++;
  else p.guesses++;
  const entry = { idx, player, hint, n: game.list.length + 1 };
  game.list.push(entry);
  game.guessed.add(idx);
  if (!hint && idx === game.puzzle.target) {
    game.over = true;
    game.winner = player;
  }
  if (!silent) {
    // 雙人搶答：猜完（或用提示）換人
    if (game.mode === "versus" && !game.over) game.current = 1 - game.current;
    save();
  }
  return entry;
}

function submitGuess(raw) {
  if (!game || game.over) return;
  const msg = $("msg");
  const idx = Engine.lookup(raw);
  if (idx < 0) {
    msg.textContent = `「${raw.trim()}」不在詞庫內，試另一個詞。`;
    return;
  }
  if (game.guessed.has(idx)) {
    msg.textContent = `「${Engine.words[idx]}」已經猜過了。`;
    return;
  }
  msg.textContent = "";
  const entry = addGuess(idx, game.current);
  renderGame(entry);
  if (game.over) setTimeout(() => finish(true), 600);
}

function useHint() {
  if (!game || game.over) return;
  const pz = game.puzzle;
  let best = Infinity;
  for (const g of game.list) best = Math.min(best, pz.rank[g.idx]);
  let r = best <= TOP_N ? Math.floor(best / 2) : 500;
  while (r >= 1 && game.guessed.has(pz.order[r])) r--;
  if (r < 1) {
    $("msg").textContent = "已經沒有更接近的提示了！";
    return;
  }
  const entry = addGuess(pz.order[r], game.current, true);
  $("msg").textContent = `提示：「${Engine.words[entry.idx]}」排第 ${r} 名。`;
  renderGame(entry);
}

function giveUp() {
  if (!game || game.over) return;
  if (!confirm("確定放棄？會直接公佈答案。")) return;
  game.over = true;
  game.winner = -1;
  save();
  finish(true);
}

// ---------------- 畫面 ----------------
function guessRow(g) {
  const pz = game.puzzle;
  const rank = pz.rank[g.idx];
  const tr = el("tr", heat(rank));
  if (g.hint) tr.classList.add("hint");
  tr.append(el("td", "n", g.hint ? "💡" : String(g.n)));
  const wordTd = el("td", "w");
  if (game.players.length > 1 && game.mode === "versus") wordTd.append(el("span", `dot p${g.player}`));
  wordTd.append(document.createTextNode(Engine.words[g.idx]));
  tr.append(wordTd);
  tr.append(el("td", "s", (pz.sims[g.idx] * 100).toFixed(2)));
  const rt = el("td", "r");
  if (rank === 0) rt.textContent = "🎯 答案";
  else if (rank <= TOP_N) {
    const bar = el("div", "meter");
    const fill = el("div");
    fill.style.width = `${((TOP_N - rank + 1) / TOP_N) * 100}%`;
    bar.append(fill);
    rt.append(el("span", null, `第 ${rank} 名`), bar);
  } else rt.textContent = "冷";
  tr.append(rt);
  return tr;
}

function renderGame(latest) {
  $("game-title").textContent = game.title;
  $("game-sub").textContent = game.sub || "";

  const turn = $("turn");
  if (game.mode === "versus") {
    turn.classList.remove("hidden");
    turn.className = `turn p${game.current}`;
    turn.textContent = game.over ? "" : `輪到 ${game.players[game.current].name}`;
  } else turn.classList.add("hidden");

  const last = $("last");
  if (latest) {
    last.classList.remove("hidden");
    last.replaceChildren(el("table", "guesses"));
    last.firstChild.append(guessRow(latest));
  } else last.classList.add("hidden");

  const body = $("guess-list");
  const sorted = game.list.slice().sort((a, b) => game.puzzle.sims[b.idx] - game.puzzle.sims[a.idx]);
  body.replaceChildren(...sorted.map(guessRow));

  const p = game.players[game.current];
  $("stats").textContent = game.mode === "versus"
    ? game.players.map((x) => `${x.name}：${x.guesses} 次`).join(" · ")
    : `已猜 ${p.guesses} 次${p.hints ? `，提示 ${p.hints} 次` : ""}`;

  const canHint = !(game.mode === "versus" && game.players[game.current].hints >= 1);
  $("btn-hint").disabled = game.over || !canHint;
  $("btn-hint").textContent = game.mode === "versus" ? "提示（每人一次，會用掉回合）"
    : game.mode === "duel" ? `提示（+${DUEL_HINT_PENALTY} 次）` : "提示";
  $("guess-input").disabled = game.over;
  $("guess-input").value = "";
}

function finish() {
  const pz = game.puzzle;
  const answer = Engine.words[pz.target];
  const p = game.players[0];
  let title, body;
  if (game.mode === "versus") {
    title = game.winner >= 0 ? `🏆 ${game.players[game.winner].name} 勝出！` : "無人猜中";
    body = `答案是「${answer}」。共猜了 ${game.list.filter((g) => !g.hint).length} 次。`;
  } else if (game.mode === "duel") {
    const score = game.winner >= 0 ? p.guesses + p.hints * DUEL_HINT_PENALTY : null;
    duel.scores[duel.round] = score;
    title = score != null ? `${p.name} 猜中了！` : `${p.name} 放棄了`;
    body = `答案是「${answer}」。` + (score != null ? `成績：${score} 次（含提示罰分）。` : "此回合當作落敗。");
    if (duel.round === 1) {
      const [a, b] = duel.scores; // a = 玩家 2 解題成績, b = 玩家 1 解題成績
      const verdict = a == null && b == null ? "雙方都放棄，平手！"
        : a == null ? "🏆 玩家 1 勝出！" : b == null ? "🏆 玩家 2 勝出！"
        : a === b ? "平手！" : a < b ? "🏆 玩家 2 勝出！" : "🏆 玩家 1 勝出！";
      body += `\n\n最終：玩家 2 用了 ${a ?? "—"} 次，玩家 1 用了 ${b ?? "—"} 次。${verdict}`;
    }
  } else {
    title = game.winner >= 0 ? "🎉 猜中了！" : "下次再來";
    body = `答案是「${answer}」。` + (game.winner >= 0 ? `你用了 ${p.guesses} 次猜中${p.hints ? `（提示 ${p.hints} 次）` : ""}。` : "");
  }
  $("result-title").textContent = title;
  $("result-body").textContent = body;
  $("nearest").replaceChildren(
    ...pz.order.slice(1, 21).map((i) => el("li", null, `${Engine.words[i]}（${(pz.sims[i] * 100).toFixed(1)}）`)),
  );
  $("btn-next").textContent = game.mode === "duel" && duel.round === 0 ? "下一回合：交換出題" : "再玩一局";
  $("btn-next").classList.toggle("hidden", game.mode === "daily");
  show("result");
}

// ---------------- 模式 ----------------
function randomAnswer() {
  return Engine.answers[Math.floor(Math.random() * Engine.answers.length)];
}

const Modes = {
  daily() {
    const day = todayNumber();
    const order = dailyOrder();
    const target = order[((day % order.length) + order.length) % order.length];
    newGame({
      mode: "daily", target, players: ["你"],
      title: `每日一題 #${day + 1}`, sub: "今日所有人的答案都一樣", saveKey: `daily-${day}`,
    });
  },
  random() {
    newGame({ mode: "random", target: randomAnswer(), players: ["你"], title: "隨機練習", sub: "" });
  },
  versus() {
    newGame({
      mode: "versus", target: randomAnswer(), players: ["玩家 1", "玩家 2"],
      title: "雙人搶答", sub: "輪流猜同一題，先猜中者勝。小心——你的好線索，對手也看得到。",
    });
  },
  duel() {
    duel = { round: 0, scores: [null, null] };
    startSetter();
  },
};

function startSetter() {
  const setter = duel.round === 0 ? 1 : 2;
  $("setter-title").textContent = `玩家 ${setter} 出題`;
  $("setter-msg").textContent = "";
  $("setter-input").value = "";
  show("setter");
  $("setter-input").focus();
}

$("setter-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const idx = Engine.lookup($("setter-input").value);
  if (idx < 0) {
    $("setter-msg").textContent = "這個詞不在詞庫內，換一個吧。";
    return;
  }
  const solver = duel.round === 0 ? 2 : 1;
  newGame({
    mode: "duel", target: idx, players: [`玩家 ${solver}`],
    title: `出題對決 · 第 ${duel.round + 1} 回合`, sub: `玩家 ${solver} 解題（提示每次 +${DUEL_HINT_PENALTY} 次）`,
  });
});

// ---------------- 事件 ----------------
for (const b of document.querySelectorAll("[data-mode]")) {
  b.addEventListener("click", () => Modes[b.dataset.mode]());
}
$("guess-form").addEventListener("submit", (e) => {
  e.preventDefault();
  submitGuess($("guess-input").value);
  $("guess-input").focus();
});
$("btn-hint").addEventListener("click", useHint);
$("btn-giveup").addEventListener("click", giveUp);
$("btn-menu").addEventListener("click", () => show("menu"));
$("btn-result-menu").addEventListener("click", () => show("menu"));
$("btn-next").addEventListener("click", () => {
  if (game.mode === "duel") {
    if (duel.round === 0) { duel.round = 1; startSetter(); } else Modes.duel();
  } else Modes[game.mode]();
});

Engine.load((p) => ($("load-bar").style.width = `${Math.round(p * 100)}%`))
  .then(() => show("menu"))
  .catch((err) => {
    $("loading").replaceChildren(el("p", "msg", `載入失敗：${err.message}。請重新整理頁面再試。`));
  });
