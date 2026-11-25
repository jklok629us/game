#!/usr/bin/env python3
"""
建置《猜詞作戰》的前端資料檔。

流程：
  1. 從 jieba 詞典取常用詞（按詞頻排序，按詞性過濾，剔除人名、數詞等）
  2. 跟詞向量模型取交集（預設用 spaCy zh_core_web_lg 的 50 萬個 300 維詞向量）
  3. 簡體轉繁體（OpenCC s2hk），並記錄簡體／台灣繁體寫法作別名
  4. 向量正規化 → （可選）PCA 降維 → 每行獨立量化為 int8
  5. 詞性標注（spaCy 中文 tagger）：n = 一般名詞、p = 專有名詞、o = 其他
     跟節目規則：提示只可用名詞（專有名詞可以）；答案只可用一般名詞
  6. 為每個答案計算難度（易／中／難），前端派題時會平均分配
  7. 輸出到 web/data/：
       vocab.json   詞表 + 別名 + 詞性
       vectors.bin  int8 向量矩陣（row-major，N x dim）
       answers.json [[答案 index, 難度 0易/1中/2難], ...]

前端只需要做 cosine similarity，不需要伺服器或 GPU。

用法：
  python tools/build_data.py --spacy-vocab <zh_core_web_lg 的 vocab 目錄>
  python tools/build_data.py --w2v <word2vec 文字格式檔>   # 例如 Tencent / fastText .vec
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# jieba 詞性：保留名詞、動詞、形容詞、地名、成語等；去掉人名(nr*)、數詞(m)、量詞(q)、虛詞
ALLOWED_POS = {
    "n", "v", "a", "vn", "an", "ns", "nz", "nt", "i", "l", "s", "t", "f",
    "ad", "d", "j", "vd", "ng", "vg", "ag", "b", "z",
}
HAN_RE = re.compile(r"^[一-鿿]+$")


def load_jieba_dict(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 3:
                continue
            word, freq, pos = parts
            if pos not in ALLOWED_POS:
                continue
            if not HAN_RE.match(word) or not (1 <= len(word) <= 4):
                continue
            rows.append((word, int(freq)))
    rows.sort(key=lambda r: -r[1])
    return rows


class SpacyVectors:
    def __init__(self, vocab_dir: str):
        from spacy.vocab import Vocab

        self.vocab = Vocab().from_disk(vocab_dir)
        self.vectors = self.vocab.vectors
        self.dim = self.vectors.shape[1]

    def get(self, word: str):
        key = self.vocab.strings.add(word)
        if key in self.vectors:
            return np.asarray(self.vectors[key], dtype=np.float32)
        return None


class W2VVectors:
    """word2vec 文字格式（首行可選 "N dim"）。只載入需要的詞，節省記憶體。"""

    def __init__(self, path: str, wanted: set[str]):
        self.table: dict[str, np.ndarray] = {}
        opener = open
        if path.endswith(".gz"):
            import gzip

            opener = gzip.open
        with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
            first = f.readline().split()
            if len(first) != 2:  # 沒有 header
                self._add(first, wanted)
            for line in f:
                self._add(line.rstrip().split(" "), wanted)
        self.dim = len(next(iter(self.table.values())))

    def _add(self, parts, wanted):
        if parts and parts[0] in wanted and parts[0] not in self.table:
            self.table[parts[0]] = np.asarray(parts[1:], dtype=np.float32)

    def get(self, word: str):
        return self.table.get(word)


def default_jieba_dict() -> Path:
    import jieba

    return Path(jieba.__file__).parent / "dict.txt"


def read_answers(path: Path) -> list[str]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


# spaCy 中文 tagger（CTB 標籤）→ 我們的三類
NOUN_TAGS = {"NN": "n", "NT": "n", "NR": "p"}
# 沒有 spaCy tagger 時，退而用 jieba 詞性（較不準）
JIEBA_NOUN = {"n": "n", "ng": "n", "vn": "n", "an": "n", "s": "n", "t": "n",
              "ns": "p", "nz": "p", "nt": "p", "nr": "p", "nrt": "p", "j": "p"}


FRAMES = [("", ""), ("这是", "。"), ("一个", ""), ("我喜欢", "。")]


def tag_words(simps: list[str], jieba_pos: dict[str, str]) -> str:
    """回傳每個詞的類別字元：n 一般名詞 / p 專有名詞 / o 其他。

    單詞冇上下文時 tagger 會出錯（例如將「烏龜」當專有名詞、「草莓」當動詞），所以：
      - 將詞放入 4 個句式分別標注，數有幾多次被標成名詞（NN/NT/NR）
      - jieba 詞性係名詞類就加 2 票
      - 總票數 ≥ 2 → 名詞；其中 NR（專有名詞）≥ 3 次 → 專有名詞
    提示用寬鬆標準，避免正常名詞被拒。"""
    try:
        import spacy
        from spacy.tokens import Doc

        nlp = spacy.load("zh_core_web_lg", exclude=["parser", "ner", "attribute_ruler", "senter"])
        procs = [p for _, p in nlp.pipeline]
    except Exception as e:  # noqa: BLE001
        print(f"  [警告] 無法載入 spaCy tagger（{e}），改用 jieba 詞性", file=sys.stderr)
        return "".join(JIEBA_NOUN.get(jieba_pos.get(w, ""), "o") for w in simps)

    noun = [0] * len(simps)
    proper = [0] * len(simps)
    for pre, post in FRAMES:
        docs, at = [], []
        for w in simps:
            toks = [t for t in (pre, w, post) if t]
            docs.append(Doc(nlp.vocab, words=toks))
            at.append(1 if pre else 0)
        for proc in procs:
            docs = list(proc.pipe(docs, batch_size=2000))
        for i, (d, j) in enumerate(zip(docs, at)):
            tag = d[j].tag_
            noun[i] += tag in NOUN_TAGS
            proper[i] += tag == "NR"
    out = []
    for i, w in enumerate(simps):
        jp = JIEBA_NOUN.get(jieba_pos.get(w, ""))
        votes = noun[i] + (2 if jp else 0)
        if votes < 2:
            out.append("o")
        elif proper[i] >= 3 or (jp == "p" and proper[i] >= 2):
            out.append("p")
        else:
            out.append("n")
    return "".join(out)


def answer_difficulty(m: np.ndarray, answers: list[int], words: list[str], pos: str,
                      freq: list[int], k: int = 10) -> list[float]:
    """估計每個答案有幾易估（越大越易）。

    想法：出題者會用最接近答案的名詞做提示；如果那些提示詞「回頭看」時，答案亦排得很前，
    玩家就容易由提示聯想到答案（例如 醫生 ↔ 護士）。另外常用詞亦較易估。
      reciprocity = 平均 1 / log2(答案喺提示詞眼中的排名 + 2)
    """
    out = []
    for a in answers:
        sims = m @ m[a]
        order = np.argsort(-sims)
        hints = [i for i in order[1:400]
                 if pos[i] in "np" and not set(words[i]) & set(words[a])][:k]
        rec = []
        for h in hints:
            sh = m @ m[h]
            r = int((sh > sh[a]).sum())  # 答案喺提示詞眼中排第幾（扣除自己）
            rec.append(1.0 / np.log2(r + 2))
        out.append((float(np.mean(rec)) if rec else 0.0, np.log10(freq[a] + 1)))
    rec = np.array([o[0] for o in out]); lf = np.array([o[1] for o in out])
    z = lambda x: (x - x.mean()) / (x.std() or 1)
    return list(z(rec) + 0.5 * z(lf))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--spacy-vocab", help="spaCy 模型的 vocab 目錄（例如 zh_core_web_lg-3.8.0/vocab）")
    src.add_argument("--w2v", help="word2vec 文字格式向量檔（可 .gz）")
    ap.add_argument("--jieba-dict", type=Path, default=None)
    ap.add_argument("--answers", type=Path, default=ROOT / "data" / "answers.txt")
    ap.add_argument("--extra", type=Path, default=ROOT / "data" / "extra_words.txt",
                    help="額外加入詞庫的詞（例如專有名詞）")
    ap.add_argument("--out", type=Path, default=ROOT / "web" / "data")
    ap.add_argument("--size", type=int, default=30000, help="詞庫大小（不計答案及額外詞）")
    ap.add_argument("--dim", type=int, default=128, help="PCA 降維後的維度；0 = 不降維")
    args = ap.parse_args()

    import opencc

    s2hk = opencc.OpenCC("s2hk")
    s2tw = opencc.OpenCC("s2tw")
    t2s = opencc.OpenCC("t2s")

    jieba_path = args.jieba_dict or default_jieba_dict()
    dict_rows = load_jieba_dict(jieba_path)
    jieba_pos, jieba_freq = {}, {}
    for line in jieba_path.open(encoding="utf-8"):
        parts = line.split()
        if len(parts) == 3 and parts[0] not in jieba_pos:
            jieba_pos[parts[0]], jieba_freq[parts[0]] = parts[2], int(parts[1])
    extra_trad = read_answers(args.extra) if args.extra.exists() else []
    answers_trad = read_answers(args.answers)
    answers_simp = [t2s.convert(w) for w in answers_trad]
    print(f"jieba 候選詞：{len(dict_rows)}；答案候選：{len(answers_trad)}", file=sys.stderr)

    if args.spacy_vocab:
        vecs = SpacyVectors(args.spacy_vocab)
    else:
        wanted = {w for w, _ in dict_rows} | set(answers_simp) | {t2s.convert(w) for w in extra_trad}
        vecs = W2VVectors(args.w2v, wanted)

    words: list[str] = []          # 繁體顯示用
    simps: list[str] = []          # 對應簡體（查向量、標詞性用）
    aliases: dict[str, int] = {}   # 其他寫法 -> index
    mats: list[np.ndarray] = []
    seen: dict[str, int] = {}      # 繁體 -> index

    def add(simp: str, trad: str | None = None) -> int | None:
        trad = trad or s2hk.convert(simp)
        if trad in seen:
            return seen[trad]
        v = vecs.get(simp)
        if v is None or not np.any(v):
            return None
        idx = len(words)
        words.append(trad)
        simps.append(simp)
        seen[trad] = idx
        mats.append(v)
        for alt in {simp, s2tw.convert(simp)}:
            if alt != trad and alt not in aliases:
                aliases[alt] = idx
        return idx

    # 先放答案，確保一定在詞庫中
    answer_idx = []
    for trad, simp in zip(answers_trad, answers_simp):
        idx = add(simp, trad)
        if idx is None:
            print(f"  [略過答案] 找不到向量：{trad}", file=sys.stderr)
        else:
            answer_idx.append(idx)

    n_extra = 0
    for trad in extra_trad:
        before = len(words)
        if add(t2s.convert(trad), trad) is None:
            print(f"  [略過額外詞] 找不到向量：{trad}", file=sys.stderr)
        n_extra += len(words) > before

    limit = args.size + len(words)
    for simp, _freq in dict_rows:
        if len(words) >= limit:
            break
        add(simp)

    pos = list(tag_words(simps, jieba_pos))
    # 答案清單係人手揀的一般名詞：以人手判斷為準，只提示 tagger 有異議的詞
    doubt = [words[i] for i in answer_idx if pos[i] != "n"]
    if doubt:
        print(f"  [提示] tagger 認為以下答案唔係一般名詞（照用，請人手確認）：{'、'.join(doubt)}", file=sys.stderr)
    for i in answer_idx:
        pos[i] = "n"
    pos = "".join(pos)

    # 別名不可跟正式詞衝突
    aliases = {k: v for k, v in aliases.items() if k not in seen}

    m = np.stack(mats).astype(np.float32)
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    if args.dim and args.dim < m.shape[1]:
        mean = m.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(m - mean, full_matrices=False)
        m = (m - mean) @ vt[: args.dim].T
        m /= np.linalg.norm(m, axis=1, keepdims=True)

    # 每行獨立量化：cosine 只看方向，前端會重新計算 norm，所以不需存 scale
    scale = 127.0 / np.abs(m).max(axis=1, keepdims=True)
    q = np.clip(np.round(m * scale), -127, 127).astype(np.int8)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "vectors.bin").write_bytes(q.tobytes())

    # 難度：用量化後的向量計，跟前端一致
    mq = q.astype(np.float32)
    mq /= np.linalg.norm(mq, axis=1, keepdims=True)
    freq = [jieba_freq.get(s_, 0) for s_ in simps]
    ease = answer_difficulty(mq, answer_idx, words, pos, freq)
    cut = np.quantile(ease, [1 / 3, 2 / 3])
    tiers = [0 if e >= cut[1] else 1 if e >= cut[0] else 2 for e in ease]
    for t, name in enumerate("易中難"):
        sample = [words[i] for i, tt, _ in sorted(zip(answer_idx, tiers, ease), key=lambda x: -x[2]) if tt == t]
        print(f"  難度「{name}」{len(sample)} 個：{'、'.join(sample[:12])}…", file=sys.stderr)
    (args.out / "vocab.json").write_text(
        json.dumps({"dim": int(q.shape[1]), "words": words, "aliases": aliases, "pos": pos}, ensure_ascii=False,
                   separators=(",", ":")),
        encoding="utf-8",
    )
    (args.out / "answers.json").write_text(json.dumps([[i, t] for i, t in zip(answer_idx, tiers)]),
                                           encoding="utf-8")

    mb = os.path.getsize(args.out / "vectors.bin") / 1e6
    print(f"詞性：一般名詞 {pos.count('n')}、專有名詞 {pos.count('p')}、其他 {pos.count('o')}；額外詞 {n_extra}",
          file=sys.stderr)
    print(f"完成：詞庫 {len(words)} 詞、別名 {len(aliases)}、答案 {len(answer_idx)}、"
          f"維度 {q.shape[1]}、vectors.bin {mb:.1f} MB", file=sys.stderr)


if __name__ == "__main__":
    main()
