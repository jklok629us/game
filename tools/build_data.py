#!/usr/bin/env python3
"""
建置《猜詞作戰》的前端資料檔。

流程：
  1. 從 jieba 詞典取常用詞（按詞頻排序，按詞性過濾，剔除人名、數詞等）
  2. 跟詞向量模型取交集（預設用 spaCy zh_core_web_lg 的 50 萬個 300 維詞向量）
  3. 簡體轉繁體（OpenCC s2hk），並記錄簡體／台灣繁體寫法作別名
  4. 向量正規化 → （可選）PCA 降維 → 每行獨立量化為 int8
  5. 輸出到 web/data/：
       vocab.json   詞表 + 別名
       vectors.bin  int8 向量矩陣（row-major，N x dim）
       answers.json 答案在詞表中的 index

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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--spacy-vocab", help="spaCy 模型的 vocab 目錄（例如 zh_core_web_lg-3.8.0/vocab）")
    src.add_argument("--w2v", help="word2vec 文字格式向量檔（可 .gz）")
    ap.add_argument("--jieba-dict", type=Path, default=None)
    ap.add_argument("--answers", type=Path, default=ROOT / "data" / "answers.txt")
    ap.add_argument("--out", type=Path, default=ROOT / "web" / "data")
    ap.add_argument("--size", type=int, default=30000, help="詞庫大小")
    ap.add_argument("--dim", type=int, default=128, help="PCA 降維後的維度；0 = 不降維")
    args = ap.parse_args()

    import opencc

    s2hk = opencc.OpenCC("s2hk")
    s2tw = opencc.OpenCC("s2tw")
    t2s = opencc.OpenCC("t2s")

    dict_rows = load_jieba_dict(args.jieba_dict or default_jieba_dict())
    answers_trad = read_answers(args.answers)
    answers_simp = [t2s.convert(w) for w in answers_trad]
    print(f"jieba 候選詞：{len(dict_rows)}；答案候選：{len(answers_trad)}", file=sys.stderr)

    if args.spacy_vocab:
        vecs = SpacyVectors(args.spacy_vocab)
    else:
        wanted = {w for w, _ in dict_rows} | set(answers_simp)
        vecs = W2VVectors(args.w2v, wanted)

    words: list[str] = []          # 繁體顯示用
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

    for simp, _freq in dict_rows:
        if len(words) >= args.size:
            break
        add(simp)

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
    (args.out / "vocab.json").write_text(
        json.dumps({"dim": int(q.shape[1]), "words": words, "aliases": aliases}, ensure_ascii=False,
                   separators=(",", ":")),
        encoding="utf-8",
    )
    (args.out / "answers.json").write_text(json.dumps(answer_idx), encoding="utf-8")

    mb = os.path.getsize(args.out / "vectors.bin") / 1e6
    print(f"完成：詞庫 {len(words)} 詞、別名 {len(aliases)}、答案 {len(answer_idx)}、"
          f"維度 {q.shape[1]}、vectors.bin {mb:.1f} MB", file=sys.stderr)


if __name__ == "__main__":
    main()
