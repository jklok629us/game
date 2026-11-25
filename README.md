# 猜詞作戰 — 中文 AI 語意聯想猜詞遊戲

靈感來自《魔鬼的詭計》中用 AI 判斷詞語聯想的遊戲，以及 Semantle / Contexto / 꼬맨틀 一類的猜詞遊戲。
玩家輸入任何中文詞，AI 會計算它和秘密答案的**語意相似度**，玩家靠分數和排名一步步逼近答案。

## 快速開始

遊戲是純靜態網頁，資料已經預先建置在 `web/data/`，不需要伺服器運算或 GPU：

```bash
cd web
python3 -m http.server 8000
# 打開 http://localhost:8000
```

> 要用 HTTP 伺服器開啟，直接雙擊 `index.html`（`file://`）時瀏覽器會阻擋讀取資料檔。
> 也可以把 `web/` 目錄直接部署到 GitHub Pages / Netlify / Cloudflare Pages。

## 遊戲模式

| 模式 | 玩法 |
|---|---|
| 每日一題 | 所有人同一個答案，進度會存在瀏覽器 |
| 隨機練習 | 隨機抽一個答案 |
| 雙人搶答 | 同一題輪流猜，先猜中者勝。所有線索公開，你的好線索也會幫到對手 |
| 出題對決 | 玩家互相出秘密答案，用最少次數猜中者勝（提示有罰分） |

## AI 系統怎樣運作

```
[離線：tools/build_data.py]
jieba 常用詞（按詞頻、詞性過濾）
   ∩ spaCy zh_core_web_lg 詞向量（50 萬詞 × 300 維）
   → OpenCC 簡轉繁（保留簡體 / 台灣寫法作別名）
   → 正規化 → PCA 降到 128 維 → int8 量化
   → web/data/vocab.json（3 萬詞）+ vectors.bin（3.8 MB）+ answers.json

[瀏覽器：web/app.js]
選定答案 → 計算所有詞和答案的 cosine similarity → 排序得出排名
玩家輸入 → 查詞表 → 顯示相似度（×100）和排名（只顯示最接近的 1000 名）
```

- **不需要訓練 LLM**：我們只需要一把「量度兩個詞意思有多接近」的尺，現成的詞向量（word embedding）已經足夠。
- **相似度**：兩個詞向量的夾角餘弦值（cosine similarity）。答案本身是 100。
- **排名**：答案在詞庫中所有詞裏，你的詞排第幾近。
- **提示**：揭露一個排名比你目前最佳結果再近一半的詞。

## 自訂 / 重新建置資料

1. 安裝依賴（見 `requirements.txt`），並下載 spaCy 中文模型。
2. 修改答案候選：`data/answers.txt`（每行一個繁體詞）。
3. 執行：

```bash
python tools/build_data.py \
  --spacy-vocab "$(python -c 'import zh_core_web_lg,os;d=os.path.dirname(zh_core_web_lg.__file__);print([os.path.join(d,x,"vocab") for x in os.listdir(d) if x.startswith("zh_core_web_lg-")][0])')"
```

常用參數：

| 參數 | 預設 | 說明 |
|---|---|---|
| `--size` | 30000 | 詞庫大小（越大越多詞可猜，檔案越大） |
| `--dim` | 128 | PCA 降維；`0` 表示保留原本 300 維（約 9 MB） |
| `--w2v` | — | 改用其他 word2vec 文字格式向量，例如 Tencent AI Lab Embedding、fastText `cc.zh.300.vec` |

## 已知限制 / 下一步

- 詞向量用網絡文本訓練，部分聯想反映網絡用法（例如「蘋果」接近「手機」多過「香蕉」）。
- 粵語口語詞（嘢、睇、咗）覆蓋較弱，可以考慮加入粵語語料訓練的向量，或者人手補同義詞表。
- 可以改用句子向量模型（如 `bge-m3`）處理多字詞組，但需要後端或較大的前端模型。
- 可以加入節目式規則：團隊對戰、資訊不對稱、分數換籌碼、限時回合、網上即時對戰等。
