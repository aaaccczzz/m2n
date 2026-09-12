# mp3 to nbs

把 MP3（或任何音訊檔）自動轉成 Minecraft Note Block Studio 的 `.nbs` 多層樂譜。

開源、無限制執行的本機轉譜工具。音檔不上傳、不排隊、不限制長度、所有參數都能調。

---

## 這支程式在做什麼

它不是單純的「格式轉換」，而是把一首混音歌曲拆解、聽寫、重新編成 NBS 多聲部樂譜：

MP3
 └─ Demucs 4-Stem 分離        → vocals / bass / other / drums
 └─ 主唱 RMVPE 音高追蹤        → Melody
 └─ Bass pYIN 音高追蹤         → Bass
 └─ Other 副旋律（多種 backend）→ Other Low / Mid / High
 └─ 和弦分析（可選）           → Chord Root / 3rd / 5th / 7th
 └─ Drums onset 分類           → Kick / Snare / HiHat
 └─ NBS v5 封裝（12 層）

輸出是 NBS v5 格式，直接用 Note Block Studio 開啟即可。

---

## 特色

- **本機執行**：音檔不離開你的電腦，沒有上傳、沒有伺服器、沒有隱私問題。
- **無限制**：不限制歌曲長度、轉檔次數、功能。
- **開源**：全部程式碼可見、可改。
- **多 backend**：Other 聲部可選 CQT（傳統訊號處理）或 Onsets & Frames（神經網路）。
- **參數全開**：主唱音域、Bass 音域、鼓組設計、和弦樣式、八度升降……全部可用 CLI 調整。
- **持久化快取**：Demucs、HPSS、O&F 推論結果會快取，第二次跑同一首歌快很多。
- **12 層輸出**：主唱、Bass、Other 低/中/高、和弦四聲部、鼓三件，各自獨立分層。

---

## 安裝

### 需求

- Python 3.10+（建議 3.13）
- ffmpeg（Demucs 需要）
- 建議有 NVIDIA GPU（Demucs 與 RMVPE 會快很多，CPU 也可跑但慢）

### 系統需求建議

- CPU：4 核心以上
- RAM：16 GB 以上（Demucs 與 librosa 都很吃記憶體）
- GPU：NVIDIA，VRAM 6 GB 以上可跑 htdemucs；8 GB 以上較順
- 硬碟：至少 5 GB 空間（模型 + 快取）

### 安裝 ffmpeg

Windows：
- 到 ffmpeg 官網下載 build，解壓後把 `bin` 加入 PATH
- 或用 `winget install ffmpeg`
- 或 `choco install ffmpeg`

Linux：
- `sudo apt install ffmpeg`

macOS：
- `brew install ffmpeg`

### 安裝步驟

1. 建立虛擬環境

   python -m venv .venv
   .venv\Scripts\activate        # Windows
   source .venv/bin/activate     # Linux / macOS

2. 安裝依賴

   pip install -r requirements.txt

3. 安裝 Demucs

   pip install demucs

4. 安裝 RMVPE ONNX（主唱音高追蹤）

   pip install rmvpe-onnx

5. 安裝 O&F 的 TFLite runtime（Other neural backend 需要）

   pip install ai-edge-litert

`onsets_frames_to_nbs.py` 會自動下載 O&F 模型（`models/onsets_frames_uni.tflite`）到專案目錄，第一次執行需要網路。它也會嘗試尋找 ffmpeg，Windows 上可用 `--ffmpeg` 明確指定路徑。

---

## 使用方式

### 最基本

   python mp3_to_nbs.py input.mp3

會在 `input.mp3` 旁邊產生 `input.nbs`。

### 指定輸出

   python mp3_to_nbs.py input.mp3 -o output.nbs

### 啟用和弦伴奏（預設關閉）

   python mp3_to_nbs.py input.mp3 --enable-chord-analysis

未啟用時，`hybrid` / `chord` / `mix` 模式會自動退化為 `extract`。

### 常用範例

   # 使用 GPU
   python mp3_to_nbs.py input.mp3 --device cuda

   # 主唱高一個八度
   python mp3_to_nbs.py input.mp3 --melody-octave-shift 1

   # 鋼琴主導的曲子
   python mp3_to_nbs.py input.mp3 --other-preset piano

   # 純音樂，不要和弦生成，只抓實際音符
   python mp3_to_nbs.py input.mp3 --other-mode extract

   # 關閉 Rap 自適應（A/B 測試用）
   python mp3_to_nbs.py input.mp3 --no-rap_revise

   # 用 CQT 取代神經網路抓 Other
   python mp3_to_nbs.py input.mp3 --other-extract-backend cqt

---

## 輸出結構

NBS 使用 12 個 layer：

| Layer | 名稱 | 來源 |
|---|---|---|
| 0 | Melody | 主唱 RMVPE |
| 1 | Bass | Bass pYIN + 和弦根音 |
| 2 | Other Low | Other MIDI < 55 |
| 3 | Other Mid | Other MIDI 55–75 |
| 4 | Other High | Other MIDI ≥ 76 |
| 5 | Chord Root | 和弦根音 |
| 6 | Chord 3rd | 和弦三度 |
| 7 | Chord 5th | 和弦五度 |
| 8 | Chord 7th | 和弦七度 |
| 9 | Kick | Drums onset 分類 |
| 10 | Snare | Drums onset 分類 |
| 11 | HiHat | Drums onset 分類 |

MIDI → NBS key 採標準映射：`key 0 = MIDI 21 (A0)`、`key 87 = MIDI 108 (C8)`。

---

## Other 模式

`--other-mode` 控制副旋律怎麼產生：

| 模式 | 行為 |
|---|---|
| `hybrid`（預設） | 優先使用 Other 實際抓到的音符，不足處用和弦補齊 |
| `chord` | 完全用和弦 + 節拍生成琶音伴奏（穩定但機械） |
| `extract` | 逐幀從 Other 抓音符（CQT 或 neural） |
| `mix` | 有人聲用 chord、無人聲用 extract，依 vocals 能量自動切換 |

`--other-extract-backend` 控制 extract 的引擎：

| Backend | 說明 |
|---|---|
| `neural`（預設） | Onsets & Frames TFLite，複音轉譜，準確度較高 |
| `cqt` | CQT peak picking + 泛音抑制 + track continuity，純訊號處理 |

---

## 參數總覽

### 基本參數

- `mp3`：輸入音訊路徑（必填）
- `-o`, `--output`：輸出 NBS 路徑，預設與輸入同名的 `.nbs`
- `--sr`：取樣率，預設 22050
- `--tps`：每秒 tick 數，預設 40.0
- `--demucs-model`：Demucs 模型，預設 `htdemucs`
- `--device`：運算裝置，例如 `cuda` / `cpu`
- `--segment`：Demucs segment
- `--keep-temp`：保留 Demucs 輸出
- `--seed`：隨機種子，控制鼓點音量 / 副旋律抖動的隨機性，預設 0

### 音高升降八度

- `--melody-octave-shift`：主唱升降八度，0=原始 1=高一個八度 2=高兩個八度 -1=低一個八度，預設 0
- `--bass-octave-shift`：Bass 升降八度，預設 0
- `--other-octave-shift`：Other 樂器升降八度，預設 0

### 主唱 RMVPE 參數

- `--melody-fmin`：主唱最低偵測音符，預設 `C2`
- `--melody-fmax`：主唱最高偵測音符，預設 `C6`
- `--melody-min-dur`：主唱音符最短保留時間（秒），預設 0.025
- `--melody-gap-tolerance`：主唱無聲多久才視為斷句（秒），預設 0.09
- `--melody-voiced-prob`：RMVPE F0 confidence threshold，預設 0.03
- `--melody-persistence`：RMVPE 新音高至少持續多久才換音，預設 0.045
- `--melody-medfilt-kernel`：RMVPE MIDI 中值濾波窗口大小，預設 5
- `--melody-min-note-ms`：人聲音符短於此長度（毫秒）會被吸附進鄰近長音，0 表示關閉，預設 70.0
- `--rap_revise` / `--no-rap_revise`：啟用 Rap/快速人聲修正，預設開啟
- `--melody-rmvpe-model`：RMVPE 模型檔路徑，留空使用套件預設模型
- `--melody-rmvpe-half`：RMVPE 使用 FP16
- `--no-octave-correction`：關閉主唱八度校正，預設開啟校正

### Bass pYIN 參數

- `--bass-fmin`：Bass 最低偵測音符，預設 `E1`
- `--bass-fmax`：Bass 最高偵測音符，預設 `C4`
- `--bass-frame-length`：Bass pYIN frame_length，預設 2048
- `--bass-hop-length`：Bass pYIN hop_length，預設 256
- `--bass-min-dur`：Bass 音符最短保留時間（秒），預設 0.05
- `--bass-gap-tolerance`：Bass 無聲多久才視為斷句（秒），預設 0.08
- `--bass-voiced-prob`：Bass 有聲判定機率門檻，預設 0.12
- `--bass-persistence`：Bass 新音高要持續多久（秒）才算換音，預設 0.03
- `--bass-medfilt-kernel`：Bass 音高中值濾波窗口大小，預設 3

### 和弦分析參數

- `--enable-chord-analysis`：啟用和弦分析，預設關閉
- `--other-chord-subdivision`：每個 beat 切成幾個音符，1=每拍一音 2=八分音符琶音 4=十六分音符琶音，預設 2
- `--other-chord-pattern`：琶音樣式，`up` / `down` / `up_down` / `root_only`，預設 `up`
- `--other-chord-note-len-ratio`：每個音符實際發聲長度占時間格的比例，預設 0.9
- `--no-chord-rotation`：關閉琶音方向隨小節交替，預設開啟
- `--other-chord-skip`：offbeat 音符被跳過的機率，預設 0.08
- `--other-chord-humanize`：音符時間隨機抖動幅度（秒），預設 0.008

### Other 樂器參數

- `--other-preset`：Other 分析預設，`default` / `piano`，預設 `default`
- `--other-mode`：副旋律產生方式，`hybrid` / `chord` / `extract` / `mix`，預設 `hybrid`
- `--other-extract-backend`：Other 實際音符抽取引擎，`cqt` / `neural`，預設 `neural`
- `--other-neural-model`：Other neural Onsets & Frames 模型路徑，留空使用 `models/onsets_frames_uni.tflite`
- `--other-neural-onset-threshold`：Other neural onset threshold，預設 0.45
- `--other-neural-frame-threshold`：Other neural frame threshold，預設 0.40
- `--other-neural-min-duration`：Other neural 最短音符時間（秒），預設 0.035
- `--other-neural-min-velocity`：Other neural 最低 velocity，預設 24
- `--other-neural-midi-min`：Other neural 最低 MIDI 音高，預設 21
- `--other-neural-midi-max`：Other neural 最高 MIDI 音高，預設 108
- `--other-neural-max-duration`：Other neural 單顆音符最大長度（秒），預設 8.0
- `--other-neural-merge-gap`：Other neural 同音符合併間隔（秒），預設 0.03
- `--other-neural-max-polyphony`：Other neural 同一 onset 群組最多保留幾個同時音符，預設 4
- `--other-neural-onset-group-window`：Other neural 視為同一 onset 群組的時間窗口（秒），預設 0.055
- `--other-neural-raw`：Other neural 直接使用 Demucs other.wav，跳過 HPSS 與 spectral gate
- `--other-hop-length`：[extract 模式] Other CQT hop_length，預設 256
- `--other-min-note-dur`：[extract 模式] Other 音符最短保留時間（秒），預設 0.08
- `--other-energy-threshold`：[extract 模式] Other 音符能量門檻，預設 0.2
- `--other-min-note-separation`：[extract 模式] Other 音符之間最短間隔（秒），預設 0.12
- `--other-max-notes-per-frame`：[extract 模式] 每個 CQT frame 最多保留幾個音高峰，預設 1
- `--other-peak-min-separation`：[extract 模式] 不同 CQT 峰之間至少相差多少半音，預設 2
- `--other-harmonic-tolerance`：[extract 模式] 泛音匹配容許誤差（半音），預設 0.45
- `--other-harmonic-count`：[extract 模式] 每個候選基頻最多檢查幾階泛音，預設 8
- `--other-harmonic-suppression`：[extract 模式] 泛音候選最高抑制比例，預設 0.65
- `--other-track-max-gap`：[extract 模式] 多音符 track 允許短暫掉音多久（秒），預設 0.06
- `--other-cqt-fmin`：[extract 模式] CQT 最低音，預設 `C2`
- `--other-cqt-fmax`：[extract 模式] CQT 最高音，預設 `C8`
- `--other-no-filter`：[extract/mix] 關閉 Other 的時間稀疏化與音符過濾
- `--other-confidence-threshold`：[extract/hybrid] Other 音符最低可信度，預設 0.18

### NBS 輸出參數

- `--other-retrigger-sec`：Other 長音在 NBS 中重新觸發的間隔（秒），0=不人工重敲，預設 0.0
- `--max-tick-shift`：同一 layer 同一 tick 撞到其他音符時，最多往後找幾個 tick 的空位，預設 4

---

## 快取機制

程式會在專案目錄下建立 `.mp3_nbs_cache/`：

.mp3_nbs_cache/
 ├─ demucs/          Demucs 4-stem 輸出
 ├─ hpss/            Other 的 harmonic 快取
 ├─ neural_clean/    O&F 前的降噪結果
 └─ neural_notes/    O&F 模型推論結果

快取 key 由「檔案路徑 + size + mtime_ns + 相關參數」組成。
同一首歌第二次跑會直接命中快取，跳過 Demucs / HPSS / O&F 推論。

改了參數（例如 `--other-neural-onset-threshold`）會產生新的 key，不會拿到舊結果。若要強制重算，直接刪掉 `.mp3_nbs_cache/` 即可。

---

## 獨立執行 O&F 模組

`onsets_frames_to_nbs.py` 也可以單獨使用：

   python onsets_frames_to_nbs.py input.wav -o output.nbs

參數：

- `--model`：自訂模型路徑
- `--ffmpeg`：ffmpeg.exe 路徑
- `--tempo`：每秒 tick，預設 20
- `--onset-threshold`
- `--frame-threshold`
- `--min-duration`
- `--keep-wav`：保留轉出的 16 kHz WAV

---

## 常見問題

**Q：一定要 GPU 嗎？**
不一定。CPU 可以跑，但 Demucs 與 RMVPE 會慢很多。有 NVIDIA GPU 時加 `--device cuda`。

**Q：為什麼我的 NBS 聽起來像機械伴奏？**
檢查是否用了 `--other-mode chord`。預設 `hybrid` 會優先使用實際抓到的音符，`chord` 才是純生成。若未啟用 `--enable-chord-analysis`，`hybrid` / `chord` / `mix` 都會退化為 `extract`。

**Q：主唱音高抓錯八度怎麼辦？**
預設會用前後文自動修正。若仍不滿意，可手動加 `--melody-octave-shift 1` 或 `-1`。

**Q：Rap 的短音節被吃掉怎麼辦？**
預設 `--rap_revise` 已開啟。若想 A/B 比較，用 `--no-rap_revise` 關閉。

**Q：Other 抓不到東西？**
試試 `--other-preset piano`（高密度、低門檻、多音符），或改用 `--other-extract-backend cqt`。

**Q：轉檔失敗，找不到 RMVPE？**
執行 `pip install rmvpe-onnx`。若仍失敗，確認 Python 版本與 onnxruntime 相容。

**Q：找不到 ffmpeg？**
`onsets_frames_to_nbs.py` 會自動搜尋常見路徑。若都找不到，用 `--ffmpeg` 指定，或把 ffmpeg 加進 PATH。
