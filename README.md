# mp3_to_nbs

把一首 MP3 自動轉成 **Minecraft Note Block Studio（`.nbs`）**，也可以同時輸出 **MIDI（`.mid`）**。

流程是：Demucs 分軌 → 各聲部各自轉譜（主唱、Bass、鼓、伴奏、Piano / Guitar）→ 和弦分析與編曲修飾 → 封裝成 NBS / MIDI。

```
MP3 ──► Demucs 4-Stem ──► vocals ─► RMVPE 音高追蹤 ─► 主旋律
                       ├─► bass   ─► pYIN / RMVPE   ─► Bass
                       ├─► drums  ─► 節拍 / onset    ─► 鼓組重新設計
                       └─► other  ─► Onsets & Frames ─► 伴奏 + 和弦補缺
        └─► htdemucs_6s（選用）─► Piano / Guitar 獨立分軌 ─► Onsets & Frames
                                  ▼
                 調性 / 和弦 / 音域適配 / 動態力度 / panning
                                  ▼
                          .nbs（以及選用的 .mid）
```

---

## 特色

- **每個聲部各用最適合的方法**：主唱走 RMVPE、Bass 走 pYIN（可改 RMVPE）、伴奏走 Onsets & Frames 神經網路轉譜。
- **自動 TPS**：偵測 Tempo 後算出 tick 網格，讓節奏對齊拍點；也可手動指定。
- **和弦分析**：偵測和弦進行，修正與旋律衝突的和弦、產生和弦伴奏，並可輸出為 MIDI Marker。
- **動態力度**：依各 stem 的音訊能量重寫每個音符的 velocity，不是固定音量。
- **立體聲 panning**：依 stem 的立體聲能量估算各 layer 的 pan。
- **音域適配**：把各樂器折進遊戲內 2 個八度（key 33–57），並換算各樂器的八度偏移。
- **主旋律修飾**：八度校正、音域自動歸位、調內修正（修轉譜的 ±1 半音誤差）、長音延音補敲。
- **快取**：同一首歌第二次跑會跳過 Demucs / HPSS / O&F 推理，方便反覆調參。
- **MIDI 輸出**：標準 SMF Type 1，主唱 / Bass / Piano / Guitar / Other / 鼓各一軌，另有和弦 Marker 軌。
- **只分軌模式**：`--separate-only` 只輸出 4 軌 WAV，不轉譜。

---

## 需求

- Python 3.10 以上（作者環境為 Python 3.13 / Windows）
- 建議有 NVIDIA GPU（Demucs 與 RMVPE 用）。沒有 GPU 也能跑，只是慢很多。

### Python 套件

腳本直接 import 的有：

| 套件 | 用途 |
|---|---|
| `numpy`、`scipy`、`librosa`、`soundfile` | 音訊處理、HPSS、pYIN、chroma |
| `torch`、`demucs` | Demucs 分軌（`demucs.api.Separator`） |
| `rmvpe_onnx` | 主唱 RMVPE 音高追蹤（ONNX） |
| `onsets_frames_to_nbs` | Other / Piano / Guitar 的神經網路轉譜（與腳本同資料夾的模組） |

MIDI 與 NBS 的寫入都是純標準庫實作，不需要額外套件。

### 需要放好的檔案

```
mp3_to_nbs.py
onsets_frames_to_nbs.py          # O&F 轉譜模組
models/
  onsets_frames_uni.tflite       # O&F 模型；也可用 --other-neural-model 指定其他路徑
```

> TODO：這裡補上你自己的安裝指令（`pip install ...`）與 `rmvpe_onnx` 的取得方式，我沒有看到這兩個檔案，所以不亂寫。

### 第一次執行會下載

- Demucs 模型權重（`htdemucs`；開啟 Piano/Guitar 分軌時還有 `htdemucs_6s`）
- RMVPE 模型（未指定 `--melody-rmvpe-model` 時使用 `rmvpe` 套件預設）

---

## 快速開始

```bash
# 最基本：輸出 song.nbs
python mp3_to_nbs.py song.mp3 -o song.nbs

# 同時輸出 NBS 與 MIDI
python mp3_to_nbs.py song.mp3 -o song.nbs --format both

# 只分軌，不轉譜
python mp3_to_nbs.py song.mp3 --separate-only -o stems_out --separate-instrumental
```

Windows 上若中文路徑或 log 亂碼，請加 `-X utf8`：

```bash
python -X utf8 mp3_to_nbs.py "歌名.mp3" -o "歌名.nbs"
```

不指定 `--instrument-preset` 時等同 `pop`，會自動開啟下表中的大部分功能。

---

## 常用參數

完整清單請用 `python mp3_to_nbs.py --help` 查看，下面只列最常調的。

### 整體

| 參數 | 說明 |
|---|---|
| `-o` | 輸出路徑。MIDI 模式會自動換成 `.mid` |
| `--format {nbs,midi,both}` | 輸出格式（預設 `nbs`） |
| `--tps N` | 固定每秒 tick 數。**不指定**則依 Tempo 自動計算；指定會關閉自動 TPS（例如 `--tps 100` 可得到 10 ms 的時間解析度） |
| `--device cuda\|cpu` | Demucs 運算裝置 |
| `--seed N` | 隨機種子，控制鼓點音量與副旋律抖動；同一個種子可重現（預設 0） |
| `--keep-temp` | 保留 Demucs 輸出 |

### 音色

| 參數 | 說明 |
|---|---|
| `--instrument-preset` | `pop`（預設）、`rock`、`synth`、`orchestra`、`legacy` |
| `--melody-instrument`、`--bass-instrument` … | 個別覆蓋 preset；`--melody-instrument auto` 會依音域自動選（高音域 bell、其餘 flute） |
| `--melody-double {off,low,high}` | 主唱加厚：另開一層較小音量的第二音色 |

各 preset 的主要音色：

| preset | 主旋律 | Other 低 / 中 / 高 | 和弦 |
|---|---|---|---|
| `pop` | flute | guitar / harp / pling | harp |
| `rock` | flute | guitar / guitar / pling | guitar |
| `synth` | bit | iron_xylophone / pling / bit | pling |
| `orchestra` | flute | harp / chime / bell | chime |
| `legacy` | flute | guitar / harp / harp | guitar |

### 力度與混音平衡

| 參數 | 說明 |
|---|---|
| `--dynamics {off,melody,bass,both,all}` | 依音訊能量重寫 velocity。`all` = 主唱 + Bass + Piano / Guitar / Other |
| `--drum-volume 倍率` | Kick / Snare 音量倍率（非 legacy 預設 0.5） |
| `--hat-volume 倍率` | HiHat 額外倍率，再乘上 `--drum-volume`（非 legacy 預設 0.6；`0` 關閉 hat） |
| `--panning {off,layer}` | 依立體聲能量估算 panning |
| `--melody-retrigger-sec 秒` | 主旋律長音每隔幾秒補敲一次（非 legacy 預設 0.4；`0` 關閉） |
| `--other-retrigger-sec 秒` | 伴奏長音補敲間隔（非 legacy 預設 0.5） |

### 主旋律

| 參數 | 說明 |
|---|---|
| `--melody-octave-shift N` | 手動整體升降八度（可與自動音域歸位疊加） |
| `--melody-min-note-ms` | 短於此毫秒數的人聲音符會被吸附進鄰近長音，抑制滑音碎片（預設 70） |
| `--rap_revise` / `--no-rap_revise` | Rap / 快速人聲修正（預設開） |
| `--melody-key-fix` / `--no-melody-key-fix` | 調內修正 |
| `--melody-key-fix-margin 倍數` | 調內修正門檻，越大越保守（預設 1.25） |
| `--no-octave-correction` | 關閉主唱八度校正 |

### 伴奏（Other / Piano / Guitar）

| 參數 | 說明 |
|---|---|
| `--other-mode {hybrid,chord,extract,mix}` | 預設 `hybrid`：優先用實際抓到的音，不足處用和弦補缺 |
| `--other-extract-backend {cqt,neural}` | 預設 `neural`（Onsets & Frames） |
| `--other-neural-engine {auto,tflite,gpu}` | `auto` 優先 WebGPU，不行再退回 tflite CPU |
| `--other-neural-raw` | 跳過 HPSS 與 spectral gate，直接用 Demucs 的 other.wav |
| `--split-piano-guitar` / `--no-split-piano-guitar` | 用 `htdemucs_6s` 另外分出 Piano / Guitar（非 legacy 預設開） |
| `--other-neural-max-polyphony N` | 同一 onset 群組最多保留幾個同時音符（預設 4） |

### 輸出整理

| 參數 | 說明 |
|---|---|
| `--trim-silence` | 剃除前後空白（非 legacy 預設開） |
| `--trim-edge-gap 秒` | 頭尾與主體相隔超過這麼多秒的孤立音視為雜音並移除（預設 0 = 不移除，建議 5） |
| `--range-fit` | 音域適配，把音符折進 key 33–57 |
| `--max-tick-shift N` | 同 layer 同 tick 撞音時最多往後找幾個 tick（預設 4） |
| `--chord-pattern {beat,bar,arp}` | 和弦伴奏敲法 |
| `--chord-polish` | 把 sus / 七和弦改成調內三和弦，並避開與旋律差半音的和弦音 |

---

## preset 與預設值

`--instrument-preset legacy` 會維持加入 preset 之前的舊行為，輸出保持與舊版一致。其他 preset（預設 `pop`）會自動開啟下列功能，除非你個別關掉：

| 功能 | `legacy` | 其他 preset |
|---|---|---|
| 動態力度 `--dynamics` | off | all |
| 立體聲 panning | off | layer |
| Piano/Guitar 分軌 | off | on |
| 音域適配 `--range-fit` | off | on |
| 主唱音域自動歸位 | off | auto |
| 和弦分析 | off | on |
| 和弦配對檢查 | off | auto |
| 和弦優化 `--chord-polish` | off | on |
| 旋律調內修正 | off | on |
| 剃除前後空白 | off | on |
| 和弦伴奏敲法 | beat | bar |
| 鼓音量 / hat 音量 | 1.0 / 1.0 | 0.5 / 0.6 |
| 主旋律延音間隔 | 0（關） | 0.4 秒 |
| Other 延音間隔 | 0（關） | 0.5 秒 |

---

## 輸出的 NBS layer

| Layer | 內容 |
|---|---|
| 0 | 主旋律 |
| 1 | Bass |
| 2 / 3 / 4 | Other 低 / 中 / 高音域 |
| 5 / 6 / 7 / 8 | 和弦伴奏（根音 / 三音 / 五音 / 七音） |
| 9 / 10 / 11 | Kick / Snare / HiHat |
| 12–15 | Piano（4 個音域帶，分界 key 48 / 60 / 72） |
| 16–19 | Guitar（4 個音域帶） |
| 20 | 主唱加厚層（`--melody-double`） |

沒有啟用的功能不會產生對應 layer 的音符。

---

## 快取

腳本旁會建立 `.mp3_nbs_cache/`。快取鍵是輸入檔的大小、修改時間與相關參數，因此：

- 同一首歌調參數重跑，可以跳過 Demucs、HPSS、雜音抑制與 O&F 模型推理。
- 換了 mp3 或影響該階段的參數，會自動重算。
- 想強制重算時，直接刪掉 `.mp3_nbs_cache/` 即可。

---

## 效能

以一首約 5 分 50 秒的歌為例（log 實測，總計約 9 分 43 秒）：

| 階段 | 約耗時 |
|---|---|
| Demucs 分軌（4-stem + 6-stem） | 55 秒 |
| 載入 stems + HPSS 快取 | 72 秒 |
| 平行分析（主唱、Bass、純音樂旋律、和弦、鼓） | 254 秒 |
| Piano / Guitar 的 O&F 轉譜 | 166 秒 |
| 封裝 NBS / MIDI | 約 4 秒 |

時間大多花在 CPU 密集的 pYIN（Bass 與「純音樂旋律搜尋」）和每個 stem 的 HPSS + 頻譜 gate。想縮短時間：

- 確認 GPU 路徑真的生效（見下方疑難排解）。
- 反覆調參時盡量重用快取。
- Bass 可試 `--bass-backend rmvpe`（GPU 上快很多，輸出與 pYIN 略有差異，建議先 A/B 比較）。
- 不需要 Piano / Guitar 分軌時，用 `--no-split-piano-guitar`。

---

## 疑難排解

**log 說 `device=cuda`，但 RMVPE 實際跑在 CPU**
看 `RMVPE → 生效 provider` 那一行。若是 `CPUExecutionProvider` 並伴隨 `cublasLt64_13.dll` 找不到，代表 onnxruntime 的 CUDA 版缺少 CUDA 13 / cuDNN 9 執行庫。Demucs 走的是 torch 自己的 CUDA，所以不受影響。

**中文檔名或 log 亂碼（Windows）**
用 `python -X utf8 ...` 執行。

**音準有些音在調外**
先看 log 的「旋律調內修正」那幾行，確認推測的調性是否合理。旋律推測的調與人聲 chroma 推測的調不一致時，會印出來；若 chroma 的相關係數 r 不高，結果可信度有限，可用 `--no-melody-key-fix` 關掉，或調高 `--melody-key-fix-margin`。

**鼓太吵或伴奏太小聲**
調整 `--drum-volume`、`--hat-volume`；確認 `--dynamics all` 有開。

**長音聽起來太快消失**
NBS 沒有 note-off，只能靠重複敲擊。用 `--melody-retrigger-sec` 與 `--other-retrigger-sec` 調整間隔。

**尾段內容很少**
可能是 mp3 本身較短，或加了 `--trim-edge-gap` 把尾段孤立音移除。先確認音檔長度與參數。

---

## 已知限制

- 轉譜是自動的，沒有人工校正，伴奏的音符與和弦難免有錯。
- NBS 的 tick 解析度、單一 layer 單一 tick 只能放一個音，高密度的和弦與快速音型會被折疊或取捨。
- 調性、和弦與八度的判斷都是推測，結果會因歌曲而異。
- 只在 Windows + NVIDIA GPU 環境驗證過；其他平台未測試。
