#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import builtins
import time


# =========================================================
# 全域計時 print
# =========================================================
# 這兩個 import 故意放在所有大型套件之前，讓計時器從 Python
# 程式真正開始執行就啟動，而不是等 numpy / librosa / RMVPE 等
# 大型模組 import 完才開始計時。
_PRINT_START_TIME = time.perf_counter()
_original_print = builtins.print


def _timed_print(*args, **kwargs):
    elapsed = time.perf_counter() - _PRINT_START_TIME
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60
    _original_print(
        f"[{hours:02d}:{minutes:02d}:{seconds:05.2f}]",
        *args,
        **kwargs,
    )


builtins.print = _timed_print

import argparse
import hashlib
import io
import multiprocessing
import os
import struct
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# RMVPE：主唱 F0 追蹤器。
# 這個 beta 版本完全移除舊 pYIN 主唱路徑，只使用 RMVPE。
_RMVPE_IMPORT_ERROR = None

try:
    from rmvpe_onnx import RMVPE
except Exception as exc:
    RMVPE = None
    _RMVPE_IMPORT_ERROR = exc

try:
    import onsets_frames_to_nbs as _of
except Exception:
    _of = None

_RMVPE_MODEL_CACHE = {}
_OF_MODEL_CACHE = {}


# ============================================================
# 持久化快取
# ============================================================
# 快取只依賴輸入檔案的 size + mtime_ns + 相關參數，因此同一首歌
# 第二次測試時可以跳過 Demucs / HPSS / O&F 推理。
# 快取不會改變轉譜演算法或輸出，只避免重算完全相同的中間結果。
_CACHE_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".mp3_nbs_cache",
)


def _cache_key_for_file(path, *parts):
    try:
        st = os.stat(path)
        identity = (
            os.path.abspath(path),
            int(st.st_size),
            int(st.st_mtime_ns),
            *parts,
        )
    except OSError:
        identity = (os.path.abspath(path), *parts)
    return hashlib.sha1(
        repr(identity).encode("utf-8", errors="replace")
    ).hexdigest()


def _ensure_cache_dir(name):
    path = os.path.join(_CACHE_ROOT, name)
    os.makedirs(path, exist_ok=True)
    return path


# ============================================================
# 音高調整
# ============================================================

# 主唱升降八度
# 0  = 原始
# 1  = 高一個八度
# 2  = 高兩個八度
# -1 = 低一個八度
#
# 舊版預設 1 是為了補償 midi_to_nbs_key() 的「-33」映射錯誤
# （所有音符被整體多降一個八度）。映射已修正為 NBS 標準的
# 「-21」（key 0 = MIDI 21 / A0）之後，0 才是真正的原始音高。
# 若刻意想讓主唱高一個八度，可改這裡或用 --melody-octave-shift 1。
MELODY_OCTAVE_SHIFT = 0

# Bass 升降八度
BASS_OCTAVE_SHIFT = 0

# Other 樂器升降八度
# Piano / Guitar / Strings / Synth 等
OTHER_OCTAVE_SHIFT = 0

# Other 相對主唱的基準音量。預設稍微壓低，避免伴奏蓋過主唱。
OTHER_VELOCITY_SCALE = 0.72


# ============================================================
# 進度顯示
# ============================================================

def show_progress(current, total, prefix="", width=30):
    if total <= 0:
        return

    percent = current / total
    filled = int(width * percent)

    bar = "#" * filled + "-" * (width - filled)

    print(
        f"\r{prefix} [{bar}] {percent * 100:5.1f}%",
        end="",
        flush=True,
    )

    if current >= total:
        print()


def progress_step(
    current,
    total,
    last_value,
):
    """
    控制進度條不要每一幀都刷新。

    大約每 1% 更新一次。
    """

    if total <= 0:
        return last_value

    percent = int(
        current / total * 100
    )

    if percent != last_value:
        show_progress(
            current,
            total,
        )
        return percent

    return last_value


# ============================================================
# Demucs 4-Stem
# ============================================================

def separate_4stems(
    mp3_path: str,
    work_dir: str,
    model: str = "htdemucs",
    device: str | None = None,
    segment: int | None = None,
):
    base = os.path.splitext(os.path.basename(mp3_path))[0]
    cache_key = _cache_key_for_file(
        mp3_path,
        "demucs-v1",
        model,
        device or "auto",
        segment,
    )
    cache_root = _ensure_cache_dir("demucs")
    cache_work_dir = os.path.join(cache_root, cache_key)
    cache_out_dir = os.path.join(cache_work_dir, model, base)
    expected = {
        name: os.path.join(cache_out_dir, f"{name}.wav")
        for name in ("vocals", "drums", "bass", "other")
    }

    if all(os.path.isfile(path) for path in expected.values()):
        print(
            f"Demucs 4-Stem 快取命中 (model={model})"
        )
        return expected

    print(
        f"Demucs 4-Stem 分離 "
        f"(model={model}) ..."
    )

    # Demucs 直接輸出到持久化 cache；這樣 TemporaryDirectory 不會讓
    # 下一次執行又把 4 個 stem 全部重新算一次。
    work_dir = cache_work_dir
    os.makedirs(work_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "demucs",
        "-n",
        model,
        "-o",
        work_dir,
    ]

    if device:
        cmd += [
            "-d",
            device,
        ]

    if segment:
        cmd += [
            "--segment",
            str(segment),
        ]

    cmd.append(mp3_path)

    subprocess.run(
        cmd,
        check=True,
    )

    out_dir = os.path.join(
        work_dir,
        model,
        base,
    )

    stems = {
        "vocals": os.path.join(
            out_dir,
            "vocals.wav",
        ),
        "drums": os.path.join(
            out_dir,
            "drums.wav",
        ),
        "bass": os.path.join(
            out_dir,
            "bass.wav",
        ),
        "other": os.path.join(
            out_dir,
            "other.wav",
        ),
    }

    for name, path in stems.items():

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Demucs 找不到 {name}.wav:\n"
                f"{path}"
            )

    print(
        "Demucs 4-Stem 完成"
    )

    return stems


# ============================================================
# RMVPE 單音旋律
# ============================================================

def _get_rmvpe_model(device=None, model_path=None, is_half=False):
    if RMVPE is None:
        raise RuntimeError(
            "找不到 RMVPE ONNX。請先執行：python -m pip install rmvpe-onnx；"
            f"原始錯誤：{_RMVPE_IMPORT_ERROR}"
        )

    # rmvpe-onnx 使用 ONNX Runtime provider，不使用舊 PyTorch RMVPE API。
    # 預設先使用套件自己的 provider 選擇；模型會在第一次 predict 時自動下載。
    cache_key = (model_path or "__default__", bool(is_half))
    if cache_key not in _RMVPE_MODEL_CACHE:
        print("  RMVPE → 載入 ONNX 模型 ...")
        kwargs = {}
        if model_path:
            kwargs["model_path"] = model_path
        _RMVPE_MODEL_CACHE[cache_key] = RMVPE(**kwargs)
        print("  RMVPE → ONNX 模型載入完成")
    return _RMVPE_MODEL_CACHE[cache_key]


def _absorb_short_notes(
    raw_notes,
    min_len,
    merge_gap=0.05,
    ratio_guard=1.5,
):
    """把滑音/過渡產生的碎片音符吸附進鄰近長音。

    一個唱音常被切成「長音 → 短過渡音 → 長音」三段，聽起來像滑音。
    候選碎片有兩層判定（皆需至少一側鄰音 >= 碎片 ratio_guard 倍長，
    避免吃掉等長的快速樂段）：

    - 一般碎片：長度 < min_len；
    - 慢滑音簽名：長度 < min_len * 2，且音高「嚴格夾在前後兩音
      之間」（單調滑行的特徵），且與兩側音程都 <= 3 半音
      （琶音、大跳音不會誤判）——較長的慢滑音 step 也會被清除。

    反覆取出最短的候選碎片併入相鄰長音：

    - 優先併入同音高的鄰居，其次併入較長的一側；
    - 沒有合格鄰音的候選保留原狀（Rap / 裝飾音群）。

    回傳合併後的 (start, end, pitch) list。
    """
    if not raw_notes or min_len <= 0:
        return list(raw_notes)

    notes = [[float(s), float(e), int(p)] for s, e, p in raw_notes]
    glide_max = float(min_len) * 2.0
    blocked = set()

    def _candidate(i):
        """判定 notes[i] 是否為可吸附的滑音/碎片候選。"""
        s, e, p = notes[i]
        d = e - s
        if d < min_len:
            return True
        if d >= glide_max or i == 0 or i + 1 >= len(notes):
            return False
        prev_p = notes[i - 1][2]
        next_p = notes[i + 1][2]
        lo, hi = (prev_p, next_p) if prev_p < next_p else (next_p, prev_p)
        if not (lo < p < hi):
            return False
        return abs(p - prev_p) <= 3 and abs(next_p - p) <= 3

    while True:
        best_i = -1
        best_d = None
        for i in range(len(notes)):
            if id(notes[i]) in blocked:
                continue
            if not _candidate(i):
                continue
            d = notes[i][1] - notes[i][0]
            if best_d is None or d < best_d:
                best_d, best_i = d, i

        if best_i < 0:
            break

        s, e, p = notes[best_i]
        prev_n = notes[best_i - 1] if best_i > 0 else None
        next_n = notes[best_i + 1] if best_i + 1 < len(notes) else None

        cands = []
        if prev_n is not None and (prev_n[1] - prev_n[0]) >= best_d * ratio_guard:
            cands.append(prev_n)
        if next_n is not None and (next_n[1] - next_n[0]) >= best_d * ratio_guard:
            cands.append(next_n)

        if not cands:
            # 快速樂段：兩側都不夠長 → 保留這個碎片，不再重試。
            blocked.add(id(notes[best_i]))
            continue

        def _absorb_key(nb, _p=p):
            return (0 if nb[2] == _p else 1, -(nb[1] - nb[0]))

        target = min(cands, key=_absorb_key)
        if target is prev_n:
            target[1] = e      # 前音延伸，蓋掉過渡碎片
        else:
            target[0] = s      # 後音提前開始
        del notes[best_i]

    merged = []
    for s, e, p in notes:
        if merged and p == merged[-1][2] and s - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], e, p)
        else:
            merged.append((s, e, p))
    return merged


def extract_melody_notes_rmvpe(
    y,
    sr,
    fmin_note="C2",
    fmax_note="C6",
    min_dur=0.025,
    gap_tolerance=0.09,
    voiced_prob_thresh=0.03,
    octave_shift=0,
    pitch_persistence_sec=0.045,
    medfilt_kernel_size=5,
    min_note_dur=0.070,
    rap_revise=True,
    name="主唱",
    device=None,
    model_path=None,
    is_half=False,
    verbose_progress=True,
):
    """RMVPE → F0 → note segmentation。

    RMVPE 原生以 16 kHz / 10 ms 左右的時間解析度輸出 F0；這裡不再
    使用 pYIN 的 voiced_flag / vprob，而是使用 RMVPE 的 0-F0 及
    salience threshold，然後做短缺口補齊（nearest 填補，避免線性
    內插製造假滑音）、median smoothing、persistence note
    segmentation，最後把過渡產生的短碎片音符吸附進鄰近長音
    （滑音抑制）。
    """
    import librosa
    from scipy.ndimage import median_filter

    if y is None or len(y) == 0:
        return []

    print(f"{name} RMVPE 音高追蹤 ...")
    model = _get_rmvpe_model(
        device=device,
        model_path=model_path,
        is_half=is_half,
    )

    # RMVPE 模型要求 16 kHz 音訊。
    if sr != 16000:
        y16 = librosa.resample(
            np.asarray(y, dtype=np.float32),
            orig_sr=sr,
            target_sr=16000,
        )
    else:
        y16 = np.asarray(y, dtype=np.float32)

    # rmvpe-onnx API 回傳 time / frequency / confidence / activation。
    times, frequency, confidence, _activation = model.predict(
        audio=y16,
        sr=16000,
    )
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    f0 = np.asarray(frequency, dtype=np.float32).reshape(-1)
    confidence = np.asarray(confidence, dtype=np.float32).reshape(-1)

    # API 會回傳原始 F0，不會自動套 confidence threshold；這裡自己套。
    n = min(len(times), len(f0), len(confidence))
    times = times[:n]
    f0 = f0[:n]
    confidence = confidence[:n]

    # 以 RMVPE 實際回傳的時間軸計算 frame 間隔，避免硬編碼 hop_sec。
    # 正常約為 10 ms；若只回傳單一 frame，退回 10 ms。
    if len(times) >= 2:
        diffs = np.diff(times)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        hop_sec = float(np.median(diffs)) if len(diffs) else 0.01
    else:
        hop_sec = 0.01

    f0[confidence < float(voiced_prob_thresh)] = 0.0
    valid = np.isfinite(f0) & (f0 > 0.0)

    # 額外限制音域，避免 RMVPE 偶爾在無聲/子音位置產生極端 F0。
    fmin = float(librosa.note_to_hz(fmin_note))
    fmax = float(librosa.note_to_hz(fmax_note))
    valid &= (f0 >= fmin) & (f0 <= fmax)

    midi = np.full(len(f0), np.nan, dtype=np.float64)
    midi[valid] = librosa.hz_to_midi(f0[valid])

    # 只補短缺口；長空白保持空白，避免 Rap/歌詞停頓被上一音拖過去。
    # 填補方式用 nearest（前後各半 hold）：原本的線性內插會在兩音
    # 之間製造連續坡道，量化後正是「滑音」碎音的來源之一。
    filled = midi.copy()
    valid_idx = np.flatnonzero(np.isfinite(filled))
    max_gap = max(0, int(round(float(gap_tolerance) / hop_sec)))
    if len(valid_idx) >= 2 and max_gap:
        for a, b in zip(valid_idx[:-1], valid_idx[1:]):
            gap = b - a - 1
            if 0 < gap <= max_gap:
                half = gap // 2
                if half:
                    filled[a + 1:a + 1 + half] = filled[a]
                filled[a + 1 + half:b] = filled[b]

    # 中值濾波只作用在有 F0 的區段，避免整段靜音被填值後產生假音符。
    smooth = filled.copy()
    finite = np.isfinite(smooth)
    kernel = max(1, int(medfilt_kernel_size))
    if kernel % 2 == 0:
        kernel += 1
    if finite.any() and kernel > 1:
        tmp = smooth.copy()
        tmp[~finite] = np.nanmedian(smooth[finite])
        filt = median_filter(tmp, size=kernel, mode="nearest")
        smooth[finite] = filt[finite]

    # ------------------------------------------------------------
    # 人聲 onset 偵測（完全自動，不需要每首歌另外調參數）
    # ------------------------------------------------------------
    # Rap 的字頭子音/爆破音通常有清楚的能量起振，但音高本身很短、
    # 抖動大，容易被下面的音高穩定度判斷擋下來，導致整串音節被
    # 吃掉。這裡用能量 onset 偵測標出這些起振點，segmentation 在
    # 這些點附近會放寬「音高要先穩定」的要求，讓短促音節也能成立。
    try:
        onset_times_sec = librosa.onset.onset_detect(
            y=np.asarray(y, dtype=np.float32),
            sr=sr,
            backtrack=True,
            units="time",
            hop_length=512,
            delta=0.04,
        )
    except Exception:
        onset_times_sec = np.array([], dtype=np.float64)

    onset_boost = np.zeros(len(times), dtype=bool)
    if len(onset_times_sec):
        onset_window = max(hop_sec, 0.025)
        oi = 0
        n_onsets = len(onset_times_sec)
        for i, t in enumerate(times):
            while oi < n_onsets - 1 and onset_times_sec[oi] < t - onset_window:
                oi += 1
            if abs(onset_times_sec[oi] - t) <= onset_window:
                onset_boost[i] = True

    # ------------------------------------------------------------
    # 自適應 Rap / 密集人聲 segmentation
    # ------------------------------------------------------------
    # rap_revise=True 時，快速音節會降低新音高所需的 persistence，
    # 避免 Rap 的短音節被一般歌唱用的 45~55ms 門檻吃掉。
    # 關閉後恢復固定 persistence，方便 A/B 比較。
    # 不把 persistence 固定成單一值：先從局部 F0 的變化速度估計
    # 人聲是否處於快速音符區，再在每一個 pitch change 上決定需要
    # 幾個連續 frame 才接受新音高。
    #
    # slow  : 40~50 ms，抑制一般歌唱的小抖動
    # medium: 25~35 ms
    # fast  : 10~20 ms，避免 Rap 的短音被 persistence 吃掉
    # ------------------------------------------------------------
    pitch_delta = np.zeros(len(smooth), dtype=np.float64)
    finite_idx = np.flatnonzero(np.isfinite(smooth))
    if len(finite_idx) >= 2:
        d = np.abs(np.diff(smooth[finite_idx]))
        pitch_delta[finite_idx[1:]] = d

    # 用約 300 ms 的局部窗口計算「音高變化密度」。
    # 每秒越多次明顯 pitch change，就越接近 Rap / 快速唱法。
    density_window = max(3, int(round(0.30 / hop_sec)))
    change_mask = pitch_delta >= 1.0
    change_count = np.convolve(
        change_mask.astype(np.float32),
        np.ones(density_window, dtype=np.float32),
        mode="same",
    )
    local_density = change_count / max(0.30, density_window * hop_sec)

    # 連續密度自適應（取代原本的 3 段離散門檻）。
    # Rap 區不能把門檻降得太低，否則 RMVPE 的連續滑動 F0 會被
    # 切成一串相鄰半音，聽起來反而像「滑音」——所以下限仍保留。
    base_persistence = max(0.020, float(pitch_persistence_sec))
    if rap_revise:
        slow_persistence = max(0.040, base_persistence)
        fast_persistence = min(slow_persistence, max(0.024, base_persistence * 0.55))
    else:
        # A/B 測試用：完全停用 Rap 自適應，只使用原本的固定 persistence。
        slow_persistence = base_persistence
        fast_persistence = base_persistence

    _density_lo, _density_hi = 2.0, 8.0

    def adaptive_persistence_at(index):
        """依局部音高變化密度連續調整 persistence，並在偵測到人聲
        onset 時再進一步縮短。完全自動，不需要為任何段落手動填
        參數——全部由 F0 與能量 onset 即時決定，且用 smoothstep
        內插避免密度在門檻邊界抖動造成音符切分忽快忽慢。"""
        density = float(local_density[index]) if len(local_density) else 0.0
        t = (density - _density_lo) / (_density_hi - _density_lo)
        t = float(np.clip(t, 0.0, 1.0))
        t = t * t * (3.0 - 2.0 * t)  # smoothstep
        persistence = slow_persistence * (1.0 - t) + fast_persistence * t

        if rap_revise and index < len(onset_boost) and onset_boost[index]:
            # Rap 字頭有明確能量起振但音高很短、抖動大，容易被
            # 穩定度門檻擋下來整串漏掉；偵測到 onset 時大幅縮短
            # 所需 persistence，讓音符能立刻成立。
            persistence = min(persistence, max(hop_sec * 1.5, persistence * 0.35))
        return persistence

    def candidate_is_stable(index, pitch):
        """確認新音高已經穩定，避免把滑音 transition 當成 note。

        人聲 onset 附近（Rap 字頭最明顯）放寬穩定度要求：onset 本身
        已經是很強的「這裡開始新音節」證據，不必再等音高完全穩定，
        否則短促的 Rap 音節會被這個門檻整串濾掉。
        """
        persistence = adaptive_persistence_at(index)
        stable_frames = max(2, int(round(persistence / hop_sec)))
        near_onset = rap_revise and index < len(onset_boost) and onset_boost[index]

        left = max(0, index - stable_frames + 1)
        segment = smooth[left:index + 1]
        segment = segment[np.isfinite(segment)]
        min_needed = max(2, stable_frames // 2)
        if near_onset:
            min_needed = min(min_needed, 2)
        if len(segment) < min_needed:
            return False

        rounded = np.rint(segment).astype(np.int32)
        same_ratio = float(np.mean(rounded == int(pitch)))
        ratio_thresh = 0.45 if near_onset else 0.60
        if same_ratio < ratio_thresh:
            return False

        center = float(np.median(segment))
        tol = 0.45 if near_onset else 0.30
        return abs(center - float(pitch)) <= tol

    def find_stable_start(index, pitch):
        """往前尋找新音高真正穩定的位置，切掉兩音之間的滑行區。"""
        search_frames = max(2, int(round(0.080 / hop_sec)))
        start_i = max(0, index - search_frames)
        target = int(pitch)

        # 從後往前找最後一個仍然偏離 target 的 frame。
        # 新 note 從它後面開始，因此 transition 不會被歸給新 note。
        stable_i = index
        for j in range(index, start_i - 1, -1):
            if not np.isfinite(smooth[j]):
                break
            if abs(float(smooth[j]) - target) > 0.30:
                break
            stable_i = j

        return stable_i

    notes = []
    current = None
    start = None
    last_t = None
    pending = None
    pending_count = 0
    needed = max(1, int(round(slow_persistence / hop_sec)))

    def flush(end_t):
        nonlocal current, start
        if current is None or start is None:
            return
        end_t = max(float(end_t), float(start))
        if end_t - start >= float(min_dur):
            notes.append((
                float(start),
                end_t,
                int(round(current)) + int(octave_shift) * 12,
            ))
        current = None
        start = None

    total = len(times)
    last_percent = -1
    for i, t in enumerate(times):
        if verbose_progress and total:
            pct = int(i / total * 100)
            if pct != last_percent:
                show_progress(i, total, prefix=f"{name} RMVPE")
                last_percent = pct

        if valid[i] and np.isfinite(smooth[i]):
            p = int(round(smooth[i]))
            if current is None:
                current, start = p, float(t)
                pending, pending_count = None, 0
            elif p == current:
                pending, pending_count = None, 0
            else:
                if pending == p:
                    pending_count += 1
                else:
                    pending, pending_count = p, 1

                # 每次 pitch change 都重新判斷目前的人聲密度。
                # Rap 區縮短 persistence；一般歌唱區維持較保守的門檻。
                adaptive_persistence = adaptive_persistence_at(i)
                adaptive_needed = max(
                    1,
                    int(round(adaptive_persistence / hop_sec)),
                )

                if pending_count >= adaptive_needed and candidate_is_stable(i, p):
                    stable_i = find_stable_start(i, p)
                    change_t = float(times[stable_i])
                    old = current
                    old_start = start

                    # 舊 note 在 transition 開始前結束；transition 本身留白，
                    # 避免 NBS 把連續 F0 聽成從舊音滑到新音。
                    transition_guard = min(
                        0.030,
                        max(0.0, change_t - old_start) * 0.35,
                    )
                    old_end = max(old_start, change_t - transition_guard)

                    if old_end - old_start >= float(min_dur):
                        notes.append((
                            old_start,
                            old_end,
                            int(round(old)) + int(octave_shift) * 12,
                        ))

                    new_start = change_t + min(0.010, hop_sec)
                    current, start = p, new_start
                    pending, pending_count = None, 0
            last_t = float(t)
        elif current is not None and last_t is not None:
            if float(t) - last_t > float(gap_tolerance):
                flush(last_t + hop_sec)
                pending, pending_count, last_t = None, 0, None

    if verbose_progress:
        show_progress(total, total, prefix=f"{name} RMVPE")
    if current is not None and last_t is not None:
        flush(last_t + hop_sec)

    # 第一層：合併相鄰同音碎片（gap <= 20ms），並丟棄 < min_dur 的音。
    merged = []
    for s, e, p in notes:
        if merged and p == merged[-1][2] and s - merged[-1][1] <= 0.02:
            merged[-1] = (merged[-1][0], e, p)
        elif e - s >= float(min_dur):
            merged.append((s, e, p))

    # 第二層：滑音抑制核心 —— 把過渡產生的短碎片吸附進鄰近長音，
    # 讓「長音 + 2~3 個滑行碎音」還原成單一音符。
    absorbed = _absorb_short_notes(
        merged,
        min_len=float(min_note_dur),
        merge_gap=0.05,
    )

    print(
        f"{name}：RMVPE 提取 {len(merged)} 個音符，"
        f"滑音吸附後 {len(absorbed)} 個"
    )
    return absorbed


# ============================================================
# Other 多音符分析
# ============================================================

def extract_other_notes(
    y_other: np.ndarray,
    sr: int,
    y_harm: np.ndarray | None = None,
    hop_length: int = 256,
    min_note_dur: float = 0.08,
    energy_threshold: float = 0.20,
    max_notes_per_frame: int = 1,
    min_note_separation: float = 0.12,
    octave_shift: int = 0,
    cqt_fmin: str = "C2",
    cqt_fmax: str = "C8",
    peak_min_separation: int = 2,
    harmonic_tolerance: float = 0.45,
    harmonic_count: int = 8,
    harmonic_suppression: float = 0.65,
    track_max_gap: float = 0.06,
    no_filter: bool = False,
    confidence_threshold: float = 0.18,
):
    """
    Other 樂器簡化轉譜。

    目標不是把 Other 所有音符全部抓出來，
    而是只留下少量、比較穩定的背景音。

    適合：
      Piano
      Guitar
      Strings
      Synth
      Pad
    """

    import librosa

    print(
        "Other 樂器簡化音符分析 ..."
    )

    # --------------------------------------------------------
    # HPSS：只取 harmonic
    # --------------------------------------------------------

    print(
        "  Other → HPSS ..."
    )

    if y_harm is None:
        y_harm, _ = librosa.effects.hpss(
            y_other,
            margin=(1.0, 3.0),
        )
        print(
            "  Other → HPSS 完成"
        )
    else:
        print(
            "  Other → 使用已快取的 harmonic stem"
        )

    # --------------------------------------------------------
    # CQT
    # --------------------------------------------------------

    print(
        "  Other → CQT 分析 ..."
    )

    cqt_bins_per_octave = 12
    cqt_fmin_hz = librosa.note_to_hz(cqt_fmin)
    cqt_fmax_hz = librosa.note_to_hz(cqt_fmax)
    if cqt_fmin_hz <= 0 or cqt_fmax_hz <= cqt_fmin_hz:
        raise ValueError(
            f"Other CQT 音域無效：{cqt_fmin} ~ {cqt_fmax}"
        )
    n_bins = max(
        12,
        int(np.floor(
            cqt_bins_per_octave
            * np.log2(cqt_fmax_hz / cqt_fmin_hz)
        )) + 1,
    )

    C = np.abs(
        librosa.cqt(
            y_harm,
            sr=sr,
            hop_length=hop_length,
            fmin=cqt_fmin_hz,
            n_bins=n_bins,
            bins_per_octave=cqt_bins_per_octave,
        )
    )

    print(
        "  Other → CQT 完成"
    )

    if C.size == 0:

        print(
            "Other：沒有 CQT 資料"
        )

        return []

    # --------------------------------------------------------
    # 正規化
    # --------------------------------------------------------

    # 不再逐 frame 用自己的最大值正規化。
    # 舊方法會讓「幾乎只有噪聲的 frame」也被放大到 max=1，
    # 導致 energy_threshold 失去意義；對鋼琴尤其容易把殘響/泛音抓成音符。
    # 改成整段 CQT 的 robust global scale，讓 threshold 真正代表相對能量。
    cqt_scale = float(np.percentile(C, 99.0))
    if not np.isfinite(cqt_scale) or cqt_scale <= 1e-9:
        cqt_scale = float(np.max(C)) if np.max(C) > 1e-9 else 1.0
    C_norm = C / (cqt_scale + 1e-9)
    C_norm = np.clip(C_norm, 0.0, 1.5)

    # 鋼琴泛音抑制：高頻 peak 若能由較低頻 fundamental 的整數倍解釋，
    # 優先視為同一個音的 harmonic，而不是新的獨立音符。
    # tolerance 以半音計，避免 CQT bin 對不準時誤殺真正音符。
    harmonic_tolerance = max(0.0, float(harmonic_tolerance))
    harmonic_count = max(2, int(harmonic_count))
    harmonic_suppression = float(np.clip(harmonic_suppression, 0.0, 1.0))

    times = librosa.frames_to_time(
        np.arange(C.shape[1]),
        sr=sr,
        hop_length=hop_length,
    )

    midi_start = int(round(librosa.hz_to_midi(cqt_fmin_hz)))
    midi_numbers = midi_start + np.arange(C.shape[0])

    # ========================================================
    # 第一階段：
    # 每一幀選前 N 個可信音符。
    # max_notes_per_frame=1 時維持舊的單音模式；
    # 2~4 時可以讓鋼琴的和弦/雙音更容易留下來。
    # ========================================================

    print(
        "  Other → 音高追蹤 ..."
    )

    frame_pitches = []
    # 前一幀的候選音高，用來讓多音符 peak 選擇具有時間連續性。
    # 沒有這層時，CQT 每幀只看「當下誰最大」，鋼琴長音很容易在
    # 基頻 / 泛音之間來回跳，最後變成大量碎 note。
    previous_selected = []

    total_frames = C.shape[1]
    last_percent = -1

    for frame in range(
        total_frames
    ):

        # 進度
        if (
            total_frames > 0
            and (
                frame == 0
                or frame == total_frames - 1
                or frame % max(
                    1,
                    total_frames // 100
                ) == 0
            )
        ):

            percent = int(
                frame / total_frames * 100
            )

            if percent != last_percent:

                show_progress(
                    frame,
                    total_frames,
                    prefix="Other 音高追蹤",
                )

                last_percent = percent

        spectrum = C_norm[
            :,
            frame
        ]

        if (
            spectrum.max()
            < energy_threshold
        ):

            frame_pitches.append(
                None
            )
            # 真正的低能量 frame 不應該把 continuity 記憶永久帶過去。
            previous_selected = []
            continue

        # ----------------------------------------------------
        # 找 local peak
        # ----------------------------------------------------

        peaks = []

        for i in range(
            1,
            len(spectrum) - 1,
        ):

            if (
                spectrum[i]
                >= spectrum[i - 1]
                and spectrum[i]
                >= spectrum[i + 1]
                and spectrum[i]
                >= energy_threshold
            ):

                peaks.append(i)

        if not peaks:

            frame_pitches.append(
                None
            )
            previous_selected = []
            continue

        # ----------------------------------------------------
        # 排序 + 多峰選擇 + 泛音抑制
        #
        # 鋼琴最麻煩的地方：一個真正的 C4 同時會在 C5/G5/C6...
        # 產生很強的泛音峰。如果直接取最高的 N 個 peak，
        # max_notes_per_frame 越大反而越容易把泛音當成獨立音符。
        #
        # 這裡對每個候選 peak 做 harmonic support 檢查：
        # 若較低音高的候選可以解釋目前峰值的大部分能量，
        # 就降低目前峰的獨立音符分數。真正的和弦音若有自己的
        # 基頻能量，通常仍會勝過單純泛音。
        # ----------------------------------------------------
        # 鋼琴泛音預處理：先把能被更低頻峰值解釋的候選標記為弱候選。
        # 不直接刪除，避免真正高音基頻被誤殺；後面的 max_keep 會自然優先選擇
        # 有較強基頻支持的音。這比單純取最高 N 個 peak 穩定。
        if harmonic_suppression > 0 and len(peaks) > 1:
            peak_strength = {int(p): float(spectrum[p]) for p in peaks}
            harmonic_limit = max(2, min(16, int(harmonic_count)))
            tolerance = max(0.05, float(harmonic_tolerance))
            harmonic_penalty = {}
            for p in peaks:
                pm = float(midi_numbers[p])
                penalty = 0.0
                for q in peaks:
                    if q >= p:
                        continue
                    qm = float(midi_numbers[q])
                    ratio = 2.0 ** ((pm - qm) / 12.0)
                    h = int(round(ratio))
                    if h < 2 or h > harmonic_limit:
                        continue
                    cents = abs(12.0 * np.log2(max(ratio, 1e-9) / h))
                    if cents > tolerance:
                        continue
                    q_strength = peak_strength[q]
                    p_strength = peak_strength[p]
                    if q_strength >= p_strength * 0.15:
                        penalty = max(penalty, min(1.0, q_strength / (p_strength + 1e-9) / np.sqrt(h)))
                harmonic_penalty[p] = penalty
            peaks.sort(
                key=lambda i: peak_strength[i] * (1.0 - float(harmonic_suppression) * harmonic_penalty.get(i, 0.0)),
                reverse=True,
            )
        else:
            peaks.sort(key=lambda i: spectrum[i], reverse=True)
        max_keep = max(1, int(max_notes_per_frame))

        def harmonic_support_score(p):
            """估計某 peak 有多少能量可以由更低的基頻/泛音解釋。"""
            strength = float(spectrum[p])
            if strength <= 0:
                return 0.0

            midi = float(midi_numbers[p])
            count = max(2, min(16, int(harmonic_count)))
            tolerance = max(0.05, float(harmonic_tolerance))
            support = 0.0

            # 若目前峰值接近某個較低音的第 2~N 階泛音，
            # 就把它視為「可能是泛音」，而不是新的基頻。
            for harmonic in range(2, count + 1):
                fundamental_midi = midi - 12.0 * np.log2(harmonic)
                fp = int(round(fundamental_midi - midi_start))
                if fp < 0 or fp >= len(spectrum):
                    continue
                actual_midi = midi_start + fp
                cents = abs(actual_midi - fundamental_midi)
                if cents > tolerance:
                    continue
                fundamental_strength = float(spectrum[fp])
                if fundamental_strength <= strength * 0.05:
                    continue
                ratio = min(1.0, fundamental_strength / (strength + 1e-9))
                # 低階泛音比高階泛音更有解釋力。
                support += ratio / np.sqrt(harmonic)

            return min(3.0, support)

        candidates = []
        for p in peaks:
            midi = int(midi_numbers[p])
            strength = float(spectrum[p])

            if any(
                abs(midi - int(midi_numbers[q])) < peak_min_separation
                for q in [c[0] for c in candidates]
            ):
                continue

            harmonic_support = harmonic_support_score(p)
            # 只做可調的軟抑制，不直接刪除；真正的高音若有足夠基頻
            # 仍然可以保留下來。
            suppression = max(0.0, min(1.0, float(harmonic_suppression)))
            # 有明顯低頻 fundamental 支持的 peak 優先視為泛音；
            # 抑制採非線性曲線，讓 piano 的 2~4 階泛音在 max_notes=4 時
            # 不會輕易擠掉真正的其他和弦音。
            harmonic_factor = min(0.95, harmonic_support / 1.5)
            score = strength * (1.0 - suppression * harmonic_factor)
            candidates.append((p, score, strength, harmonic_support))

        # ----------------------------------------------------
        # 時間連續性：優先保留上一幀已經存在的音高。
        #
        # 不是硬鎖 pitch，而是給 continuity bonus：
        #   同 pitch       → 強烈優先
        #   ±1 半音        → 小幅優先（CQT 量化/漂移）
        #   新音符         → 仍可依自己的 spectrum score 進入
        #
        # 這樣 chord 改變時新音仍然能進來，不會變成「永遠黏住舊音」。
        # ----------------------------------------------------
        def continuity_score(p):
            if not previous_selected:
                return 0.0
            distances = [abs(int(midi_numbers[p]) - int(old)) for old in previous_selected]
            d = min(distances)
            if d == 0:
                return 0.22
            if d == 1:
                return 0.07
            if d == 2:
                return 0.025
            return 0.0

        candidates.sort(
            key=lambda x: x[1] + continuity_score(x[0]),
            reverse=True,
        )
        selected = [c[0] for c in candidates[:max_keep]]

        if not selected:
            frame_pitches.append(None)
        elif max_keep == 1:
            frame_pitches.append(
                int(midi_numbers[selected[0]]) + octave_shift * 12
            )
        else:
            selected_midi = [
                int(midi_numbers[p]) + octave_shift * 12
                for p in selected
            ]
            frame_pitches.append(selected_midi)

        # 保存這一幀真正選中的音高，供下一幀 continuity scoring 使用。
        if selected:
            previous_selected = [
                int(midi_numbers[p]) + octave_shift * 12
                for p in selected
            ]
        else:
            previous_selected = []

    show_progress(
        total_frames,
        total_frames,
        prefix="Other 音高追蹤",
    )

    # ========================================================
    # 第二階段：
    # 相同音高連續幀合併
    # ========================================================

    print(
        "  Other → 音符合併 ..."
    )

    raw_notes = []

    current_pitch = None
    start_frame = None
    last_frame = None

    def flush():

        nonlocal current_pitch
        nonlocal start_frame
        nonlocal last_frame

        if (
            current_pitch is None
            or start_frame is None
            or last_frame is None
        ):

            return

        start_time = times[
            start_frame
        ]

        end_time = (
            times[last_frame]
            + hop_length / sr
        )

        if no_filter or (
            end_time
            - start_time
            >= min_note_dur
        ):

            raw_notes.append(
                (
                    float(start_time),
                    float(end_time),
                    int(current_pitch),
                )
            )

        current_pitch = None
        start_frame = None
        last_frame = None

    total_frames = len(frame_pitches)
    primary_frame_pitches = [
        (p[0] if isinstance(p, (list, tuple)) and p else p)
        for p in frame_pitches
    ]

    last_percent = -1

    for frame, pitch in enumerate(primary_frame_pitches):

        if (
            total_frames > 0
            and (
                frame == 0
                or frame == total_frames - 1
                or frame % max(
                    1,
                    total_frames // 100
                ) == 0
            )
        ):

            percent = int(
                frame / total_frames * 100
            )

            if percent != last_percent:

                show_progress(
                    frame,
                    total_frames,
                    prefix="Other 音符合併",
                )

                last_percent = percent

        if pitch is None:
            flush()
            continue

        if current_pitch is None:
            current_pitch = int(pitch)
            start_frame = frame
            last_frame = frame
            continue

        p = int(pitch)
        if p == current_pitch:
            last_frame = frame
            continue

        flush()
        current_pitch = p
        start_frame = frame
        last_frame = frame

    flush()

    # max_notes_per_frame > 1：使用「每音高獨立 track」而不是把
    # secondary peak 當成一條永遠跟著 frame 的清單。
    # 每個 pitch track 允許短暫掉音，並優先連接到最近的 pitch，
    # 能明顯減少鋼琴和弦在不同 frame 間抖成大量碎音。
    if max(1, int(max_notes_per_frame)) > 1:
        # 使用真正的 track ID，而不是 dict[pitch]。
        # 同一時間/相鄰時間可能有兩個相同 pitch 的聲部，
        # 用 pitch 當 key 會互相覆蓋；track ID 可以讓每個聲部獨立延續。
        tracks = []
        secondary_notes = []
        max_gap_frames = max(
            1,
            int(round(max(0.0, track_max_gap) / (hop_length / sr))),
        )
        max_keep = max(1, int(max_notes_per_frame)) - 1

        def close_track(state):
            s = float(times[state["start"]])
            e = float(times[state["last"]] + hop_length / sr)
            if no_filter or e - s >= min_note_dur:
                secondary_notes.append((s, e, int(state["pitch"])))

        for frame, value in enumerate(frame_pitches):
            selected = value if isinstance(value, (list, tuple)) else []
            # secondary track 仍只接手 primary 之外的候選，但泛音判斷會把
            # primary 一起納入參考，避免「primary=C4、secondary=C5」時
            # secondary 看不到 C4 而錯把 C5 當獨立音符。
            all_selected = list(dict.fromkeys(int(p) for p in selected))
            # primary 也必須參與泛音判斷，但不能把 primary 本身送進
            # secondary track。舊版只用 pitches 當 base，會讓
            # primary=C4、secondary=C5 時看不到 C4，因而把 C5 泛音誤當獨立音。
            primary_pitch = all_selected[0] if all_selected else None
            pitches = all_selected[1:1 + max_keep]

            # 同一 frame 內若高音能被「primary 或更低的 secondary」的
            # 整數倍泛音解釋，降低它成為獨立 note 的優先級；
            # 真正的和弦音仍可保留，因為這裡只是軟性過濾。
            if pitches and harmonic_suppression > 0:
                filtered = []
                bases = ([primary_pitch] if primary_pitch is not None else []) + [
                    p for p in pitches if p != primary_pitch
                ]
                for pitch in pitches:
                    freq = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
                    is_harmonic = False
                    for base_pitch in bases:
                        if base_pitch >= pitch:
                            continue
                        base_freq = 440.0 * (2.0 ** ((base_pitch - 69) / 12.0))
                        ratio = freq / max(base_freq, 1e-9)
                        h = int(round(ratio))
                        cents = abs(12.0 * np.log2(max(ratio, 1e-9) / max(h, 1))) if h > 0 else 999.0
                        if 2 <= h <= harmonic_count and cents <= harmonic_tolerance:
                            is_harmonic = True
                            break
                    if not is_harmonic:
                        filtered.append(pitch)
                pitches = filtered

            # 先依 pitch 距離做一對一 greedy matching。
            # CQT 已經量化到半音；這裡採「同 pitch 才延續」的保守策略。
            # 不把 ±1/±2 半音直接串成同一顆 note，避免真正的旋律變化
            # 被錯誤拉成超長音。track_max_gap 只負責處理短暫掉幀。
            unmatched_tracks = [
                j for j, state in enumerate(tracks)
                if frame - state["last"] <= max_gap_frames
            ]
            used_tracks = set()
            matches = []
            for p in pitches:
                best = None
                for j in unmatched_tracks:
                    if j in used_tracks:
                        continue
                    old = tracks[j]["pitch"]
                    distance = abs(old - p)
                    if distance == 0 and (best is None or distance < best[0]):
                        best = (distance, j)
                if best is not None:
                    matches.append((p, best[1]))
                    used_tracks.add(best[1])

            matched_pitches = set()
            for p, j in matches:
                tracks[j]["last"] = frame
                tracks[j]["pitch"] = p
                matched_pitches.add(p)

            # 未匹配的 pitch 建立新 track。
            for p in pitches:
                if p in matched_pitches:
                    continue
                tracks.append({
                    "start": frame,
                    "last": frame,
                    "pitch": p,
                })

            # 太久沒出現的 track 結束。
            keep_tracks = []
            for state in tracks:
                if frame - state["last"] > max_gap_frames:
                    close_track(state)
                else:
                    keep_tracks.append(state)
            tracks = keep_tracks

        for state in tracks:
            close_track(state)

        raw_notes.extend(secondary_notes)

    show_progress(
        total_frames,
        total_frames,
        prefix="Other 音符合併",
    )

    # ========================================================
    # 可選：完全關閉「時間稀疏化 + 音符過濾」
    #
    # no_filter 時仍保留 CQT 的基本峰值/有效音高判定，
    # 但不再用 min_note_separation、min_note_dur、音域限制
    # 把已抓到的音符刪掉。
    # ========================================================

    if no_filter:
        result = sorted(
            [
                (float(s), float(e), int(p))
                for s, e, p in raw_notes
                if np.isfinite(s) and np.isfinite(e) and np.isfinite(p)
            ],
            key=lambda x: (x[0], x[2]),
        )
        print(
            f"Other：no-filter 模式，保留 {len(result)} 個原始音符"
        )
        return result

    # ========================================================
    # 第三階段：
    # 時間稀疏化
    #
    # 防止：
    #
    # C4 C4 C4 C4 E4 E4 G4...
    #
    # 變成一堆密集音符。
    # ========================================================

    print(
        "  Other → 時間稀疏化 ..."
    )

    filtered = []

    total_notes = len(
        raw_notes
    )

    last_percent = -1

    for index, note in enumerate(
        raw_notes
    ):

        if (
            total_notes > 0
            and (
                index == 0
                or index == total_notes - 1
                or index % max(
                    1,
                    total_notes // 100
                ) == 0
            )
        ):

            percent = int(
                index / total_notes * 100
            )

            if percent != last_percent:

                show_progress(
                    index,
                    total_notes,
                    prefix="Other 時間稀疏化",
                )

                last_percent = percent

        start, end, pitch = note[:3]

        if not filtered:

            filtered.append(
                note
            )

            continue

        prev = filtered[-1]

        # ----------------------------------------------------
        # 如果新音符距離上一個音太近
        # ----------------------------------------------------

        if (
            start - prev[1]
            < min_note_separation
        ):

            # 如果是同一個音，直接合併
            if pitch == prev[2]:

                filtered[-1] = (
                    prev[0],
                    max(
                        prev[1],
                        end,
                    ),
                    prev[2],
                )

            # 不同音：
            # 保留持續時間比較長的
            elif (
                end - start
                > prev[1] - prev[0]
            ):

                filtered[-1] = note

            continue

        filtered.append(
            note
        )

    show_progress(
        total_notes,
        total_notes,
        prefix="Other 時間稀疏化",
    )

    # ========================================================
    # 第四階段：
    # 過濾過高/過低的可疑音
    # ========================================================

    final_notes = []

    total_notes = len(
        filtered
    )

    last_percent = -1

    for index, note in enumerate(filtered):
        start, end, pitch = note[:3]

        if (
            total_notes > 0
            and (
                index == 0
                or index == total_notes - 1
                or index % max(
                    1,
                    total_notes // 100
                ) == 0
            )
        ):

            percent = int(
                index / total_notes * 100
            )

            if percent != last_percent:

                show_progress(
                    index,
                    total_notes,
                    prefix="Other 音符過濾",
                )

                last_percent = percent

        duration = (
            end - start
        )

        # 太短直接丟掉
        if duration < min_note_dur:
            continue

        # Other 背景通常不需要極端音域
        if pitch < 40:
            continue

        if pitch > 90:
            continue

        # ----------------------------------------------------
        # Confidence：把「音量、持續時間、是否接近穩定長音」
        # 合併成 0~1 的可信度。
        #
        # CQT peak 本身不是絕對可靠的樂器辨識，因此這裡不把
        # confidence 當成「這一定是正確音符」，只拿來：
        #   1. 丟掉非常弱的候選
        #   2. 把較不確定的音符轉成較低 velocity
        #
        # 這比單純把所有 CQT peak 等音量輸出自然很多。
        # ----------------------------------------------------
        frame_index = int(
            np.clip(
                round(((start + end) * 0.5) * sr / hop_length),
                0,
                C_norm.shape[1] - 1,
            )
        )
        cqt_index = int(
            np.clip(
                pitch - (midi_start + octave_shift * 12),
                0,
                C_norm.shape[0] - 1,
            )
        )
        local_energy = float(C_norm[cqt_index, frame_index])
        energy_conf = float(np.clip(local_energy / max(0.45, energy_threshold), 0.0, 1.0))
        duration_conf = float(np.clip(duration / 0.20, 0.0, 1.0))
        confidence = 0.72 * energy_conf + 0.28 * duration_conf

        if confidence < float(confidence_threshold):
            continue

        # 可信度映射到 NBS velocity：強音保持存在感，
        # 弱音不直接消失，而是降低存在感。
        velocity = int(round(32 + 60 * confidence))

        final_notes.append(
            (
                start,
                end,
                pitch,
                _clip(velocity, 24, 92),
            )
        )

    show_progress(
        total_notes,
        total_notes,
        prefix="Other 音符過濾",
    )

    print(
        f"Other："
        f"簡化後 {len(final_notes)} 個音符"
    )

    return final_notes


# ============================================================
# Other 音符抽取（神經網路版本）
#
# CQT + peak picking 對重疊聲部（尤其是有人聲同時存在時）
# 效果不穩定：容易把泛音當成獨立音符，或在人聲蓋住背景樂器時
# 直接偵測不到，觸發 hybrid/mix 模式的和弦生成 fallback，
# 聽起來像「程序生成」的空虛伴奏。
#
# 這裡改用 Onsets & Frames（Google Magenta）TFLite 模型做
# 複音轉譜，準確度通常遠高於 CQT peak-picking，
# 回傳格式對齊 extract_other_notes()：(start, end, midi_pitch, velocity)。
# ============================================================

def _get_of_model(model_path=None):
    if _of is None:
        raise RuntimeError(
            "找不到 onsets_frames_to_nbs.py，"
            "請確認它與 mp3_to_nbs.py 放在同一個資料夾。"
        )

    cache_key = model_path or "__default__"
    if cache_key in _OF_MODEL_CACHE:
        return _OF_MODEL_CACHE[cache_key]

    base = os.path.dirname(os.path.abspath(__file__))
    path = model_path or os.path.join(
        base, "models", "onsets_frames_uni.tflite"
    )

    if not os.path.isfile(path):
        _of.download_model(path)

    print(f"  Other(neural) → 載入 Onsets & Frames 模型：{path}")
    model = _of.OFModel(path)
    _OF_MODEL_CACHE[cache_key] = model
    return model


def _limit_neural_polyphony(
    notes,
    max_polyphony=4,
    onset_group_window=0.055,
):
    """限制 O&F 在複雜 Other stem 上產生的過度多音。"""
    if not notes or max_polyphony <= 0:
        return notes

    notes = sorted(notes, key=lambda x: (x[0], -x[3], x[2]))
    kept = []
    active = []
    window = max(0.0, float(onset_group_window))
    max_polyphony = max(1, int(max_polyphony))

    for note in notes:
        s, e, p, v = note
        active = [a for a in active if a[1] > s]
        group = [a for a in active if abs(a[0] - s) <= window]

        if len(group) < max_polyphony:
            kept.append(note)
            active.append(note)
            continue

        weakest = min(group, key=lambda x: (x[3], x[2]))
        if v > weakest[3]:
            try:
                idx = kept.index(weakest)
            except ValueError:
                idx = -1
            if idx >= 0:
                kept[idx] = note
                active.remove(weakest)
                active.append(note)

    kept.sort(key=lambda x: (x[0], x[2]))
    return kept


def _postprocess_neural_notes(
    notes,
    min_velocity=24,
    midi_min=21,
    midi_max=108,
    max_duration=8.0,
    merge_gap=0.03,
    max_polyphony=4,
    onset_group_window=0.055,
):
    """
    過濾/清理 Onsets & Frames 輸出的雜音。

    模型偶爾會在极安靜或極端音域產生一些站不住腳的短音，
    也可能把同一顆音拆成好幾段幾乎連在一起的碎音（rapid retrigger），
    聽起來像雜訊。這裡做三件事：

    1. 濾掉音量太小、音域超出範圍的音符。
    2. 把過長（可能是模型判斷卡住）的音符裁切到 max_duration。
    3. 合併同音高、間隔小於 merge_gap 的相鄰音符，減少碎音雜訊。
    """

    if not notes:
        return notes

    filtered = [
        (s, min(e, s + max_duration), p, v)
        for (s, e, p, v) in notes
        if v >= min_velocity and midi_min <= p <= midi_max
    ]

    if not filtered:
        return filtered

    # 第二層：去除「突然跳到極高/極低音域、只出現一下」的異常音。
    # 這類 note 很常是 O&F 對 Other 殘響/雜訊的誤判。
    # 不用固定禁止高音或低音，而是看時間鄰近的音符是否支持它，
    # 因此真正的高音旋律/低音樂器仍有機會保留。
    filtered.sort(key=lambda x: (x[0], x[2]))
    supported = []
    for i, note in enumerate(filtered):
        s, e, p, v = note
        duration = max(0.0, e - s)
        if duration >= 0.22 or v >= 72:
            supported.append(note)
            continue

        nearby = []
        # 只看前後約 0.35 秒內的其他音符。
        for j in range(max(0, i - 24), min(len(filtered), i + 25)):
            if i == j:
                continue
            os_, oe, op, ov = filtered[j]
            if abs(os_ - s) <= 0.35 or abs(oe - e) <= 0.35:
                nearby.append(op)

        if nearby:
            nearest = min(abs(p - op) for op in nearby)
            # 突然跨超過一個半八度、又很短、又沒有相近音支持 → 丟掉。
            if nearest > 18:
                continue
        else:
            # 完全孤立的短音也是高風險雜訊。
            if duration < 0.10 and v < 60:
                continue

        supported.append(note)

    filtered = supported
    if not filtered:
        return filtered

    before_polyphony = len(filtered)
    filtered = _limit_neural_polyphony(
        filtered,
        max_polyphony=max_polyphony,
        onset_group_window=onset_group_window,
    )
    if before_polyphony != len(filtered):
        print(
            f"  Other(neural) → 多音限制：{before_polyphony} → {len(filtered)}"
        )

    if not filtered:
        return filtered

    filtered.sort(key=lambda x: (x[2], x[0]))

    merged = []
    for s, e, p, v in filtered:
        if (
            merged
            and merged[-1][2] == p
            and s - merged[-1][1] <= merge_gap
        ):
            prev_s, prev_e, prev_p, prev_v = merged[-1]
            merged[-1] = (prev_s, max(prev_e, e), prev_p, max(prev_v, v))
        else:
            merged.append((s, e, p, v))

    merged.sort(key=lambda x: (x[0], x[2]))
    return merged


def clean_other_for_neural(
    y_other: np.ndarray,
    sr: int,
    harmonic_mix: float = 0.55,
    noise_reduction: float = 0.45,
):
    """在送進 Onsets & Frames 前，保守地削弱 Other 的背景雜訊。

    只作用於 neural backend，不改動原始 other.wav，也不影響 CQT/hybrid
    的 harmonic 快取。處理分兩層：
      1. HPSS 偏向 harmonic，降低鼓/瞬態殘留。
      2. 頻譜 gate 壓低整段都很弱的背景頻率，保留強音的 attack。

    這不是破壞性的硬降噪：弱聲仍會保留一部分，避免把真正的鋼琴/吉他
    長音一起刪掉。
    """
    if y_other is None or len(y_other) == 0:
        return y_other

    import librosa

    y = np.asarray(y_other, dtype=np.float32)
    if not np.any(np.isfinite(y)):
        return np.zeros_like(y, dtype=np.float32)
    y = np.nan_to_num(y, copy=False)

    harmonic_mix = float(np.clip(harmonic_mix, 0.0, 1.0))
    noise_reduction = float(np.clip(noise_reduction, 0.0, 1.0))

    print("  Other(neural) → 雜音抑制 ...")

    # 先取 harmonic；margin 越偏向 harmonic，鼓與短促瞬態越容易被壓掉。
    harmonic, _ = librosa.effects.hpss(
        y,
        margin=(1.0, 3.0),
    )

    # 頻譜 gate：以每個頻率 bin 的低百分位作為背景 floor。
    # 不使用硬 0/1 mask，而是保留 noise_reduction 後的殘留量，避免
    # 把真正很輕的伴奏長音一起切掉。
    n_fft = 2048
    hop = 256
    S = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
    )
    mag = np.abs(S)
    # 提高背景估計百分位，對「整段都有的底噪/殘響」更嚴格。
    floor = np.percentile(mag, 25.0, axis=1, keepdims=True)
    floor = np.maximum(floor, 1e-8)

    # 更強的 soft gate：低於背景約 1.6 倍的成分大幅壓低。
    # 不直接歸零，避免真正的弱鋼琴長音被完全切掉。
    ratio = mag / floor
    gate_strength = np.clip((ratio - 1.35) / 1.25, 0.0, 1.0)
    keep_floor = max(0.04, 1.0 - 0.95 * noise_reduction)
    mask = keep_floor + (1.0 - keep_floor) * gate_strength

    # 頻率/時間方向做很輕的平滑，避免 gate 自己製造顫抖與碎片。
    from scipy.ndimage import median_filter
    mask = median_filter(mask, size=(3, 5), mode="nearest")

    gated = librosa.istft(
        S * mask,
        hop_length=hop,
        win_length=n_fft,
        length=len(y),
    ).astype(np.float32)

    cleaned = harmonic_mix * harmonic + (1.0 - harmonic_mix) * gated

    # 保持和原 Other 大致相同的峰值，避免後面的 O&F 因整體音量突然改變
    # 而改變行為；只在必要時縮放。
    old_peak = float(np.max(np.abs(y)))
    new_peak = float(np.max(np.abs(cleaned)))
    if old_peak > 1e-6 and new_peak > 1e-6:
        gain = min(1.20, old_peak / new_peak)
        cleaned *= gain

    print("  Other(neural) → 雜音抑制完成")
    return np.nan_to_num(cleaned.astype(np.float32))


def extract_other_notes_neural(
    y_other: np.ndarray,
    sr: int,
    octave_shift: int = 0,
    model_path: str | None = None,
    onset_threshold: float = 0.45,
    frame_threshold: float = 0.40,
    min_duration: float = 0.035,
    min_velocity: int = 24,
    midi_min: int = 21,
    midi_max: int = 108,
    max_duration: float = 8.0,
    merge_gap: float = 0.03,
    max_polyphony: int = 4,
    onset_group_window: float = 0.055,
    raw_input: bool = False,
    cache_key: str | None = None,
):
    """用 Onsets & Frames 模型對 Other 軌做複音轉譜。"""

    if _of is None:
        print(
            "  [WARN] 找不到 onsets_frames_to_nbs.py，"
            "neural backend 無法使用，改回空結果。"
        )
        return []

    if y_other is None or len(y_other) == 0:
        return []

    print("Other(neural) → Onsets & Frames 轉譜 ...")

    # Neural backend 專用：先削弱 Other 中的背景/瞬態殘留。
    # 原始 y_other 不會被修改，因此 CQT / harmonic / chord 等其他流程
    # 仍使用原本的 Other 資料。
    # 預設採輕度清理，避免鋼琴/吉他 decay 被切碎。
    # raw_input 時完全跳過清理，直接將 Demucs other.wav 送入 O&F。
    if raw_input:
        print("  Other(neural) → RAW：跳過 HPSS / spectral gate")
        y_clean = np.asarray(y_other, dtype=np.float32)
    else:
        clean_cache = None
        if cache_key:
            clean_cache = os.path.join(
                _ensure_cache_dir("neural_clean"),
                f"{cache_key}_v1.npy",
            )
        if clean_cache and os.path.isfile(clean_cache):
            print("  Other(neural) → 使用持久化雜音抑制快取")
            y_clean = np.load(clean_cache, mmap_mode=None).astype(np.float32, copy=False)
        else:
            y_clean = clean_other_for_neural(
                y_other,
                sr,
                harmonic_mix=0.55,
                noise_reduction=0.45,
            )
            if clean_cache:
                np.save(clean_cache, np.asarray(y_clean, dtype=np.float32), allow_pickle=False)

    # 模型推理本身約佔這個階段最長的時間。只快取「模型轉譜結果」，
    # 不快取最後的 post-process，這樣之後調整 max_polyphony / merge_gap
    # 仍然會重新套用新參數，不會被舊結果綁死。
    model = _get_of_model(model_path)
    model_file = model_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models",
        "onsets_frames_uni.tflite",
    )
    model_key = _cache_key_for_file(
        model_file,
        "of-model-v1",
        onset_threshold,
        frame_threshold,
        min_duration,
        raw_input,
    )
    inference_cache = None
    if cache_key:
        inference_cache = os.path.join(
            _ensure_cache_dir("neural_notes"),
            f"{cache_key}_{model_key}.npz",
        )

    if inference_cache and os.path.isfile(inference_cache):
        print("  Other(neural) → 使用持久化 O&F 轉譜快取（跳過模型推理）")
        cached = np.load(inference_cache)
        result = [
            (
                float(s), float(e), int(p) + int(octave_shift) * 12, int(v)
            )
            for s, e, p, v in zip(
                cached["start"],
                cached["end"],
                cached["pitch"],
                cached["velocity"],
            )
        ]
        cached.close()
        before = len(result)
        result = _postprocess_neural_notes(
            result,
            min_velocity=min_velocity,
            midi_min=midi_min,
            midi_max=midi_max,
            max_duration=max_duration,
            merge_gap=merge_gap,
            max_polyphony=max_polyphony,
            onset_group_window=onset_group_window,
        )
        print(
            f"Other(neural)：快取模型輸出 {before} 個 → "
            f"過濾/合併後 {len(result)} 個音符"
        )
        return result

    y16 = y_clean
    if sr != _of.MODEL_SR:
        import librosa
        y16 = librosa.resample(
            y_clean.astype(np.float32),
            orig_sr=sr,
            target_sr=_of.MODEL_SR,
        )

    pred, times = _of.infer_audio(model, y16)
    notes = _of.predictions_to_notes(
        pred,
        times,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        min_duration=min_duration,
    )

    if inference_cache:
        np.savez_compressed(
            inference_cache,
            start=np.asarray([n.start for n in notes], dtype=np.float32),
            end=np.asarray([n.end for n in notes], dtype=np.float32),
            pitch=np.asarray([n.pitch for n in notes], dtype=np.int16),
            velocity=np.asarray([n.velocity for n in notes], dtype=np.uint8),
        )

    result = [
        (
            float(n.start),
            float(n.end),
            int(n.pitch) + int(octave_shift) * 12,
            int(n.velocity),
        )
        for n in notes
    ]

    before = len(result)
    result = _postprocess_neural_notes(
        result,
        min_velocity=min_velocity,
        midi_min=midi_min,
        midi_max=midi_max,
        max_duration=max_duration,
        merge_gap=merge_gap,
        max_polyphony=max_polyphony,
        onset_group_window=onset_group_window,
    )

    print(
        f"Other(neural)：模型輸出 {before} 個 → "
        f"過濾/合併後 {len(result)} 個音符"
    )
    return result


def run_other_extraction(y_other, sr, args, y_harm=None, cache_key=None):
    """依 --other-extract-backend 選擇 CQT 或神經網路來抽取 Other 音符。"""

    backend = getattr(args, "other_extract_backend", "neural")

    if backend == "neural":
        return extract_other_notes_neural(
            y_other,
            sr,
            octave_shift=args.other_octave_shift,
            model_path=args.other_neural_model,
            onset_threshold=args.other_neural_onset_threshold,
            frame_threshold=args.other_neural_frame_threshold,
            min_duration=args.other_neural_min_duration,
            min_velocity=args.other_neural_min_velocity,
            midi_min=args.other_neural_midi_min,
            midi_max=args.other_neural_midi_max,
            max_duration=args.other_neural_max_duration,
            merge_gap=args.other_neural_merge_gap,
            max_polyphony=args.other_neural_max_polyphony,
            onset_group_window=args.other_neural_onset_group_window,
            raw_input=args.other_neural_raw,
            cache_key=cache_key,
        )

    return extract_other_notes(
        y_other,
        sr,
        y_harm=y_harm,
        hop_length=args.other_hop_length,
        min_note_dur=args.other_min_note_dur,
        energy_threshold=args.other_energy_threshold,
        max_notes_per_frame=args.other_max_notes_per_frame,
        min_note_separation=args.other_min_note_separation,
        octave_shift=args.other_octave_shift,
        cqt_fmin=args.other_cqt_fmin,
        cqt_fmax=args.other_cqt_fmax,
        peak_min_separation=args.other_peak_min_separation,
        harmonic_tolerance=args.other_harmonic_tolerance,
        harmonic_count=args.other_harmonic_count,
        harmonic_suppression=args.other_harmonic_suppression,
        track_max_gap=args.other_track_max_gap,
        no_filter=args.other_no_filter,
        confidence_threshold=args.other_confidence_threshold,
    )


# ============================================================
# 和弦分析
# ============================================================



# ============================================================
# 主旋律八度校正
# ============================================================
#
# pYIN 偶爾會把主唱音高抓錯八度（±12 / ±24 半音）。
# 大部分時間準，但偶爾跳一個八度特別突兀。
#
# 這裡用「前後文音高」推測主旋律的真正走向：
# 如果某個音跟前後文差至少一個八度、又像孤立跳點，
# 就把它移回正確的八度；同時用和弦當輔助判斷，
# 避免誤傷真正的音程跳動。
# ============================================================

def _find_active_chord(
    chords,
    t,
):

    for (
        s,
        e,
        root,
        quality,
    ) in chords:

        if s <= t < e:
            return (
                root,
                quality,
            )

    return None


def _chord_fit_score(
    pitch,
    root,
    quality,
):
    """
    回傳 0~11：0 = 正好是合弦音，越大離和弦越遠。
    用來當八度修正的輔助判斷。
    """

    intervals = {
        "maj": (0, 4, 7),
        "min": (0, 3, 7),
        "7": (0, 4, 7, 10),
        "maj7": (0, 4, 7, 11),
        "min7": (0, 3, 7, 10),
        "sus2": (0, 2, 7),
        "sus4": (0, 5, 7),
        "dim": (0, 3, 6),
    }.get(quality, (0, 4, 7))

    root_pc = root % 12
    pc = pitch % 12

    best = 12

    for iv in intervals:

        d = (pc - root_pc - iv) % 12

        d = min(
            d,
            12 - d,
        )

        if d < best:
            best = d

    return best


def _pick_octave(
    pitch,
    neighbors,
):
    """
    從 pitch 的多個八度副本中，選出跟前後文最一致的那個。

    對每個候選 C（pitch ±12, ±24），把每個鄰居移到
    離 C 最近的八度副本，然後比較：
      1. 需要移動八度的鄰居數目（越少越一致）
      2. 移動後的總半音距離（越小越一致）

    兩個條件都「明顯」比原始音高好才回傳新八度，
    否則回傳原始音高。
    """

    if not neighbors:
        return pitch

    def _score(
        target,
    ):
        shifts = 0
        dist = 0.0

        for nb in neighbors:
            best_d = abs(nb - target)
            best_shift = 0

            for k in (-1, 1):
                copy = nb + k * 12
                d = abs(copy - target)
                if d < best_d:
                    best_d = d
                    best_shift = k

            if best_shift != 0:
                shifts += 1

            dist += best_d

        return shifts, dist

    best = pitch
    best_score = _score(pitch)

    for shift in (
        -24,
        -12,
        12,
        24,
    ):

        cand = pitch + shift
        score = _score(cand)

        # 必須「明顯」更貼才改（少一個八度移動，或距離少 4 以上）
        better_shifts = (
            score[0] < best_score[0] - 1
        )

        better_dist = (
            score[0] <= best_score[0]
            and score[1]
            < best_score[1] - 4
        )

        if (
            better_shifts
            or better_dist
        ):

            best_score = score
            best = cand

    return best


def correct_melody_octaves(
    notes,
    chords,
):
    """
    修正孤立的主唱八度跳錯。

    對每個音，用「前後文的音高」來評分它的每個八度副本，
    如果某個副本明顯更貼近前後文才移動該音；
    另外再用和弦當輔助（修正後不能大幅偏離和弦音）。
    """

    if len(notes) < 3:
        return [
            (s, e, p)
            for s, e, p in notes
        ]

    corrected = []

    for i in range(len(notes)):

        start, end, pitch = notes[i]

        # 左右鄰居（左邊用已校正的音高，修正能往前傳遞）
        left = [
            p
            for s, e, p in corrected[-4:]
        ]

        right = [
            p
            for s, e, p
            in notes[i + 1:i + 4]
        ]

        # 只把「孤立的八度跳點」視為可疑。
        # 如果前後音本身也真的形成大跳，不要因為八度關係把正常旋律改掉。
        immediate = []
        if left:
            immediate.append(left[-1])
        if right:
            immediate.append(right[0])

        best = pitch
        if len(immediate) == 2:
            a, b = immediate
            context_span = abs(a - b)
            if context_span <= 5 and abs(pitch - a) >= 9 and abs(pitch - b) >= 9:
                best = _pick_octave(pitch, immediate)
        elif len(immediate) == 1:
            q = immediate[0]
            if abs(pitch - q) >= 18:
                best = _pick_octave(pitch, immediate)

        p = pitch

        # 八度確實被調整了 → 再用和弦作輔助，而不是反過來決定音高。
        if best != pitch and abs(best - pitch) in (12, 24):
            ch = _find_active_chord(chords, (start + end) / 2.0)
            if ch is None:
                p = best
            else:
                root, quality = ch
                old_fit = _chord_fit_score(pitch, root, quality)
                new_fit = _chord_fit_score(best, root, quality)
                if new_fit <= old_fit + 1:
                    p = best

        corrected.append(
            (
                start,
                end,
                p,
            )
        )

    return corrected


# ============================================================
# 和弦生成副旋律（穩定版，取代逐幀抓音）
# ============================================================
#
# 逐幀從 Other 音軌抓音符（extract_other_notes）本質上是
# 猜測，音源越複雜（多樂器疊在一起）越容易抓錯、抖動、
# 音符亂跳。
#
# 這裡改用已經算好的和弦進行（chords）+ 節拍（beat_times），
# 用樂理規則（琶音）直接「生成」一段穩定的副旋律，
# 不需要再去猜測音訊裡實際彈的是什麼音。
#
# 因為和弦分析是以「每個 beat 區段」為單位統計音高分佈，
# 天生就比逐幀 pYIN / CQT 抓音穩定很多。
# ============================================================



# ============================================================
# 鼓分析
# ============================================================



# ============================================================
# 鼓組樣式設計
#
# 不要再每拍都一樣大聲：
#  - Kick 落在 1、3 拍，Snare 落在 2、4 拍（backbeat）
#  - 音量依每個拍子的 onset strength 起伏
#  - downbeat 大聲、其他拍子輕一點
#  - 鼓點時間稍微加一點 humanize（抖動）
#
# 回傳 [(layer, time_sec, velocity), ...]
# ============================================================



# ============================================================
# NBS Instrument
# ============================================================

L_MELODY = 0
L_BASS = 1

# Other 聲部分層：不硬猜「這是鋼琴/吉他」，
# 而是依音域先拆成低、中、高三個獨立聲部。
# 這樣即使樂器辨識不準，也能避免所有 Other 擠在同一層。
L_OTHER_LOW = 2
L_OTHER_MID = 3
L_OTHER_HIGH = 4

# 舊名稱保留作為相容別名；新的輸出會依音域實際分層。
L_OTHER = L_OTHER_MID

L_CH_ROOT = 5
L_CH_3RD = 6
L_CH_5TH = 7
L_CH_7TH = 8
L_KICK = 9
L_SNARE = 10
L_HAT = 11


INSTR = {
    'harp': 0,
    'bass': 1,
    'bd': 2,
    'snare': 3,
    'hat': 4,
    'guitar': 5,
    'flute': 6,
}


LAYER_NAMES = {
    L_MELODY: "Melody",
    L_BASS: "Bass",
    L_OTHER_LOW: "Other Low",
    L_OTHER_MID: "Other Mid",
    L_OTHER_HIGH: "Other High",
    L_CH_ROOT: "Chord Root",
    L_CH_3RD: "Chord 3rd",
    L_CH_5TH: "Chord 5th",
    L_CH_7TH: "Chord 7th",
    L_KICK: "Kick",
    L_SNARE: "Snare",
    L_HAT: "HiHat",
}


def _clip(
    v,
    lo,
    hi,
):
    return max(
        lo,
        min(
            hi,
            int(round(v)),
        ),
    )


# ============================================================
# MIDI → NBS
# ============================================================

def midi_to_nbs_key(
    midi_key,
):
    # NBS 標準映射：key 0 = MIDI 21 (A0)、key 87 = MIDI 108 (C8)。
    # 舊版使用 midi - 33，會讓所有音符整體低一個八度，
    # 且 MIDI < 33 的低音（例如 Bass 預設音域 E1=28）全部被夾成 key 0。
    return _clip(
        midi_key - 21,
        0,
        87,
    )


# ============================================================
# NBS Builder
# ============================================================

def build_nbs_bytes(
    melody_notes,
    bass_notes,
    other_notes,
    chords,
    drum_hits,
    beat_times,
    tps=20.0,
    song_name="auto_convert",
    song_author="mp3_to_nbs",
    original_author="",
    description="",
    max_tick_shift=4,
    song_duration=None,
    other_retrigger_sec=0.25,
):

    notes_by_layer_tick = {}

    def add_note(
        layer,
        tick,
        instrument,
        key,
        velocity,
        panning=100,
        pitch=0,
        collision_mode="shift",
    ):

        key = _clip(
            key,
            0,
            87,
        )

        velocity = _clip(
            velocity,
            0,
            100,
        )

        d = notes_by_layer_tick.setdefault(
            layer,
            {},
        )

        # --------------------------------------------------------
        # 同一 layer + tick 只能放一個 note。
        #
        # 重要修正：不能所有來源都一律往後 shift。
        # Other / chord 是多音符聲部；如果和弦的兩顆音撞在同一 tick，
        # 把第二顆推到下一 tick 會把「同時發聲」變成琶音，聽感會明顯失真。
        #
        # 因此：
        #   shift         → Melody / Bass 等單音快速線條，保留快速音符
        #   replace_weaker → Other / Drum，寧可丟較弱碰撞音，也不製造假琶音
        #   drop           → 保守地丟棄碰撞音
        #
        # chord 本身使用不同 layer（root/3rd/5th/7th），正常和弦不會互撞。
        # --------------------------------------------------------

        final_tick = tick

        if collision_mode == "shift":
            shift = 0
            while (
                final_tick in d
                and shift < max_tick_shift
            ):
                final_tick += 1
                shift += 1

            if final_tick in d:
                return

        elif final_tick in d:
            existing = d[final_tick]
            existing_velocity = int(existing[2])

            if collision_mode == "replace_weaker":
                # 同一 tick 的兩個不同音高不能同時存在於同一 layer。
                # 保留較強的音，避免把和弦攤成連續琶音。
                if velocity <= existing_velocity:
                    return
            else:  # drop
                return

        d[final_tick] = (
            instrument,
            key,
            velocity,
            panning,
            pitch,
        )

    # ========================================================
    # 主唱
    # ========================================================

    for start, end, pitch in melody_notes:

        tick = int(
            round(
                start * tps
            )
        )

        duration = (
            end - start
        )

        velocity = (
            100
            if duration < 0.1
            else 92
        )

        add_note(
            L_MELODY,
            tick,
            INSTR["flute"],
            midi_to_nbs_key(
                pitch
            ),
            velocity,
        )

    # ========================================================
    # Bass
    # ========================================================

    for start, end, pitch in bass_notes:

        tick = int(
            round(
                start * tps
            )
        )

        duration = (
            end - start
        )

        velocity = (
            88
            if duration < 0.15
            else 78
        )

        add_note(
            L_BASS,
            tick,
            INSTR["bass"],
            midi_to_nbs_key(
                pitch
            ),
            velocity,
        )

    # ========================================================
    # Other 樂器 → 聲部分層
    #
    # 目前不假裝能從混合 Other stem 精準辨識「鋼琴/吉他/弦樂」。
    # 改用穩健的音域 + 音符特徵分成 Low / Mid / High 三層。
    # 低音域偏 Bass-like，中音域偏 Piano/Guitar/Pad，高音域偏 Lead/Synth。
    # 這只是聲部分群，不是樂器分類；目的是讓同一個 tick 能同時
    # 容納不同音域的 Other 音符，並讓 NBS 播放時不全部擠成一條線。
    # ========================================================

    def classify_other_layer(pitch, duration):
        p = int(pitch)
        d = max(0.0, float(duration))

        # 主要依 MIDI 音域；邊界附近用持續時間作輕微偏好。
        # 長音在中音域更常見於 pad/chord，所以不要輕易丟到 high。
        if p < 55:
            return L_OTHER_LOW
        if p >= 76:
            return L_OTHER_HIGH
        if p >= 70 and d < 0.10:
            return L_OTHER_HIGH
        return L_OTHER_MID

    for note in other_notes:

        start = note[0]
        end = note[1]
        pitch = note[2]
        other_layer = classify_other_layer(pitch, end - start)

        base_velocity = (
            note[3]
            if len(note) >= 4
            else None
        )

        start_tick = int(
            round(
                start * tps
            )
        )

        end_tick = int(
            round(
                end * tps
            )
        )

        if (
            end_tick
            <= start_tick
        ):
            continue

        key = midi_to_nbs_key(
            pitch
        )

        # NBS 沒有 MIDI 式 note-off / duration。
        # 舊版無論音符多短都每 0.25 秒重敲，長音很容易變成「叮、叮、叮」。
        # 現在預設只觸發一次；只有明確指定 >0 才啟用週期性 sustain 重觸發。
        retrigger = float(other_retrigger_sec)
        if retrigger <= 0.0:
            ticks_to_play = (start_tick,)
        else:
            step = max(1, int(round(retrigger * tps)))
            ticks_to_play = range(start_tick, end_tick, step)

        for tick in ticks_to_play:

            if tick == start_tick:

                if base_velocity is not None:
                    velocity = base_velocity
                else:
                    velocity = 72

            else:

                if base_velocity is not None:
                    velocity = max(
                        30,
                        base_velocity - 18,
                    )
                else:
                    velocity = 52

            # Other 整體比主唱再退一些，降低伴奏蓋過低男聲的機率。
            velocity = _clip(
                velocity * OTHER_VELOCITY_SCALE,
                18,
                88,
            )

            # 聲部使用不同音色只是「角色化」而非樂器辨識：
            # Low 偏 guitar、Mid 保持 harp、High 偏 flute。
            # 這能讓三個 Other layer 在實際播放時真的有可聽見的區別，
            # 同時不宣稱我們已經知道原曲究竟是鋼琴、吉他還是 synth。
            # Other 不使用 flute：高音也統一使用 harp，避免 flute 過於突出。
            if other_layer == L_OTHER_LOW:
                other_instrument = INSTR["guitar"]
            else:
                other_instrument = INSTR["harp"]

            add_note(
                other_layer,
                tick,
                other_instrument,
                key,
                velocity,
                collision_mode="replace_weaker",
            )

    # ========================================================
    # 和弦
    # ========================================================

    for beat in beat_times:

        tick = int(
            round(
                beat * tps
            )
        )

        for (
            s,
            e,
            root,
            quality,
        ) in chords:

            if s <= beat < e:

                # ------------------------------------------------
                # Bass 根音
                # ------------------------------------------------

                bass_midi = (
                    48 + root
                )

                # 有真實 Bass 音符落在附近時，不再額外塞一個
                # root，避免同一條 Bass 軌變成「真 Bass + 機械 root」雙重音。
                has_real_bass = any(
                    abs(float(bs) - float(beat)) < 0.08
                    and int(round(bp)) % 12 == int(bass_midi) % 12
                    for bs, be, bp in bass_notes
                )
                if not has_real_bass:
                    add_note(
                        L_BASS,
                        tick,
                        INSTR["bass"],
                        midi_to_nbs_key(
                            bass_midi
                        ),
                        52,
                        collision_mode="shift",
                    )

                # ------------------------------------------------
                # 和弦
                # ------------------------------------------------

                chord_intervals = {
                    "maj": (0, 4, 7),
                    "min": (0, 3, 7),
                    "7": (0, 4, 7, 10),
                    "maj7": (0, 4, 7, 11),
                    "min7": (0, 3, 7, 10),
                    "sus2": (0, 2, 7),
                    "sus4": (0, 5, 7),
                    "dim": (0, 3, 6),
                }
                intervals = list(chord_intervals.get(quality, (0, 4, 7)))

                chord_layers = (
                    L_CH_ROOT,
                    L_CH_3RD,
                    L_CH_5TH,
                    L_CH_7TH,
                )

                for (
                    layer,
                    iv,
                ) in zip(
                    chord_layers,
                    intervals,
                ):

                    chord_midi = (
                        60
                        + root
                        + iv
                    )

                    add_note(
                        layer,
                        tick,
                        INSTR["guitar"],
                        midi_to_nbs_key(
                            chord_midi
                        ),
                        42,
                        collision_mode="drop",
                    )

                break

    # ========================================================
    # 鼓（設計好的鼓組節奏，含音量起伏）
    # ========================================================

    for layer, t, velocity in drum_hits:

        tick = int(
            round(
                t * tps
            )
        )

        if layer == L_KICK:
            instrument = INSTR["bd"]
        elif layer == L_SNARE:
            instrument = INSTR["snare"]
        else:
            instrument = INSTR["hat"]

        add_note(
            layer,
            tick,
            instrument,
            45,
            _clip(
                velocity,
                12,
                100,
            ),
            collision_mode="replace_weaker",
        )

    # ========================================================
    # Tick Map
    # ========================================================

    tick_map = {}

    for layer, d in (
        notes_by_layer_tick.items()
    ):

        for tick, note in d.items():

            tick_map.setdefault(
                tick,
                [],
            ).append(
                (layer,) + note
            )

    all_ticks = sorted(
        tick_map.keys()
    )

    # 有原始音訊長度時保留整首歌曲的尾端靜音，不讓 NBS 提前結束。
    audio_length_tick = 0
    if song_duration is not None:
        audio_length_tick = max(0, int(round(float(song_duration) * tps)))

    song_length = max(
        all_ticks[-1] if all_ticks else 0,
        audio_length_tick,
    )

    # ========================================================
    # Binary
    # ========================================================

    buf = io.BytesIO()

    def w_u8(v):
        buf.write(
            struct.pack(
                "<B",
                v,
            )
        )

    def w_i16(v):
        buf.write(
            struct.pack(
                "<h",
                v,
            )
        )

    def w_u16(v):
        buf.write(
            struct.pack(
                "<H",
                v,
            )
        )

    def w_i32(v):
        buf.write(
            struct.pack(
                "<i",
                v,
            )
        )

    def w_str(s):

        b = s.encode(
            "utf-8"
        )

        w_i32(
            len(b)
        )

        buf.write(b)

    VERSION = 5
    VANILLA_COUNT = 16
    LAYER_COUNT = 12

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    w_u16(0)
    w_u8(VERSION)
    w_u8(VANILLA_COUNT)

    w_u16(
        min(
            song_length,
            65535,
        )
    )

    w_u16(
        LAYER_COUNT
    )

    w_str(song_name)
    w_str(song_author)
    w_str(original_author)
    w_str(description)

    w_u16(
        int(
            round(
                tps * 100
            )
        )
    )

    w_u8(0)
    w_u8(0)
    w_u8(4)

    for _ in range(5):
        w_i32(0)

    w_str("")

    w_u8(0)
    w_u8(0)
    w_u16(0)

    # --------------------------------------------------------
    # Notes
    # --------------------------------------------------------

    last_tick = -1

    for tick in all_ticks:

        w_u16(
            tick
            - last_tick
        )

        last_tick = tick

        entries = sorted(
            tick_map[tick],
            key=lambda x:
                x[0],
        )

        last_layer = -1

        for (
            layer,
            instrument,
            key,
            velocity,
            panning,
            pitch,
        ) in entries:

            w_u16(
                layer
                - last_layer
            )

            last_layer = layer

            w_u8(
                instrument
            )

            w_u8(key)
            w_u8(velocity)
            w_u8(panning)
            w_i16(pitch)

        # End of tick
        w_u16(0)

    # End of notes
    w_u16(0)

    # --------------------------------------------------------
    # Layers
    # --------------------------------------------------------

    for i in range(
        LAYER_COUNT
    ):

        w_str(
            LAYER_NAMES.get(
                i,
                f"Layer {i}",
            )
        )

        w_u8(0)
        w_u8(100)
        w_u8(100)

    w_u8(0)

    print(
        f"產生 "
        f"{len(all_ticks)} 個 tick"
    )

    return buf.getvalue()


# ============================================================
# Other mix 模式：依 vocals.wav 能量自動切換
#
# 有人聲：只允許 chord accompaniment
# 無人聲：只允許 Other extract
#
# 使用 hysteresis（開啟/關閉門檻不同）+ RMS 平滑，避免
# 人聲剛好落在門檻附近時 chord/extract 每幾幀來回切換。
# ============================================================

def build_vocal_activity_mask(
    y_vocal,
    sr,
    times,
    hop_length=256,
    on_db_offset=10.0,
    min_on_db=-39.0,
    hysteresis_db=5.0,
):
    import librosa

    times = np.asarray(times, dtype=float)
    if len(times) == 0:
        return np.zeros(0, dtype=bool)

    rms = librosa.feature.rms(
        y=y_vocal,
        frame_length=2048,
        hop_length=hop_length,
        center=True,
    )[0]

    rms_times = librosa.times_like(
        rms,
        sr=sr,
        hop_length=hop_length,
    )

    if len(rms) == 0:
        return np.zeros(len(times), dtype=bool)

    db = librosa.amplitude_to_db(
        rms + 1e-8,
        ref=1.0,
    )

    # 以較低分位數估計 vocals stem 的底噪，再往上抓一個安全距離。
    # 同時設絕對下限，避免非常安靜的歌曲把底噪誤判成人聲。
    noise_floor = float(np.percentile(db, 20))
    on_db = max(
        min_on_db,
        noise_floor + on_db_offset,
    )
    off_db = on_db - hysteresis_db

    target_db = np.interp(
        times,
        rms_times,
        db,
        left=float(db[0]),
        right=float(db[-1]),
    )

    # 約 3 個 frame 的平滑，降低單一子音/呼吸造成的快速切換。
    smooth = target_db.copy()
    if len(smooth) >= 3:
        kernel = min(7, len(smooth) if len(smooth) % 2 else len(smooth) - 1)
        if kernel >= 3:
            from scipy.ndimage import median_filter
            smooth = median_filter(smooth, size=kernel, mode="nearest")

    active = np.zeros(len(smooth), dtype=bool)
    state = False
    for i, level in enumerate(smooth):
        if not state:
            if level >= on_db:
                state = True
        else:
            if level < off_db:
                state = False
        active[i] = state

    # 不讓單一 frame 的閃爍形成獨立區段。
    frame_dur = hop_length / sr
    min_run = max(1, int(round(0.06 / frame_dur)))
    padded = np.r_[False, active, False]
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    for s, e in zip(starts, ends):
        if e - s < min_run:
            active[s:e] = False

    print(
        f"  Vocal 能量切換門檻：on={on_db:.1f} dB, off={off_db:.1f} dB"
    )
    print(
        f"  Vocal active：{100.0 * float(active.mean()):.1f}%"
    )
    return active


def filter_notes_by_vocal_activity(
    notes,
    times,
    vocal_active,
    keep_when_vocal,
):
    """依 vocals 狀態裁切音符，而不是只看 note 中點。

    這很重要：一個 chord note 可能跨過人聲開始/結束點。
    只看中點會讓它整顆穿過切換邊界；現在會按 vocal mask
    把 note 裁成對應區段，避免 chord/extract 互相滲透。
    """
    if not notes:
        return []
    times = np.asarray(times, dtype=float)
    active = np.asarray(vocal_active, dtype=bool)
    if len(times) == 0 or len(active) == 0:
        return [] if keep_when_vocal else list(notes)

    result = []
    grid_step = float(np.median(np.diff(times))) if len(times) > 1 else 0.0

    for note in notes:
        start, end = float(note[0]), float(note[1])
        if end <= start:
            continue

        # 找到與 note 重疊的 mask 區間。
        left = max(0, int(np.searchsorted(times, start, side="right") - 1))
        right = min(len(times) - 1, int(np.searchsorted(times, end, side="left")))

        segment_start = None
        last_state = None
        for idx in range(left, right + 1):
            state = bool(active[idx]) == keep_when_vocal
            cell_start = float(times[idx])
            cell_end = float(times[idx + 1]) if idx + 1 < len(times) else float(end)
            cell_start = max(cell_start, start)
            cell_end = min(cell_end, end)
            if cell_end <= cell_start:
                continue

            if state:
                if segment_start is None:
                    segment_start = cell_start
                elif last_state is False:
                    segment_start = cell_start
            elif segment_start is not None:
                seg_end = cell_start
                if seg_end > segment_start:
                    result.append((segment_start, seg_end, *note[2:]))
                segment_start = None
            last_state = state

        if segment_start is not None:
            seg_end = end
            if seg_end > segment_start:
                result.append((segment_start, seg_end, *note[2:]))

    # 去掉因 mask 邊界產生的極短碎片；保留原始 note 其餘欄位。
    if grid_step > 0:
        min_piece = min(grid_step * 0.5, 0.02)
        result = [n for n in result if n[1] - n[0] >= min_piece]

    return result


# ============================================================
# Main
# ============================================================

def main():

    conversion_start = time.perf_counter()

    ap = argparse.ArgumentParser(
        description=(
            "MP3 → Demucs 4-Stem → NBS"
        )
    )

    ap.add_argument(
        "mp3",
        help="mp3 路徑",
    )

    ap.add_argument(
        "-o",
        "--output",
        default=None,
        help="nbs 路徑",
    )

    ap.add_argument(
        "--sr",
        type=int,
        default=22050,
        help="取樣率",
    )

    ap.add_argument(
        "--tps",
        type=float,
        default=40.0,
        help="每秒 tick",
    )

    ap.add_argument(
        "--demucs-model",
        default="htdemucs",
        help="Demucs 模型",
    )

    ap.add_argument(
        "--device",
        default=None,
        help=(
            "運算裝置，例如 "
            "cuda / cpu"
        ),
    )

    ap.add_argument(
        "--segment",
        type=int,
        default=None,
        help="Demucs segment",
    )

    ap.add_argument(
        "--keep-temp",
        action="store_true",
        help="保留 Demucs 輸出",
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "隨機種子，控制鼓點音量 / 副旋律抖動的隨機性。"
            "同一個種子輸出可重現"
            "（預設: %(default)s）"
        ),
    )

    # ========================================================
    # 音高升降八度
    # ========================================================

    octave_group = ap.add_argument_group(
        "音高升降八度"
    )

    octave_group.add_argument(
        "--melody-octave-shift",
        type=int,
        default=MELODY_OCTAVE_SHIFT,
        help=(
            "主唱升降八度。"
            "0=原始 1=高一個八度 "
            "2=高兩個八度 -1=低一個八度"
            "（預設: %(default)s）"
        ),
    )

    octave_group.add_argument(
        "--bass-octave-shift",
        type=int,
        default=BASS_OCTAVE_SHIFT,
        help=(
            "Bass 升降八度"
            "（預設: %(default)s）"
        ),
    )

    octave_group.add_argument(
        "--other-octave-shift",
        type=int,
        default=OTHER_OCTAVE_SHIFT,
        help=(
            "Other 樂器升降八度"
            "（預設: %(default)s）"
        ),
    )

    # ========================================================
    # 主唱 RMVPE 參數
    # ========================================================

    melody_group = ap.add_argument_group(
        "主唱 RMVPE 參數"
    )

    melody_group.add_argument(
        "--melody-fmin",
        default="C2",
        help="主唱最低偵測音符（預設: %(default)s）",
    )

    melody_group.add_argument(
        "--melody-fmax",
        default="C6",
        help="主唱最高偵測音符（預設: %(default)s）",
    )

    melody_group.add_argument(
        "--melody-min-dur",
        type=float,
        default=0.025,
        help=(
            "主唱音符最短保留時間（秒）。"
            "太短的音符會被丟棄"
            "（預設: %(default)s）"
        ),
    )

    melody_group.add_argument(
        "--melody-gap-tolerance",
        type=float,
        default=0.09,
        help=(
            "主唱無聲多久才視為斷句（秒）。"
            "rap 換氣快可以調大一點"
            "（預設: %(default)s）"
        ),
    )

    melody_group.add_argument(
        "--melody-voiced-prob",
        type=float,
        default=0.03,
        help="RMVPE F0 confidence threshold（預設: %(default)s）",
    )

    melody_group.add_argument(
        "--melody-persistence",
        type=float,
        default=0.045,
        help="RMVPE 新音高至少持續多久才換音（預設: %(default)s）",
    )

    melody_group.add_argument(
        "--melody-medfilt-kernel",
        type=int,
        default=5,
        help="RMVPE MIDI 中值濾波窗口大小（預設: %(default)s）",
    )

    melody_group.add_argument(
        "--melody-min-note-ms",
        type=float,
        default=70.0,
        help=(
            "人聲音符短於此長度（毫秒）會被吸附進鄰近長音，用來抑制"
            "滑音/過渡碎片；另外長度 2 倍以內、音高恰好夾在前後兩音"
            "之間的慢滑音也會一併清除（Rap 等長音群不受影響）；"
            "0 表示關閉（預設: %(default)s）"
        ),
    )

    melody_group.add_argument(
        "--rap_revise",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "啟用 Rap/快速人聲修正：依局部音高變化密度自適應縮短"
            "新音高 persistence，避免 Rap 短音節被吞掉。"
            "預設開啟；使用 --no-rap_revise 可關閉。"
        ),
    )

    melody_group.add_argument(
        "--melody-rmvpe-model",
        default=None,
        help="RMVPE 模型檔路徑；留空使用 rmvpe 套件預設模型。",
    )

    melody_group.add_argument(
        "--melody-rmvpe-half",
        action="store_true",
        help="RMVPE 使用 FP16（GPU 可用時可降低 VRAM/提高速度）。",
    )

    melody_group.add_argument(
        "--no-octave-correction",
        action="store_false",
        dest="octave_correction",
        default=True,
        help=(
            "關閉主唱八度校正。"
            "預設會用前後文修正主唱偶爾抓錯八度的孤立音；"
            "若有啟用和弦分析（--enable-chord-analysis），"
            "會額外用和弦當輔助判斷。"
        ),
    )

    # ========================================================
    # Bass pYIN 參數
    # ========================================================

    bass_group = ap.add_argument_group(
        "Bass pYIN 參數"
    )

    bass_group.add_argument(
        "--bass-fmin",
        default="E1",
        help="Bass 最低偵測音符（預設: %(default)s）",
    )

    bass_group.add_argument(
        "--bass-fmax",
        default="C4",
        help="Bass 最高偵測音符（預設: %(default)s）",
    )

    bass_group.add_argument(
        "--bass-frame-length",
        type=int,
        default=2048,
        help="Bass pYIN frame_length（預設: %(default)s）",
    )

    bass_group.add_argument(
        "--bass-hop-length",
        type=int,
        default=256,
        help="Bass pYIN hop_length（預設: %(default)s）",
    )

    bass_group.add_argument(
        "--bass-min-dur",
        type=float,
        default=0.05,
        help="Bass 音符最短保留時間（秒）（預設: %(default)s）",
    )

    bass_group.add_argument(
        "--bass-gap-tolerance",
        type=float,
        default=0.08,
        help="Bass 無聲多久才視為斷句（秒）（預設: %(default)s）",
    )

    bass_group.add_argument(
        "--bass-voiced-prob",
        type=float,
        default=0.12,
        help="Bass 有聲判定機率門檻（預設: %(default)s）",
    )

    bass_group.add_argument(
        "--bass-persistence",
        type=float,
        default=0.03,
        help="Bass 新音高要持續多久（秒）才算換音（預設: %(default)s）",
    )

    bass_group.add_argument(
        "--bass-medfilt-kernel",
        type=int,
        default=3,
        help="Bass 音高中值濾波窗口大小（預設: %(default)s）",
    )

    # ========================================================
    # 和弦分析參數
    # ========================================================

    chord_group = ap.add_argument_group(
        "和弦分析參數"
    )

    chord_group.add_argument(
        "--enable-chord-analysis",
        action="store_true",
        default=False,
        dest="enable_chord_analysis",
        help=(
            "啟用和弦分析（預設: 關閉）。"
            "關閉時：跳過 chroma CQT 與和弦辨識，"
            "不會產生和弦琶音伴奏（L_CH_ROOT/3rd/5th/7th）、"
            "不會用和弦輔助主唱八度校正與 Bass 修正；"
            "Other 副旋律若是 hybrid/chord/mix 模式，"
            "會自動退化為純 extract。"
            "啟用時：恢復原本完整和弦進行分析與伴奏生成。"
        ),
    )

    chord_group.add_argument(
        "--other-chord-subdivision",
        type=int,
        default=2,
        help=(
            "[chord 模式] 每個 beat 切成幾個音符。"
            "1=每拍一音 2=八分音符琶音 4=十六分音符琶音"
            "（預設: %(default)s）"
        ),
    )

    chord_group.add_argument(
        "--other-chord-pattern",
        choices=[
            "up",
            "down",
            "up_down",
            "root_only",
        ],
        default="up",
        help=(
            "[chord 模式] 琶音樣式。"
            "up=由低到高循環 down=由高到低循環 "
            "up_down=上下來回 root_only=只彈根音（最保守）"
            "（預設: %(default)s）"
        ),
    )

    chord_group.add_argument(
        "--other-chord-note-len-ratio",
        type=float,
        default=0.9,
        help=(
            "[chord 模式] 每個音符實際發聲長度占時間格的比例，"
            "越小越斷奏(staccato)，越接近1越圓滑(legato)"
            "（預設: %(default)s）"
        ),
    )

    chord_group.add_argument(
        "--no-chord-rotation",
        action="store_false",
        dest="other_chord_rotation",
        default=True,
        help=(
            "[chord 模式] 關閉琶音方向隨小節交替"
            "（預設: 開啟，每個小節的反拍反轉方向）"
        ),
    )

    chord_group.add_argument(
        "--other-chord-skip",
        type=float,
        default=0.08,
        help=(
            "[chord 模式] offbeat 音符被跳過（留白）的機率，"
            "製造呼吸感。0 = 全不放過"
            "（預設: %(default)s）"
        ),
    )

    chord_group.add_argument(
        "--other-chord-humanize",
        type=float,
        default=0.008,
        help=(
            "[chord 模式] 音符時間隨機抖動幅度（秒），"
            "消除節拍器般的整齊感"
            "（預設: %(default)s）"
        ),
    )

    # ========================================================
    # Other 樂器參數
    # ========================================================

    other_group = ap.add_argument_group(
        "Other 樂器參數"
    )

    other_group.add_argument(
        "--other-preset",
        choices=[
            "default",
            "piano",
        ],
        default="default",
        help=(
            "Other 分析預設。"
            "piano = 高時間解析度、較低能量門檻、更多同時音符、"
            "較短音符與較短間隔，並自動啟用 --other-no-filter。"
            "可再用其他 --other-* 參數覆蓋個別值"
            "（預設: %(default)s）"
        ),
    )

    other_group.add_argument(
        "--other-mode",
        choices=[
            "hybrid",
            "chord",
            "extract",
            "mix",
        ],
        default="hybrid",
        help=(
            "副旋律產生方式。"
            "hybrid = 優先使用 Other 實際抓到的音符，不足處再用和弦補齊（推薦）；"
            "chord = 用和弦生成琶音伴奏（穩定）；"
            "extract = 用 CQT 逐幀抓音符；"
            "mix = 有人聲用 chord、無人聲用 extract"
            "（預設: %(default)s）。"
            "注意：若未啟用 --enable-chord-analysis，"
            "hybrid/chord/mix 會自動退化為純 extract。"
        ),
    )

    other_group.add_argument(
        "--other-extract-backend",
        choices=["cqt", "neural"],
        default="neural",
        help="Other 實際音符抽取引擎：cqt / neural（預設: %(default)s）。",
    )

    other_group.add_argument(
        "--other-neural-model",
        default=None,
        help="Other neural Onsets & Frames 模型路徑；留空使用 models/onsets_frames_uni.tflite。",
    )

    other_group.add_argument(
        "--other-neural-onset-threshold",
        type=float,
        default=0.45,
        help="Other neural onset threshold（預設: %(default)s）",
    )

    other_group.add_argument(
        "--other-neural-frame-threshold",
        type=float,
        default=0.40,
        help="Other neural frame threshold（預設: %(default)s）",
    )

    other_group.add_argument(
        "--other-neural-min-duration",
        type=float,
        default=0.035,
        help="Other neural 最短音符時間（秒）（預設: %(default)s）",
    )

    other_group.add_argument(
        "--other-neural-min-velocity",
        type=int,
        default=24,
        help="Other neural 最低 velocity（預設: %(default)s）",
    )

    other_group.add_argument(
        "--other-neural-midi-min",
        type=int,
        default=21,
        help="Other neural 最低 MIDI 音高（預設: %(default)s）",
    )

    other_group.add_argument(
        "--other-neural-midi-max",
        type=int,
        default=108,
        help="Other neural 最高 MIDI 音高（預設: %(default)s）",
    )

    other_group.add_argument(
        "--other-neural-max-duration",
        type=float,
        default=8.0,
        help="Other neural 單顆音符最大長度（秒）（預設: %(default)s）",
    )

    other_group.add_argument(
        "--other-neural-merge-gap",
        type=float,
        default=0.03,
        help="Other neural 同音符合併間隔（秒）（預設: %(default)s）",
    )

    other_group.add_argument(
        "--other-neural-max-polyphony",
        type=int,
        default=4,
        help="Other neural 同一 onset 群組最多保留幾個同時音符（預設: %(default)s）",
    )

    other_group.add_argument(
        "--other-neural-onset-group-window",
        type=float,
        default=0.055,
        help="Other neural 視為同一 onset 群組的時間窗口（秒）（預設: %(default)s）",
    )

    other_group.add_argument(
        "--other-neural-raw",
        action="store_true",
        help="Other neural 直接使用 Demucs other.wav，跳過 HPSS 與 spectral gate。",
    )

    other_group.add_argument(
        "--other-hop-length",
        type=int,
        default=256,
        help="[extract 模式] Other CQT hop_length（預設: %(default)s）",
    )

    other_group.add_argument(
        "--other-min-note-dur",
        type=float,
        default=0.08,
        help="[extract 模式] Other 音符最短保留時間（秒）（預設: %(default)s）",
    )

    other_group.add_argument(
        "--other-energy-threshold",
        type=float,
        default=0.2,
        help=(
            "[extract 模式] Other 音符能量門檻，"
            "越低越容易抓到弱音"
            "（預設: %(default)s）"
        ),
    )

    other_group.add_argument(
        "--other-min-note-separation",
        type=float,
        default=0.12,
        help=(
            "[extract 模式] Other 音符之間最短間隔（秒），"
            "越小越容易保留快速音符"
            "（預設: %(default)s）"
        ),
    )

    other_group.add_argument(
        "--other-max-notes-per-frame",
        type=int,
        default=1,
        help=(
            "[extract 模式] 每個 CQT frame 最多保留幾個音高峰。"
            "1=單音；2=雙音；3/4 可增加鋼琴和弦捕捉率"
            "（預設: %(default)s）"
        ),
    )

    other_group.add_argument(
        "--other-peak-min-separation",
        type=int,
        default=2,
        help=(
            "[extract 模式] 不同 CQT 峰之間至少相差多少半音，"
            "避免同一音附近的頻率峰重複計算"
            "（預設: %(default)s）"
        ),
    )

    other_group.add_argument(
        "--other-harmonic-tolerance",
        type=float,
        default=0.45,
        help=(
            "[extract 模式] 泛音匹配容許誤差（半音）。"
            "越大越積極合併泛音，鋼琴可用 0.35~0.60"
            "（預設: %(default)s）"
        ),
    )

    other_group.add_argument(
        "--other-harmonic-count",
        type=int,
        default=8,
        help=(
            "[extract 模式] 每個候選基頻最多檢查幾階泛音。"
            "越大越能辨識鋼琴泛音，但計算量也增加"
            "（預設: %(default)s）"
        ),
    )

    other_group.add_argument(
        "--other-harmonic-suppression",
        type=float,
        default=0.65,
        help=(
            "[extract 模式] 泛音候選最高抑制比例。"
            "0=不抑制，1=最多完全抑制"
            "（預設: %(default)s）"
        ),
    )

    other_group.add_argument(
        "--other-track-max-gap",
        type=float,
        default=0.06,
        help=(
            "[extract 模式] 多音符 track 允許短暫掉音多久（秒）。"
            "越大越容易把同一個鋼琴音連起來"
            "（預設: %(default)s）"
        ),
    )

    other_group.add_argument(
        "--other-cqt-fmin",
        default="C2",
        help="[extract 模式] CQT 最低音（預設: %(default)s）",
    )

    other_group.add_argument(
        "--other-cqt-fmax",
        default="C8",
        help="[extract 模式] CQT 最高音（預設: %(default)s）",
    )

    other_group.add_argument(
        "--other-no-filter",
        action="store_true",
        help=(
            "[extract/mix] 關閉 Other 的時間稀疏化與音符過濾。"
            "仍保留 CQT 峰值本身的基本判定"
        ),
    )

    other_group.add_argument(
        "--other-confidence-threshold",
        type=float,
        default=0.18,
        help=(
            "[extract/hybrid] Other 音符最低可信度。"
            "越高越乾淨，越低越容易保留弱音"
            "（預設: %(default)s）"
        ),
    )

    # ========================================================
    # NBS 輸出參數
    # ========================================================

    nbs_group = ap.add_argument_group(
        "NBS 輸出參數"
    )

    nbs_group.add_argument(
        "--other-retrigger-sec",
        type=float,
        default=0.0,
        help=(
            "Other 長音在 NBS 中重新觸發的間隔（秒）。"
            "0=不人工重敲（推薦，可避免迴音）；"
            ">0 才啟用週期性 sustain 重觸發"
            "（預設: %(default)s）"
        ),
    )

    nbs_group.add_argument(
        "--max-tick-shift",
        type=int,
        default=4,
        help=(
            "同一 layer 同一 tick 撞到其他音符時，"
            "最多往後找幾個 tick 的空位。"
            "設 0 等同回到舊行為（直接丟棄撞到的音符）"
            "（預設: %(default)s）"
        ),
    )

    args = ap.parse_args()

    # ========================================================
    # Other preset
    # ========================================================
    # preset 只提供一組方便的基準值；如果使用者在命令列
    # 明確指定某個 --other-* 參數，就保留使用者自己的值。
    # ========================================================
    if args.other_preset == "piano":
        argv = sys.argv[1:]

        def _arg_was_set(*names):
            return any(
                token == name or token.startswith(name + "=")
                for token in argv
                for name in names
            )

        piano_defaults = {
            "other_hop_length": 128,
            "other_min_note_dur": 0.025,
            "other_energy_threshold": 0.07,
            "other_min_note_separation": 0.025,
            "other_max_notes_per_frame": 4,
            "other_peak_min_separation": 2,
            "other_harmonic_tolerance": 0.45,
            "other_harmonic_count": 8,
            "other_harmonic_suppression": 0.65,
            "other_track_max_gap": 0.06,
            "other_cqt_fmin": "A0",
            "other_cqt_fmax": "C8",
        }

        option_names = {
            "other_hop_length": ("--other-hop-length",),
            "other_min_note_dur": ("--other-min-note-dur",),
            "other_energy_threshold": ("--other-energy-threshold",),
            "other_min_note_separation": ("--other-min-note-separation",),
            "other_max_notes_per_frame": ("--other-max-notes-per-frame",),
            "other_peak_min_separation": ("--other-peak-min-separation",),
            "other_harmonic_tolerance": ("--other-harmonic-tolerance",),
            "other_harmonic_count": ("--other-harmonic-count",),
            "other_harmonic_suppression": ("--other-harmonic-suppression",),
            "other_track_max_gap": ("--other-track-max-gap",),
            "other_cqt_fmin": ("--other-cqt-fmin",),
            "other_cqt_fmax": ("--other-cqt-fmax",),
        }

        for attr, value in piano_defaults.items():
            if not _arg_was_set(*option_names[attr]):
                setattr(args, attr, value)

        # Piano preset 預設開啟 no-filter；若使用者沒有明確關閉
        # （目前此 flag 是單向開啟），就直接啟用。
        args.other_no_filter = True
        if not _arg_was_set("--other-retrigger-sec"):
            args.other_retrigger_sec = 0.0

        print(
            "Other preset：PIANO（高密度、多音符、保留短音）"
        )
        print(
            f"  hop={args.other_hop_length}, "
            f"threshold={args.other_energy_threshold}, "
            f"max_notes={args.other_max_notes_per_frame}, "
            f"neural_max_polyphony={args.other_neural_max_polyphony}, "
            f"harmonic_suppression={args.other_harmonic_suppression}, "
            f"CQT={args.other_cqt_fmin}~{args.other_cqt_fmax}"
        )

    import librosa

    mp3_path = args.mp3

    out_path = (
        args.output
        or os.path.splitext(
            mp3_path
        )[0]
        + ".nbs"
    )

    song_name = (
        os.path.splitext(
            os.path.basename(
                mp3_path
            )
        )[0]
    )

    print()
    print(
        "========================================"
    )
    print(
        " MP3 → NBS 音樂轉譜"
    )
    print(
        "========================================"
    )
    print()

    # ========================================================
    # 工作目錄
    # ========================================================

    with tempfile.TemporaryDirectory() as tmp:

        if args.keep_temp:

            work_dir = os.path.join(
                os.path.dirname(
                    os.path.abspath(
                        mp3_path
                    )
                ),
                "_demucs_out",
            )

            os.makedirs(
                work_dir,
                exist_ok=True,
            )

        else:

            work_dir = tmp

        # ====================================================
        # [1/7] Demucs 4 Stem
        # ====================================================

        print()
        print(
            "[1/7] Demucs 4-Stem 分離"
        )

        stems = separate_4stems(
            mp3_path,
            work_dir,
            model=args.demucs_model,
            device=args.device,
            segment=args.segment,
        )

        print(
            "[1/7] 完成"
        )

        # ====================================================
        # [2/7] Load
        # ====================================================

        print()
        print(
            "[2/7] 載入 stems ..."
        )

        print(
            "  載入 vocals.wav ..."
        )

        y_vocal, sr_v = (
            librosa.load(
                stems["vocals"],
                sr=args.sr,
                mono=True,
            )
        )

        print(
            "  vocals.wav 完成"
        )

        print(
            "  載入 bass.wav ..."
        )

        y_bass, sr_b = (
            librosa.load(
                stems["bass"],
                sr=args.sr,
                mono=True,
            )
        )

        print(
            "  bass.wav 完成"
        )

        print(
            "  載入 other.wav ..."
        )

        y_other, sr_o = (
            librosa.load(
                stems["other"],
                sr=args.sr,
                mono=True,
            )
        )

        print(
            "  other.wav 完成"
        )

        # Other 的 harmonic stem 會被純音樂旋律 / 和弦 / CQT 重複使用。
        # 將結果持久化，第二次執行可直接載入，避免再花 ~18 秒做 HPSS。
        hpss_key = _cache_key_for_file(
            stems["other"],
            "hpss-v1",
            args.sr,
        )
        hpss_cache = os.path.join(
            _ensure_cache_dir("hpss"),
            hpss_key + ".npy",
        )
        if os.path.isfile(hpss_cache):
            print("  Other → 使用持久化 harmonic 快取")
            y_other_harm = np.load(hpss_cache, mmap_mode=None).astype(np.float32, copy=False)
            y_other_perc = None
        else:
            print("  Other → 建立 harmonic/percussive 快取 ...")
            y_other_harm, y_other_perc = librosa.effects.hpss(
                y_other,
                margin=(1.0, 3.0),
            )
            np.save(hpss_cache, np.asarray(y_other_harm, dtype=np.float32), allow_pickle=False)
            print("  Other → harmonic/percussive 快取完成")

        print(
            "  載入 drums.wav ..."
        )

        y_drums, sr_d = (
            librosa.load(
                stems["drums"],
                sr=args.sr,
                mono=True,
            )
        )

        print(
            "  drums.wav 完成"
        )

        assert (
            sr_v
            == sr_b
            == sr_o
            == sr_d
            == args.sr
        )

        print(
            "[2/7] 完成"
        )

        other_cache_key = _cache_key_for_file(
            stems["other"],
            "other-neural-v1",
            args.sr,
        )

        # ====================================================
        # [3/7] + [4/7] 主唱 + Bass pYIN
        #
        # 兩者是各自獨立的音軌分析，互不依賴，
        # 用多進程平行跑可以讓總時間接近
        # max(主唱時間, Bass時間) 而不是兩者相加。
        # ====================================================

        print()
        print(
            "[3/7][4/7] 主唱 + Bass pYIN 音高追蹤"
        )

        melody_kwargs = dict(
            fmin_note=args.melody_fmin,
            fmax_note=args.melody_fmax,
            min_dur=args.melody_min_dur,
            gap_tolerance=args.melody_gap_tolerance,
            voiced_prob_thresh=args.melody_voiced_prob,
            octave_shift=args.melody_octave_shift,
            pitch_persistence_sec=args.melody_persistence,
            medfilt_kernel_size=args.melody_medfilt_kernel,
            min_note_dur=max(0.0, float(args.melody_min_note_ms)) / 1000.0,
            rap_revise=args.rap_revise,
            name="主唱",
            device=args.device,
            model_path=args.melody_rmvpe_model,
            is_half=args.melody_rmvpe_half,
        )

        bass_kwargs = dict(
            fmin_note=args.bass_fmin,
            fmax_note=args.bass_fmax,

            frame_length=args.bass_frame_length,
            hop_length=args.bass_hop_length,

            min_dur=args.bass_min_dur,
            gap_tolerance=args.bass_gap_tolerance,

            voiced_prob_thresh=args.bass_voiced_prob,

            octave_shift=(
                args.bass_octave_shift
            ),

            pitch_persistence_sec=(
                args.bass_persistence
            ),

            medfilt_kernel_size=(
                args.bass_medfilt_kernel
            ),

            name="Bass",
        )

        # --------------------------------------------------
        # Vocal RMVPE + Bass pYIN 平行運算
        # --------------------------------------------------
        # 兩條 stem 完全獨立。改用 ThreadPoolExecutor 而不是
        # multiprocessing，避免 Windows spawn 複製大型 numpy array，
        # 也避免目前檔案裡原本不存在的 _melody_pyin_worker。
        # RMVPE ONNX 與 librosa/NumPy 的重運算大多會釋放 GIL，
        # 在這台多核心 CPU 上可以明顯縮短總等待時間。
        melody_kwargs["verbose_progress"] = False
        print("平行模式：主唱 RMVPE + Bass pYIN 同時運算中 ...")
        parallel_start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=2) as executor:
            melody_future = executor.submit(
                extract_melody_notes_rmvpe,
                y_vocal,
                args.sr,
                **melody_kwargs,
            )
            bass_future = executor.submit(
                extract_bass_notes_pyin,
                y_bass,
                args.sr,
                verbose_progress=False,
                **bass_kwargs,
            )
            melody_notes = melody_future.result()
            bass_notes = bass_future.result()

        print(
            f"平行分析完成：主唱 {len(melody_notes)} 個、"
            f"Bass {len(bass_notes)} 個，"
            f"耗時 {time.perf_counter() - parallel_start:.2f} 秒"
        )

        # ----------------------------------------------------
        # 純音樂段落 fallback
        #
        # mix 模式會在 [7/7] 自己依 vocals 能量選擇 extract，
        # 因此這裡不能再先補一次，否則同一段 Other 旋律會重複。
        # 其他模式維持原本的純音樂 fallback 行為。
        # ----------------------------------------------------
        if args.other_mode == "mix":
            print(
                "mix 模式：延後到 [7/7]，依 vocals.wav 能量切換 Other 策略"
            )
        else:
            instrumental_melody = extract_instrumental_melody(
                y_other,
                y_vocal,
                args.sr,
                hop_length=256,
                octave_shift=args.other_octave_shift,
                y_harm=y_other_harm,
            )

            if instrumental_melody:
                melody_notes = merge_melody_sources(
                    melody_notes,
                    instrumental_melody,
                )
                print(
                    f"純音樂段落：補入 {len(instrumental_melody)} 個 Other 旋律音符"
                )
            else:
                print("純音樂段落：沒有需要補入的旋律")

        print(
            "[3/7][4/7] 完成"
        )

        # ====================================================
        # [5/7] 和弦 + Drums 可並行
        #
        # 兩者都只讀自己的音訊；和弦也只依賴已完成的 Other harmonic
        # cache，因此不需要互相等待。
        #
        # 若 --enable-chord-analysis 未指定，analyze_chords 會退化
        # 為純 beat tracking，仍回傳 other_beat_times / chord_tempo，
        # 但 chords = []，因此後續和弦伴奏 / Bass 聯動 / 八度校正
        # 都會自動跳過和弦相關分支。
        # ====================================================

        if args.enable_chord_analysis:
            print()
            print("[5/7] Other 和弦分析 + [6/7] Drums 分析（平行）")
        else:
            print()
            print(
                "[5/7] Other 節拍分析（和弦分析已關閉）"
                " + [6/7] Drums 分析（平行）"
            )
        chord_drum_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as executor:
            chord_future = executor.submit(
                analyze_chords,
                y_other,
                args.sr,
                y_harm=y_other_harm,
                chord_analysis=args.enable_chord_analysis,
            )
            drums_future = executor.submit(
                analyze_drums,
                y_drums,
                args.sr,
            )
            (
                chords,
                other_beat_times,
                chord_tempo,
            ) = chord_future.result()
            (
                onset_times,
                onset_env,
                env_times,
                drum_beat_times,
                drum_tempo,
            ) = drums_future.result()

        print(
            f"和弦 + Drums 平行完成，耗時 {time.perf_counter() - chord_drum_start:.2f} 秒"
        )

        # ------------------------------------------------
        # 修正主唱偶爾抓錯八度的孤立音
        # ------------------------------------------------

        if (
            args.octave_correction
            and melody_notes
        ):

            if args.enable_chord_analysis:
                print(
                    "  主唱八度校正（前後文 + 和弦輔助） ..."
                )
            else:
                print(
                    "  主唱八度校正（僅前後文，和弦分析已關閉） ..."
                )

            melody_notes = (
                correct_melody_octaves(
                    melody_notes,
                    chords,
                )
            )

            print(
                f"  八度校正完成："
                f"{len(melody_notes)} 個音符"
            )

        # ------------------------------------------------
        # Bass 與和弦聯動：只做「八度 / 明顯錯音」修正，
        # 不把經過音全部硬改成 root，保留原曲 bass line。
        # 和弦關閉時 chords=[]，這個區塊會自動跳過。
        # ------------------------------------------------
        if bass_notes and chords:
            corrected_bass = []
            for start, end, pitch in bass_notes:
                mid = (start + end) * 0.5
                chord = _find_active_chord(chords, mid)
                p = int(pitch)
                if chord is not None:
                    root, quality = chord
                    root_pc = (48 + root) % 12
                    candidates = [p + 12 * k for k in range(-2, 3)]
                    # 不再依和弦 root 強迫改寫 Bass。
                    # 只在前後相鄰音符明顯支持 ±12/±24 的孤立八度錯誤時修正。
                    idx = len(corrected_bass)
                    neighbors = []
                    if idx > 0:
                        neighbors.append(int(corrected_bass[-1][2]))
                    if idx + 1 < len(bass_notes):
                        neighbors.append(int(bass_notes[idx + 1][2]))
                    if neighbors:
                        def context_score(candidate):
                            return sum(
                                min(
                                    abs(candidate - q),
                                    abs(candidate - (q - 12)),
                                    abs(candidate - (q + 12)),
                                )
                                for q in neighbors
                            )
                        best = min(candidates, key=context_score)
                        if abs(best - p) in (12, 24) and context_score(best) + 4 < context_score(p):
                            p = best
                corrected_bass.append((start, end, p))
            bass_notes = corrected_bass
            print(f"  Bass/Chord 聯動完成：{len(bass_notes)} 個音符")

        if args.enable_chord_analysis:
            print("[5/7] 和弦分析完成")
        else:
            print("[5/7] 和弦分析已關閉（僅節拍追蹤）")
        print("[6/7] Drums 分析完成")

        # ====================================================
        # Beat 選擇
        # ====================================================

        print()
        print(
            "選擇節拍來源 ..."
        )

        # 鼓的 beat 優先
        if len(
            drum_beat_times
        ) > 0:

            beat_times = (
                drum_beat_times
            )

            print(
                "使用 Drums Beat"
            )

        else:

            beat_times = (
                other_beat_times
            )

            print(
                "Drums 沒有 Beat，"
                "使用 Other Beat"
            )

        # ====================================================
        # 設計鼓組節奏（含音量起伏，避免節拍器感）
        # ====================================================

        print()
        print(
            "設計鼓組節奏 ..."
        )

        drum_hits = (
            design_drum_hits(
                beat_times,
                onset_env,
                env_times,
                hop_length=512,
                sr=args.sr,
                seed=args.seed,
                onset_times=onset_times,
                y_drums=y_drums,
            )
        )

        print(
            f"鼓組設計完成："
            f"{len(drum_hits)} 個鼓點"
        )

        # ====================================================
        # [7/7] 副旋律（Other）
        #
        # 四種模式：
        #   hybrid  優先使用 Other 實際音符，再用 chord 補缺
        #   chord   用和弦 + 節拍生成琶音伴奏
        #   extract 逐幀從 Other 音軌抓音符
        #   mix     有 vocals → chord；無 vocals → extract
        #
        # 若 --enable-chord-analysis 未啟用，hybrid/chord/mix
        # 都需要和弦資料，因此自動退化為 extract 模式，並印出警告。
        # ====================================================

        print()

        # ------------------------------------------------
        # 和弦關閉時的退化邏輯
        # ------------------------------------------------
        _chord_dependent_modes = ("hybrid", "chord", "mix")
        if (
            not args.enable_chord_analysis
            and args.other_mode in _chord_dependent_modes
        ):
            print(
                f"[警告] --other-mode={args.other_mode} 需要和弦資料，"
                f"但 --enable-chord-analysis 未啟用。"
            )
            print(
                "       自動退化為 extract 模式（純 CQT/Neural 抓音）。"
            )
            print(
                "       若需要完整和弦伴奏，請加上 --enable-chord-analysis。"
            )
            effective_other_mode = "extract"
        else:
            effective_other_mode = args.other_mode

        if effective_other_mode == "hybrid":

            print(
                "[7/7] Other HYBRID：實際音符優先 + 和弦補缺"
            )

            extracted_notes = run_other_extraction(
                y_other,
                args.sr,
                args,
                y_harm=y_other_harm,
                cache_key=other_cache_key,
            )

            generated_chords = generate_chord_accompaniment(
                chords,
                beat_times,
                subdivision=args.other_chord_subdivision,
                pattern=args.other_chord_pattern,
                octave_shift=args.other_octave_shift,
                note_len_ratio=args.other_chord_note_len_ratio,
                pattern_rotation=args.other_chord_rotation,
                skip_ratio=args.other_chord_skip,
                humanize_sec=args.other_chord_humanize,
                seed=args.seed,
            )

            # 只用和弦填「實際 Other 音符沒有覆蓋」的部分。
            # 若兩者在時間上重疊且 pitch 相同/相差一個八度，
            # 視為原曲已經有這個聲部，不再重複塞一顆機械音。
            other_notes = list(extracted_notes)
            fallback_count = 0
            for chord_note in generated_chords:
                cs, ce, cp = chord_note[:3]
                covered = False
                for extracted in extracted_notes:
                    es, ee, ep = extracted[:3]
                    if cs < ee and ce > es:
                        if abs(int(cp) - int(ep)) % 12 == 0:
                            covered = True
                            break
                if not covered:
                    other_notes.append(chord_note)
                    fallback_count += 1

            other_notes.sort(key=lambda x: (x[0], x[2]))
            print(
                f"  HYBRID：實際音符 {len(extracted_notes)} 個，"
                f"和弦補缺 {fallback_count} 個，"
                f"合計 {len(other_notes)} 個"
            )

        elif effective_other_mode == "chord":

            print(
                "[7/7] 用和弦生成副旋律"
            )

            other_notes = (
                generate_chord_accompaniment(
                    chords,
                    beat_times,

                    subdivision=(
                        args.other_chord_subdivision
                    ),

                    pattern=(
                        args.other_chord_pattern
                    ),

                    octave_shift=(
                        args.other_octave_shift
                    ),

                    note_len_ratio=(
                        args.other_chord_note_len_ratio
                    ),

                    pattern_rotation=(
                        args.other_chord_rotation
                    ),

                    skip_ratio=(
                        args.other_chord_skip
                    ),

                    humanize_sec=(
                        args.other_chord_humanize
                    ),

                    seed=args.seed,
                )
            )

        elif effective_other_mode == "extract":

            print(
                "[7/7] Other 樂器簡化音符分析（extract 模式）"
            )

            other_notes = run_other_extraction(
                y_other,
                args.sr,
                args,
                y_harm=y_other_harm,
                cache_key=other_cache_key,
            )

        else:
            # ------------------------------------------------
            # MIX：有人聲 → chord；無人聲 → extract
            # ------------------------------------------------
            print(
                "[7/7] Other MIX：依 vocals.wav 能量自動切換"
            )

            # chord 先整段生成，再依 vocal mask 只留下「有人聲」區段。
            chord_notes_all = generate_chord_accompaniment(
                chords,
                beat_times,
                subdivision=args.other_chord_subdivision,
                pattern=args.other_chord_pattern,
                octave_shift=args.other_octave_shift,
                note_len_ratio=args.other_chord_note_len_ratio,
                pattern_rotation=args.other_chord_rotation,
                skip_ratio=args.other_chord_skip,
                humanize_sec=args.other_chord_humanize,
                seed=args.seed,
            )

            # 用密集的 256-sample 時間軸判斷 vocals，而不是只看 beat。
            # 這樣一個 beat 中間開始/結束人聲時，也能正確切換。
            mix_times = np.arange(
                max(1, int(np.ceil(max(len(y_vocal), len(y_other)) / 256))),
                dtype=float,
            ) * 256.0 / args.sr

            vocal_active = build_vocal_activity_mask(
                y_vocal,
                args.sr,
                mix_times,
                hop_length=256,
            )

            vocal_chord_notes = filter_notes_by_vocal_activity(
                chord_notes_all,
                mix_times,
                vocal_active,
                keep_when_vocal=True,
            )

            print(
                f"  MIX chord：保留 {len(vocal_chord_notes)} / {len(chord_notes_all)} 個音符（有人聲區段）"
            )

            # extract 只在無人聲區段啟用。
            extracted_all = run_other_extraction(
                y_other,
                args.sr,
                args,
                y_harm=y_other_harm,
                cache_key=other_cache_key,
            )

            # extract 也使用同一個密集 vocal mask，保證切換邏輯一致。
            instrumental_extract_notes = filter_notes_by_vocal_activity(
                extracted_all,
                mix_times,
                vocal_active,
                keep_when_vocal=False,
            )

            print(
                f"  MIX extract：保留 {len(instrumental_extract_notes)} / {len(extracted_all)} 個音符（無人聲區段）"
            )

            # 兩邊的條件互斥，因此人聲段不會被 extract 污染，
            # 純音樂段也不會被 chord 生成的機械伴奏覆蓋。
            other_notes = (
                vocal_chord_notes
                + instrumental_extract_notes
            )
            other_notes.sort(key=lambda x: x[0])

            print(
                f"  MIX 完成：{len(other_notes)} 個 Other 音符"
            )

        print(
            "[7/7] 完成"
        )

        # ====================================================
        # NBS
        # ====================================================

        print()
        print(
            "封裝 NBS ..."
        )

        data = build_nbs_bytes(
            melody_notes=melody_notes,

            bass_notes=bass_notes,

            other_notes=other_notes,

            chords=chords,

            drum_hits=drum_hits,

            beat_times=beat_times,

            tps=args.tps,

            song_name=song_name,

            song_author="mp3_to_nbs",

            original_author="",

            description=(
                "Demucs 4-Stem "
                "vocals+bass+other+drums"
            ),

            song_duration=(
                len(y_other) / args.sr
            ),

            max_tick_shift=(
                args.max_tick_shift
            ),

            other_retrigger_sec=(
                args.other_retrigger_sec
            ),
        )

        print(
            "寫入 NBS ..."
        )

        with open(
            out_path,
            "wb",
        ) as f:

            f.write(data)

        print(
            "NBS 寫入完成"
        )

    elapsed = time.perf_counter() - conversion_start
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    print(
        f"轉換耗時：{minutes} 分 {seconds:.2f} 秒"
    )

    print()
    print(
        "================================"
    )
    print(
        "完成"
    )
    print(
        f"NBS：{out_path}"
    )
    print(
        "================================"
    )


def extract_bass_notes_pyin(y, sr, fmin_note="C2", fmax_note="C6",
                         frame_length=1024, hop_length=128,
                         min_dur=0.035, gap_tolerance=0.09,
                         voiced_prob_thresh=0.10, octave_shift=0,
                         name="旋律", pitch_persistence_sec=0.055,
                         medfilt_kernel_size=5, verbose_progress=True):
    """pYIN：不跨長靜音 forward-fill，最後才量化 pitch。"""
    import librosa
    from scipy.ndimage import median_filter

    print(f"{name} pYIN 音高追蹤 ...")
    f0, vflag, vprob = librosa.pyin(
        y, sr=sr,
        fmin=librosa.note_to_hz(fmin_note),
        fmax=librosa.note_to_hz(fmax_note),
        frame_length=frame_length,
        hop_length=hop_length,
        fill_na=np.nan,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    valid = vflag & ~np.isnan(f0) & (vprob >= voiced_prob_thresh)
    midi = np.full(len(f0), np.nan, dtype=float)
    midi[valid] = librosa.hz_to_midi(f0[valid])

    # 只填補短缺口；長靜音保持 NaN，避免把上一個音拖過整段休止。
    filled = midi.copy()
    valid_idx = np.flatnonzero(~np.isnan(filled))
    max_gap = max(0, int(round(gap_tolerance / (hop_length / sr))))
    if len(valid_idx) >= 2 and max_gap:
        for a, b in zip(valid_idx[:-1], valid_idx[1:]):
            gap = b - a - 1
            if 0 < gap <= max_gap:
                filled[a:b + 1] = np.linspace(filled[a], filled[b], b - a + 1)

    smooth = filled.copy()
    finite = ~np.isnan(smooth)
    kernel = max(1, int(medfilt_kernel_size))
    if kernel % 2 == 0:
        kernel += 1
    if finite.any() and kernel > 1:
        tmp = smooth.copy()
        tmp[~finite] = np.nanmedian(smooth[finite])
        filt = median_filter(tmp, size=kernel, mode="nearest")
        smooth[finite] = filt[finite]

    notes = []
    current = None
    start = None
    last_t = None
    pending = None
    pending_count = 0
    needed = max(1, int(round(pitch_persistence_sec / (hop_length / sr))))

    def flush(end_t):
        nonlocal current, start
        if current is None or start is None:
            return
        end_t = max(float(end_t), float(start))
        if end_t - start >= min_dur:
            notes.append((float(start), end_t, int(round(current)) + octave_shift * 12))
        current = None
        start = None

    total = len(times)
    last_percent = -1
    for i, t in enumerate(times):
        if verbose_progress and total:
            pct = int(i / total * 100)
            if pct != last_percent:
                show_progress(i, total, prefix=f"{name} pYIN")
                last_percent = pct

        if valid[i] and not np.isnan(smooth[i]):
            p = int(round(smooth[i]))
            if current is None:
                current, start = p, float(t)
                pending, pending_count = None, 0
            elif p == current:
                pending, pending_count = None, 0
            else:
                if pending == p:
                    pending_count += 1
                else:
                    pending, pending_count = p, 1
                if pending_count >= needed:
                    change_t = float(t - (needed - 1) * hop_length / sr)
                    old = current
                    old_start = start
                    if change_t - old_start >= min_dur:
                        notes.append((old_start, change_t, int(round(old)) + octave_shift * 12))
                    current, start = p, change_t
                    pending, pending_count = None, 0
            last_t = float(t)
        elif current is not None and last_t is not None and float(t) - last_t > gap_tolerance:
            flush(last_t + hop_length / sr)
            pending, pending_count, last_t = None, 0, None

    if verbose_progress:
        show_progress(total, total, prefix=f"{name} pYIN")
    if current is not None and last_t is not None:
        flush(last_t + hop_length / sr)

    merged = []
    for s, e, p in notes:
        if merged and p == merged[-1][2] and s - merged[-1][1] <= 0.02:
            merged[-1] = (merged[-1][0], e, p)
        elif e - s >= min_dur:
            merged.append((s, e, p))
    print(f"{name}：提取 {len(merged)} 個音符")
    return merged


def analyze_chords(y_other, sr, hop_length=512, y_harm=None, chord_analysis=True):
    """和弦：Chroma 時間平滑 + root bias + 低置信度保持上一和弦。

    當 chord_analysis=False 時，跳過 chroma CQT 與和弦辨識，
    只做 beat tracking，回傳 ([], beat_times, tempo)。
    這樣下游的 beat 選擇仍然有 other_beat_times 可用，
    但不會浪費時間計算 chords / 也不會產生和弦伴奏。
    """
    import librosa
    from scipy.ndimage import uniform_filter1d

    if not chord_analysis:
        # ------------------------------------------------
        # 純 beat tracking 模式：跳過所有 chroma 工作
        # ------------------------------------------------
        print("Other 和弦分析已關閉（僅做 beat tracking） ...")
        tempo, beat_frames = librosa.beat.beat_track(
            y=y_other, sr=sr, hop_length=hop_length, trim=False
        )
        beat_times = librosa.frames_to_time(
            beat_frames, sr=sr, hop_length=hop_length
        )
        tempo_val = float(np.atleast_1d(tempo)[0])
        print(f"  Tempo 約 {tempo_val:.1f} BPM")
        print("  和弦分析已關閉：chords = []")
        return [], beat_times, tempo_val

    print("Other 和弦分析 ...")
    if y_harm is None:
        y_harm, _ = librosa.effects.hpss(y_other, margin=(1.0, 3.0))
        print("  Other 和弦 → HPSS 完成")
    else:
        print("  Other 和弦 → 使用已快取的 harmonic stem")
    tempo, beat_frames = librosa.beat.beat_track(
        y=y_other, sr=sr, hop_length=hop_length, trim=False
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    tempo_val = float(np.atleast_1d(tempo)[0])
    duration = len(y_other) / sr

    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=hop_length)
    # 約 1.5 拍的平滑，抑制瞬間泛音造成的和弦跳動。
    beat_sec = 60.0 / max(tempo_val, 40.0)
    smooth_frames = max(3, int(round(beat_sec * 1.5 * sr / hop_length)))
    if smooth_frames % 2 == 0:
        smooth_frames += 1
    chroma_s = uniform_filter1d(chroma, size=smooth_frames, axis=1, mode="nearest")
    frame_times = np.arange(chroma_s.shape[1]) * hop_length / sr

    candidates = []
    chord_templates = (
        ("maj",   (0, 4, 7)),
        ("min",   (0, 3, 7)),
        ("7",     (0, 4, 7, 10)),
        ("maj7",  (0, 4, 7, 11)),
        ("min7",  (0, 3, 7, 10)),
        ("sus2",  (0, 2, 7)),
        ("sus4",  (0, 5, 7)),
        ("dim",   (0, 3, 6)),
    )
    for root in range(12):
        for quality, intervals in chord_templates:
            mask = np.zeros(12, dtype=float)
            for iv in intervals:
                mask[(root + iv) % 12] = 1.0
            candidates.append((root, quality, mask / np.linalg.norm(mask)))

    segs = np.unique(np.concatenate(([0.0], beat_times, [duration])))
    raw = []
    prev = None
    for i in range(len(segs) - 1):
        s, e = float(segs[i]), float(segs[i + 1])
        if e - s < 0.05:
            continue
        mask = (frame_times >= s) & (frame_times < e)
        if not mask.any():
            continue
        vec = chroma_s[:, mask].mean(axis=1)
        norm = np.linalg.norm(vec)
        if norm <= 1e-9:
            continue
        vec /= norm

        scored = []
        for root, quality, tmpl in candidates:
            score = float(np.dot(vec, tmpl))
            # Root 小幅加權，避免三和弦轉位時 root 被第三音搶走。
            score += float(vec[root]) * 0.12
            if prev == (root, quality):
                score += 0.035
            scored.append((score, root, quality))
        scored.sort(reverse=True)
        best_score, root, quality = scored[0]
        second = scored[1][0]
        if prev is not None and best_score - second < 0.025:
            root, quality = prev
        prev = (root, quality)
        raw.append((s, e, root, quality))

    merged = []
    for c in raw:
        if merged and c[2:] == merged[-1][2:] and c[0] - merged[-1][1] < 0.08:
            merged[-1] = (merged[-1][0], c[1], c[2], c[3])
        else:
            merged.append(c)

    print(f"  Tempo 約 {tempo_val:.1f} BPM")
    print(f"  偵測到 {len(merged)} 個和弦區段")
    return merged, beat_times, tempo_val


def generate_chord_accompaniment(
    chords, beat_times, subdivision=2, pattern="up", octave_shift=0,
    note_len_ratio=0.9, min_note_dur=0.05, downbeat_phase=0,
    pattern_rotation=True, velocity_base=48, accent_on_downbeat=14,
    accent_on_beat=6, skip_ratio=0.08, humanize_sec=0.008, seed=0,
):
    """伴奏：四小節輪換 pattern，和弦變化時強調 root，保留空拍。"""
    rng = np.random.default_rng(seed)
    notes = []
    if not chords or len(beat_times) < 2:
        return notes
    beats = np.asarray(sorted(beat_times), dtype=float)

    def chord_at(t):
        for s, e, root, quality in chords:
            if s <= t < e:
                return root, quality
        return None

    patterns = ([0, 2, 1, 2], [0, 1, 2, 1], [0, 2, 0, 1], [2, 1, 0, 1])
    subdivision = max(1, int(subdivision))
    for i in range(len(beats) - 1):
        s, e = float(beats[i]), float(beats[i + 1])
        if e <= s:
            continue
        active = chord_at((s + e) * 0.5)
        if active is None:
            continue
        root, quality = active
        quality_intervals = {
            "maj": (0, 4, 7),
            "min": (0, 3, 7),
            "7": (0, 4, 7, 10),
            "maj7": (0, 4, 7, 11),
            "min7": (0, 3, 7, 10),
            "sus2": (0, 2, 7),
            "sus4": (0, 5, 7),
            "dim": (0, 3, 6),
        }
        intervals = quality_intervals.get(quality, (0, 4, 7))
        tones = [60 + root + iv + octave_shift * 12 for iv in intervals]
        bar = (i - downbeat_phase) // 4
        phase = (i - downbeat_phase) % 4

        if pattern == "root_only":
            seq = [0]
        elif pattern == "down":
            seq = [2, 1, 0, 1]
        elif pattern == "up_down":
            seq = [0, 1, 2, 1]
        else:
            seq = list(patterns[bar % len(patterns)])
            if pattern_rotation and bar % 4 == 3:
                seq = list(reversed(seq))

        slot = (e - s) / subdivision
        for k in range(subdivision):
            if subdivision >= 4 and k in (1, 3) and (bar + phase) % 2:
                continue
            if k > 0 and rng.random() < skip_ratio * (1.0 + 0.15 * (bar % 3)):
                continue
            p = tones[seq[(phase + k) % len(seq)] % len(tones)]
            next_chord = chord_at(e + 1e-5)
            if k == 0 and next_chord is not None and next_chord != active:
                p = tones[0]
            t = max(s, min(e - 0.001, s + k * slot + rng.uniform(-humanize_sec, humanize_sec)))
            dur = min(slot, max(min_note_dur, slot * note_len_ratio))
            vel = velocity_base + (accent_on_downbeat if phase == 0 else accent_on_beat) if k == 0 else velocity_base - 5
            if phase == 0:
                vel += 4
            if next_chord is not None and next_chord != active and k == 0:
                vel += 4
            vel = _clip(vel + rng.integers(-3, 4), 20, 96)
            notes.append((float(t), float(min(e, t + dur)), int(p), int(vel)))

    # 相鄰和弦若品質/根音相同，避免重複生成完全相同的 attack。
    notes.sort(key=lambda x: x[0])
    deduped = []
    for note in notes:
        if deduped and note[2] == deduped[-1][2] and abs(note[0] - deduped[-1][0]) < 0.015:
            if note[3] > deduped[-1][3]:
                deduped[-1] = note
        else:
            deduped.append(note)
    print(f"和弦生成副旋律：產生 {len(deduped)} 個音符")
    return deduped


def analyze_drums(y_drums, sr, hop_length=512):
    """鼓分析保留原始 onset，供後面的頻譜分類使用。"""
    import librosa
    print("Drums 分析 ...")
    tempo, beat_frames = librosa.beat.beat_track(
        y=y_drums, sr=sr, hop_length=hop_length, trim=False
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    tempo_val = float(np.atleast_1d(tempo)[0])
    onset_env = librosa.onset.onset_strength(
        y=y_drums, sr=sr, hop_length=hop_length, aggregate=np.median
    )
    env_times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(
        y=y_drums, sr=sr, hop_length=hop_length, backtrack=False,
        units="frames", delta=0.08, wait=max(1, int(0.04 * sr / hop_length))
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    print(f"  Drums → Beat：{len(beat_times)}")
    print(f"  Drums → Onset：{len(onset_times)}")
    return onset_times, onset_env, env_times, beat_times, tempo_val


def design_drum_hits(beat_times, onset_strength_env, env_times,
                     hop_length=512, sr=22050, seed=0,
                     onset_times=None, y_drums=None):
    """依原始 drums onset + 頻譜重心分類 Kick/Snare/Hat。"""
    import librosa
    rng = np.random.default_rng(seed)
    if y_drums is None or onset_times is None or len(onset_times) == 0:
        # 舊 API 的安全 fallback。
        beats = np.asarray(beat_times, dtype=float)
        if len(beats) == 0:
            return []
        gap = float(np.median(np.diff(beats))) if len(beats) > 1 else 0.5
        return [
            (L_KICK if i % 4 in (0, 2) else L_SNARE,
             float(t),
             _clip(78 if i % 4 in (0, 2) else 72, 35, 100))
            for i, t in enumerate(beats)
        ]

    y = np.asarray(y_drums, dtype=float)
    onset_times = np.asarray(onset_times, dtype=float)
    n_fft = 2048
    hop = max(128, hop_length)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    low = freqs < 180.0
    mid = (freqs >= 180.0) & (freqs < 4000.0)
    high = freqs >= 4000.0
    env = np.asarray(onset_strength_env)
    env_ref = max(1e-6, float(np.percentile(env, 98))) if len(env) else 1.0

    hits = []
    for t in onset_times:
        frame = int(np.clip(round(t * sr / hop), 0, S.shape[1] - 1))
        spec = S[:, frame]
        total = float(spec.sum()) + 1e-9
        low_ratio = float(spec[low].sum()) / total
        mid_ratio = float(spec[mid].sum()) / total
        high_ratio = float(spec[high].sum()) / total
        env_i = int(np.clip(round(t * sr / hop_length), 0, len(env) - 1)) if len(env) else 0
        strength = float(env[env_i]) if len(env) else 0.0
        norm = min(1.0, strength / env_ref)

        # 一個 onset 可以同時包含 Kick + Hat、Snare + Hat 等多層鼓聲。
        # 不同 layer 可以同時存在；只有同一 layer 的近鄰重複 onset 才合併。
        layers = []
        if low_ratio >= 0.34 and low_ratio > high_ratio * 1.05:
            layers.append((L_KICK, 62))
        if high_ratio >= 0.30 and high_ratio > low_ratio * 0.65:
            layers.append((L_HAT, 36))
        if mid_ratio >= 0.20 and (mid_ratio + high_ratio) >= 0.32:
            layers.append((L_SNARE, 55))
        if not layers:
            centroid = float((freqs * spec).sum() / total)
            layers.append((L_KICK, 55) if centroid < 220 else (L_HAT, 34) if centroid > 5000 else (L_SNARE, 50))

        for layer, base in layers:
            vel = _clip(base + norm * 38 + rng.integers(-3, 4), 16, 100)
            duplicate = False
            for j in range(len(hits) - 1, -1, -1):
                old_layer, old_t, old_vel = hits[j]
                if float(t) - old_t >= 0.025:
                    break
                if old_layer == layer:
                    duplicate = True
                    if vel > old_vel:
                        hits[j] = (layer, float(t), vel)
                    break
            if not duplicate:
                hits.append((layer, float(t), vel))

    hits.sort(key=lambda x: x[1])
    print(f"  鼓組：依原始 onset 還原 {len(hits)} 個鼓點")
    return hits


def extract_instrumental_melody(
    y_other,
    y_vocal,
    sr,
    hop_length=256,
    octave_shift=0,
    fmin_note="C2",
    fmax_note="C7",
    y_harm=None,
):
    """在 vocals 缺席的區段，從 Other stem 找最突出的單音旋律。"""
    import librosa
    from scipy.ndimage import median_filter

    if len(y_other) == 0:
        return []

    print("Other 純音樂旋律搜尋 ...")

    # Other 先取 harmonic，盡量降低殘留鼓聲/瞬態的干擾。
    if y_harm is None:
        y_harm, _ = librosa.effects.hpss(y_other, margin=(1.0, 3.0))
        print("  Other 純音樂旋律 → HPSS 完成")
    else:
        print("  Other 純音樂旋律 → 使用已快取的 harmonic stem")

    # 用 pYIN 找「可能的主旋律」。這裡不直接整段採用，
    # 後面還會用 vocals 能量把真正的人聲區段排除。
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y_harm,
        sr=sr,
        fmin=librosa.note_to_hz(fmin_note),
        fmax=librosa.note_to_hz(fmax_note),
        frame_length=2048,
        hop_length=hop_length,
        fill_na=np.nan,
    )

    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    other_valid = (
        voiced_flag
        & np.isfinite(f0)
        & (voiced_prob >= 0.18)
    )
    other_midi = np.full(len(f0), np.nan, dtype=float)
    other_midi[other_valid] = librosa.hz_to_midi(f0[other_valid])

    # vocals 能量用 RMS；取較長窗口避免單一子音造成誤判。
    vocal_rms = librosa.feature.rms(
        y=y_vocal,
        frame_length=2048,
        hop_length=hop_length,
    )[0]
    vocal_times = librosa.times_like(
        vocal_rms,
        sr=sr,
        hop_length=hop_length,
    )

    if len(vocal_rms) == 0:
        vocal_db = np.full(len(times), -80.0)
    else:
        vocal_db = np.interp(
            times,
            vocal_times,
            librosa.amplitude_to_db(
                vocal_rms + 1e-8,
                ref=1.0,
            ),
            left=-80.0,
            right=-80.0,
        )

    # Other 自己也要有足夠能量，否則容易從 pad/殘響中抓到假旋律。
    other_rms = librosa.feature.rms(
        y=y_harm,
        frame_length=2048,
        hop_length=hop_length,
    )[0]
    other_times = librosa.times_like(
        other_rms,
        sr=sr,
        hop_length=hop_length,
    )
    other_db = np.interp(
        times,
        other_times,
        librosa.amplitude_to_db(
            other_rms + 1e-8,
            ref=1.0,
        ),
        left=-80.0,
        right=-80.0,
    )

    # 只有 vocals 明顯低於 Other 的區段才啟用 fallback。
    # -32 dB 已經是很保守的「近似沒有 vocals」。
    instrumental = (
        other_valid
        & (vocal_db <= -32.0)
        & (other_db >= -30.0)
    )

    # 短暫漏檢不要形成一堆碎音：只保留連續至少約 70ms 的候選。
    frame_dur = hop_length / sr
    min_frames = max(1, int(round(0.07 / frame_dur)))
    mask = instrumental.copy()
    run_start = None
    for i, ok in enumerate(np.r_[mask, False]):
        if ok and run_start is None:
            run_start = i
        elif not ok and run_start is not None:
            if i - run_start < min_frames:
                mask[run_start:i] = False
            run_start = None

    midi = other_midi.copy()
    finite = np.isfinite(midi) & mask
    if finite.any():
        tmp = midi.copy()
        fill = float(np.nanmedian(tmp[finite]))
        tmp[~np.isfinite(tmp)] = fill
        smooth = median_filter(tmp, size=5, mode="nearest")
        midi[finite] = smooth[finite]

    notes = []
    current = None
    start = None
    last_time = None
    pending = None
    pending_count = 0
    persistence = max(1, int(round(0.055 / frame_dur)))

    def flush(end_time):
        nonlocal current, start
        if current is not None and start is not None and end_time - start >= 0.06:
            notes.append((
                float(start),
                float(end_time),
                int(round(current)) + octave_shift * 12,
            ))
        current = None
        start = None

    for i, t in enumerate(times):
        if mask[i] and np.isfinite(midi[i]):
            p = int(round(midi[i]))
            if current is None:
                current, start = p, float(t)
                pending, pending_count = None, 0
            elif p == current:
                pending, pending_count = None, 0
            else:
                if pending == p:
                    pending_count += 1
                else:
                    pending, pending_count = p, 1
                if pending_count >= persistence:
                    change_t = float(t - (persistence - 1) * frame_dur)
                    flush(change_t)
                    current, start = p, change_t
                    pending, pending_count = None, 0
            last_time = float(t)
        elif current is not None and last_time is not None:
            if float(t) - last_time > 0.09:
                flush(last_time + frame_dur)
                pending, pending_count, last_time = None, 0, None

    if current is not None and last_time is not None:
        flush(last_time + frame_dur)

    # 太短/太密的候選再清一次。
    result = []
    for note in notes:
        if note[1] - note[0] < 0.06:
            continue
        if result and note[0] - result[-1][1] < 0.02 and note[2] == result[-1][2]:
            result[-1] = (result[-1][0], note[1], note[2])
        else:
            result.append(note)

    print(f"Other 純音樂旋律：找到 {len(result)} 個候選音符")
    return result


def merge_melody_sources(vocal_notes, instrumental_notes):
    """以 vocals 優先；只有沒有 vocals 的時間才加入 instrumental melody。"""
    if not instrumental_notes:
        return vocal_notes
    if not vocal_notes:
        return sorted(instrumental_notes, key=lambda x: x[0])

    result = list(vocal_notes)
    for ins_start, ins_end, ins_pitch in instrumental_notes:
        # 只要與任何人聲音符有實質重疊，就讓 vocals 優先。
        overlaps = False
        for v_start, v_end, _ in vocal_notes:
            if ins_start < v_end and ins_end > v_start:
                overlaps = True
                break
        if not overlaps:
            result.append((ins_start, ins_end, ins_pitch))

    result.sort(key=lambda x: (x[0], x[2]))
    return result


if __name__ == "__main__":
    main()
