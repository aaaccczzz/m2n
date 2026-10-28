#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Onsets & Frames (TFLite) -> NBS

純鋼琴/鋼琴主導音樂轉 NBS。
NBS 格式統一對齊 mp3_to_nbs.py 的 build_nbs_bytes()。
"""
from __future__ import annotations
import argparse, contextlib, io, os, shutil, struct, subprocess, sys, tempfile, urllib.request, wave
from dataclasses import dataclass
import numpy as np

MODEL_URL = "https://storage.googleapis.com/magentadata/models/onsets_frames_transcription/tflite/onsets_frames_wavinput.tflite"
MODEL_SR = 16000
MIDI_MIN, MIDI_MAX = 21, 108
FALLBACK_FFMPEG = r"C:\Users\acz\games\ffmpeg\ffmpeg.exe"

@dataclass
class Note:
    pitch: int
    start: float
    end: float
    velocity: int

def sigmoid(x):
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))

def find_ffmpeg(explicit=None):
    import shutil
    candidates = [explicit, FALLBACK_FFMPEG, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ffmpeg", "ffmpeg.exe")), shutil.which("ffmpeg")]
    for p in candidates:
        if p and os.path.isfile(p): return os.path.abspath(p)
    raise RuntimeError("找不到 ffmpeg。請使用 --ffmpeg 指定 ffmpeg.exe。")

def convert_to_wav(src, dst, ffmpeg_path=None):
    ffmpeg = find_ffmpeg(ffmpeg_path)
    print(f"使用 FFmpeg：{ffmpeg}")
    p = subprocess.run([ffmpeg, "-y", "-i", src, "-vn", "-ac", "1", "-ar", str(MODEL_SR), "-sample_fmt", "s16", dst], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode:
        err = p.stderr.decode("utf-8", errors="replace")
        raise RuntimeError("ffmpeg 轉 WAV 失敗：\n" + err[-3000:])

def load_wav(path):
    with wave.open(path, "rb") as w:
        channels, width, sr = w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(w.getnframes())
    if width == 2: y = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4: y = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else: raise RuntimeError(f"只支援 16/32-bit WAV，目前是 {width * 8}-bit")
    if channels > 1: y = y.reshape(-1, channels).mean(axis=1)
    if sr != MODEL_SR: raise RuntimeError(f"模型需要 {MODEL_SR} Hz WAV，但收到 {sr} Hz")
    return np.nan_to_num(y.astype(np.float32))

def download_model(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    print(f"下載 Onsets & Frames 模型：{MODEL_URL}\n→ {path}")
    tmp = path + ".part"
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
        if os.path.getsize(tmp) < 1024 * 1024: raise RuntimeError("下載檔案異常，檔案太小")
        os.replace(tmp, path)
    except Exception as e:
        try: os.remove(tmp)
        except OSError: pass
        raise RuntimeError(f"模型下載失敗：{e}")

def get_interpreter(model_path, num_threads=None):
    """取得 TFLite/LiteRT interpreter。

    優先使用新版 ai-edge-litert，因為它提供 Python 3.13 + Windows
    x86-64 的 wheel；其次才嘗試 TensorFlow 與舊版 tflite-runtime。
    這樣整個專案可以維持 Python 3.13，不需要為了這個模型退回 3.11。

    num_threads：CPU 推理執行緒數。None 時用 min(8, CPU核心)。
    多 worker 平行時，每顆 interpreter 應給較小的 num_threads，
    避免總執行緒數超過核心數造成過度訂閱。
    """
    errors = []

    # 目前推薦的獨立 LiteRT runtime。
    if num_threads is None:
        num_threads = max(1, min(8, os.cpu_count() or 4))
    else:
        num_threads = max(1, int(num_threads))

    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter(model_path=model_path, num_threads=num_threads)
    except ImportError as e:
        errors.append(f"ai_edge_litert: {e}")
    except Exception as e:
        errors.append(f"ai_edge_litert: {type(e).__name__}: {e}")

    # 舊環境仍可使用 TensorFlow 內建的 TFLite interpreter。
    try:
        import tensorflow as tf
        return tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
    except ImportError as e:
        errors.append(f"tensorflow: {e}")
    except Exception as e:
        errors.append(f"tensorflow: {type(e).__name__}: {e}")

    # 最後保留舊版 tflite-runtime 相容路徑。
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter(model_path=model_path, num_threads=num_threads)
    except ImportError as e:
        errors.append(f"tflite_runtime: {e}")
    except Exception as e:
        errors.append(f"tflite_runtime: {type(e).__name__}: {e}")

    raise RuntimeError(
        "缺少可用的 TFLite/LiteRT runtime。\n"
        "Python 3.13 + Windows 建議安裝：py -3.13 -m pip install ai-edge-litert\n"
        "嘗試結果：\n  - " + "\n  - ".join(errors)
    )

@contextlib.contextmanager
def _suppress_stderr_fd():
    """暫時把 C 層 stderr（fd 2）導向 os.devnull。

    LiteRT / Dawn 初始化會直接對 fd 2 印大量 INFO 噪音，
    Python 層的 redirect_stderr 攔不到。任何步驟失敗就直接放行。
    """
    try:
        sys.stderr.flush()
    except Exception:
        pass
    saved = None
    devnull = None
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        saved = os.dup(2)
        os.dup2(devnull, 2)
    except OSError:
        pass
    try:
        yield
    finally:
        if saved is not None:
            try:
                os.dup2(saved, 2)
            except OSError:
                pass
            os.close(saved)
        if devnull is not None:
            os.close(devnull)


def ensure_webgpu_runtime_dlls():
    """把 tools/dxc 的 DXC DLL 複製到 LiteRT 套件目錄（WebGPU 著色器編譯用）。

    Dawn（WebGPU backend）在自己的模組目錄搜尋 dxil.dll /
    dxcompiler.dll；把檔案放進 site-packages/ai_edge_litert 是實測
    有效的做法（設定 PATH 無效）。目標已存在就跳過。
    回傳 None 代表就緒，否則回傳錯誤訊息（不丟例外）。
    """
    try:
        import ai_edge_litert
    except Exception as e:
        return f"ai_edge_litert 未安裝：{e}"

    pkg_dir = os.path.dirname(os.path.abspath(ai_edge_litert.__file__))
    src_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "tools", "dxc"
    )
    if not os.path.isdir(src_dir):
        if os.path.isfile(os.path.join(pkg_dir, "dxil.dll")):
            return None
        return f"找不到 {src_dir}，且套件目錄沒有 dxil.dll"

    copied = []
    for name in sorted(os.listdir(src_dir)):
        if not name.lower().endswith(".dll"):
            continue
        dst = os.path.join(pkg_dir, name)
        if os.path.isfile(dst):
            continue
        try:
            shutil.copy2(os.path.join(src_dir, name), dst)
            copied.append(name)
        except OSError as e:
            return f"複製 {name} 失敗：{e}"
    if copied:
        print(f"  WebGPU：已複製 {', '.join(copied)} → {pkg_dir}")
    return None


class WebGPUOFModel:
    """LiteRT CompiledModel（WebGPU / D3D12）後端，介面與 OFModel 相容。

    - 以 enforce_f32=True 執行：與 tflite CPU 的 logits 差約 1e-4。
    - 少數 op（GATHER/PAD）在 CPU 執行，其餘 2715 個 op 在 GPU。
    - 構造時先跑一顆零窗暖機（觸發著色器編譯並驗證可執行）。
    - 需要 dxil.dll / dxcompiler.dll（見 ensure_webgpu_runtime_dlls）。
    """

    backend = "webgpu"

    def __init__(self, model_path, num_threads=None):
        self.model_path = os.path.abspath(model_path)
        self.num_threads = None  # GPU 推理不使用 CPU 執行緒數

        err = ensure_webgpu_runtime_dlls()
        if err:
            print(f"  [WARN] WebGPU DLL 準備失敗：{err}")

        try:
            with _suppress_stderr_fd():
                from ai_edge_litert.compiled_model import CompiledModel
                from ai_edge_litert.hardware_accelerator import HardwareAccelerator
                from ai_edge_litert import options as _options

                opt = _options.Options(
                    hardware_accelerators=(
                        HardwareAccelerator.GPU
                        | HardwareAccelerator.CPU
                    )
                )
                try:
                    opt.gpu_options.enforce_f32 = True
                except Exception:
                    pass

                self.cm = CompiledModel.from_file(self.model_path, options=opt)
                sigs = self.cm.get_signature_list()
                self.sig = list(sigs)[0]
                info = sigs[self.sig]
                self.in_name = info["inputs"][0]
                self.out_names = [
                    n
                    for n in (
                        "frame_logits",
                        "onset_logits",
                        "offset_logits",
                        "velocity_values",
                    )
                    if n in info["outputs"]
                ]
                if len(self.out_names) != 4:
                    raise RuntimeError(
                        f"模型輸出缺少 O&F logits：{info['outputs']}"
                    )
                in_details = self.cm.get_input_tensor_details(self.sig)
                out_details = self.cm.get_output_tensor_details(self.sig)
                self.in_buf = self.cm.create_input_buffer_by_name(
                    self.sig, self.in_name
                )
                self.out_bufs = {
                    n: self.cm.create_output_buffer_by_name(self.sig, n)
                    for n in self.out_names
                }
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"WebGPU 模型初始化失敗（{type(e).__name__}: {e}）。"
                "常見原因：顯卡/驅動不支援 D3D12，或缺少 "
                "tools/dxc/dxcompiler.dll 與 dxil.dll。"
            )

        in_shape = [int(v) for v in in_details[self.in_name]["shape"]]
        self.input_len = int(np.prod(in_shape)) if in_shape else 0
        out_shape = [
            int(v) for v in out_details[self.out_names[0]]["shape"]
        ]
        self.output_len = int(out_shape[1]) if len(out_shape) >= 2 else 0
        self.output_channels = int(out_shape[-1]) if out_shape else 0
        self.window = 2048
        if (
            self.output_len <= 1
            or (self.input_len - self.window) % (self.output_len - 1)
        ):
            raise RuntimeError(
                f"無法從模型推導 hop："
                f"input={self.input_len}, output={self.output_len}"
            )
        self.hop = (self.input_len - self.window) // (self.output_len - 1)
        self.timestep = self.hop / MODEL_SR
        print(f"O&F WebGPU：input {self.input_len} samples")
        print(f"O&F WebGPU：output {self.output_len} frames")
        print(
            f"O&F WebGPU 時間解析度：{self.timestep * 1000:.2f} ms"
            "（enforce_f32）"
        )

        # 暖機：第一次執行含著色器編譯（數秒），先跑一顆零窗，
        # 讓真正的推理路徑不會被首窗拖慢；同時驗證 GPU 可執行。
        with _suppress_stderr_fd():
            self.infer(np.zeros(self.input_len, dtype=np.float32))

    def infer(self, samples):
        x = np.asarray(samples, dtype=np.float32)
        if len(x) < self.input_len:
            x = np.pad(x, (0, self.input_len - len(x)))
        elif len(x) > self.input_len:
            x = x[: self.input_len]
        self.in_buf.write(x)
        self.cm.run_by_name(
            self.sig, {self.in_name: self.in_buf}, self.out_bufs
        )
        outs = []
        for name in self.out_names:
            flat = self.out_bufs[name].read(
                self.output_len * self.output_channels, np.float32
            )
            outs.append(
                flat.reshape(1, self.output_len, self.output_channels)
            )
        return np.transpose(np.concatenate(outs, axis=0), [1, 2, 0])


class OFModel:
    def __init__(self, model_path, num_threads=None):
        self.model_path = os.path.abspath(model_path)
        self.num_threads = num_threads
        self.interpreter = get_interpreter(model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.inputs = self.interpreter.get_input_details()
        self.outputs = self.interpreter.get_output_details()
        self.output_by_name = {x["name"]: x["index"] for x in self.outputs}
        inp = self.inputs[0]
        self.input_index = inp["index"]
        self.input_len = int(inp["shape"][0])
        # 官方 realtime 模型：output shape [1, frames, 88]
        shape = self.outputs[0]["shape"]
        self.output_len = int(shape[1])
        self.window = 2048
        if self.output_len <= 1 or (self.input_len - self.window) % (self.output_len - 1):
            raise RuntimeError(f"無法從模型推導 hop：input={self.input_len}, output={self.output_len}")
        self.hop = (self.input_len - self.window) // (self.output_len - 1)
        self.timestep = self.hop / MODEL_SR
        print(f"模型 input：{self.input_len} samples")
        print(f"模型 output：{self.output_len} frames")
        print(f"時間解析度：{self.timestep * 1000:.2f} ms")
    def infer(self, samples):
        x = np.asarray(samples, dtype=np.float32)
        if len(x) < self.input_len: x = np.pad(x, (0, self.input_len-len(x)))
        elif len(x) > self.input_len: x = x[:self.input_len]
        self.interpreter.set_tensor(self.input_index, x)
        self.interpreter.invoke()
        def get(name):
            if name not in self.output_by_name: raise RuntimeError(f"模型缺少輸出 {name}；實際輸出：{list(self.output_by_name)}")
            return self.interpreter.get_tensor(self.output_by_name[name])
        return np.transpose(np.concatenate([get("frame_logits"), get("onset_logits"), get("offset_logits"), get("velocity_values")], axis=0), [1,2,0])

def _resolve_jobs(jobs=None):
    """jobs<=0（auto）：CPU 核心數/2，上限 4。單核機器回 1。"""
    try:
        jobs = int(jobs or 0)
    except (TypeError, ValueError):
        jobs = 0
    if jobs > 0:
        return jobs
    try:
        cpu = int(os.cpu_count() or 2)
    except Exception:
        cpu = 2
    return max(1, min(4, cpu // 2))


def infer_audio(model, samples, overlap_timesteps=4, jobs=1, models=None, verbose=True):
    """O&F 全曲推理。jobs>1 時用多顆獨立 interpreter 平行跑 windows。

    每顆 interpreter 有自己的 session（TFLite session 非執行緒安全，
    不可共用）。models 可傳入預建的 OFModel list；沒傳則按需建立。
    輸出順序與序列版完全一致（按全域 window 編號保序拼接）。
    verbose=False 時不印 windows/chunk 進度（供背景執行緒使用）。
    """
    overlap = model.hop * overlap_timesteps + model.window
    stride = max(1, model.input_len - overlap)
    x = np.pad(samples, (0, max(0, model.input_len-len(samples))))
    starts = list(range(0, max(1, len(x)-model.input_len+1), stride))
    last = max(0, len(x)-model.input_len)
    if not starts or starts[-1] != last: starts.append(last)
    results, times = [], []
    if verbose:
        print(f"Onsets & Frames 推理：{len(starts)} 個 windows")
    jobs = max(1, int(jobs or 1))

    if jobs <= 1 or len(starts) <= 1:
        for n,start in enumerate(starts,1):
            pred = model.infer(x[start:start+model.input_len])
            cut = overlap_timesteps if n > 1 else 0
            pred = pred[cut:]
            local_start = start + cut*model.hop
            results.append(pred)
            times.append(local_start/MODEL_SR + np.arange(len(pred))*model.timestep)
            if verbose:
                print(f"\r  window {n}/{len(starts)}", end="", flush=True)
        if verbose:
            print()
        return np.concatenate(results), np.concatenate(times)

    # ---- 平行路徑 ----
    import concurrent.futures as _cf
    workers = list(models) if models else [model]
    while len(workers) < jobs:
        workers.append(
            OFModel(model.model_path, num_threads=getattr(model, "num_threads", None))
        )
    workers = workers[:jobs]

    indexed = list(enumerate(starts, 1))  # (全域編號n, start)
    chunk_size = max(1, (len(indexed) + jobs - 1) // jobs)
    chunks = [indexed[i:i + chunk_size] for i in range(0, len(indexed), chunk_size)]

    def _run(chunk, worker):
        out = []
        for n, start in chunk:
            pred = worker.infer(x[start:start + worker.input_len])
            cut = overlap_timesteps if n > 1 else 0
            pred = pred[cut:]
            local_start = start + cut * worker.hop
            out.append((
                pred,
                local_start / MODEL_SR + np.arange(len(pred)) * worker.timestep,
            ))
        return out

    with _cf.ThreadPoolExecutor(max_workers=len(chunks)) as ex:
        futs = [
            ex.submit(_run, ch, workers[i % len(workers)])
            for i, ch in enumerate(chunks)
        ]
        done = 0
        for f in futs:  # 按 chunk 順序取回，保序拼接
            for pred, tt in f.result():
                results.append(pred)
                times.append(tt)
            done += 1
            if verbose:
                print(f"\r  chunk {done}/{len(chunks)}", end="", flush=True)
    if verbose:
        print()
    return np.concatenate(results), np.concatenate(times)

def predictions_to_notes(pred, times, onset_threshold=.45, frame_threshold=.40, min_duration=.035, max_duration=12.0):
    frame_p, onset_p = sigmoid(pred[:,:,0]), sigmoid(pred[:,:,1])
    offset_p, velocity_p = sigmoid(pred[:,:,2]), np.clip(pred[:,:,3],0,1)
    notes=[]; n_frames=len(times)
    if n_frames == 0: return notes
    dt = float(np.median(np.diff(times))) if n_frames > 1 else .032
    for pi,midi in enumerate(range(MIDI_MIN,MIDI_MAX+1)):
        active=False; start_idx=0; velocity=64
        for t in range(n_frames):
            on=onset_p[t,pi]>=onset_threshold; fr=frame_p[t,pi]>=frame_threshold; off=offset_p[t,pi]>=.5
            if not active:
                if on: active=True; start_idx=t; velocity=int(round(20+107*velocity_p[t,pi]))
                continue
            if on and t>start_idx:
                end=float(times[t])
                if end-float(times[start_idx])>=min_duration: notes.append(Note(midi,float(times[start_idx]),end,velocity))
                start_idx=t; velocity=int(round(20+107*velocity_p[t,pi])); continue
            if off and not fr:
                end=float(times[min(t+1,n_frames-1)])
                if end-float(times[start_idx])>=min_duration: notes.append(Note(midi,float(times[start_idx]),min(end,float(times[start_idx])+max_duration),velocity))
                active=False; continue
            if not fr and np.max(frame_p[t:min(n_frames,t+3),pi]) < frame_threshold:
                end=float(times[t])
                if end-float(times[start_idx])>=min_duration: notes.append(Note(midi,float(times[start_idx]),end,velocity))
                active=False
        if active:
            end=float(times[-1]+dt)
            if end-float(times[start_idx])>=min_duration: notes.append(Note(midi,float(times[start_idx]),min(end,float(times[start_idx])+max_duration),velocity))
    notes.sort(key=lambda n:(n.start,n.pitch))
    return notes

def quantize_notes(notes, ticks_per_second):
    return [(int(round(n.start*ticks_per_second)), int(round(n.end*ticks_per_second)), n.pitch, n.velocity) for n in notes]

def _clip(v, lo, hi):
    return max(lo, min(hi, int(round(v))))

def midi_to_nbs_key(midi):
    """MIDI note -> NBS key。對齊 mp3_to_nbs.py 的映射。

    NBS 標準：key 0 = MIDI 21 (A0)、key 87 = MIDI 108 (C8)。
    舊版使用 midi - 33 會整體低一個八度，且 MIDI < 33 的音
    （鋼琴低音域 A0~A1）全部被夾成 key 0。
    """
    return _clip(midi - 21, 0, 87)

L_MELODY = 0
LAYER_COUNT = 1

LAYER_NAMES = {
    L_MELODY: "Piano",
}

INSTR = {
    'harp': 0,
}

def build_nbs_bytes(notes, tps=20.0, song_name="Onsets & Frames",
                    max_tick_shift=4, song_duration=None):
    """將 quantized notes 打包成 NBS v5 bytes，格式對齊 mp3_to_nbs.py。"""
    notes_by_tick = {}

    def add_note(tick, instrument, key, velocity, panning=100, pitch=0,
                 collision_mode="shift"):
        key = _clip(key, 0, 87)
        velocity = _clip(velocity, 0, 100)
        d = notes_by_tick.setdefault(tick, {})

        final_tick = tick
        if collision_mode == "shift":
            shift = 0
            while final_tick in d and shift < max_tick_shift:
                final_tick += 1
                shift += 1
            if final_tick in d:
                return
        elif final_tick in d:
            existing = d[final_tick]
            existing_velocity = int(existing[2])
            if collision_mode == "replace_weaker":
                if velocity <= existing_velocity:
                    return
            else:
                return

        d[final_tick] = (instrument, key, velocity, panning, pitch)

    for tick, midi, velocity in notes:
        key = midi_to_nbs_key(midi)
        add_note(tick, INSTR["harp"], key, velocity, collision_mode="shift")

    all_ticks = sorted(notes_by_tick.keys())
    if not all_ticks:
        raise RuntimeError("沒有可輸出的 NBS 音符")

    audio_length_tick = 0
    if song_duration is not None:
        audio_length_tick = max(0, int(round(float(song_duration) * tps)))

    song_length = max(all_ticks[-1], audio_length_tick)
    song_length = min(song_length, 65535)

    buf = io.BytesIO()

    def w_u8(v): buf.write(struct.pack("<B", v))
    def w_i16(v): buf.write(struct.pack("<h", v))
    def w_u16(v): buf.write(struct.pack("<H", v))
    def w_i32(v): buf.write(struct.pack("<i", v))
    def w_str(s):
        b = s.encode("utf-8")
        w_i32(len(b))
        buf.write(b)

    VERSION = 5
    VANILLA_COUNT = 16

    w_u16(0)
    w_u8(VERSION)
    w_u8(VANILLA_COUNT)
    w_u16(song_length)
    w_u16(LAYER_COUNT)

    w_str(song_name)
    w_str("Onsets & Frames")
    w_str("")
    w_str("Neural piano transcription")

    w_u16(int(round(tps * 100)))
    w_u8(0)
    w_u8(0)
    w_u8(4)

    for _ in range(5):
        w_i32(0)

    w_str("")
    w_u8(0)
    w_u8(0)
    w_u16(0)

    last_tick = -1
    for tick in all_ticks:
        w_u16(tick - last_tick)
        last_tick = tick

        entries = sorted(notes_by_tick[tick].items(), key=lambda x: x[0])
        last_layer = -1

        for layer, (instrument, key, velocity, panning, pitch) in entries:
            w_u16(layer - last_layer)
            last_layer = layer
            w_u8(instrument)
            w_u8(key)
            w_u8(velocity)
            w_u8(panning)
            w_i16(pitch)

        w_u16(0)

    w_u16(0)

    for i in range(LAYER_COUNT):
        w_str(LAYER_NAMES.get(i, f"Layer {i}"))
        w_u8(0)
        w_u8(100)
        w_u8(100)

    w_u8(0)

    return buf.getvalue()

def main():
    ap=argparse.ArgumentParser(description='Onsets & Frames TFLite → NBS')
    ap.add_argument('input'); ap.add_argument('-o','--output',required=True); ap.add_argument('--model',default=None); ap.add_argument('--ffmpeg',default=None)
    ap.add_argument('--tempo',type=int,default=20); ap.add_argument('--onset-threshold',type=float,default=.45); ap.add_argument('--frame-threshold',type=float,default=.40); ap.add_argument('--min-duration',type=float,default=.035); ap.add_argument('--keep-wav',action='store_true'); ap.add_argument('--jobs',type=int,default=0,help='O&F 推理平行度（獨立 interpreter 數）；0=自動（CPU核心/2，上限4），1=序列')
    args=ap.parse_args()
    if not os.path.isfile(args.input): print(f'[ERROR] 找不到輸入：{args.input}'); return 2
    base=os.path.dirname(os.path.abspath(__file__)); model_path=args.model or os.path.join(base,'models','onsets_frames_uni.tflite')
    try:
        if not os.path.isfile(model_path): download_model(model_path)
        model=OFModel(model_path)
        with tempfile.TemporaryDirectory(prefix='of_nbs_') as td:
            wav=os.path.join(td,'input.wav'); print('[1/4] FFmpeg → 16 kHz mono WAV ...'); convert_to_wav(args.input,wav,args.ffmpeg); samples=load_wav(wav); print(f'音訊長度：{len(samples)/MODEL_SR:.2f} 秒')
            print('[2/4] Onsets & Frames neural inference ...'); jobs=_resolve_jobs(args.jobs); print(f'推理平行度：{jobs}'); pred,times=infer_audio(model,samples,jobs=jobs)
            print('[3/4] onset/frame → piano notes ...'); notes=predictions_to_notes(pred,times,args.onset_threshold,args.frame_threshold,args.min_duration); print(f'偵測到 {len(notes)} 個音符')
            if not notes: raise RuntimeError('模型沒有偵測到音符；可嘗試降低 --onset-threshold')
            qnotes = quantize_notes(notes, args.tempo)
            tick_midi_vel = [(t, p, v) for t, _, p, v in qnotes]
            print('[4/4] 寫入 NBS ...')
            nbs_data = build_nbs_bytes(
                tick_midi_vel,
                tps=float(args.tempo),
                song_name=os.path.splitext(os.path.basename(args.input))[0],
                max_tick_shift=4,
                song_duration=len(samples) / MODEL_SR,
            )
            with open(os.path.abspath(args.output), 'wb') as f:
                f.write(nbs_data)
            print(f'NBS：{args.output}')
            print(f'音符：{len(tick_midi_vel)}')
            if args.keep_wav:
                import shutil; kept=os.path.splitext(os.path.abspath(args.output))[0]+'_16k.wav'; shutil.copy2(wav,kept); print(f'WAV：{kept}')
        print('完成。'); return 0
    except KeyboardInterrupt: print('\n[ERROR] 使用者中止'); return 130
    except Exception as e: print(f'[ERROR] {type(e).__name__}: {e}'); return 1

if __name__=='__main__': raise SystemExit(main())
