"""
Mako G-032B Camera Control - One H5 file per trigger (STREAMING-SAFE)
======================================================================
Software trigger: collects next N frames from the live stream.
Hardware trigger: each incoming frame = one trigger.
Saves one HDF5 file per trigger, plus optional PNGs.

Fix history
-----------
v4 -> v5:
  - start_streaming buffer_count raised to 10 (was default 3) to prevent
    frame-buffer starvation under hardware trigger.
  - H5 filenames now use microsecond-precision timestamps to avoid
    collisions when two triggers arrive within the same second.
  - _configure_camera_trigger: added TriggerOverlap="Off" to prevent
    camera state-machine lockup after 2-3 hardware triggers on GigE.
  - Added AcquisitionFrameRateAbs / AcquisitionFrameRateLimit (10 Hz
    default) to cap acquisition rate and avoid GigE overruns.
  - Added GevSCPSPacketSize and GevSCPD conservative defaults for GigE
    transport stability.
v5 -> v6:
  - Added LineDebounceTime on Line1 (default 100 us, user-adjustable).
    Eliminates spurious self-triggering when Line1 is floating/unconnected
    or when the signal has noise/bounce shorter than the debounce window.
  - Added "Debounce time (us)" spinbox in the Acquisition tab.
v6 -> v7:
  - Fixed root cause of spurious hardware triggers when GPIO connector is
    physically disconnected: free-run frames already buffered in the VmbPy
    pipeline at the moment TriggerMode=On is applied were being incorrectly
    counted as real hardware triggers. Added a 0.5 s post-arm discard window
    in frame_handler to reject these pre-buffered frames.
v7 -> v8:
  - LineDebounce block made firmware-resilient: tries known feature name
    variants and prints a clear message if none are supported.
  - Added 'Dump All Camera Features to Console' button in Diagnostics
    section of Acquisition tab to identify correct feature names for any
    firmware version.
v8 -> v9:
  - SyncInGlitchFilter (correct Mako G-032B feature name, via SyncIn1
    selector) replaces the incorrect LineSelector/LineDebounceTime names.
    Default 100 us, user-adjustable via Debounce Time spinbox.
  - Fixed camera hang after Stop Streaming: added explicit stop_streaming()
    call before the camera context exits, ensuring all in-flight
    frame_handler callbacks complete and VmbPy releases its locks cleanly.
    Previously the camera required a power-cycle after every session.
  - Fixed 0 fps live view after Stop Acquisition in hardware trigger mode:
    TriggerMode is now set back to Off when acquisition stops, returning
    the camera to free-run for the live preview.
v9 -> v10:
  - Fixed fps dropping to 10 after Stop Acquisition: AcquisitionFrameRateAbs
    and AcquisitionFrameRateLimit are now restored to their camera-reported
    maximum when acquisition stops, undoing the 10 Hz cap applied during
    hardware trigger mode.
v10 -> v11:
  - Improved frame rate restore: original values are now saved before
    applying the 10 Hz cap and restored exactly on stop, instead of
    relying on get_range() which was silently failing.
v13 -> v14:
  - Fixed fps dropping to 10 after Stop Acquisition: when TriggerMode
    switches back to Off (free-run), the camera re-applies the stored
    AcquisitionFrameRateAbs value. Now explicitly set to
    AcquisitionFrameRateLimit (camera maximum) on every stop.
v14 -> v15:
  - Delay scan (own tab): steps the hardware trigger delay from a start to an
    end value and records a fixed number of frames, or a fixed time, per step.
    Each step goes to its own subfolder (stepNNN_delay<X>us) of the run, so
    the PXI-viewer lists every delay as a run; scan_log.csv in the run folder
    lists the steps. Frames arriving within the settle time after a delay
    change are discarded.
v12 -> v13:
  - TriggerSource now tries SyncIn1 before Line1: on Mako G-032B the
    physical trigger input routes through SyncIn1, not Line1.
  - Added live SyncIn1 HIGH/LOW indicator in the status bar during
    hardware acquisition so signal presence can be verified electrically.
v11 -> v12:
  - Removed AcquisitionFrameRateAbs cap entirely: it was persisting at
    10 Hz across sessions (camera retains feature values) and
    AcquisitionFrameRateLimit is read-only so could never be restored.
    TriggerOverlap=Off alone is sufficient to prevent lockups.
  - On connect, AcquisitionFrameRateAbs is now reset to the camera's
    computed maximum (AcquisitionFrameRateLimit) to clear any stale
    value left by a previous session.
  - Fixed spurious "outside of 'with' context" warnings on startup:
    _stop_acquisition now guards camera feature calls with self.streaming.

Dependencies:
    pip install vmbpy opencv-python pillow h5py numpy
"""

import queue
import time
import threading
import concurrent.futures
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import csv
import math
import os
import re
from datetime import datetime, timezone

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from vmbpy import *


# ------------------------------------------------------------------------------
def build_metadata(frame_index: int,
                   trigger_index: int,
                   wall_time_utc: float,
                   camera_timestamp_ns: int,
                   exposure_us: float,
                   gain: float,
                   gamma: float,
                   black_level: float,
                   bin_h: int,
                   bin_v: int,
                   width: int,
                   height: int,
                   pixel_format: str,
                   trigger_mode: str = "",
                   trigger_delay_us: float = 0.0) -> dict:
    return dict(
        trigger_mode=trigger_mode,
        trigger_delay_us=trigger_delay_us,
        frame_index=frame_index,
        trigger_index=trigger_index,
        wall_time_utc_s=wall_time_utc,
        wall_time_iso=datetime.fromtimestamp(wall_time_utc, tz=timezone.utc).isoformat(),
        camera_timestamp_ns=camera_timestamp_ns,
        exposure_us=exposure_us,
        gain_dB=gain,
        gamma=gamma,
        black_level=black_level,
        binning_h=bin_h,
        binning_v=bin_v,
        width=width,
        height=height,
        pixel_format=pixel_format,
    )


def save_trigger_h5(filepath: str,
                    frames: list[np.ndarray],
                    metas: list[dict],
                    trigger_index: int,
                    exposure_frames: int):
    """Thread‑safe HDF5 writer: saves one trigger’s frames.

    Written under a temporary name and renamed when complete, so programs that
    watch the folder (e.g. the PXI-viewer live tab) never see a half-written file.
    """
    tmp = filepath + ".part"
    with h5py.File(tmp, "w") as f:
        grp = f.create_group("frames")
        grp.attrs["trigger_index"] = trigger_index
        grp.attrs["exposure_frames"] = exposure_frames
        grp.attrs["saved_time_utc"] = time.time()
        for i, (img, meta) in enumerate(zip(frames, metas)):
            ds = grp.create_dataset(f"frame_{i:04d}", data=img,
                                    compression="gzip", compression_opts=4)
            for k, v in meta.items():
                ds.attrs[k] = v
    os.replace(tmp, filepath)


def next_run_number(folder: str, base: str) -> int:
    """Next free run number for names like <base><N>_... or <base><N>.ext in folder."""
    pattern = re.compile(re.escape(base) + r"(\d+)(?:[_.]|$)")
    numbers = [int(m.group(1)) for name in os.listdir(folder)
               if (m := pattern.match(name))]
    return max(numbers, default=0) + 1


def scan_delays(start: float, end: float, step: float, max_steps: int = 10000) -> list:
    """Delays of a scan from start towards end in steps of |step| (µs).

    The direction follows start -> end, so a scan can also run downwards. The end
    is included when it lies on the step grid; otherwise the last delay is the last
    grid point before it. Raises ValueError for a step <= 0, a negative delay or
    more than max_steps steps.
    """
    step = abs(step)
    if step <= 0:
        raise ValueError("step must be > 0")
    if min(start, end) < 0:
        raise ValueError("delays must be >= 0")
    n = int(math.floor(abs(end - start) / step + 1e-9)) + 1
    if n > max_steps:
        raise ValueError(f"{n} steps; at most {max_steps}")
    sign = 1.0 if end >= start else -1.0
    return [round(start + sign * i * step, 6) for i in range(n)]


def delay_label(delay: float) -> str:
    """Delay as a file-name part: 120 -> '120us', 12.5 -> '12p5us'."""
    return f"{delay:g}".replace(".", "p") + "us"


# ------------------------------------------------------------------------------
class MakoCameraGUI:
    TRIGGER_MODES = ["Software (manual button)", "Hardware Line1 (GPIO)"]
    MAX_ZOOM = 32.0
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Mako G-032B — One H5 per trigger")
        self.root.geometry("1100x780")
        self.root.minsize(900, 620)
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # Camera & streaming
        self.cam = None
        self.streaming = False
        self.frame_queue: queue.Queue = queue.Queue(maxsize=4)
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self._binning_pending = False   # set by the GUI, handled by the stream worker

        # Live view zoom (display only): factor and window centre as image fraction
        self._zoom, self._zoom_cx, self._zoom_cy = 1.0, 0.5, 0.5
        self._drag_start = None
        self._last_disp_img = None
        self._disp_geom = None          # ((h, w), crop w, crop h, shown w, shown h)

        # Acquisition state
        self.acquiring = False
        self.trigger_index = 0
        self.total_frames_saved = 0
        self._png_dir = ""
        self._run_dir = ""        # day folder (YY-MM-DD) of the current run
        self._run_name = ""       # base filename + run number of the current run
        self._hw_arm_time = 0.0   # wall time when hardware acquisition was armed

        # Software trigger burst collection (streaming‑safe)
        self._sw_trigger_pending = False
        self._sw_frames_needed = 0
        self._sw_collected_frames: list[np.ndarray] = []
        self._sw_collected_metas: list[dict] = []

        # Thread pool for saving HDF5 files (non‑blocking)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        # GUI variables
        # Data goes to <save_folder>/<YY-MM-DD>/<base_filename><run number>_...
        self.save_folder = tk.StringVar(value=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data_camera"))
        self.base_filename = tk.StringVar(value="camera_test_run")
        self.trigger_mode_var = tk.StringVar(value=self.TRIGGER_MODES[0])
        self.exposure_frames = tk.IntVar(value=1)      # frames per trigger
        self.trigger_debounce_ns = tk.IntVar(value=100)   # SyncIn1 glitch filter in nanoseconds (camera max 50000)
        self.trigger_delay_us = tk.DoubleVar(value=0.0)   # delay from trigger edge to exposure start, microseconds
        self.live_hw_var = tk.BooleanVar(value=False)     # live view waits for hardware triggers
        self._delay_job = None                            # pending trigger-delay update (typing)
        self._delay_value = 0.0                           # delay the camera uses (for metadata)
        self.trigger_delay_us.trace_add("write", self._trigger_delay_changed)
        self.save_png_var = tk.BooleanVar(value=True)
        self.save_h5_var = tk.BooleanVar(value=True)

        # Delay scan (hardware trigger)
        self.scan_start_us = tk.DoubleVar(value=0.0)
        self.scan_end_us = tk.DoubleVar(value=100.0)
        self.scan_step_us = tk.DoubleVar(value=10.0)
        self.scan_mode = tk.StringVar(value="frames")     # "frames" or "seconds" per step
        self.scan_frames = tk.IntVar(value=100)
        self.scan_seconds = tk.DoubleVar(value=10.0)
        self.scan_settle_s = tk.DoubleVar(value=0.5)
        self._scan = None                       # state dict while a scan runs
        self._scan_lock = threading.Lock()      # shared with the frame handler thread

        self.exp_var = tk.DoubleVar(value=1000)
        self.gain_var = tk.DoubleVar(value=0.0)
        self.gamma_var = tk.DoubleVar(value=1.0)
        self.black_var = tk.DoubleVar(value=0)
        self.bin_h_var = tk.IntVar(value=1)
        self.bin_v_var = tk.IntVar(value=1)

        self._create_gui()

    # --------------------------------------------------------------------------
    # GUI
    # --------------------------------------------------------------------------
    def _create_gui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        self._build_live_tab()
        self._build_settings_tab()
        self._build_acquisition_tab()
        self._build_scan_tab()

        if not PIL_AVAILABLE:
            messagebox.showwarning("Missing Library", "Install Pillow for live preview")
        if not H5PY_AVAILABLE:
            messagebox.showwarning("Missing Library", "Install h5py for HDF5 saving")

        self.root.mainloop()

    def _build_live_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=" Live View ")

        top = ttk.Frame(tab, padding=8)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Camera:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        self.cam_label = ttk.Label(top, text="Not connected")
        self.cam_label.pack(side=tk.LEFT, padx=12)
        self.connect_btn = ttk.Button(top, text="Connect & Start Streaming",
                                      command=self.toggle_streaming)
        self.connect_btn.pack(side=tk.RIGHT)

        feed = ttk.LabelFrame(tab, text=" Live Camera Feed ", padding=5)
        feed.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)
        self.live_label = tk.Label(feed, background="black", relief="sunken")
        self.live_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Zoom affects only the live view, never the saved data
        zoom_bar = ttk.Frame(feed)
        zoom_bar.pack(fill=tk.X, padx=5)
        self.zoom_label = ttk.Label(zoom_bar, text="Zoom: 1.0×")
        self.zoom_label.pack(side=tk.LEFT)
        ttk.Label(zoom_bar, text="mouse wheel = zoom · drag = move · double-click = reset",
                  foreground="gray").pack(side=tk.LEFT, padx=12)
        ttk.Button(zoom_bar, text="Reset zoom", command=self._reset_live_zoom).pack(side=tk.RIGHT)
        self.live_label.bind("<MouseWheel>", self._on_live_wheel)
        self.live_label.bind("<ButtonPress-1>", self._on_live_press)
        self.live_label.bind("<B1-Motion>", self._on_live_drag)
        self.live_label.bind("<Double-Button-1>", self._reset_live_zoom)

        # Live view on the hardware trigger (preview only, nothing is saved)
        trig = ttk.Frame(tab, padding=(10, 0))
        trig.pack(fill=tk.X)
        ttk.Checkbutton(trig, text="Live view on hardware trigger (nothing is saved)",
                        variable=self.live_hw_var,
                        command=self._live_hw_toggled).pack(side=tk.LEFT)
        ttk.Label(trig, text="Trigger delay (µs):").pack(side=tk.LEFT, padx=(20, 4))
        ttk.Spinbox(trig, from_=0, to=1000000, increment=1, width=10,
                    textvariable=self.trigger_delay_us).pack(side=tk.LEFT)
        ttk.Label(trig, text="(applied at once; same setting as on the Acquisition tab)",
                  foreground="gray").pack(side=tk.LEFT, padx=8)

        self.status = ttk.Label(tab, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(fill=tk.X, padx=10, pady=(0, 8))

    def _build_settings_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=" Camera Settings ")

        cf = ttk.LabelFrame(tab, text="Camera Controls", padding=12)
        cf.pack(fill=tk.X, padx=15, pady=15)
        cf.columnconfigure(1, weight=1)

        def row_slider(r, label, var, lo, hi, cmd, fmt_fn, step, typed_hi=None):
            """Slider plus a box to type the value (Enter or leaving the box applies it,
            the arrows step by `step`). Typed values may go up to typed_hi (default hi)."""
            ttk.Label(cf, text=label).grid(row=r, column=0, sticky=tk.W, padx=5, pady=6)
            ttk.Scale(cf, from_=lo, to=hi, orient=tk.HORIZONTAL,
                      variable=var, command=cmd).grid(row=r, column=1, sticky=tk.EW, padx=8)
            top = hi if typed_hi is None else typed_hi
            box = ttk.Spinbox(cf, from_=lo, to=top, increment=step, width=10)
            box.set(fmt_fn(var.get()))
            box.grid(row=r, column=2, padx=10)

            def apply_typed(event=None):
                try:
                    value = float(box.get().strip().replace(",", "."))
                except ValueError:
                    box.set(fmt_fn(var.get()))   # not a number: show the current value again
                    return
                var.set(min(max(value, lo), top))
                cmd()

            box.configure(command=apply_typed)   # arrow buttons
            for ev in ("<Return>", "<KP_Enter>", "<FocusOut>"):
                box.bind(ev, apply_typed)
            # slider moved or value applied: show it in the box
            var.trace_add("write", lambda *a: box.set(fmt_fn(var.get())))

        row_slider(0, "Exposure Time (µs):", self.exp_var, 10, 100000,
                   self.update_exposure, lambda v: f"{int(v)}", step=10, typed_hi=60_000_000)
        # ranges as reported by the Mako G-032B
        row_slider(1, "Gain (dB):", self.gain_var, 0, 30,
                   self.update_gain, lambda v: f"{v:.2f}", step=1)
        row_slider(2, "Gamma:", self.gamma_var, 0.25, 4.0,
                   self.update_gamma, lambda v: f"{v:.3f}", step=0.01)
        row_slider(3, "Black Level:", self.black_var, 0, 255.75,
                   self.update_blacklevel, lambda v: f"{v:.2f}", step=0.25)
        ttk.Label(cf, text="Type a value and press Enter (or use the arrows). Exposure can be "
                           "typed up to 60 s; the slider goes to 100 ms.\nThe boxes show the "
                           "value the camera actually uses (it rounds Gain to whole dB).",
                  foreground="gray").grid(row=5, column=0, columnspan=4, sticky=tk.W, padx=5)

        ttk.Label(cf, text="Binning H:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=6)
        bh = ttk.Combobox(cf, textvariable=self.bin_h_var, values=[1, 2, 4], state="readonly", width=6)
        bh.grid(row=4, column=1, sticky=tk.W, padx=8)
        bh.bind("<<ComboboxSelected>>", self.update_binning)

        ttk.Label(cf, text="Binning V:").grid(row=4, column=2, sticky=tk.W, padx=5, pady=6)
        bv = ttk.Combobox(cf, textvariable=self.bin_v_var, values=[1, 2, 4], state="readonly", width=6)
        bv.grid(row=4, column=3, sticky=tk.W, padx=8)
        bv.bind("<<ComboboxSelected>>", self.update_binning)

        ttk.Button(tab, text="Apply All Settings", command=self.apply_all_settings).pack(pady=10)

    def _build_acquisition_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=" Acquisition ")

        loc = ttk.LabelFrame(tab, text="Save Location", padding=12)
        loc.pack(fill=tk.X, padx=15, pady=(15, 5))
        ttk.Label(loc, text="Folder:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(loc, textvariable=self.save_folder).grid(row=0, column=1, sticky=tk.EW, padx=6)
        ttk.Button(loc, text="Browse…", command=self._browse_folder).grid(row=0, column=2)
        ttk.Label(loc, text="Base filename:").grid(row=1, column=0, sticky=tk.W, pady=(8, 0))
        ttk.Entry(loc, textvariable=self.base_filename).grid(row=1, column=1, sticky=tk.EW, padx=6, pady=(8, 0))
        ttk.Label(loc, text="Saved to <folder>/YY-MM-DD/<base filename><run number>/, "
                            "one folder per run.",
                  foreground="gray").grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=(6, 0))
        loc.columnconfigure(1, weight=1)

        fmt = ttk.LabelFrame(tab, text="Output Formats", padding=12)
        fmt.pack(fill=tk.X, padx=15, pady=5)
        ttk.Checkbutton(fmt, text="Save PNG per frame", variable=self.save_png_var).pack(anchor=tk.W)
        ttk.Checkbutton(fmt, text="Save HDF5 per trigger", variable=self.save_h5_var).pack(anchor=tk.W)

        trig = ttk.LabelFrame(tab, text="Trigger Settings", padding=12)
        trig.pack(fill=tk.X, padx=15, pady=5)
        ttk.Label(trig, text="Trigger mode:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=6)
        ttk.Combobox(trig, textvariable=self.trigger_mode_var,
                     values=self.TRIGGER_MODES, state="readonly", width=30).grid(row=0, column=1, sticky=tk.W, padx=8)
        ttk.Label(trig, text="Frames per trigger:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=6)
        ttk.Spinbox(trig, from_=1, to=1000, increment=1,
                    textvariable=self.exposure_frames, width=8).grid(row=1, column=1, sticky=tk.W, padx=8)
        ttk.Label(trig, text="Debounce time (ns):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=6)
        ttk.Spinbox(trig, from_=0, to=50000, increment=100,
                    textvariable=self.trigger_debounce_ns, width=10).grid(row=2, column=1, sticky=tk.W, padx=8)
        ttk.Label(trig, text="(Hardware mode only — rejects pulses shorter than this, max 50000 ns)",
                  foreground="gray").grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(0, 4))
        ttk.Label(trig, text="Trigger delay (µs):").grid(row=4, column=0, sticky=tk.W, padx=5, pady=6)
        ttk.Spinbox(trig, from_=0, to=1000000, increment=1,
                    textvariable=self.trigger_delay_us, width=10).grid(row=4, column=1, sticky=tk.W, padx=8)
        ttk.Label(trig, text="(Hardware mode only — wait between trigger edge and start of exposure)",
                  foreground="gray").grid(row=5, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(0, 4))

        ctrl = ttk.LabelFrame(tab, text="Acquisition Control", padding=12)
        ctrl.pack(fill=tk.X, padx=15, pady=5)
        btn_row = ttk.Frame(ctrl)
        btn_row.pack(fill=tk.X)
        self.acq_btn = ttk.Button(btn_row, text="Start Acquisition",
                                  command=self.toggle_acquisition, state="disabled")
        self.acq_btn.pack(side=tk.LEFT, padx=(0, 10))
        self.sw_trigger_btn = ttk.Button(btn_row, text="⚡ Software Trigger",
                                         command=self._fire_software_trigger, state="disabled")
        self.sw_trigger_btn.pack(side=tk.LEFT)

        stats = ttk.LabelFrame(tab, text="Statistics", padding=12)
        stats.pack(fill=tk.X, padx=15, pady=5)
        self.stat_triggers = ttk.Label(stats, text="Triggers received: 0")
        self.stat_triggers.pack(anchor=tk.W)
        self.stat_frames = ttk.Label(stats, text="Frames saved: 0")
        self.stat_frames.pack(anchor=tk.W)
        self.stat_h5dir = ttk.Label(stats, text="H5 folder: —", foreground="gray")
        self.stat_h5dir.pack(anchor=tk.W)

        diag = ttk.LabelFrame(tab, text="Diagnostics", padding=12)
        diag.pack(fill=tk.X, padx=15, pady=5)
        ttk.Button(diag, text="Dump All Camera Features to Console",
                   command=self._dump_camera_features).pack(anchor=tk.W)
        ttk.Label(diag, text="Use this to find correct feature names for your firmware version.",
                  foreground="gray").pack(anchor=tk.W, pady=(2, 0))

    def _build_scan_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=" Delay Scan ")

        rng = ttk.LabelFrame(tab, text="Trigger delay range", padding=12)
        rng.pack(fill=tk.X, padx=15, pady=(15, 5))
        for col, (label, var) in enumerate((("Start (µs):", self.scan_start_us),
                                            ("End (µs):", self.scan_end_us),
                                            ("Step (µs):", self.scan_step_us))):
            ttk.Label(rng, text=label).grid(row=0, column=2 * col, sticky=tk.W, padx=(0 if col == 0 else 15, 4))
            ttk.Spinbox(rng, from_=0, to=1000000, increment=1, width=10,
                        textvariable=var).grid(row=0, column=2 * col + 1, sticky=tk.W)
        ttk.Label(rng, text="The scan runs from start towards end (also downwards). The end is "
                            "included when it lies on the step grid. The camera rounds the delay "
                            "to whole µs.", foreground="gray", wraplength=900,
                  justify=tk.LEFT).grid(row=1, column=0, columnspan=6, sticky=tk.W, pady=(6, 0))

        per = ttk.LabelFrame(tab, text="Per step", padding=12)
        per.pack(fill=tk.X, padx=15, pady=5)
        ttk.Radiobutton(per, text="Number of frames:", value="frames",
                        variable=self.scan_mode).grid(row=0, column=0, sticky=tk.W)
        ttk.Spinbox(per, from_=1, to=1000000, increment=10, width=10,
                    textvariable=self.scan_frames).grid(row=0, column=1, sticky=tk.W, padx=8)
        ttk.Radiobutton(per, text="Time (s):", value="seconds",
                        variable=self.scan_mode).grid(row=1, column=0, sticky=tk.W, pady=(6, 0))
        ttk.Spinbox(per, from_=0.1, to=86400, increment=1, width=10,
                    textvariable=self.scan_seconds).grid(row=1, column=1, sticky=tk.W, padx=8, pady=(6, 0))
        ttk.Label(per, text="Settle time after each delay change (s):").grid(
            row=2, column=0, sticky=tk.W, pady=(10, 0))
        ttk.Spinbox(per, from_=0, to=60, increment=0.1, width=10,
                    textvariable=self.scan_settle_s).grid(row=2, column=1, sticky=tk.W, padx=8, pady=(10, 0))
        ttk.Label(per, text="Frames arriving within the settle time may still have been "
                            "exposed with the previous delay and are discarded. Time per step "
                            "is counted after the settle time.", foreground="gray",
                  wraplength=900, justify=tk.LEFT).grid(row=3, column=0, columnspan=3,
                                                        sticky=tk.W, pady=(4, 0))

        run = ttk.LabelFrame(tab, text="Scan", padding=12)
        run.pack(fill=tk.X, padx=15, pady=5)
        self.scan_summary = ttk.Label(run, text="")
        self.scan_summary.pack(anchor=tk.W)
        self.scan_btn = ttk.Button(run, text="Start Delay Scan",
                                   command=self.toggle_delay_scan, state="disabled")
        self.scan_btn.pack(anchor=tk.W, pady=(8, 4))
        self.scan_progress = ttk.Progressbar(run, mode="determinate", maximum=1.0)
        self.scan_progress.pack(fill=tk.X, pady=4)
        self.scan_status = ttk.Label(run, text="Not running")
        self.scan_status.pack(anchor=tk.W)
        ttk.Label(run, text="Uses the hardware trigger (Line1) with the debounce time, save "
                            "location, base filename and output formats of the Acquisition tab. "
                            "Saved to <folder>/YY-MM-DD/<base filename><run number>/"
                            "stepNNN_delay<X>us/, one subfolder per step, with scan_log.csv in "
                            "the run folder. Afterwards the trigger delay goes back to the value "
                            "on the Acquisition tab.", foreground="gray", wraplength=900,
                  justify=tk.LEFT).pack(anchor=tk.W, pady=(8, 0))

        for var in (self.scan_start_us, self.scan_end_us, self.scan_step_us, self.scan_mode,
                    self.scan_frames, self.scan_seconds, self.scan_settle_s):
            var.trace_add("write", self._update_scan_summary)
        self._update_scan_summary()

    # --------------------------------------------------------------------------
    # Streaming
    # --------------------------------------------------------------------------
    def toggle_streaming(self):
        if not self.streaming:
            threading.Thread(target=self._stream_worker, daemon=True).start()
        else:
            self.streaming = False

    def _stream_worker(self):
        try:
            with VmbSystem.get_instance() as vmb:
                cams = vmb.get_all_cameras()
                if not cams:
                    raise RuntimeError("No cameras found")
                self.cam = next((c for c in cams if "Mako G-032B" in c.get_name()), cams[0])
                self.root.after(0, lambda: self.cam_label.config(text=self.cam.get_name()))

                with self.cam:
                    # Negotiate the GigE packet size with the network adapter.
                    # The camera powers up with 8228 (jumbo); if the NIC has
                    # jumbo packets disabled every frame arrives Incomplete.
                    try:
                        stream = self.cam.get_streams()[0]
                        stream.GVSPAdjustPacketSize.run()
                        while not stream.GVSPAdjustPacketSize.is_done():
                            time.sleep(0.01)
                    except Exception as e:
                        print("GVSPAdjustPacketSize warning:", e)
                    self.cam.get_feature_by_name("PixelFormat").set("Mono8")
                    try:
                        self.cam.get_feature_by_name("ExposureAuto").set("Off")
                        self.cam.get_feature_by_name("GainAuto").set("Off")
                    except Exception:
                        pass
                    self.apply_all_settings()
                    # Reset AcquisitionFrameRateAbs to the camera's computed
                    # maximum in case a previous session left it at 10 Hz.
                    try:
                        max_fps = self.cam.get_feature_by_name("AcquisitionFrameRateLimit").get()
                        self.cam.get_feature_by_name("AcquisitionFrameRateAbs").set(max_fps)
                    except Exception as e:
                        print("AcquisitionFrameRateAbs reset warning:", e)
                    # Start in the chosen live mode. The camera keeps its trigger
                    # settings between sessions, so a previous run can have left
                    # TriggerMode=On, which would make the live view stay dark.
                    self._apply_live_trigger_mode()
                    self.streaming = True
                    self.fps_counter = 0
                    self.fps_start_time = time.time()

                    self.root.after(0, lambda: self.connect_btn.config(text="Stop Streaming"))
                    self.root.after(0, lambda: self.acq_btn.config(state="normal"))
                    self.root.after(0, lambda: self.scan_btn.config(state="normal"))
                    self.root.after(0, lambda: self.status.config(text="Streaming…"))

                    def frame_handler(cam, stream, frame):
                        if not self.streaming:
                            stream.queue_frame(frame)
                            return
                        try:
                            self.fps_counter += 1
                            img = frame.as_numpy_ndarray()
                            if img.ndim == 3 and img.shape[2] == 1:
                                img = img.squeeze(axis=2)
                            if img.dtype != np.uint8:
                                img = img.astype(np.uint8)

                            # --- Acquisition handling ---
                            if self.acquiring and self._scan is not None:
                                # Delay scan: hardware trigger, saved per step
                                self._scan_frame(img, frame)
                            elif self.acquiring:
                                # Hardware trigger mode: each frame is a trigger.
                                # Discard frames that arrived within the first
                                # HW_ARM_DISCARD_S seconds after arming — these
                                # are free-run frames already buffered in the
                                # VmbPy pipeline before TriggerMode=On took effect
                                # and must not be counted as real triggers.
                                if "Hardware" in self.trigger_mode_var.get():
                                    if time.time() - self._hw_arm_time > 0.5:
                                        self._save_hardware_trigger_frame(img, frame)
                                # Software trigger mode: collect burst if pending
                                elif self._sw_trigger_pending:
                                    self._collect_software_trigger_frame(img, frame)

                            # Live display (subsampled)
                            if self.fps_counter % 2 == 0:
                                disp = (cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                                        if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                                try:
                                    self.frame_queue.put_nowait(disp)
                                except queue.Full:
                                    pass
                        except Exception as e:
                            print("Frame error:", e)
                        finally:
                            stream.queue_frame(frame)

                    self.cam.start_streaming(frame_handler, buffer_count=10)
                    self.root.after(10, self._update_display)

                    while self.streaming:
                        # Binning changes the image size, so the stream is
                        # restarted with buffers of the new size.
                        if self._binning_pending:
                            self._binning_pending = False
                            try:
                                self.cam.stop_streaming()
                                self._apply_binning()
                                self.cam.start_streaming(frame_handler, buffer_count=10)
                            except Exception as e:
                                print("Binning change failed:", e)
                        time.sleep(0.01)

                    # Explicitly stop streaming before exiting the camera context.
                    # This ensures all in-flight frame_handler callbacks finish
                    # and VmbPy releases its internal locks cleanly, preventing
                    # the camera from hanging on the next connect attempt.
                    try:
                        self.cam.stop_streaming()
                    except Exception as e:
                        print("stop_streaming warning:", e)

        except Exception as e:
            # Copy the message now: 'e' no longer exists when the dialog callback runs.
            msg = str(e)
            self.root.after(0, lambda: messagebox.showerror("Camera Error", msg))
        finally:
            self.streaming = False
            self._stop_acquisition()
            self.root.after(0, lambda: self.connect_btn.config(text="Connect & Start Streaming"))
            self.root.after(0, lambda: self.acq_btn.config(state="disabled"))
            self.root.after(0, lambda: self.scan_btn.config(state="disabled"))

    # --------------------------------------------------------------------------
    # Hardware trigger: each frame is one trigger
    # --------------------------------------------------------------------------
    def _save_hardware_trigger_frame(self, img: np.ndarray, raw_frame, folder=None,
                                     png_dir=None, delay_us=None, extra=None, prefix=None):
        """Save one hardware-triggered frame. folder/png_dir/prefix default to the
        current run; delay_us to the delay the camera uses; extra adds attributes."""
        folder = self._run_dir if folder is None else folder
        png_dir = self._png_dir if png_dir is None else png_dir
        prefix = self._run_name if prefix is None else prefix
        self.trigger_index += 1
        self.total_frames_saved += 1
        self.root.after(0, self._update_stats)

        # Build metadata
        meta = build_metadata(
            frame_index=self.total_frames_saved,
            trigger_index=self.trigger_index,
            wall_time_utc=time.time(),
            camera_timestamp_ns=raw_frame.get_timestamp(),
            exposure_us=self.exp_var.get(),
            gain=self.gain_var.get(),
            gamma=self.gamma_var.get(),
            black_level=self.black_var.get(),
            bin_h=self.bin_h_var.get(),
            bin_v=self.bin_v_var.get(),
            width=img.shape[1],
            height=img.shape[0],
            pixel_format="Mono8",
            trigger_mode="hardware",
            trigger_delay_us=self._delay_value if delay_us is None else delay_us,
        )
        if extra:
            meta.update(extra)

        # PNG
        if self.save_png_var.get():
            fname = (f"trigger{self.trigger_index:06d}"
                     f"_frame0000_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}.png")
            path = os.path.join(png_dir, fname)
            threading.Thread(target=_save_png, args=(img, path), daemon=True).start()

        # HDF5 (one file per trigger)
        if self.save_h5_var.get():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            h5_filename = f"{prefix}_{timestamp}_trigger{self.trigger_index:06d}.h5"
            h5_path = os.path.join(folder, h5_filename)
            self._executor.submit(
                save_trigger_h5,
                h5_path,
                [img],
                [meta],
                self.trigger_index,
                1
            )

    # --------------------------------------------------------------------------
    # Software trigger: collect next N frames from the live stream
    # --------------------------------------------------------------------------
    def _fire_software_trigger(self):
        if not self.acquiring or "Software" not in self.trigger_mode_var.get():
            return
        if self._sw_trigger_pending:
            # Already collecting – ignore new trigger
            return
        # Arm the burst collector
        self._sw_trigger_pending = True
        self._sw_frames_needed = self.exposure_frames.get()
        self._sw_collected_frames = []
        self._sw_collected_metas = []

    def _collect_software_trigger_frame(self, img: np.ndarray, raw_frame):
        """Collect one frame for the pending software trigger burst."""
        # Build metadata
        frame_idx_in_burst = len(self._sw_collected_frames)
        meta = build_metadata(
            frame_index=self.total_frames_saved + frame_idx_in_burst,
            trigger_index=self.trigger_index + 1,  # will be incremented when finalised
            wall_time_utc=time.time(),
            camera_timestamp_ns=raw_frame.get_timestamp(),
            exposure_us=self.exp_var.get(),
            gain=self.gain_var.get(),
            gamma=self.gamma_var.get(),
            black_level=self.black_var.get(),
            bin_h=self.bin_h_var.get(),
            bin_v=self.bin_v_var.get(),
            width=img.shape[1],
            height=img.shape[0],
            pixel_format="Mono8",
            trigger_mode="software",   # free run: no trigger delay applies
            trigger_delay_us=0.0,
        )
        self._sw_collected_frames.append(img)
        self._sw_collected_metas.append(meta)

        self._sw_frames_needed -= 1
        if self._sw_frames_needed == 0:
            # Burst complete – save now
            self._finalize_software_trigger()

    def _finalize_software_trigger(self):
        if not self._sw_collected_frames:
            self._sw_trigger_pending = False
            return

        self.trigger_index += 1
        n_frames = len(self._sw_collected_frames)
        self.total_frames_saved += n_frames
        self.root.after(0, self._update_stats)

        # Update metadata with final trigger index
        for meta in self._sw_collected_metas:
            meta["trigger_index"] = self.trigger_index

        # PNGs (optional)
        if self.save_png_var.get():
            for idx, (img, meta) in enumerate(zip(self._sw_collected_frames, self._sw_collected_metas)):
                fname = (f"trigger{self.trigger_index:06d}"
                         f"_frame{idx:04d}"
                         f"_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}.png")
                path = os.path.join(self._png_dir, fname)
                threading.Thread(target=_save_png, args=(img, path), daemon=True).start()

        # HDF5 (one file per trigger)
        if self.save_h5_var.get():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            h5_filename = f"{self._run_name}_{timestamp}_trigger{self.trigger_index:06d}.h5"
            h5_path = os.path.join(self._run_dir, h5_filename)
            self._executor.submit(
                save_trigger_h5,
                h5_path,
                self._sw_collected_frames.copy(),
                self._sw_collected_metas.copy(),
                self.trigger_index,
                len(self._sw_collected_frames)
            )

        # Clear pending flag
        self._sw_trigger_pending = False
        self._sw_collected_frames = []
        self._sw_collected_metas = []

    # --------------------------------------------------------------------------
    # Acquisition start / stop
    # --------------------------------------------------------------------------
    def toggle_acquisition(self):
        if not self.streaming:
            messagebox.showwarning("Warning", "Start streaming first")
            return
        if self.acquiring:
            self._stop_acquisition()
        else:
            self._start_acquisition()

    def _new_run_folder(self):
        """Create the next run folder and make it the current run.

        One subfolder per day (YY-MM-DD) inside the chosen folder, and in it one
        folder per run: <day>/<base><N>/ holds that run's H5 files and PNGs.
        Returns the folder, or None (after an error dialog) if it cannot be made.
        """
        day_folder = os.path.join(self.save_folder.get(), datetime.now().strftime("%y-%m-%d"))
        base = self.base_filename.get()
        try:
            os.makedirs(day_folder, exist_ok=True)
            # Runs are numbered within the day: <base>1, <base>2, ...
            run_name = f"{base}{next_run_number(day_folder, base)}"
            folder = os.path.join(day_folder, run_name)
            os.makedirs(folder, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Error", f"Cannot create folder:\n{e}")
            return None
        self._run_name, self._run_dir = run_name, folder
        return folder

    def _start_acquisition(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = self._new_run_folder()
        if folder is None:
            return

        # PNG subfolder
        self._png_dir = os.path.join(folder, f"{self._run_name}_{ts}_frames")
        if self.save_png_var.get():
            os.makedirs(self._png_dir, exist_ok=True)

        self.trigger_index = 0
        self.total_frames_saved = 0
        self._hw_arm_time = 0.0
        self._sw_trigger_pending = False
        self._sw_collected_frames = []
        self._sw_collected_metas = []

        self.acquiring = True
        self.acq_btn.config(text="Stop Acquisition")
        self.scan_btn.config(state="disabled")

        if "Software" in self.trigger_mode_var.get():
            self.sw_trigger_btn.config(state="normal")
        else:
            self.sw_trigger_btn.config(state="disabled")

        self._configure_camera_trigger()
        if "Hardware" in self.trigger_mode_var.get():
            # Record arm time so frame_handler can discard frames that were
            # already buffered in the VmbPy pipeline before TriggerMode=On.
            self._hw_arm_time = time.time()
        self.status.config(text="Acquisition active — waiting for trigger…")
        self.stat_h5dir.config(text=f"H5 folder: {folder}  (run: {self._run_name})")
        self._update_stats()

    def _stop_acquisition(self):
        if self._scan is not None:
            self._finish_delay_scan("stopped")   # ends the scan, then comes back here
            return
        self.acquiring = False
        self._sw_trigger_pending = False
        self.acq_btn.config(text="Start Acquisition")
        state = "normal" if self.streaming else "disabled"
        self.acq_btn.config(state=state)
        self.scan_btn.config(state=state)
        self.sw_trigger_btn.config(state="disabled")
        self.status.config(text=f"Stopped. {self.total_frames_saved} frames saved.")
        # Back to the live-view mode: free run, or the hardware trigger if
        # "Live view on hardware trigger" is ticked. Leaving TriggerMode=On
        # otherwise causes 0 fps in live view. Guard with self.streaming so we
        # don't call camera features outside the VmbPy 'with' context (which
        # happens on program startup/shutdown).
        if self.cam and self.streaming:
            self._apply_live_trigger_mode()

    def _apply_live_trigger_mode(self):
        """Trigger mode for the live view outside an acquisition."""
        if self.live_hw_var.get():
            self._configure_camera_trigger(hardware=True)
        else:
            self._set_free_run()

    def _set_free_run(self):
        if not self.cam:
            return
        try:
            self.cam.get_feature_by_name("TriggerMode").set("Off")
        except Exception as e:
            print("TriggerMode->Off warning:", e)
        # Restore full frame rate. When TriggerMode=On the camera ignores
        # AcquisitionFrameRateAbs, but once back in free-run it re-applies
        # whatever value is stored — which may be a stale 10 Hz.
        # Set it explicitly to the camera's computed maximum.
        try:
            max_fps = self.cam.get_feature_by_name("AcquisitionFrameRateLimit").get()
            self.cam.get_feature_by_name("AcquisitionFrameRateAbs").set(max_fps)
            print(f"Frame rate restored to {max_fps:.1f} fps")
        except Exception as e:
            print(f"Frame rate restore warning: {e}")

    def _live_hw_toggled(self):
        """Live view switched between free run and hardware trigger (no saving)."""
        if not (self.cam and self.streaming):
            return   # applied when streaming starts
        if self.acquiring:
            # the acquisition's own trigger mode stays; the live mode follows on stop
            self.status.config(text="Live trigger mode applies after the acquisition stops")
            return
        self._apply_live_trigger_mode()
        self.status.config(text="Live view on hardware trigger — waiting for trigger…"
                           if self.live_hw_var.get() else "Live view: free run")

    def _trigger_delay_changed(self, *args):
        """Trigger delay edited: send it to the camera shortly after typing stops."""
        if self._delay_job is not None:
            self.root.after_cancel(self._delay_job)
        self._delay_job = self.root.after(400, self._apply_trigger_delay)

    def _apply_trigger_delay(self):
        self._delay_job = None
        try:
            delay = float(self.trigger_delay_us.get())
        except (tk.TclError, ValueError):
            return   # half-typed value
        if not (self.cam and self.streaming):
            return   # applied with the hardware trigger configuration
        if self._scan is not None:
            self.status.config(text="Delay scan running: the typed trigger delay applies "
                                    "after the scan")
            return
        self._send_trigger_delay(delay)

    def _send_trigger_delay(self, delay):
        """Set TriggerDelayAbs and remember the value the camera uses (it rounds to 1 µs).
        Returns that value, or None if the camera refused it."""
        try:
            f = self.cam.get_feature_by_name("TriggerDelayAbs")
            f.set(max(delay, 0.0))
            self._delay_value = float(f.get())
            print(f"TriggerDelayAbs set to {self._delay_value} us")
            return self._delay_value
        except Exception as e:
            print(f"TriggerDelayAbs warning: {e}")
            return None

    def _configure_camera_trigger(self, hardware=None):
        """Set camera trigger source based on GUI selection.

        Hardware trigger notes (Mako G-032B / GigE):
        - TriggerOverlap="Off"  -> camera ignores edges that arrive while it is
          still reading out the previous frame.  Without this the state machine
          can lock up after 2-3 triggers.
        - AcquisitionFrameRateAbs / AcquisitionFrameRateLimit cap the internal
          acquisition rate so the camera never attempts to run faster than it
          can sustain over GigE, which is another common lockup source.
          Adjust the value (default 10 Hz) to match your actual trigger rate.
        - GevSCPSPacketSize=1500 and GevSCPD (inter-packet delay) are set
          conservatively to prevent GigE buffer overruns that also manifest
          as 0 fps hangs.
        """
        if not self.cam:
            return
        if hardware is None:   # default: the acquisition tab's trigger mode
            hardware = "Hardware" in self.trigger_mode_var.get()
        try:
            if hardware:
                self.cam.get_feature_by_name("TriggerSelector").set("FrameStart")
                self.cam.get_feature_by_name("TriggerMode").set("On")
                # On this Mako G-032B (firmware 00.01.54) the trigger input
                # (pin 4, SyncIn1) is the TriggerSource entry "Line1"; SyncIn1
                # is kept as a fallback for other firmware versions.
                for src in ("Line1", "SyncIn1"):
                    try:
                        self.cam.get_feature_by_name("TriggerSource").set(src)
                        print(f"TriggerSource set to {src}")
                        break
                    except Exception as e:
                        print(f"TriggerSource {src} failed: {e}")
                self.cam.get_feature_by_name("TriggerActivation").set("RisingEdge")

                # Prevent state-machine lockup: discard triggers that arrive
                # during readout rather than queuing them.
                try:
                    self.cam.get_feature_by_name("TriggerOverlap").set("Off")
                except Exception as e:
                    print("TriggerOverlap warning:", e)

                # Glitch filter on SyncIn1 (same physical pin as Line1/trigger input).
                # Rejects pulses shorter than this value in nanoseconds.
                # This is the correct feature name for Mako G-032B firmware.
                try:
                    self.cam.get_feature_by_name("SyncInSelector").set("SyncIn1")
                    self.cam.get_feature_by_name("SyncInGlitchFilter").set(
                        self.trigger_debounce_ns.get()
                    )
                    print(f"SyncInGlitchFilter set to {self.trigger_debounce_ns.get()} ns")
                except Exception as e:
                    print(f"SyncInGlitchFilter warning: {e}")

                # Delay between the trigger edge and the start of exposure.
                try:
                    self._send_trigger_delay(float(self.trigger_delay_us.get()))
                except (tk.TclError, ValueError):
                    self._send_trigger_delay(self._delay_value)   # field half-typed

                # Note: AcquisitionFrameRateAbs is not capped here.
                # TriggerOverlap=Off is sufficient to prevent lockups, and
                # capping the frame rate causes persistent 10 fps after stop.

                # GigE transport tuning: conservative packet size + inter-packet
                # delay to avoid NIC buffer overruns on the host side.
                try:
                    self.cam.get_feature_by_name("GevSCPSPacketSize").set(1500)
                except Exception:
                    pass
                try:
                    self.cam.get_feature_by_name("GevSCPD").set(1000)
                except Exception:
                    pass

            else:
                # Software mode: free run, no hardware trigger
                self.cam.get_feature_by_name("TriggerMode").set("Off")
                try:
                    self.cam.get_feature_by_name("AcquisitionFrameRateAbs").set(30.0)
                except Exception:
                    pass
        except Exception as e:
            print("Trigger config warning:", e)

    # --------------------------------------------------------------------------
    # Delay scan: step the hardware trigger delay, N frames or T seconds per step
    # --------------------------------------------------------------------------
    def _scan_settings(self):
        """Scan settings from the GUI as a dict; raises ValueError if invalid."""
        try:
            delays = scan_delays(float(self.scan_start_us.get()), float(self.scan_end_us.get()),
                                 float(self.scan_step_us.get()))
            mode = self.scan_mode.get()
            n = int(self.scan_frames.get())
            seconds = float(self.scan_seconds.get())
            settle = float(self.scan_settle_s.get())
        except tk.TclError:
            raise ValueError("a field is empty or not a number")
        if mode == "frames" and n < 1:
            raise ValueError("frames per step must be >= 1")
        if mode == "seconds" and seconds <= 0:
            raise ValueError("time per step must be > 0")
        if settle < 0:
            raise ValueError("settle time must be >= 0")
        return dict(delays=delays, mode=mode, n=n, seconds=seconds, settle=settle)

    def _update_scan_summary(self, *args):
        if self._scan is not None:
            return   # keeps describing the running scan's own settings
        try:
            p = self._scan_settings()
        except ValueError as e:
            self.scan_summary.config(text=f"Invalid settings: {e}", foreground="red")
            return
        d = p["delays"]
        shown = ", ".join(f"{x:g}" for x in d) if len(d) <= 8 else \
            f"{d[0]:g}, {d[1]:g}, {d[2]:g}, …, {d[-1]:g}"
        if p["mode"] == "frames":
            per = f"{p['n']} frames per step = {p['n'] * len(d)} frames"
        else:
            total = len(d) * (p["seconds"] + p["settle"])
            per = f"{p['seconds']:g} s per step ≈ " + (f"{total:.0f} s" if total < 120 else
                                                       f"{total / 60:.1f} min") + " in total"
        self.scan_summary.config(text=f"{len(d)} steps: {shown} µs — {per}",
                                 foreground="")

    def toggle_delay_scan(self):
        if self._scan is not None:
            self._finish_delay_scan("stopped")
        else:
            self._start_delay_scan()

    def _start_delay_scan(self):
        if not self.streaming or not self.cam:
            messagebox.showwarning("Warning", "Start streaming first")
            return
        if self.acquiring:
            messagebox.showwarning("Warning", "Stop the running acquisition first")
            return
        if not (self.save_h5_var.get() or self.save_png_var.get()):
            messagebox.showwarning("Warning", "Neither HDF5 nor PNG saving is ticked "
                                              "(Acquisition tab)")
            return
        try:
            p = self._scan_settings()
        except ValueError as e:
            messagebox.showerror("Delay scan", f"Invalid settings: {e}")
            return
        folder = self._new_run_folder()
        if folder is None:
            return
        log_path = os.path.join(folder, "scan_log.csv")
        try:
            with open(log_path, "w", newline="") as f:
                f.write(f"# delay scan {self._run_name}, started "
                        f"{datetime.now(timezone.utc).isoformat()}\n")
                f.write(f"# start {self.scan_start_us.get():g} us, end {self.scan_end_us.get():g} us, "
                        f"step {self.scan_step_us.get():g} us, per step: "
                        + (f"{p['n']} frames" if p["mode"] == "frames" else f"{p['seconds']:g} s")
                        + f", settle {p['settle']:g} s, debounce "
                        f"{self.trigger_debounce_ns.get()} ns, exposure {self.exp_var.get():g} us\n")
                csv.writer(f).writerow(["step", "delay_set_us", "delay_camera_us", "frames",
                                        "start_utc", "end_utc", "status", "folder"])
        except OSError as e:
            messagebox.showerror("Delay scan", f"Cannot write {log_path}:\n{e}")
            return

        self.trigger_index = 0
        self.total_frames_saved = 0
        self._scan = dict(p, idx=-1, phase="switching", count=0, t0=0.0, measure_t0=0.0,
                          actual=None, completed=0,
                          step_dir="", png_dir="", start_utc="", log=log_path,
                          n_steps=len(p["delays"]))
        self.acquiring = True
        self.acq_btn.config(state="disabled")
        self.sw_trigger_btn.config(state="disabled")
        self.scan_btn.config(text="Stop Delay Scan")
        self.scan_progress.config(value=0.0)
        self._configure_camera_trigger(hardware=True)
        self._hw_arm_time = time.time()
        self.stat_h5dir.config(text=f"H5 folder: {folder}  (delay scan: {self._run_name})")
        self._update_stats()
        self._scan_next_step()
        if self._scan is not None:
            self.root.after(50, self._scan_poll)

    def _scan_log_step(self, s, status):
        """Append the current step of scan state s to scan_log.csv."""
        if s["idx"] < 0 or s["idx"] >= s["n_steps"]:
            return
        if status == "complete":
            s["completed"] += 1
        try:
            with open(s["log"], "a", newline="") as f:
                csv.writer(f).writerow([
                    s["idx"] + 1, f"{s['delays'][s['idx']]:g}",
                    "" if s["actual"] is None else f"{s['actual']:g}", s["count"],
                    s["start_utc"], datetime.now(timezone.utc).isoformat(), status,
                    os.path.basename(s["step_dir"])])
        except OSError as e:
            print(f"scan_log.csv warning: {e}")

    def _scan_next_step(self):
        """Close the current step (if any) and set up the next one, or finish."""
        s = self._scan
        with self._scan_lock:
            s["phase"] = "switching"   # the frame handler saves nothing meanwhile
        self._scan_log_step(s, "complete")
        idx = s["idx"] + 1
        if idx >= s["n_steps"]:
            s["idx"] = idx   # already logged
            self._finish_delay_scan("complete")
            return
        delay = s["delays"][idx]
        actual = self._send_trigger_delay(delay)
        if actual is None:
            s["idx"] = s["n_steps"]
            self._finish_delay_scan("error")
            messagebox.showerror("Delay scan", f"The camera did not accept trigger delay "
                                               f"{delay:g} µs; scan stopped.")
            return
        name = f"step{idx + 1:03d}_delay{delay_label(delay)}"
        step_dir = os.path.join(self._run_dir, name)
        png_dir = os.path.join(step_dir, f"{name}_frames")
        try:
            os.makedirs(step_dir, exist_ok=True)
            if self.save_png_var.get():
                os.makedirs(png_dir, exist_ok=True)
        except OSError as e:
            s["idx"] = s["n_steps"]
            self._finish_delay_scan("error")
            messagebox.showerror("Delay scan", f"Cannot create {step_dir}:\n{e}")
            return
        now = time.time()
        # Frames count from the end of the settle time, and never within 0.5 s of arming
        # (free-run frames from before TriggerMode=On), so the first step of a timed scan
        # gets its full time too.
        measure_t0 = max(now + s["settle"], self._hw_arm_time + 0.5)
        with self._scan_lock:
            s.update(idx=idx, count=0, actual=actual, step_dir=step_dir, png_dir=png_dir,
                     prefix=f"{self._run_name}_{name}", t0=now, measure_t0=measure_t0,
                     start_utc=datetime.now(timezone.utc).isoformat(), phase="measure")

    def _scan_frame(self, img: np.ndarray, raw_frame):
        """Frame handler thread: save the frame into the current step, if it counts."""
        s = self._scan
        if s is None:
            return
        now = time.time()
        with self._scan_lock:
            if (s["phase"] != "measure" or now < s["measure_t0"]
                    or (s["mode"] == "frames" and s["count"] >= s["n"])):
                return
            s["count"] += 1
            idx, step_dir, png_dir, prefix, actual = (s["idx"], s["step_dir"], s["png_dir"],
                                                      s["prefix"], s["actual"])
        self._save_hardware_trigger_frame(
            img, raw_frame, folder=step_dir, png_dir=png_dir, delay_us=actual, prefix=prefix,
            extra=dict(scan_step=idx + 1, scan_n_steps=s["n_steps"],
                       scan_delay_set_us=s["delays"][idx]))

    def _scan_poll(self):
        """GUI thread: show progress and move on when the step has enough frames or time."""
        s = self._scan
        if s is None or not self.acquiring:
            return
        now = time.time()
        with self._scan_lock:
            idx, count, phase = s["idx"], s["count"], s["phase"]
            measure_t0 = s.get("measure_t0", 0.0)
        if phase == "measure":
            if s["mode"] == "frames":
                frac = count / s["n"]
                done = count >= s["n"]
                txt = f"{count}/{s['n']} frames"
            else:
                elapsed = max(now - measure_t0, 0.0)
                frac = min(elapsed / s["seconds"], 1.0)
                done = now >= measure_t0 + s["seconds"]
                txt = f"{count} frames, {max(s['seconds'] - elapsed, 0):.1f} s left"
            if now < measure_t0:
                txt = "settling…"
            self.scan_status.config(text=f"Step {idx + 1}/{s['n_steps']}: delay "
                                         f"{s['actual']:g} µs — {txt}")
            self.scan_progress.config(value=(idx + frac) / s["n_steps"])
            if done:
                self._scan_next_step()
        if self._scan is not None:
            self.root.after(50, self._scan_poll)

    def _finish_delay_scan(self, status):
        """End the scan ("complete", "stopped" or "error") and return to live view."""
        s = self._scan
        if s is None:
            return
        with self._scan_lock:
            s["phase"] = "done"
        if status != "complete":
            self._scan_log_step(s, status)   # the interrupted step
        self._scan = None
        try:
            with open(s["log"], "a") as f:
                f.write(f"# {status}, {datetime.now(timezone.utc).isoformat()}, "
                        f"{self.total_frames_saved} frames saved\n")
        except OSError:
            pass
        # back to the delay typed on the Acquisition tab
        if self.cam and self.streaming:
            try:
                self._send_trigger_delay(float(self.trigger_delay_us.get()))
            except (tk.TclError, ValueError):
                pass
        self.scan_btn.config(text="Start Delay Scan")
        if status == "complete":
            self.scan_progress.config(value=1.0)
        self.scan_status.config(text=f"Scan {status}: {s['completed']}/{s['n_steps']} steps complete, "
                                     f"{self.total_frames_saved} frames in {self._run_dir}")
        self._stop_acquisition()
        self._update_scan_summary()

    # --------------------------------------------------------------------------
    # Display & stats
    # --------------------------------------------------------------------------
    def _render_live(self, img: np.ndarray):
        """Draw img into the live view, cropped to the current zoom window."""
        if not PIL_AVAILABLE:
            return
        try:
            oh, ow = img.shape[:2]
            z = self._zoom
            cw, ch = max(int(round(ow / z)), 1), max(int(round(oh / z)), 1)
            # keep the zoom window inside the image
            self._zoom_cx = min(max(self._zoom_cx, cw / 2 / ow), 1 - cw / 2 / ow)
            self._zoom_cy = min(max(self._zoom_cy, ch / 2 / oh), 1 - ch / 2 / oh)
            x0 = int(round(self._zoom_cx * ow - cw / 2))
            y0 = int(round(self._zoom_cy * oh - ch / 2))
            crop = img[y0:y0 + ch, x0:x0 + cw]

            aw = max(self.live_label.winfo_width() - 10, 1)
            ah = max(self.live_label.winfo_height() - 10, 1)
            # unzoomed: never enlarge; zoomed: fill the view, show pixels as blocks
            scale = min(aw / cw, ah / ch) if z > 1 else min(aw / cw, ah / ch, 1.0)
            nw, nh = max(int(cw * scale), 1), max(int(ch * scale), 1)
            interp = cv2.INTER_NEAREST if scale > 1 else cv2.INTER_AREA
            resized = cv2.resize(crop, (nw, nh), interpolation=interp)
            photo = ImageTk.PhotoImage(image=Image.fromarray(resized))
            self.live_label.config(image=photo, anchor="center")
            self.live_label.image = photo
            self._disp_geom = (img.shape[:2], cw, ch, nw, nh)
        except Exception as e:
            print("Display error:", e)

    def _view_to_image_fraction(self, ex, ey):
        """Mouse position in the live view -> position in the full image (0..1), or None."""
        if self._disp_geom is None:
            return None
        (oh, ow), cw, ch, nw, nh = self._disp_geom
        px = (ex - (self.live_label.winfo_width() - nw) / 2) / nw    # 0..1 across the shown crop
        py = (ey - (self.live_label.winfo_height() - nh) / 2) / nh
        px, py = min(max(px, 0.0), 1.0), min(max(py, 0.0), 1.0)
        return (self._zoom_cx + (px - 0.5) * cw / ow,
                self._zoom_cy + (py - 0.5) * ch / oh, px, py)

    def _on_live_wheel(self, event):
        pos = self._view_to_image_fraction(event.x, event.y)
        factor = 1.25 if event.delta > 0 else 1 / 1.25
        new_zoom = min(max(self._zoom * factor, 1.0), self.MAX_ZOOM)
        if pos is not None and new_zoom != self._zoom:
            fx, fy, px, py = pos
            # keep the image point under the cursor where it is
            self._zoom_cx = fx - (px - 0.5) / new_zoom
            self._zoom_cy = fy - (py - 0.5) / new_zoom
        self._zoom = new_zoom
        self._refresh_live_zoom()

    def _on_live_press(self, event):
        self._drag_start = (event.x, event.y, self._zoom_cx, self._zoom_cy)

    def _on_live_drag(self, event):
        if self._drag_start is None or self._disp_geom is None or self._zoom <= 1:
            return
        x0, y0, cx0, cy0 = self._drag_start
        (oh, ow), cw, ch, nw, nh = self._disp_geom
        self._zoom_cx = cx0 - (event.x - x0) / nw * cw / ow
        self._zoom_cy = cy0 - (event.y - y0) / nh * ch / oh
        self._refresh_live_zoom()

    def _reset_live_zoom(self, event=None):
        self._zoom, self._zoom_cx, self._zoom_cy = 1.0, 0.5, 0.5
        self._refresh_live_zoom()

    def _refresh_live_zoom(self):
        self.zoom_label.config(text=f"Zoom: {self._zoom:.1f}×")
        # redraw at once, so zooming also works when no new frames arrive (e.g. waiting for a trigger)
        if self._last_disp_img is not None:
            self._render_live(self._last_disp_img)

    def _update_display(self):
        if not self.streaming:
            return
        img_to_show = None
        while not self.frame_queue.empty():
            try:
                img_to_show = self.frame_queue.get_nowait()
            except queue.Empty:
                break
        if img_to_show is not None:
            self._last_disp_img = img_to_show
            self._render_live(img_to_show)

        if time.time() - self.fps_start_time > 1.0:
            fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = time.time()
            acq_str = "  |  ACQUIRING" if self.acquiring else ""
            sync_str = ""
            hw_live = self.live_hw_var.get() and not self.acquiring
            if hw_live:
                acq_str = "  |  LIVE ON HARDWARE TRIGGER (not saved)"
            if self.cam and (hw_live or (self.acquiring and "Hardware" in self.trigger_mode_var.get())):
                try:
                    levels = self.cam.get_feature_by_name("SyncInLevels").get()
                    sync_str = f"  |  SyncIn1={'HIGH' if (levels & 1) else 'LOW'}"
                except Exception:
                    pass
            self.status.config(text=f"Streaming… FPS: {fps}{acq_str}{sync_str}")
        self.root.after(16, self._update_display)

    def _update_stats(self):
        self.stat_triggers.config(text=f"Triggers received: {self.trigger_index}")
        self.stat_frames.config(text=f"Frames saved: {self.total_frames_saved}")

    # --------------------------------------------------------------------------
    # Camera settings helpers
    # --------------------------------------------------------------------------
    def apply_all_settings(self):
        if not self.cam:
            return
        try:
            self.cam.get_feature_by_name("ExposureTimeAbs").set(int(self.exp_var.get()))
            self.cam.get_feature_by_name("Gain").set(self.gain_var.get())
            self.cam.get_feature_by_name("Gamma").set(self.gamma_var.get())
            self.cam.get_feature_by_name("BlackLevel").set(self.black_var.get())
        except Exception as e:
            print("Settings warning:", e)
        if self.streaming:
            self._binning_pending = True   # applied by the stream worker
        else:
            self._apply_binning()

    def _apply_binning(self):
        """Set binning and reset the image to the full (binned) sensor.

        The camera shrinks Width/Height when binning goes up but does not
        enlarge them again when it goes down, which leaves a small crop in
        the top-left corner. Call only while the stream is stopped.
        """
        try:
            self.cam.get_feature_by_name("BinningHorizontal").set(self.bin_h_var.get())
            self.cam.get_feature_by_name("BinningVertical").set(self.bin_v_var.get())
            self.cam.get_feature_by_name("OffsetX").set(0)
            self.cam.get_feature_by_name("OffsetY").set(0)
            self.cam.get_feature_by_name("Width").set(self.cam.get_feature_by_name("WidthMax").get())
            self.cam.get_feature_by_name("Height").set(self.cam.get_feature_by_name("HeightMax").get())
        except Exception as e:
            print("Binning warning:", e)

    def _set_and_read_back(self, feature, value, var):
        """Send a value to the camera and show the value it actually uses (the camera
        rounds some settings, e.g. Gain to whole dB)."""
        if not (self.cam and self.streaming):
            return
        try:
            f = self.cam.get_feature_by_name(feature)
            f.set(value)
            actual = f.get()
            if abs(actual - var.get()) > 1e-6:
                var.set(actual)
        except Exception as e:
            print(f"{feature} warning:", e)

    def update_exposure(self, *args):
        self._set_and_read_back("ExposureTimeAbs", int(self.exp_var.get()), self.exp_var)

    def update_gain(self, *args):
        self._set_and_read_back("Gain", self.gain_var.get(), self.gain_var)

    def update_gamma(self, *args):
        self._set_and_read_back("Gamma", self.gamma_var.get(), self.gamma_var)

    def update_blacklevel(self, *args):
        self._set_and_read_back("BlackLevel", self.black_var.get(), self.black_var)

    def update_binning(self, event=None):
        if self.cam and self.streaming:
            self._binning_pending = True   # applied by the stream worker

    def _dump_camera_features(self):
        """Print all available camera feature names to the console for diagnostics."""
        if not self.cam:
            messagebox.showwarning("Warning", "Connect camera first")
            return
        try:
            features = self.cam.get_all_features()
            print("\n=== Camera Features ===")
            for f in sorted(features, key=lambda x: x.get_name()):
                try:
                    name = f.get_name()
                    val  = f.get()
                    print(f"  {name} = {val}")
                except Exception:
                    try:
                        print(f"  {f.get_name()} (not readable)")
                    except Exception:
                        pass
            print("=== End of Features ===\n")
            messagebox.showinfo("Done", "Feature list printed to console.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _browse_folder(self):
        folder = filedialog.askdirectory(initialdir=self.save_folder.get() or os.path.expanduser("~"))
        if folder:
            self.save_folder.set(folder)


def _save_png(img: np.ndarray, path: str):
    try:
        # imwrite returns False instead of raising, e.g. if the path is too long
        if not cv2.imwrite(path, img):
            print(f"PNG save failed ({path})")
    except Exception as e:
        print(f"PNG save error ({path}):", e)


if __name__ == "__main__":
    MakoCameraGUI()