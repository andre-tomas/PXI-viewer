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
                            if self.acquiring:
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

    # --------------------------------------------------------------------------
    # Hardware trigger: each frame is one trigger
    # --------------------------------------------------------------------------
    def _save_hardware_trigger_frame(self, img: np.ndarray, raw_frame):
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
            trigger_delay_us=self._delay_value,
        )

        # PNG
        if self.save_png_var.get():
            fname = (f"trigger{self.trigger_index:06d}"
                     f"_frame0000_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}.png")
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

    def _start_acquisition(self):
        # One subfolder per day (YY-MM-DD) inside the chosen folder, and in it
        # one folder per run: <day>/<base><N>/ holds that run's H5 files and PNGs
        day_folder = os.path.join(self.save_folder.get(), datetime.now().strftime("%y-%m-%d"))
        base = self.base_filename.get()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            os.makedirs(day_folder, exist_ok=True)
            # Runs are numbered within the day: <base>1, <base>2, ...
            self._run_name = f"{base}{next_run_number(day_folder, base)}"
            folder = os.path.join(day_folder, self._run_name)
            os.makedirs(folder, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Error", f"Cannot create folder:\n{e}")
            return
        self._run_dir = folder

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
        self.acquiring = False
        self._sw_trigger_pending = False
        self.acq_btn.config(text="Start Acquisition")
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
        self._send_trigger_delay(delay)

    def _send_trigger_delay(self, delay):
        """Set TriggerDelayAbs and remember the value the camera uses (it rounds to 1 µs)."""
        try:
            f = self.cam.get_feature_by_name("TriggerDelayAbs")
            f.set(max(delay, 0.0))
            self._delay_value = float(f.get())
            print(f"TriggerDelayAbs set to {self._delay_value} us")
        except Exception as e:
            print(f"TriggerDelayAbs warning: {e}")

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