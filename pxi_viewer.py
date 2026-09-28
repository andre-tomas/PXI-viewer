#!/usr/bin/env python
"""PXI-viewer: interactive viewer for 2D detector frames stored as HDF5 files (one frame per file).

Usage:
    python pxi_viewer.py [DATA_DIR ...] [--key DATASET]

Every directory below a DATA_DIR that contains *.h5 / *.hdf5 files is listed as a
run; folders can also be added from the GUI ("Add folder…"). Files in a run are
ordered by natural sort of their names (numbers compared numerically). The frame
is the first 2D dataset found in the file, or the dataset given with --key
(e.g. --key entry/data/frame). All frames of a run must have the same shape.
Directories whose name contains "dark" are offered as dark runs.
Several runs can be selected (Ctrl/Shift-click) and are then loaded together as
one set of frames; each frame remembers which run it came from.

Hits tab: a hit is a connected group of unmasked pixels >= threshold in the frame
as displayed (dark subtracted, masked, binned); its position is the centre of its
brightest binned pixel. Positions are given relative to a centre (typed, picked
on the image or found automatically) in mm, using the detector diameter and the
fitted active circle, and exported to HDF5 (see HITS_CONVENTION).
Needs numpy, h5py, pyqtgraph (with PyQt5) and scipy.

Keys:  Left/Right  prev/next pattern     PgUp/PgDn  jump 10
       F  single frame   M  mean   D  median   S  std
       A  auto levels (percentile)   L  toggle log scale
       +/-  coarser/finer binning (never finer than the native pixels)
       B  toggle sum / max binning
       E  toggle mask painting       Ctrl+Z  undo last mask edit

Mask: stored at native detector resolution, True/1 = masked (excluded) pixel.
While painting, a click masks the displayed pixel, i.e. a whole b×b block when
binned, so one can mask coarsely at a high bin factor and then refine at lower
ones. Right-click or Shift+click removes. In binned views a bin sums only its
unmasked pixels; it is hidden once the masked fraction reaches the threshold,
otherwise (sum binning) it can be rescaled by b²/n_valid.
"""
import os
import re
import sys
from glob import glob

import h5py
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

H5_EXT = (".h5", ".hdf5")
FRAME_KEY = None  # dataset path inside each file; None = first 2D dataset (set by --key)

VIEW_FRAME, VIEW_MEAN, VIEW_MEDIAN, VIEW_STD = "Frame", "Mean", "Median", "Std"
SORT_FILE, SORT_DESC, SORT_ASC = ("File order", "Total intensity (high → low)",
                                     "Total intensity (low → high)")
HIST_CURRENT, HIST_ALL, HIST_SEL = ("Current view", "All frames in run",
                                    "Stats selection (first N)")
MASK_CONVENTION = "1 = masked (excluded) pixel, 0 = good pixel; native detector resolution"


def find_runs(root):
    """Map '<root name>/<relative dir>' -> path for every dir below root holding *.h5."""
    root = os.path.normpath(os.path.abspath(root))
    base = os.path.basename(root)
    runs = {}
    for dirpath, _, files in os.walk(root):
        if any(f.endswith(H5_EXT) for f in files):
            rel = os.path.relpath(dirpath, root)
            runs[base if rel == "." else os.path.join(base, rel)] = dirpath
    return runs


def rebin(img, b, mode="sum"):
    """Bin the last two axes of img by an integer factor b (b=1: unchanged).

    mode "sum" adds the b×b pixels of a bin, "max" takes the brightest one.
    Edge rows/columns that do not fill a whole bin are cropped.
    """
    if b <= 1:
        return img
    h, w = (img.shape[-2] // b) * b, (img.shape[-1] // b) * b
    a = img[..., :h, :w].reshape(img.shape[:-2] + (h // b, b, w // b, b))
    return a.max(axis=(-3, -1)) if mode == "max" else a.sum(axis=(-3, -1))


def masked_fraction(mask, b):
    """Fraction of masked native pixels in each b×b bin."""
    return rebin(mask.astype(np.float32), b) / float(b * b)


def masked_rebin(img, mask, b, threshold=0.5, rescale=True, mode="sum"):
    """Bin img (..., H, W) (mode "sum" or "max") using only unmasked pixels.

    Bins whose masked fraction is >= threshold become NaN. For sum binning,
    partially masked bins are multiplied by b²/n_valid when rescale is True, so
    they stay comparable to fully valid bins (max binning needs no rescaling).
    """
    img = np.asarray(img, dtype=np.float32)
    if mask is None or not mask.any():
        return rebin(img, b, mode)
    fill = np.float32(-np.inf if mode == "max" else 0)
    out = rebin(np.where(mask, fill, img), b, mode)
    frac = masked_fraction(mask, b)
    dead = frac >= threshold
    if rescale and mode == "sum":
        part = (frac > 0) & ~dead
        out[..., part] /= (1 - frac[part])
    out[..., dead] = np.nan
    return out


def outline_segments(region, b):
    """Boundary of a boolean bin map as line segments in native-pixel coordinates.

    Returns x, y arrays of point pairs (for connect="pairs"): one segment per bin
    edge between a region bin and a non-region bin.
    """
    d = np.pad(region, 1, mode="edge")  # no outline along the sensor border
    iv, jv = np.nonzero(d[1:-1, 1:] != d[1:-1, :-1])   # vertical edges at x=j
    ih, jh = np.nonzero(d[1:, 1:-1] != d[:-1, 1:-1])   # horizontal edges at y=i
    x = np.concatenate([np.repeat(jv, 2), np.stack([jh, jh + 1], 1).ravel()]) * b
    y = np.concatenate([np.stack([iv, iv + 1], 1).ravel(), np.repeat(ih, 2)]) * b
    return x.astype(np.float64), y.astype(np.float64)


def box_filter(img, r):
    """Mean over a (2r+1)² box (edges padded by replication)."""
    if r <= 0:
        return img.astype(np.float64)
    k = 2 * r + 1
    c = np.pad(img.astype(np.float64), r, mode="edge").cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)))
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def dilate(mask, n):
    """Grow a boolean mask by n pixels (8-connected steps, i.e. roughly octagonal)."""
    out = mask.copy()
    for _ in range(int(n)):
        m = out.copy()
        m[1:] |= out[:-1]
        m[:-1] |= out[1:]
        m[:, 1:] |= out[:, :-1]
        m[:, :-1] |= out[:, 1:]
        m[1:, 1:] |= out[:-1, :-1]
        m[:-1, :-1] |= out[1:, 1:]
        m[1:, :-1] |= out[:-1, 1:]
        m[:-1, 1:] |= out[1:, :-1]
        out = m
    return out


def robust_sigma(v):
    """Median and MAD-based sigma (falls back to the plain std when the MAD is 0)."""
    med = np.median(v)
    sig = 1.4826 * np.median(np.abs(v - med))
    return med, (sig if sig > 0 else v.std())


def fit_active_circle(act, level=0.5, smooth=3):
    """Find the circular active area of the detector in an activity map (std or mean).

    The map is smoothed and thresholded at `level` between the outside level (2nd
    percentile) and the inside level. The inside level is taken from the dim part
    of the active area (10th percentile of the pixels above the outside level), so
    bright signal in the middle does not pull the edge inwards. A circle is fitted
    (least squares, with outlier rejection) to the left/right edges of the
    thresholded region in every row, which works even when the circle is cut off
    at the top/bottom of the sensor. Returns (cx, cy, R), or None when there is no
    usable edge (e.g. a run with (almost) no counts).
    """
    S = box_filter(act, smooth)
    h, w = S.shape
    lo = np.percentile(S, 2)
    inside = S[S > lo + 1e-12 * max(abs(lo), 1)]
    if inside.size < 0.05 * S.size:  # hardly anything active: no circle to find
        return None
    hi = np.percentile(inside, 10)
    if hi <= lo:
        return None
    region = S > lo + level * (hi - lo)
    xs, ys = [], []
    for y in range(h):
        idx = np.flatnonzero(region[y])
        if idx.size < 2:
            continue
        for x in (idx[0], idx[-1]):
            if 0 < x < w - 1:  # points on the sensor border are not circle edges
                xs.append(x + 0.5)
                ys.append(y + 0.5)
    x, y = np.array(xs), np.array(ys)
    if x.size < 10:
        return None

    def fit(x, y):
        A = np.c_[2 * x, 2 * y, np.ones_like(x)]
        cx, cy, c = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)[0]
        return cx, cy, np.sqrt(max(c + cx ** 2 + cy ** 2, 0))

    keep = np.ones(x.size, bool)
    for _ in range(4):
        cx, cy, R = fit(x[keep], y[keep])
        res = np.abs(np.hypot(x - cx, y - cy) - R)
        keep = res < max(2.0, 3 * np.median(res))
    return fit(x[keep], y[keep])


def edge_contrast(act, circle, width=10):
    """Median of the map in a band just inside the circle / just outside it."""
    cx, cy, R = circle
    yy, xx = np.indices(act.shape)
    r = np.hypot(xx + 0.5 - cx, yy + 0.5 - cy)
    band_in = act[(r > R - 3 * width) & (r < R - width)]
    band_out = act[(r > R + width) & (r < R + 3 * width)]
    if band_in.size < 50 or band_out.size < 50:
        return np.nan
    i, o = np.median(band_in), np.median(band_out)
    return np.inf if o <= 0 and i > 0 else (i / o if o > 0 else np.nan)


def natural_key(path):
    """Sort key comparing the numbers in a file name numerically (img2 < img10)."""
    parts = re.split(r"(\d+)", os.path.basename(path))
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def list_frames(run_dir):
    """HDF5 files of a run in natural-sort order."""
    files = [f for f in glob(os.path.join(run_dir, "*")) if f.endswith(H5_EXT)]
    return sorted(files, key=natural_key)


def frame_dataset_name(h5, key=None):
    """Path of the frame dataset: `key` if given, else the first 2D numeric dataset."""
    if key:
        return key
    found = []

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim == 2 and obj.dtype.kind in "uif":
            found.append(name)
            return True  # stop at the first match
    h5.visititems(visit)
    if not found:
        raise KeyError(f"no 2D dataset in {h5.filename}; use --key to name it")
    return found[0]


def scalar_attrs(ds, limit=12):
    """Scalar attributes of a dataset, for display."""
    out = {}
    for k, v in ds.attrs.items():
        if np.ndim(v) == 0:
            out[k] = v.decode(errors="replace") if isinstance(v, bytes) else v
            if len(out) >= limit:
                break
    return out


def save_mask(path, mask):
    if path.endswith(".npy"):
        np.save(path, mask.astype(np.uint8))
    else:
        with h5py.File(path, "w") as h5:
            ds = h5.create_dataset("mask", data=mask.astype(np.uint8), compression="gzip")
            ds.attrs["convention"] = MASK_CONVENTION


def load_mask(path):
    """Read a mask from .npy or .h5 (dataset 'mask', else the first 2D dataset)."""
    if path.endswith(".npy"):
        return np.load(path).astype(bool)
    with h5py.File(path, "r") as h5:
        if "mask" in h5:
            return h5["mask"][()].astype(bool)
        found = []
        h5.visititems(lambda n, o: found.append(o) if isinstance(o, h5py.Dataset)
                      and o.ndim == 2 else None)
        if not found:
            raise ValueError(f"no 2D dataset in {path}")
        return found[0][()].astype(bool)


class Loader(QtCore.QThread):
    """Reads all frames of one or more runs in the background (concatenated)."""
    progress = QtCore.pyqtSignal(int, int)
    done = QtCore.pyqtSignal(object, object, object, object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, run_dirs):
        super().__init__()
        self.run_dirs = run_dirs

    def run(self):
        try:
            files, run_of = [], []
            for r, d in enumerate(self.run_dirs):
                fs = list_frames(d)
                files += fs
                run_of += [r] * len(fs)
            frames, meta, key = None, [], FRAME_KEY
            for i, f in enumerate(files):
                with h5py.File(f, "r") as h5:
                    key = frame_dataset_name(h5, key)  # detected once, then reused
                    ds = h5[key]
                    if frames is None:
                        frames = np.empty((len(files),) + ds.shape, dtype=ds.dtype)
                    elif ds.shape != frames.shape[1:]:
                        raise ValueError(f"{os.path.basename(f)}: frame shape {ds.shape} "
                                         f"differs from {frames.shape[1:]}")
                    frames[i] = ds[()]
                    meta.append(scalar_attrs(ds))
                if i % 25 == 0:
                    self.progress.emit(i, len(files))
            self.done.emit(frames, files, meta, np.array(run_of, dtype=int))
        except Exception as e:  # surface I/O errors in the GUI
            self.failed.emit(repr(e))


def auto_mask(mean, std, circle=None, margin=0.0, k=10.0, smooth=2, grow=2,
              use_mean=True, use_std=True):
    """First-guess mask from the mean and std images of a background run.

    - outside the active circle (cx, cy, R), shrunk by `margin` px, is masked;
    - inside, pixels whose box-smoothed mean and/or std exceeds the median by more
      than k robust sigmas (MAD) are masked, then grown by `grow` px.
    Returns (mask, background_part).
    """
    h, w = mean.shape
    if circle is not None:
        cx, cy, R = circle
        yy, xx = np.indices((h, w))
        inside = np.hypot(xx + 0.5 - cx, yy + 0.5 - cy) <= R - margin
    else:
        inside = np.ones((h, w), bool)
    bg = np.zeros((h, w), bool)
    for img, use in ((mean, use_mean), (std, use_std)):
        if use and inside.any():
            v = box_filter(img, smooth)
            med, sig = robust_sigma(v[inside])
            bg |= v > med + k * sig
    bg = dilate(bg & inside, grow) & inside
    return ~inside | bg, bg


def find_hits(img, threshold, min_size=1):
    """Hits in one (binned, masked) image: connected groups of pixels >= threshold.

    Pixels that are NaN (masked) never belong to a hit. Groups are 8-connected
    (diagonal neighbours touch). Each hit is reported at its brightest pixel.
    Returns dict of arrays: ix, iy (pixel index of the peak), peak, size, sum.
    """
    from scipy import ndimage as ndi  # only needed for hit finding
    valid = np.isfinite(img)
    above = valid & (img >= threshold)
    lab, n = ndi.label(above, structure=np.ones((3, 3), bool))
    empty = dict(ix=np.zeros(0, int), iy=np.zeros(0, int), peak=np.zeros(0),
                 size=np.zeros(0, int), sum=np.zeros(0))
    if n == 0:
        return empty
    # work on the (few) hit pixels only: much faster than ndi.maximum_position
    flat = np.flatnonzero(lab)
    lbl = lab.ravel()[flat]
    val = img.ravel()[flat]
    size = np.bincount(lbl, minlength=n + 1)[1:]
    total = np.bincount(lbl, weights=val, minlength=n + 1)[1:]
    order = np.lexsort((val, lbl))                      # by label, then value
    last = np.r_[np.flatnonzero(np.diff(lbl[order])), len(order) - 1]
    peak_flat = flat[order[last]]                       # brightest pixel per label
    keep = size >= min_size
    iy, ix = np.unravel_index(peak_flat[keep], img.shape)
    return dict(ix=ix, iy=iy, peak=img.ravel()[peak_flat[keep]], size=size[keep],
                sum=total[keep])


def find_center(mean, mask=None, radius=10):
    """Centre of the brightest region: argmax of the box-smoothed (masked) mean
    image, refined by the intensity-weighted centroid within `radius` of it.
    Returns (x, y) in native pixel coordinates (pixel i spans [i, i+1))."""
    img = np.where(mask, 0.0, mean) if mask is not None else mean.astype(np.float64)
    sm = box_filter(img, radius)
    iy, ix = np.unravel_index(np.argmax(sm), sm.shape)
    y0, y1 = max(iy - radius, 0), min(iy + radius + 1, sm.shape[0])
    x0, x1 = max(ix - radius, 0), min(ix + radius + 1, sm.shape[1])
    win = np.clip(sm[y0:y1, x0:x1] - sm[y0:y1, x0:x1].min(), 0, None)
    if win.sum() <= 0:
        return ix + 0.5, iy + 0.5
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return (float((win * (xx + 0.5)).sum() / win.sum()),
            float((win * (yy + 0.5)).sum() / win.sum()))


HITS_CONVENTION = (
    "hits[f, h, 0, :] = (x, y) position of hit h in frame f relative to the centre, in mm "
    "(x to the right, y downwards as in the image); hits[f, h, 1, :] = (ix, iy) index of "
    "the hit's brightest binned pixel. Position = centre of that binned pixel: native "
    "pixel i spans [i, i+1), binned pixel j (bin factor b) has centre (j + 0.5) * b. "
    "Unused hit slots are NaN; n_hits[f] gives the number of hits in frame f. Frames are "
    "in file order (see 'files').")


class WheelGuard(QtCore.QObject):
    """Stops the mouse wheel from changing spin boxes, combo boxes and sliders.

    The wheel event is passed on to the enclosing scroll area instead, so scrolling
    the side panel never changes a setting by accident.
    """

    def eventFilter(self, obj, ev):
        if ev.type() != QtCore.QEvent.Wheel:
            return False
        w = obj.parentWidget()
        while w is not None and not isinstance(w, QtWidgets.QScrollArea):
            w = w.parentWidget()
        if w is not None:
            QtWidgets.QApplication.sendEvent(w.verticalScrollBar(), ev)
        return True  # never let the control itself see the wheel


def install_wheel_guard(root, guard):
    """Guard every spin box, combo box and slider below `root`."""
    for w in root.findChildren((QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox,
                                QtWidgets.QSlider)):
        w.setFocusPolicy(QtCore.Qt.StrongFocus)  # no focus grab by wheel either
        w.installEventFilter(guard)


class AutoMaskDialog(QtWidgets.QDialog):
    """Parameters + live preview for auto_mask() on the loaded run."""

    def __init__(self, viewer):
        super().__init__(viewer)
        self.v = viewer
        self.setWindowTitle("Auto mask from background run")
        self.mean = self.std = None
        self.candidate = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(QtWidgets.QLabel(
            "Uses the mean and std of the loaded run (the frames selected by\n"
            "'Stats over first N'; dark subtracted if that is on).\n"
            "Load a background run (no signal of interest) first."))

        g = QtWidgets.QGroupBox("Active area (circle)")
        g.setCheckable(True)
        f = QtWidgets.QFormLayout(g)
        self.circle_grp = g
        self.circ_src = QtWidgets.QComboBox()
        self.circ_src.addItems(["std", "mean"])
        self.level = self._dspin(1, 99, 50, " %", "Edge = where the smoothed map crosses "
                                 "this fraction between the outside and inside levels")
        self.circ_smooth = self._ispin(0, 20, 3, " px")
        self.fit_btn = QtWidgets.QPushButton("Fit circle")
        self.fit_btn.clicked.connect(self.fit_circle)
        self.cx = self._dspin(-5000, 5000, 0, " px")
        self.cy = self._dspin(-5000, 5000, 0, " px")
        self.R = self._dspin(0, 5000, 0, " px")
        self.margin = self._dspin(-100, 200, 10, " px", "Shrink the circle by this much, to "
                                  "also mask the soft edge (ring); negative grows it")
        self.circ_info = QtWidgets.QLabel()
        self.circ_info.setWordWrap(True)
        f.addRow("Fit on", self.circ_src)
        f.addRow("Edge level", self.level)
        f.addRow("Smoothing", self.circ_smooth)
        f.addRow(self.fit_btn)
        f.addRow("Centre x", self.cx)
        f.addRow("Centre y", self.cy)
        f.addRow("Radius", self.R)
        f.addRow("Margin (shrink)", self.margin)
        f.addRow(self.circ_info)
        lay.addWidget(g)

        g = QtWidgets.QGroupBox("Background / hot pixels (robust σ cut)")
        g.setCheckable(True)
        f = QtWidgets.QFormLayout(g)
        self.bg_grp = g
        self.use_mean = QtWidgets.QCheckBox("mean")
        self.use_std = QtWidgets.QCheckBox("std")
        for c in (self.use_mean, self.use_std):
            c.setChecked(True)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.use_mean)
        row.addWidget(self.use_std)
        self.k = self._dspin(0.5, 1000, 10, " σ", "Mask where the smoothed value exceeds "
                             "median + k·σ, with σ from the MAD inside the circle")
        self.bg_smooth = self._ispin(0, 20, 2, " px", "Box smoothing radius before the cut "
                                     "(0 = per pixel, catches isolated hot pixels)")
        self.grow = self._ispin(0, 50, 2, " px", "Grow the masked regions by this much")
        f.addRow("Cut on", row)
        f.addRow("Threshold k", self.k)
        f.addRow("Smoothing", self.bg_smooth)
        f.addRow("Grow", self.grow)
        lay.addWidget(g)

        self.info = QtWidgets.QLabel()
        lay.addWidget(self.info)
        btns = QtWidgets.QHBoxLayout()
        for text, fn in (("Replace mask", lambda: self.apply(False)),
                         ("Add to mask", lambda: self.apply(True)), ("Close", self.close)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            btns.addWidget(b)
        lay.addLayout(btns)

        for wdg in (self.cx, self.cy, self.R, self.margin, self.k):
            wdg.valueChanged.connect(self.preview)
        for wdg in (self.bg_smooth, self.grow):
            wdg.valueChanged.connect(self.preview)
        for wdg in (self.use_mean, self.use_std, self.circle_grp, self.bg_grp):
            wdg.toggled.connect(self.preview)
        for wdg in (self.circ_src, ):
            wdg.currentTextChanged.connect(self.fit_circle)
        self.level.valueChanged.connect(self.fit_circle)
        self.circ_smooth.valueChanged.connect(self.fit_circle)

        # preview drawn on the image: candidate outline (cyan) and fitted circle (green)
        self.prev_curve = pg.PlotCurveItem(connect="pairs", pen=pg.mkPen((0, 230, 255), width=1.5))
        self.circ_curve = pg.PlotCurveItem(pen=pg.mkPen((80, 255, 80), width=2,
                                                        style=QtCore.Qt.DashLine))
        for it in (self.prev_curve, self.circ_curve):
            it.setZValue(20)
            viewer.imv.view.addItem(it)

    @staticmethod
    def _dspin(lo, hi, val, suffix="", tip=""):
        s = QtWidgets.QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(1)
        s.setValue(val)
        s.setSuffix(suffix)
        s.setKeyboardTracking(False)
        s.setToolTip(tip)
        return s

    @staticmethod
    def _ispin(lo, hi, val, suffix="", tip=""):
        s = QtWidgets.QSpinBox()
        s.setRange(lo, hi)
        s.setValue(val)
        s.setSuffix(suffix)
        s.setKeyboardTracking(False)
        s.setToolTip(tip)
        return s

    def start(self):
        """(Re)read mean/std of the loaded run and fit the circle."""
        v = self.v
        if v.frames is None:
            QtWidgets.QMessageBox.information(v, "Auto mask", "Load a (background) run first.")
            return False
        self.mean = v._stat_image(VIEW_MEAN)[0]
        self.std = v._stat_image(VIEW_STD)[0]
        self.fit_circle()
        return True

    def fit_circle(self, *_):
        if self.mean is None:
            return
        src = self.std if self.circ_src.currentText() == "std" else self.mean
        c = fit_active_circle(src, self.level.value() / 100.0, self.circ_smooth.value())
        h, w = src.shape
        if c is None:
            self.circ_info.setText("No circular edge found — enter the circle by hand "
                                   "or untick this box.")
            self.preview()
            return
        cx, cy, R = c
        con = edge_contrast(box_filter(src, 3), c)
        weak = not (con >= 3)  # also True for NaN (no room outside the circle)
        self.circ_info.setText(
            f"fit: centre ({cx:.1f}, {cy:.1f}), R = {R:.1f} px, edge contrast "
            f"in/out = {con:.3g}" +
            ("<br><b>Weak or no edge: this run probably does not show the active area "
             "(e.g. long exposure / dark offset). Use a short-exposure background run, or "
             "enter the circle by hand.</b>" if weak else ""))
        for s_, val in ((self.cx, cx), (self.cy, cy), (self.R, R)):
            s_.blockSignals(True)
            s_.setValue(val)
            s_.blockSignals(False)
        self.preview()

    def compute(self):
        circle = ((self.cx.value(), self.cy.value(), self.R.value())
                  if self.circle_grp.isChecked() else None)
        bg_on = self.bg_grp.isChecked()
        mask, bg = auto_mask(self.mean, self.std, circle, self.margin.value(),
                             self.k.value(), self.bg_smooth.value(), self.grow.value(),
                             bg_on and self.use_mean.isChecked(),
                             bg_on and self.use_std.isChecked())
        return mask, bg, circle

    def preview(self, *_):
        if self.mean is None:
            return
        mask, bg, circle = self.compute()
        self.candidate = mask
        self.prev_curve.setData(*outline_segments(bg, 1))
        if circle is not None:
            t = np.linspace(0, 2 * np.pi, 721)
            r = circle[2] - self.margin.value()
            self.circ_curve.setData(circle[0] + r * np.cos(t), circle[1] + r * np.sin(t))
        self.circ_curve.setVisible(circle is not None)
        n = mask.size
        self.info.setText(f"candidate: {int(mask.sum())} px masked ({100.0 * mask.sum() / n:.1f} %)"
                          f", of which background/hot: {int(bg.sum())} px\n"
                          "preview: green dashed = circle, cyan = background cut")

    def apply(self, add):
        if self.candidate is None:
            return
        v = self.v
        v._push_undo()
        v.mask = (v.mask | self.candidate) if add else self.candidate.copy()
        if self.circle_grp.isChecked():  # also sets the detector scale for hits (mm/px)
            v.set_circle((self.cx.value(), self.cy.value(), self.R.value()))
        v._mask_changed()
        v.status.showMessage("Auto mask " + ("added" if add else "applied") +
                             f": {int(v.mask.sum())} px masked (Ctrl+Z to undo)")

    def closeEvent(self, ev):
        for it in (self.prev_curve, self.circ_curve):
            it.setData([], [])
        super().closeEvent(ev)


class PatternViewer(QtWidgets.QMainWindow):
    def __init__(self, roots):
        super().__init__()
        self.setWindowTitle("PXI-viewer")
        self.resize(1550, 1000)
        self.runs = {}
        for r in roots:
            self.runs.update(find_runs(r))

        self.frames = None      # (N, H, W) raw uint8
        self.files = []
        self.meta = []
        self.dark_cache = {}    # run name -> mean dark frame (float32)
        self.stat_cache = {}    # (view, n) -> image
        self.raw_totals = None  # plain sum of each frame
        self.totals = None      # total intensity per frame (dark subtracted, mask applied)
        self.order = None       # frame indices in display order
        self.loader = None
        self.shown = None       # displayed values with the mask applied (stats, levels)
        self.rendered = None    # what is drawn (masked pixels kept in outline mode)
        self.mask = None        # (H, W) bool, True = masked
        self.undo = []
        self.loaded_runs = []   # names of the runs currently loaded
        self.run_of = None      # index into loaded_runs for every frame
        self.run_lines = []     # run boundaries in the total-intensity plot
        self._last_paint = None
        self.pins = []          # pinned histogram curves

        self._build_ui()
        self._install_shortcuts()
        self._wheel_guard = WheelGuard(self)
        install_wheel_guard(self, self._wheel_guard)
        self._fill_run_combos()
        if not roots:
            self.status.showMessage("Use 'Add folder…' to open a data folder")
        elif not self.runs:
            self.status.showMessage(f"No HDF5 files found under {', '.join(roots)}")

    # ---------------------------------------------------------------- UI
    def _build_ui(self):
        pg.setConfigOptions(imageAxisOrder="row-major")

        # image + histogram/LUT (drag the handles on the histogram to set colour limits)
        self.imv = pg.ImageView(view=pg.PlotItem())
        self.imv.ui.roiBtn.hide()
        self.imv.ui.menuBtn.hide()
        self.imv.view.setLabel("bottom", "x (px)")
        self.imv.view.setLabel("left", "y (px)")
        self.imv.view.invertY(True)
        self.imv.getHistogramWidget().item.sigLevelsChanged.connect(self._levels_from_hist)
        # mask overlay: red = masked bin, orange = partially masked bin; drawn either
        # as outlines (default) or as a translucent fill
        self.overlay = pg.ImageItem()
        self.outline_dead = pg.PlotCurveItem(connect="pairs",
                                             pen=pg.mkPen((255, 40, 40), width=2))
        self.outline_part = pg.PlotCurveItem(connect="pairs", pen=pg.mkPen(
            (255, 170, 0), width=1.5, style=QtCore.Qt.DashLine))
        for it in (self.overlay, self.outline_part, self.outline_dead):
            it.setZValue(10)
            it.setAcceptedMouseButtons(QtCore.Qt.NoButton)
            self.imv.view.addItem(it)

        # total intensity vs rank; click to jump
        self.tot_plot = pg.PlotWidget()
        self.tot_plot.setLabel("left", "Total intensity")
        self.tot_plot.setLabel("bottom", "Position in current order")
        self.tot_plot.showGrid(x=True, y=True, alpha=0.3)
        self.tot_curve = self.tot_plot.plot(pen=pg.mkPen((100, 160, 255), width=1))
        self.tot_marker = pg.InfiniteLine(angle=90, movable=True,
                                          pen=pg.mkPen((255, 170, 0), width=2))
        self.tot_marker.sigPositionChangeFinished.connect(
            lambda l: self.pos_spin.setValue(int(round(l.value()))))
        self.tot_plot.addItem(self.tot_marker)
        self.tot_plot.scene().sigMouseClicked.connect(self._tot_clicked)
        self.tot_plot.setMaximumHeight(220)

        image_tab = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        image_tab.addWidget(self.imv)
        image_tab.addWidget(self.tot_plot)
        image_tab.setStretchFactor(0, 4)
        image_tab.setStretchFactor(1, 1)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(image_tab, "Image")
        self.tabs.addTab(self._build_hist_tab(), "Histograms")
        self.tabs.addTab(self._build_hits_tab(), "Hits")
        self.tabs.currentChanged.connect(lambda _: self._update_hist())

        # ---- control panel
        panel = QtWidgets.QWidget()
        form = QtWidgets.QVBoxLayout(panel)

        # data
        g = QtWidgets.QGroupBox("Data")
        lay = QtWidgets.QFormLayout(g)
        # runs: filterable list, Ctrl/Shift-click to select several (loaded together)
        self.run_filter = QtWidgets.QLineEdit()
        self.run_filter.setPlaceholderText("filter runs (space-separated words)")
        self.run_filter.setClearButtonEnabled(True)
        self.run_filter.textChanged.connect(self._filter_runs)
        self.run_list = QtWidgets.QListWidget()
        self.run_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.run_list.setMinimumHeight(140)
        self.run_list.setToolTip("Ctrl/Shift-click to select several runs; they are loaded "
                                 "together as one set of frames. Double-click loads one run.")
        self.run_list.itemSelectionChanged.connect(self._run_selection_changed)
        self.run_list.itemDoubleClicked.connect(lambda _: self.load_run())
        self.run_sel_label = QtWidgets.QLabel("no run selected")
        self._n_files = {}
        self.dark_combo = QtWidgets.QComboBox()
        for c in (self.dark_combo,):
            # editable so one can type any part of the name to filter the list
            c.setEditable(True)
            c.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
            c.completer().setFilterMode(QtCore.Qt.MatchContains)
            c.completer().setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
            c.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            c.setMinimumContentsLength(8)
        self.load_btn = QtWidgets.QPushButton("Load selected")
        self.load_btn.clicked.connect(self.load_run)
        self.add_btn = QtWidgets.QPushButton("Add folder…")
        self.add_btn.setToolTip("Scan another folder for runs (*.h5, *.hdf5)")
        self.add_btn.clicked.connect(self._add_folder)
        self.dark_chk = QtWidgets.QCheckBox("Subtract mean dark")
        # names get long: full name on hover
        self.dark_combo.currentTextChanged.connect(self.dark_combo.setToolTip)
        self.dark_chk.toggled.connect(self._dark_changed)
        self.dark_combo.currentIndexChanged.connect(self._dark_changed)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setVisible(False)
        lay.addRow(self.run_filter)
        lay.addRow(self.run_list)
        lay.addRow(self.run_sel_label)
        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(self.load_btn)
        btns.addWidget(self.add_btn)
        lay.addRow(btns)
        lay.addRow(self.dark_chk)
        lay.addRow("Dark run", self.dark_combo)
        lay.addRow(self.progress)
        form.addWidget(g)

        # navigation
        g = QtWidgets.QGroupBox("Browse")
        lay = QtWidgets.QFormLayout(g)
        self.sort_combo = QtWidgets.QComboBox()
        self.sort_combo.addItems([SORT_FILE, SORT_DESC, SORT_ASC])
        self.sort_combo.currentIndexChanged.connect(lambda _: self._resort(keep_frame=False))
        self.pos_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.pos_spin = QtWidgets.QSpinBox()
        self.pos_slider.valueChanged.connect(self.pos_spin.setValue)
        self.pos_spin.valueChanged.connect(self.pos_slider.setValue)
        self.pos_spin.valueChanged.connect(self._show_frame_at)
        nav = QtWidgets.QHBoxLayout()
        for text, step in (("◀◀", -10), ("◀", -1), ("▶", 1), ("▶▶", 10)):
            b = QtWidgets.QPushButton(text)
            b.setFixedWidth(42)
            b.clicked.connect(lambda _, s=step: self.step(s))
            nav.addWidget(b)
        lay.addRow("Sort by", self.sort_combo)
        lay.addRow("Position", self.pos_spin)
        lay.addRow(self.pos_slider)
        lay.addRow(nav)
        form.addWidget(g)

        # view mode
        g = QtWidgets.QGroupBox("Show")
        lay = QtWidgets.QVBoxLayout(g)
        row = QtWidgets.QHBoxLayout()
        self.view_group = QtWidgets.QButtonGroup(self)
        for i, name in enumerate((VIEW_FRAME, VIEW_MEAN, VIEW_MEDIAN, VIEW_STD)):
            b = QtWidgets.QPushButton(name)
            b.setCheckable(True)
            b.setChecked(i == 0)
            self.view_group.addButton(b, i)
            row.addWidget(b)
        self.view_group.buttonClicked.connect(lambda _: self.refresh())
        lay.addLayout(row)
        sub = QtWidgets.QFormLayout()
        self.stat_n = QtWidgets.QSpinBox()
        self.stat_n.setRange(0, 10 ** 6)
        self.stat_n.setSpecialValueText("all frames")
        self.stat_n.setToolTip("Mean/median/std over the first N patterns of the current "
                               "sort order (e.g. the N brightest). 0 = all.")
        self.stat_n.editingFinished.connect(self.refresh)
        sub.addRow("Stats over first N", self.stat_n)
        lay.addLayout(sub)
        form.addWidget(g)

        # colour
        g = QtWidgets.QGroupBox("Colour scale")
        lay = QtWidgets.QFormLayout(g)
        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(["magma", "inferno", "viridis", "plasma", "cividis", "grey"])
        self.cmap_combo.currentTextChanged.connect(self._set_cmap)
        self.vmin = QtWidgets.QDoubleSpinBox()
        self.vmax = QtWidgets.QDoubleSpinBox()
        for s in (self.vmin, self.vmax):
            s.setRange(-1e9, 1e9)
            s.setDecimals(2)
            s.setKeyboardTracking(False)
            s.valueChanged.connect(self._levels_from_spins)
        self.lock_chk = QtWidgets.QCheckBox("Lock limits while browsing")
        self.lock_chk.setChecked(True)
        self.plo = QtWidgets.QDoubleSpinBox()
        self.phi = QtWidgets.QDoubleSpinBox()
        for s, v in ((self.plo, 0.1), (self.phi, 99.9)):
            s.setRange(0, 100)
            s.setDecimals(2)
            s.setValue(v)
        prow = QtWidgets.QHBoxLayout()
        prow.addWidget(self.plo)
        prow.addWidget(QtWidgets.QLabel("–"))
        prow.addWidget(self.phi)
        prow.addWidget(QtWidgets.QLabel("%"))
        self.auto_btn = QtWidgets.QPushButton("Auto (percentile)")
        self.auto_btn.clicked.connect(self.auto_levels)
        self.log_chk = QtWidgets.QCheckBox("log10(1 + I)")
        self.log_chk.toggled.connect(lambda _: (self.refresh(), self.auto_levels()))
        lay.addRow("Colormap", self.cmap_combo)
        lay.addRow("Min", self.vmin)
        lay.addRow("Max", self.vmax)
        lay.addRow("Percentiles", prow)
        lay.addRow(self.auto_btn)
        lay.addRow(self.lock_chk)
        lay.addRow(self.log_chk)
        form.addWidget(g)

        # binning
        g = QtWidgets.QGroupBox("Binning")
        lay = QtWidgets.QFormLayout(g)
        self.bin_mode = QtWidgets.QComboBox()
        self.bin_mode.addItems(["Sum", "Max"])
        self.bin_mode.setToolTip("Sum: add the b×b pixels of a bin (total signal).\n"
                                 "Max: brightest pixel of the bin (keeps single hits and "
                                 "hot pixels visible, values stay in pixel units). Key: B")
        self.bin_mode.currentTextChanged.connect(self._bin_mode_changed)
        self.bin_spin = QtWidgets.QSpinBox()
        self.bin_spin.setRange(1, 64)
        self.bin_spin.setPrefix("×")
        self.bin_spin.setSpecialValueText("×1 (native)")
        self.bin_spin.setToolTip("Bin b×b blocks. 1 = native detector resolution "
                                 "(the finest possible). Keys: + / -")
        self.bin_spin.valueChanged.connect(self._bin_changed)
        self._bin = 1
        self.bin_label = QtWidgets.QLabel()
        lay.addRow("Bin factor", self.bin_spin)
        lay.addRow("Mode", self.bin_mode)
        lay.addRow(self.bin_label)
        form.addWidget(g)

        form.addWidget(self._build_mask_group())

        # info
        g = QtWidgets.QGroupBox("Info")
        lay = QtWidgets.QVBoxLayout(g)
        self.info = QtWidgets.QLabel("–")
        self.info.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.info.setWordWrap(True)
        self.info.setAlignment(QtCore.Qt.AlignTop)
        self.info.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        lay.addWidget(self.info)
        form.addWidget(g)
        form.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        # minimum only, so the splitter handle can make the panel wider (long run names)
        panel_w = panel.minimumSizeHint().width() + scroll.verticalScrollBar().sizeHint().width() + 6
        scroll.setMinimumWidth(panel_w)
        split = QtWidgets.QSplitter()
        split.addWidget(self.tabs)
        split.addWidget(scroll)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        split.setCollapsible(1, False)
        split.setHandleWidth(6)
        split.setSizes([10000, panel_w])
        self.setCentralWidget(split)
        self.status = self.statusBar()

        item = self.imv.imageItem
        item.hoverEvent = self._hover
        item.mouseClickEvent = self._img_click
        item.mouseDragEvent = self._img_drag
        self._set_cmap(self.cmap_combo.currentText())

    def _build_mask_group(self):
        g = QtWidgets.QGroupBox("Mask")
        lay = QtWidgets.QFormLayout(g)
        self.edit_btn = QtWidgets.QPushButton("Paint mask (E)")
        self.edit_btn.setCheckable(True)
        self.edit_btn.setToolTip("Left click/drag: mask, right click or Shift: unmask.\n"
                                 "Paints whole displayed pixels, i.e. b×b blocks when "
                                 "binned: mask coarsely, then lower the bin factor to refine.")
        self.edit_btn.toggled.connect(self._edit_toggled)
        self.erase_chk = QtWidgets.QCheckBox("Left button erases")
        self.brush = QtWidgets.QSpinBox()
        self.brush.setRange(1, 99)
        self.brush.setSuffix(" displayed px")
        self.brush.setToolTip("Brush width in displayed (binned) pixels")
        self.apply_mask_chk = QtWidgets.QCheckBox("Apply mask")
        self.apply_mask_chk.setToolTip("Exclude masked pixels from the image, the total "
                                       "intensities (sorting) and the histograms")
        self.apply_mask_chk.setChecked(True)
        self.apply_mask_chk.toggled.connect(lambda _: self._mask_changed())
        self.overlay_chk = QtWidgets.QCheckBox("Show overlay")
        self.overlay_chk.setChecked(True)
        self.overlay_chk.toggled.connect(lambda _: self.refresh())
        self.overlay_style = QtWidgets.QComboBox()
        self.overlay_style.addItems(["Outline", "Filled"])
        self.overlay_style.setToolTip("Outline: red = hidden bins, dashed orange = partially "
                                      "masked bins")
        self.overlay_style.currentTextChanged.connect(lambda _: self.refresh())
        orow = QtWidgets.QHBoxLayout()
        orow.addWidget(self.overlay_chk)
        orow.addWidget(self.overlay_style)
        self.bin_thresh = QtWidgets.QSpinBox()
        self.bin_thresh.setRange(1, 100)
        self.bin_thresh.setValue(50)
        self.bin_thresh.setSuffix(" % masked")
        self.bin_thresh.setToolTip("When binned: a bin is hidden once this fraction of its "
                                   "native pixels is masked. 1 % ≈ any masked pixel hides it.")
        self.bin_thresh.valueChanged.connect(lambda _: self.refresh())
        self.rescale_chk = QtWidgets.QCheckBox("Rescale partial bins by b²/n_valid (sum)")
        self.rescale_chk.setChecked(True)
        self.rescale_chk.toggled.connect(lambda _: self.refresh())

        self.thr_spin = QtWidgets.QDoubleSpinBox()
        self.thr_spin.setRange(-1e9, 1e9)
        self.thr_spin.setValue(200)
        self.thr_btn = QtWidgets.QPushButton("Mask ≥")
        self.thr_btn.setToolTip("Mask native pixels of the current image (frame / mean / "
                                "median / std, dark subtracted if on) at or above this value, "
                                "e.g. hot pixels in the std or mean image")
        self.thr_btn.clicked.connect(self._mask_threshold)
        trow = QtWidgets.QHBoxLayout()
        trow.addWidget(self.thr_btn)
        trow.addWidget(self.thr_spin)

        row1 = QtWidgets.QHBoxLayout()
        row2 = QtWidgets.QHBoxLayout()
        for row, text, fn in ((row1, "Undo", self._undo), (row1, "Invert", self._invert_mask),
                              (row1, "Clear", self._clear_mask),
                              (row2, "Load…", self._load_mask), (row2, "Save…", self._save_mask)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        self.auto_btn_mask = QtWidgets.QPushButton("Auto mask…")
        self.auto_btn_mask.setToolTip("First-guess mask from the mean/std of the loaded "
                                      "(background) run: circular active area + hot regions")
        self.auto_btn_mask.clicked.connect(self._auto_mask)
        self._auto_dlg = None
        self.mask_label = QtWidgets.QLabel("no mask")
        self.mask_label.setWordWrap(True)
        self.mask_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored,
                                      QtWidgets.QSizePolicy.Preferred)
        lay.addRow(self.edit_btn)
        lay.addRow(self.auto_btn_mask)
        lay.addRow("Brush", self.brush)
        lay.addRow(self.erase_chk)
        lay.addRow(trow)
        lay.addRow(row1)
        lay.addRow(row2)
        lay.addRow(self.apply_mask_chk)
        lay.addRow(orow)
        lay.addRow("Hide bin at ≥", self.bin_thresh)
        lay.addRow(self.rescale_chk)
        lay.addRow(self.mask_label)
        return g

    def _build_hist_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        self.hist_plot = pg.PlotWidget()
        self.hist_plot.setLabel("left", "Counts")
        self.hist_plot.getAxis("left").enableAutoSIPrefix(False)
        self.hist_plot.setLabel("bottom", "Pixel value (as displayed: dark, mask, binning, log)")
        self.hist_plot.showGrid(x=True, y=True, alpha=0.3)
        self.hist_plot.addLegend(offset=(-10, 10))
        self.hist_curve = self.hist_plot.plot(stepMode="center", connect="finite",
                                              pen=pg.mkPen("w", width=2), name="current")
        self.hist_lines = [pg.InfiniteLine(angle=90, pen=pg.mkPen((255, 170, 0), width=1,
                                                                  style=QtCore.Qt.DashLine))
                           for _ in range(2)]
        for l in self.hist_lines:
            self.hist_plot.addItem(l)
        v.addWidget(self.hist_plot, 1)

        self.hist_src = QtWidgets.QComboBox()
        self.hist_src.addItems([HIST_CURRENT, HIST_ALL, HIST_SEL])
        self.hist_src.setToolTip("Current view updates live. The frame sets are computed on "
                                 "'Compute' (takes a few s per 1000 frames).")
        self.hist_src.currentTextChanged.connect(lambda _: self._update_hist())
        self.hist_bins = QtWidgets.QSpinBox()
        self.hist_bins.setRange(5, 10000)
        self.hist_bins.setValue(256)
        self.hist_bins.setKeyboardTracking(False)
        self.hist_bins.valueChanged.connect(lambda _: self._update_hist())
        self.hist_logx = QtWidgets.QCheckBox("log x")
        self.hist_logx.setToolTip("Log value axis with log-spaced bins; values ≤ 0 are left out")
        self.hist_logx.toggled.connect(self._hist_logx_changed)
        self.hist_log = QtWidgets.QCheckBox("log y")
        self.hist_log.setChecked(True)
        self.hist_log.toggled.connect(self._hist_log_changed)
        self.hist_norm = QtWidgets.QCheckBox("normalize")
        self.hist_norm.setToolTip("Divide by the number of values, to compare a single "
                                  "frame with whole runs")
        self.hist_norm.toggled.connect(lambda on: (
            self.hist_plot.setLabel("left", "Fraction of pixels" if on else "Counts"),
            self._hist_log_changed(self.hist_log.isChecked())))
        self.hist_auto = QtWidgets.QCheckBox("auto range")
        self.hist_auto.setChecked(True)
        self.hist_lo = QtWidgets.QDoubleSpinBox()
        self.hist_hi = QtWidgets.QDoubleSpinBox()
        for s, val in ((self.hist_lo, 0), (self.hist_hi, 255)):
            s.setRange(-1e9, 1e9)
            s.setValue(val)
            s.setKeyboardTracking(False)
            s.valueChanged.connect(lambda _: self._update_hist())
        self.hist_auto.toggled.connect(lambda _: self._update_hist())
        self.hist_compute = QtWidgets.QPushButton("Compute")
        self.hist_compute.clicked.connect(lambda: self._update_hist(force=True))
        # zoom/pan only along the ticked axes (mouse wheel, drag); Fit resets the view
        self.zoom_x = QtWidgets.QCheckBox("zoom x")
        self.zoom_y = QtWidgets.QCheckBox("zoom y")
        for c in (self.zoom_x, self.zoom_y):
            c.setChecked(True)
            c.setToolTip("Untick to lock that axis while zooming/panning. Tip: scrolling "
                         "over an axis also zooms only that axis.")
            c.toggled.connect(lambda _: self.hist_plot.setMouseEnabled(
                x=self.zoom_x.isChecked(), y=self.zoom_y.isChecked()))
        fit = QtWidgets.QPushButton("Fit")
        fit.setToolTip("Reset zoom to show all curves")
        fit.clicked.connect(lambda: self.hist_plot.enableAutoRange())
        pin = QtWidgets.QPushButton("Pin")
        pin.setToolTip("Keep the current curve for comparison (other frames, runs, views)")
        pin.clicked.connect(self._pin_hist)
        clear = QtWidgets.QPushButton("Clear pins")
        clear.clicked.connect(self._clear_pins)
        for sp in (self.hist_lo, self.hist_hi):
            sp.setMaximumWidth(110)
        rows = ((QtWidgets.QLabel("Source"), self.hist_src, self.hist_compute, pin, clear,
                 self.zoom_x, self.zoom_y, fit),
                (QtWidgets.QLabel("Bins"), self.hist_bins, self.hist_logx, self.hist_log, self.hist_norm,
                 self.hist_auto, self.hist_lo, QtWidgets.QLabel("–"), self.hist_hi))
        for widgets in rows:
            row = QtWidgets.QHBoxLayout()
            for wd in widgets:
                row.addWidget(wd)
            row.addStretch(1)
            v.addLayout(row)
        self.hist_info = QtWidgets.QLabel("–")
        self.hist_info.setWordWrap(True)
        self.hist_info.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        v.addWidget(self.hist_info)
        self.hist_plot.setLogMode(y=True)
        self._hist_data = None  # (edges, counts, label)
        return w

    def selected_runs(self):
        """Selected run names, in list order."""
        return [self.run_list.item(i).text() for i in range(self.run_list.count())
                if self.run_list.item(i).isSelected()]

    def select_runs(self, names):
        """Select exactly these runs in the list (e.g. from scripts)."""
        names = set(names)
        self.run_list.blockSignals(True)
        for i in range(self.run_list.count()):
            it = self.run_list.item(i)
            it.setSelected(it.text() in names)
        self.run_list.blockSignals(False)
        self._run_selection_changed()
        sel = self.run_list.selectedItems()
        if sel:
            self.run_list.scrollToItem(sel[0])

    def _filter_runs(self, text):
        words = text.lower().split()
        for i in range(self.run_list.count()):
            it = self.run_list.item(i)
            it.setHidden(not all(w in it.text().lower() for w in words))

    def _run_selection_changed(self):
        sel = self.selected_runs()
        n = sum(self._n_files.get(r, 0) for r in sel)
        self.run_sel_label.setText("no run selected" if not sel else
                                   f"{len(sel)} run(s) selected, {n} files")
        if sel:
            self._pick_dark(sel[0])

    def _fill_run_combos(self):
        names = sorted(self.runs)
        self._n_files = {n: len(list_frames(self.runs[n])) for n in names}
        keep = set(self.selected_runs())
        self.run_list.blockSignals(True)
        self.run_list.clear()
        for n in names:
            it = QtWidgets.QListWidgetItem(n)
            it.setToolTip(f"{self.runs[n]}  ({self._n_files[n]} files)")
            self.run_list.addItem(it)
            it.setSelected(n in keep)
        self.run_list.blockSignals(False)
        self._filter_runs(self.run_filter.text())
        for c in (self.dark_combo,):
            cur = c.currentText()
            c.blockSignals(True)
            c.clear()
            c.addItems(names)
            if cur in self.runs:
                c.setCurrentText(cur)
            c.blockSignals(False)
            # the drop-down list shows full names even when the panel is narrow
            fm = c.view().fontMetrics()
            longest = max((fm.horizontalAdvance(n) for n in names), default=0)
            c.view().setMinimumWidth(longest + 40)
            c.view().setTextElideMode(QtCore.Qt.ElideNone)
            c.completer().popup().setMinimumWidth(longest + 40)
        self._run_selection_changed()

    def _add_folder(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Add data folder")
        if not path:
            return
        new = find_runs(path)
        if not new:
            self.status.showMessage(f"No HDF5 files found under {path}")
            return
        self.runs.update(new)
        self._fill_run_combos()
        self.select_runs([sorted(new)[0]])
        self.status.showMessage(f"Added {len(new)} runs from {path}")

    def _pick_dark(self, run):
        """Default to a dark run from the same parent folder as the selected run."""
        darks = [k for k in self.runs if "dark" in os.path.basename(k).lower()]
        same = [k for k in darks if os.path.dirname(k) == os.path.dirname(run)]
        if same or darks:
            self.dark_combo.setCurrentText((same or darks)[0])

    def _install_shortcuts(self):
        keys = {
            "Left": lambda: self.step(-1), "Right": lambda: self.step(1),
            "PgUp": lambda: self.step(-10), "PgDown": lambda: self.step(10),
            "F": lambda: self.set_view(0), "M": lambda: self.set_view(1),
            "D": lambda: self.set_view(2), "S": lambda: self.set_view(3),
            "A": self.auto_levels, "L": self.log_chk.toggle,
            "+": lambda: self.bin_spin.stepBy(1), "=": lambda: self.bin_spin.stepBy(1),
            "-": lambda: self.bin_spin.stepBy(-1),
            "B": lambda: self.bin_mode.setCurrentIndex(1 - self.bin_mode.currentIndex()),
            "E": self.edit_btn.toggle, "Ctrl+Z": self._undo,
        }
        for k, fn in keys.items():
            sc = QtWidgets.QShortcut(QtGui.QKeySequence(k), self)
            sc.setContext(QtCore.Qt.ApplicationShortcut)
            sc.activated.connect(fn)

    # ---------------------------------------------------------------- loading
    def load_run(self):
        if self.loader is not None and self.loader.isRunning():
            return
        names = self.selected_runs()
        if not names:
            self.status.showMessage("Select one or more runs first")
            return
        self.load_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status.showMessage(f"Loading {len(names)} run(s) …")
        self.loader = Loader([self.runs[n] for n in names])
        self.loader.progress.connect(lambda i, n: (self.progress.setMaximum(n),
                                                   self.progress.setValue(i)))
        self.loader.done.connect(lambda fr, fi, me, ro: self._loaded(names, fr, fi, me, ro))
        self.loader.failed.connect(self._load_failed)
        self.loader.start()

    def _load_failed(self, msg):
        self.load_btn.setEnabled(True)
        self.progress.setVisible(False)
        QtWidgets.QMessageBox.critical(self, "Load failed", msg)

    def _loaded(self, names, frames, files, meta, run_of):
        self.load_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.frames, self.files, self.meta = frames, files, meta
        self.loaded_runs, self.run_of = names, run_of
        # short label for titles/histograms, e.g. "run_a" or "run_a + 2 more"
        name = names[0] if len(names) == 1 else f"{names[0]} + {len(names) - 1} more"
        self.run_name = name
        self.raw_totals = frames.reshape(len(frames), -1).sum(axis=1, dtype=np.float64)
        if self.mask is None or self.mask.shape != frames.shape[1:]:
            if self.mask is not None:
                QtWidgets.QMessageBox.warning(
                    self, "Mask reset", f"Mask shape {self.mask.shape} does not match "
                    f"frames {frames.shape[1:]}; starting with an empty mask.")
            self.mask = np.zeros(frames.shape[1:], dtype=bool)
            self.undo = []
        self._update_mask_label()
        if len(names) == 1 and "dark" in name.lower():  # loaded dark run: mean for free
            self.dark_cache[name] = frames.mean(axis=0, dtype=np.float32)
        self.setWindowTitle(f"PXI-viewer — {name}")
        if self.cen_x.value() == 0 and self.cen_y.value() == 0:  # default: image centre
            self._set_center(frames.shape[2] / 2, frames.shape[1] / 2, user=False)
        self.hits = None  # hits belong to the previous frames
        self.hpf_curve.setData([], [])
        self._update_hit_map()
        self.hits_info.setText("No hits extracted yet.")
        self._dark_changed(first=True)
        self._center_changed()
        if self._auto_dlg is not None and self._auto_dlg.isVisible():
            self._auto_dlg.start()  # new run: redo the auto-mask statistics

    def _dark(self):
        if not self.dark_chk.isChecked():
            return None
        name = self.dark_combo.currentText()
        if name not in self.runs:
            return None
        if name not in self.dark_cache:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            self.status.showMessage(f"Computing mean dark from {name} …")
            QtWidgets.QApplication.processEvents()
            try:
                files = list_frames(self.runs[name])
                acc, key = None, FRAME_KEY
                for f in files:
                    with h5py.File(f, "r") as h5:
                        key = frame_dataset_name(h5, key)
                        fr = h5[key][()].astype(np.float64)
                    acc = fr if acc is None else acc + fr
                self.dark_cache[name] = (acc / len(files)).astype(np.float32)
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
        return self.dark_cache[name]

    def _active_mask(self):
        if self.mask is None or not self.apply_mask_chk.isChecked() or not self.mask.any():
            return None
        return self.mask

    def _update_totals(self, keep_frame=True):
        """Total intensity per frame over unmasked pixels, minus the dark there."""
        flat = self.frames.reshape(len(self.frames), -1)
        mask = self._active_mask()
        dark = self._dark()
        if mask is None:
            self.totals = self.raw_totals.copy()
            dark_sum = dark.sum(dtype=np.float64) if dark is not None else 0.0
        else:
            m = mask.ravel()
            if m.sum() < m.size // 2:  # cheaper to subtract the few masked pixels
                self.totals = self.raw_totals - flat[:, np.flatnonzero(m)].sum(
                    axis=1, dtype=np.float64)
            else:
                self.totals = flat[:, np.flatnonzero(~m)].sum(axis=1, dtype=np.float64)
            dark_sum = dark[~mask].sum(dtype=np.float64) if dark is not None else 0.0
        self.totals -= dark_sum
        self._resort(keep_frame=keep_frame)

    def _dark_changed(self, *_, first=False):
        if self.frames is None:
            return
        dark = self._dark()
        self.stat_cache.clear()
        self.status.showMessage(f"{self.run_name}: {len(self.frames)} patterns" +
                                (f", dark subtracted ({self.dark_combo.currentText()})"
                                 if dark is not None else ""))
        self._update_totals(keep_frame=not first)
        if first or not self.lock_chk.isChecked():
            self.auto_levels()

    # ---------------------------------------------------------------- ordering
    def _resort(self, *_, keep_frame=True):
        if self.totals is None:
            return
        current = self.current_index() if keep_frame and self.order is not None else None
        mode = self.sort_combo.currentText()
        if mode == SORT_DESC:
            self.order = np.argsort(-self.totals, kind="stable")
        elif mode == SORT_ASC:
            self.order = np.argsort(self.totals, kind="stable")
        else:
            self.order = np.arange(len(self.totals))
        self.stat_cache.clear()
        n = len(self.order)
        self.tot_curve.setData(np.arange(n), self.totals[self.order])
        self.tot_plot.setLabel("bottom", f"Position ({mode.lower()})")
        # several runs loaded in file order: dashed lines where the next run starts
        for l in self.run_lines:
            self.tot_plot.removeItem(l)
        self.run_lines = []
        if mode == SORT_FILE and len(self.loaded_runs) > 1:
            for x in np.flatnonzero(np.diff(self.run_of)) + 0.5:
                l = pg.InfiniteLine(x, angle=90, pen=pg.mkPen((180, 180, 180), width=1,
                                                              style=QtCore.Qt.DashLine))
                self.tot_plot.addItem(l)
                self.run_lines.append(l)
        for w in (self.pos_slider, self.pos_spin):
            w.blockSignals(True)
            w.setRange(0, n - 1)
            w.blockSignals(False)
        pos = int(np.where(self.order == current)[0][0]) if current is not None else 0
        self.pos_slider.setValue(pos)
        self.pos_spin.blockSignals(True)
        self.pos_spin.setValue(pos)
        self.pos_spin.blockSignals(False)
        self.refresh()

    def current_index(self):
        return int(self.order[self.pos_spin.value()])

    def step(self, s):
        if self.order is not None:
            self.pos_spin.setValue(int(np.clip(self.pos_spin.value() + s, 0,
                                               len(self.order) - 1)))

    def _tot_clicked(self, ev):
        if self.order is None or ev.button() != QtCore.Qt.LeftButton:
            return
        vb = self.tot_plot.getPlotItem().vb
        if vb.sceneBoundingRect().contains(ev.scenePos()):
            x = vb.mapSceneToView(ev.scenePos()).x()
            self.pos_spin.setValue(int(np.clip(round(x), 0, len(self.order) - 1)))

    def _show_frame_at(self, _):
        if self.view_group.checkedId() != 0:
            self.set_view(0)
        else:
            self.refresh()

    # ---------------------------------------------------------------- images
    def set_view(self, i):
        self.view_group.button(i).setChecked(True)
        self.refresh()

    def _selection(self):
        """Frame indices used for mean/median/std: first N of the current order."""
        n = self.stat_n.value() or len(self.order)
        return np.sort(self.order[:n]), n

    def _stat_image(self, view, all_frames=False):
        """Mean/median/std of the stats selection, or of every loaded frame."""
        if all_frames:
            idx, n = np.arange(len(self.frames)), len(self.frames)
        else:
            idx, n = self._selection()
        key = (view, n)  # n == all frames gives the same image whatever the order
        if key not in self.stat_cache:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            self.status.showMessage(f"Computing {view.lower()} of {len(idx)} patterns …")
            QtWidgets.QApplication.processEvents()
            try:
                stack = self.frames if len(idx) == len(self.frames) else self.frames[idx]
                if view == VIEW_MEAN:
                    img = stack.mean(axis=0, dtype=np.float64)
                else:
                    # std/median in row blocks to keep the float temporaries small
                    # (runs can have >10k frames)
                    fn = (lambda a: a.std(axis=0, dtype=np.float64)) if view == VIEW_STD \
                        else (lambda a: np.median(a, axis=0))
                    img = np.empty(stack.shape[1:], dtype=np.float64)
                    for r in range(0, stack.shape[1], 16):
                        img[r:r + 16] = fn(stack[:, r:r + 16])
                dark = self._dark()
                if dark is not None and view != VIEW_STD:
                    img = img - dark
                self.stat_cache[key] = img.astype(np.float32)
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
            self.status.showMessage(f"{view} of {len(idx)} patterns "
                                    f"({self.sort_combo.currentText().lower()})")
        return self.stat_cache[key], n

    def current_image(self):
        """Current frame or statistic at native resolution (dark subtracted if on)."""
        view = self.view_group.checkedButton().text()
        if view == VIEW_FRAME:
            i = self.current_index()
            img = self.frames[i].astype(np.float32)
            dark = self._dark()
            if dark is not None:
                img -= dark
            return img, view, None
        img, n = self._stat_image(view)
        return img, view, n

    def _bin_changed(self, b):
        old, self._bin = self._bin, b
        self.refresh()
        if self.frames is None:
            return
        if self.log_chk.isChecked():
            self.auto_levels()
        elif self._bin_mode() == "sum":  # values scale by b^2: keep relative limits
            f = (b / old) ** 2
            self._set_levels(self.vmin.value() * f, self.vmax.value() * f)
        self._set_hist_range()

    def _bin_mode(self):
        return self.bin_mode.currentText().lower()

    def _bin_mode_changed(self, _):
        self.refresh()
        if self.frames is not None:
            self.auto_levels()  # sum and max values differ by up to b^2
            self._set_hist_range()

    def _display(self, img, use_mask=True, use_log=True):
        """Native image or stack (..., H, W) -> displayed values (mask, binning, log)."""
        img = masked_rebin(img, self._active_mask() if use_mask else None, self._bin,
                           self.bin_thresh.value() / 100.0, self.rescale_chk.isChecked(),
                           self._bin_mode())
        if use_log and self.log_chk.isChecked():
            img = np.log10(1 + np.clip(img, 0, None))
        return img

    def _values_are_integer(self):
        """True when displayed values are sums/maxima of raw counts (histogram edges)."""
        if self._dark() is not None or self.log_chk.isChecked():
            return False
        mask = self._active_mask()
        return not (mask is not None and self.rescale_chk.isChecked() and self._bin > 1
                    and self._bin_mode() == "sum")

    def _set_hist_range(self):
        finite = self.shown[np.isfinite(self.shown)]
        if finite.size:
            self.imv.getHistogramWidget().item.setHistogramRange(
                float(finite.min()), float(finite.max()))

    def refresh(self):
        if self.frames is None:
            return
        img, view, n = self.current_image()
        self.native = img
        self.shown = self._display(img)
        # outline mode: draw the real values under the mask, only the outline marks it
        outline = self.overlay_chk.isChecked() and self.overlay_style.currentText() == "Outline"
        self.rendered = (self._display(img, use_mask=False)
                         if outline and self._active_mask() is not None else self.shown)
        levels = (self.vmin.value(), self.vmax.value())
        b = self._bin
        # scale keeps the axes in native-pixel units whatever the binning
        self.imv.setImage(self.rendered, autoLevels=False, autoRange=False,
                          autoHistogramRange=False, levels=levels, scale=(b, b))
        self.bin_label.setText(f"{img.shape[1]}×{img.shape[0]} → "
                               f"{self.shown.shape[1]}×{self.shown.shape[0]} px")
        if not self.lock_chk.isChecked() and view == VIEW_FRAME:
            self.auto_levels()
        self._set_hist_range()
        self._update_overlay()
        self._update_hit_markers(view)

        pos = self.pos_spin.value()
        self.tot_marker.setValue(pos)
        shown_max = np.nanmax(self.shown) if np.isfinite(self.shown).any() else np.nan
        if view == VIEW_FRAME:
            i = self.current_index()
            m = self.meta[i]
            text = (f"<b>{os.path.basename(self.files[i])}</b><br>"
                    f"frame #{i}<br>"
                    f"position {pos + 1}/{len(self.order)}<br>"
                    f"total intensity: {self.totals[i]:.4g}<br>"
                    f"rank by intensity: {int((self.totals > self.totals[i]).sum()) + 1}<br>"
                    f"max displayed pixel: {shown_max:.1f}<br>")
            text += "<br>".join(f"{k}: {v}" for k, v in m.items())
        else:
            text = (f"<b>{view} pattern</b> of {n} / {len(self.order)} patterns<br>"
                    f"({'all' if n == len(self.order) else 'first ' + str(n) + ', ' + self.sort_combo.currentText().lower()})<br>"
                    f"sum (displayed): {np.nansum(self.shown):.4g}<br>"
                    f"max displayed pixel: {shown_max:.2f}")
        if view == VIEW_FRAME:  # the run this frame comes from
            run = self.loaded_runs[self.run_of[self.current_index()]]
            head = f"run: {run}"
            if len(self.loaded_runs) > 1:
                head += f"  ({self.run_of[self.current_index()] + 1} of {len(self.loaded_runs)} loaded)"
        else:
            head = "runs: " + ", ".join(self.loaded_runs)
        # zero-width spaces let long run/file names wrap inside the narrow panel
        text = f"{head}<br>{text}".replace("/", "/\u200b").replace("_", "_\u200b")
        self.info.setText(text)
        self._update_hits_info()
        if self.tabs.currentIndex() == 1 and self.hist_src.currentText() == HIST_CURRENT:
            self._update_hist()

    # ---------------------------------------------------------------- mask
    def _update_overlay(self):
        items = (self.overlay, self.outline_dead, self.outline_part)
        if self.mask is None or not self.overlay_chk.isChecked() or not self.mask.any():
            for it in items:
                it.setVisible(False)
            return
        b = self._bin
        frac = masked_fraction(self.mask, b)
        dead = frac >= self.bin_thresh.value() / 100.0
        if self.overlay_style.currentText() == "Outline":
            self.overlay.setVisible(False)
            for it, region in ((self.outline_dead, dead), (self.outline_part, (frac > 0) & ~dead)):
                it.setData(*outline_segments(region, b))
                it.setVisible(bool(region.any()))
            return
        self.outline_dead.setVisible(False)
        self.outline_part.setVisible(False)
        rgba = np.zeros(frac.shape + (4,), dtype=np.uint8)
        rgba[dead] = (255, 40, 40, 170)                  # hidden bins
        rgba[(frac > 0) & ~dead] = (255, 170, 0, 90)     # partially masked bins
        self.overlay.setImage(rgba, autoLevels=False)
        self.overlay.setRect(QtCore.QRectF(0, 0, frac.shape[1] * b, frac.shape[0] * b))
        self.overlay.setVisible(True)

    def _update_mask_label(self):
        if self.mask is None:
            self.mask_label.setText("no mask")
            return
        n = int(self.mask.sum())
        self.mask_label.setText(f"{n} of {self.mask.size} native pixels masked "
                                f"({100.0 * n / self.mask.size:.2f} %)")

    def _mask_changed(self, final=True):
        """Mask edited: redraw; recompute totals (and sort) once an edit is finished."""
        self._mask_version += 1  # lets extracted hits notice that the mask changed
        self._update_mask_label()
        if self.frames is None:
            return
        if final:
            self._update_totals(keep_frame=True)
        else:
            self.refresh()

    def _push_undo(self):
        if self.mask is not None:
            self.undo.append(self.mask.copy())
            del self.undo[:-50]

    def _undo(self):
        if self.undo:
            self.mask = self.undo.pop()
            self._mask_changed()

    def _clear_mask(self):
        if self.mask is not None and self.mask.any():
            self._push_undo()
            self.mask[:] = False
            self._mask_changed()

    def _invert_mask(self):
        if self.mask is not None:
            self._push_undo()
            self.mask = ~self.mask
            self._mask_changed()

    def _mask_threshold(self):
        if self.frames is None:
            return
        self._push_undo()
        img, view, _ = self.current_image()
        hit = img >= self.thr_spin.value()
        self.mask |= hit
        self._mask_changed()
        self.status.showMessage(f"Masked {int(hit.sum())} pixels ≥ {self.thr_spin.value()} "
                                f"in the {view.lower()} image")

    def _auto_mask(self):
        if self._auto_dlg is None:
            self._auto_dlg = AutoMaskDialog(self)
            install_wheel_guard(self._auto_dlg, self._wheel_guard)
        if self._auto_dlg.start():
            self._auto_dlg.show()
            self._auto_dlg.raise_()

    def _save_mask(self):
        if self.mask is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save mask", "mask.h5", "HDF5 (*.h5);;NumPy (*.npy)")
        if not path:
            return
        if not path.endswith((".h5", ".npy")):
            path += ".h5"
        save_mask(path, self.mask)
        self.status.showMessage(f"Saved mask ({int(self.mask.sum())} px) to {path}")

    def _load_mask(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load mask", "", "Masks (*.h5 *.hdf5 *.npy);;All files (*)")
        if not path:
            return
        try:
            m = load_mask(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load mask failed", repr(e))
            return
        if self.frames is not None and m.shape != self.frames.shape[1:]:
            QtWidgets.QMessageBox.critical(
                self, "Load mask failed",
                f"Mask shape {m.shape} ≠ frame shape {self.frames.shape[1:]}")
            return
        self._push_undo()
        self.mask = m
        self._mask_changed()
        self.status.showMessage(f"Loaded mask from {path} ({MASK_CONVENTION.split(';')[0]}; "
                                "use Invert if the file uses 1 = good)")

    def _edit_toggled(self, on):
        # while painting, left-drag paints instead of panning the image
        self.imv.view.getViewBox().setMouseEnabled(x=not on, y=not on)
        self.status.showMessage("Mask painting ON: left = mask, right/Shift = unmask"
                                if on else "Mask painting off")

    def _erasing(self, ev):
        right = ev.button() == QtCore.Qt.RightButton
        shift = bool(ev.modifiers() & QtCore.Qt.ShiftModifier)
        return self.erase_chk.isChecked() != (right or shift)

    def _paint(self, pos, erase, last=None):
        """Set/clear brush×brush blocks of displayed pixels (b×b native each).

        With `last` (previous drag position) the whole segment is painted, so fast
        strokes leave no gaps.
        """
        b, k = self._bin, self.brush.value()
        h, w = self.mask.shape
        x1, y1 = pos.x(), pos.y()
        x0, y0 = (x1, y1) if last is None else (last.x(), last.y())
        n = int(np.ceil(2 * max(abs(x1 - x0), abs(y1 - y0)))) + 1
        cells = {(int(np.floor(x)), int(np.floor(y)))
                 for x, y in zip(np.linspace(x0, x1, n), np.linspace(y0, y1, n))}
        for cx, cy in cells:
            xa, ya = (cx - k // 2) * b, (cy - k // 2) * b
            self.mask[max(ya, 0):min(ya + k * b, h), max(xa, 0):min(xa + k * b, w)] = not erase

    def _img_click(self, ev):
        if self.pick_center_btn.isChecked() and ev.button() == QtCore.Qt.LeftButton:
            ev.accept()  # set the hit-position centre (native px) at the click
            b = self._bin
            self._set_center(ev.pos().x() * b, ev.pos().y() * b)
            self.pick_center_btn.setChecked(False)
            return
        if not self.edit_btn.isChecked() or self.mask is None or \
                ev.button() not in (QtCore.Qt.LeftButton, QtCore.Qt.RightButton):
            ev.ignore()
            return
        ev.accept()
        self._push_undo()
        self._paint(ev.pos(), self._erasing(ev))
        self._mask_changed(final=True)

    def _img_drag(self, ev):
        if not self.edit_btn.isChecked() or self.mask is None or \
                ev.button() not in (QtCore.Qt.LeftButton, QtCore.Qt.RightButton):
            ev.ignore()
            return
        ev.accept()
        if ev.isStart():
            self._push_undo()
            self._last_paint = None
        self._paint(ev.pos(), self._erasing(ev), self._last_paint)
        self._last_paint = ev.pos()
        self._mask_changed(final=ev.isFinish())

    # ---------------------------------------------------------------- histograms
    def _hist_edges(self, lo, hi, integer, pos_min=None):
        """Bin edges and their kind: 'lin' (uniform), 'log' (uniform in log10) or
        'irregular'. lo/hi: data range; pos_min: smallest positive value (log x)."""
        n = self.hist_bins.value()
        if not self.hist_auto.isChecked():
            lo, hi = self.hist_lo.value(), self.hist_hi.value()
        if self.hist_logx.isChecked():
            if lo <= 0:  # log axis: start at the smallest positive value
                lo = pos_min if pos_min is not None and pos_min > 0 else 1e-3
            if hi <= lo:
                hi = lo * 10
            if integer:  # log-spaced, but every edge half-way between whole counts
                k = np.unique(np.round(np.geomspace(max(lo, 1), hi + 1, n + 1)))
                return k - 0.5, "irregular"
            return np.geomspace(lo, hi, n + 1), "log"
        if hi <= lo:
            hi = lo + 1
        if integer:  # centre bins on integers, width a whole number of counts
            lo, hi = np.floor(lo), np.ceil(hi)
            step = max(1, int(np.ceil((hi - lo + 1) / n)))
            n = int(np.ceil((hi - lo + 1) / step))
            return lo - 0.5 + step * np.arange(n + 1), "lin"
        return np.linspace(lo, hi, n + 1), "lin"

    @staticmethod
    def _histogram(v, edges, kind, weights=None):
        """np.histogram using the fast uniform path where possible."""
        if kind == "log":
            pos = v > 0
            return np.histogram(np.log10(v[pos]), bins=len(edges) - 1,
                                range=(np.log10(edges[0]), np.log10(edges[-1])),
                                weights=None if weights is None else weights[pos])[0]
        if kind == "lin":
            return np.histogram(v, bins=len(edges) - 1, range=(edges[0], edges[-1]),
                                weights=weights)[0]
        return np.histogram(v, bins=edges, weights=weights)[0]

    def _update_hist(self, force=False):
        if self.tabs.currentIndex() != 1 or self.frames is None or self.shown is None:
            return
        src = self.hist_src.currentText()
        self.hist_compute.setEnabled(src != HIST_CURRENT)
        integer = self._values_are_integer()
        if src == HIST_CURRENT:
            v = self.shown[np.isfinite(self.shown)]
            if not v.size:
                return
            pos = v[v > 0]
            edges, kind = self._hist_edges(v.min(), v.max(), integer,
                                           pos.min() if pos.size else None)
            counts = self._histogram(v, edges, kind)
            view = self.view_group.checkedButton().text()
            what = (f"frame #{self.current_index()}" if view == VIEW_FRAME else view.lower())
            stats = dict(n=v.size, mean=v.mean(), median=np.median(v), std=v.std(),
                         min=v.min(), max=v.max())
            if self.hist_logx.isChecked():
                stats["dropped ≤0 (log x)"] = v.size - pos.size
        else:
            if not force:
                self.hist_info.setText("Press 'Compute' to histogram the frame set.")
                return
            idx = np.arange(len(self.frames)) if src == HIST_ALL else self._selection()[0]
            edges, counts, stats = self._hist_frames(idx, integer)
            what = f"{len(idx)} frames"
        label = f"{os.path.basename(self.run_name)} · {what} · ×{self._bin}" + \
            (f" {self._bin_mode()}" if self._bin > 1 else "")
        self._hist_data = (edges, counts, label)
        self._plot_hist(self.hist_curve, edges, counts)
        self.hist_curve.opts["name"] = label
        lg = self.hist_plot.getPlotItem().legend
        lg.removeItem(self.hist_curve)
        lg.addItem(self.hist_curve, "current: " + label)
        self._place_hist_lines(self.vmin.value(), self.vmax.value())
        self.hist_info.setText(
            f"{label}   |   " + "   ".join(f"{k}: {v:.4g}" for k, v in stats.items()) +
            "   |   dashed lines: colour limits")

    def _hist_frames(self, idx, integer):
        """Histogram of the displayed values of many frames, in chunks."""
        dark = self._dark()
        chunk = 128

        def chunks():
            for c in range(0, len(idx), chunk):
                st = self.frames[idx[c:c + chunk]].astype(np.float32)
                if dark is not None:
                    st -= dark
                st = self._display(st)
                yield c, st[np.isfinite(st)]

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            lo, hi, s1, s2, nval, pmin, n_le0 = np.inf, -np.inf, 0.0, 0.0, 0, np.inf, 0
            need_range = self.hist_auto.isChecked()
            if integer:
                # one pass: exact integer counts, rebinned afterwards
                total = np.zeros(1, dtype=np.int64)
                for c, v in chunks():
                    self._hist_progress(c, len(idx))
                    bc = np.bincount(v.astype(np.int64))
                    if len(bc) > len(total):
                        total = np.pad(total, (0, len(bc) - len(total)))
                    total[:len(bc)] += bc
                vals = np.arange(len(total))
                nz = np.flatnonzero(total)
                lo, hi = (nz[0], nz[-1]) if nz.size else (0, 1)
                nzp = nz[nz > 0]
                edges, kind = self._hist_edges(lo, hi, True, nzp[0] if nzp.size else None)
                counts = self._histogram(vals, edges, kind, weights=total)
                n_le0 = int(total[0])
                nval = total.sum()
                mean = (vals * total).sum() / max(nval, 1)
                std = np.sqrt(max((vals ** 2 * total).sum() / max(nval, 1) - mean ** 2, 0))
                cdf = np.cumsum(total)
                median = float(np.searchsorted(cdf, nval / 2.0))
            else:
                if need_range:
                    for c, v in chunks():
                        self._hist_progress(c, 2 * len(idx))
                        if v.size:
                            lo, hi = min(lo, v.min()), max(hi, v.max())
                            p = v[v > 0]
                            if p.size:
                                pmin = min(pmin, p.min())
                edges, kind = self._hist_edges(lo if np.isfinite(lo) else 0,
                                               hi if np.isfinite(hi) else 1, False,
                                               pmin if np.isfinite(pmin) else None)
                counts = np.zeros(len(edges) - 1, dtype=np.int64)
                for c, v in chunks():
                    self._hist_progress(c + (len(idx) if need_range else 0),
                                        (2 if need_range else 1) * len(idx))
                    counts += self._histogram(v, edges, kind)
                    n_le0 += int((v <= 0).sum())
                    s1 += v.sum(dtype=np.float64)
                    s2 += (v.astype(np.float64) ** 2).sum()
                    nval += v.size
                    if not need_range and v.size:
                        lo, hi = min(lo, v.min()), max(hi, v.max())
                mean = s1 / max(nval, 1)
                std = np.sqrt(max(s2 / max(nval, 1) - mean ** 2, 0))
                median = np.nan
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.progress.setVisible(False)
        stats = dict(n=nval, mean=mean, std=std, min=lo, max=hi)
        if np.isfinite(median):
            stats["median"] = median
        if self.hist_logx.isChecked():
            stats["dropped ≤0 (log x)"] = n_le0
        return edges, counts, stats

    def _hist_progress(self, i, n):
        self.progress.setVisible(True)
        self.progress.setMaximum(n)
        self.progress.setValue(i)
        QtWidgets.QApplication.processEvents()

    def _plot_hist(self, curve, edges, counts):
        y = counts.astype(np.float64)
        if self.hist_norm.isChecked() and counts.sum():
            y /= counts.sum()
        if self.hist_log.isChecked():
            y[y <= 0] = np.nan  # log axis: leave empty bins out
        curve.edges_counts = (edges, counts)
        if self.hist_logx.isChecked():  # e.g. pinned linear curves: drop bins at x ≤ 0
            k = int(np.searchsorted(edges, 0, side="right"))
            edges, y = edges[k:], y[k:]
            if len(edges) < 2:
                curve.setData([], [])
                return
        curve.setData(edges, y)

    def _place_hist_lines(self, lo, hi):
        """Colour-limit markers; in log-x mode positions are in log10 view units."""
        logx = self.hist_logx.isChecked()
        for l, val in zip(self.hist_lines, (lo, hi)):
            ok = val > 0 or not logx
            l.setVisible(ok)
            if ok:
                l.setValue(np.log10(val) if logx else val)

    def _hist_logx_changed(self, on):
        self.hist_plot.setLogMode(x=on, y=self.hist_log.isChecked())
        self._place_hist_lines(self.vmin.value(), self.vmax.value())
        for c in self.pins:  # pinned curves keep their bins; just re-plot
            self._plot_hist(c, *c.edges_counts)
        self._update_hist(force=self.hist_src.currentText() == HIST_CURRENT)
        self.hist_plot.enableAutoRange()

    def _hist_log_changed(self, on):
        self.hist_plot.setLogMode(x=self.hist_logx.isChecked(), y=on)
        for c in [self.hist_curve] + self.pins:
            if getattr(c, "edges_counts", None) is not None:
                self._plot_hist(c, *c.edges_counts)

    def _pin_hist(self):
        if self._hist_data is None:
            return
        edges, counts, label = self._hist_data
        color = pg.intColor(len(self.pins), hues=8, sat=200)
        c = self.hist_plot.plot(stepMode="center", connect="finite",
                                pen=pg.mkPen(color, width=1.5), name=label)
        self._plot_hist(c, edges, counts)
        self.pins.append(c)

    def _clear_pins(self):
        for c in self.pins:
            self.hist_plot.removeItem(c)
        self.pins = []

    # ---------------------------------------------------------------- hits
    def _build_hits_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        plots = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        # 2D map of all hit positions relative to the centre (mm)
        self.hit_map_plot = pg.PlotWidget()
        self.hit_map_plot.setAspectLocked(True)
        self.hit_map_plot.invertY(True)  # same orientation as the image
        self.hit_map_plot.setLabel("bottom", "x − centre (mm)")
        self.hit_map_plot.setLabel("left", "y − centre (mm)")
        self.hit_map_img = pg.ImageItem()
        self.hit_map_img.setColorMap(pg.colormap.get("magma"))
        self.hit_map_plot.addItem(self.hit_map_img)
        self.hit_map_circle = pg.PlotCurveItem(pen=pg.mkPen((80, 255, 80), width=1.5,
                                                            style=QtCore.Qt.DashLine))
        self.hit_map_plot.addItem(self.hit_map_circle)
        plots.addWidget(self.hit_map_plot)
        self.hits_per_frame = pg.PlotWidget()
        self.hits_per_frame.setLabel("left", "Hits per frame")
        self.hits_per_frame.setLabel("bottom", "Frame (file order)")
        self.hits_per_frame.showGrid(x=True, y=True, alpha=0.3)
        self.hpf_curve = self.hits_per_frame.plot(pen=pg.mkPen((100, 160, 255), width=1))
        plots.addWidget(self.hits_per_frame)
        v.addWidget(plots, 1)

        def dspin(lo, hi, val, dec=2, suffix="", tip=""):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setDecimals(dec)
            s.setValue(val)
            s.setSuffix(suffix)
            s.setKeyboardTracking(False)
            s.setToolTip(tip)
            return s

        grid = QtWidgets.QGridLayout()
        # detection
        self.hit_thr = dspin(-1e9, 1e9, 1, 2, "", "A hit is a connected group of unmasked "
                             "(binned) pixels ≥ this value, as displayed (dark subtracted, "
                             "binned; log scale ignored)")
        self.hit_thr.valueChanged.connect(lambda _: self.refresh())
        self.hit_min = QtWidgets.QSpinBox()
        self.hit_min.setRange(1, 10000)
        self.hit_min.setSuffix(" px")
        self.hit_min.setToolTip("Minimum hit size in (binned) pixels")
        self.hit_min.valueChanged.connect(lambda _: self.refresh())
        self.hit_show = QtWidgets.QCheckBox("show hits on image")
        self.hit_show.setToolTip("Circle the hits of the current frame on the image "
                                 "(off by default)")
        self.hit_show.toggled.connect(lambda _: self.refresh())
        find = QtWidgets.QPushButton("Find hits in all frames")
        find.clicked.connect(self.find_all_hits)
        export = QtWidgets.QPushButton("Export hits…")
        export.clicked.connect(self._export_hits)
        grid.addWidget(QtWidgets.QLabel("<b>Detection</b>"), 0, 0)
        grid.addWidget(QtWidgets.QLabel("threshold ≥"), 0, 1)
        grid.addWidget(self.hit_thr, 0, 2)
        grid.addWidget(QtWidgets.QLabel("min size"), 0, 3)
        grid.addWidget(self.hit_min, 0, 4)
        grid.addWidget(self.hit_show, 0, 5)
        grid.addWidget(find, 0, 6)
        grid.addWidget(export, 0, 7)
        # centre
        self.cen_x = dspin(-1e5, 1e5, 0, 2, " px", "Centre x in native pixels")
        self.cen_y = dspin(-1e5, 1e5, 0, 2, " px", "Centre y in native pixels")
        self._center_set = False  # crosshair only once the centre was picked/auto/typed
        for s in (self.cen_x, self.cen_y):
            s.valueChanged.connect(lambda _: self._center_changed(user=True))
        self.pick_center_btn = QtWidgets.QPushButton("Pick on image")
        self.pick_center_btn.setCheckable(True)
        self.pick_center_btn.setToolTip("Then click on the image (Image tab) to set the centre")
        self.pick_center_btn.toggled.connect(self._pick_center_toggled)
        self.cen_radius = QtWidgets.QSpinBox()
        self.cen_radius.setRange(1, 500)
        self.cen_radius.setValue(10)
        self.cen_radius.setSuffix(" px")
        self.cen_radius.setToolTip("Smoothing / centroid radius for the automatic centre")
        auto = QtWidgets.QPushButton("Auto centre")
        auto.setToolTip("Brightest region of the (masked) mean of all loaded frames "
                        "(every selected run, independent of 'Stats over first N')")
        auto.clicked.connect(self._auto_center)
        self.cen_show = QtWidgets.QCheckBox("show centre")
        self.cen_show.setChecked(True)
        self.cen_show.toggled.connect(lambda _: self._center_changed())
        grid.addWidget(QtWidgets.QLabel("<b>Centre</b>"), 1, 0)
        grid.addWidget(QtWidgets.QLabel("x"), 1, 1)
        grid.addWidget(self.cen_x, 1, 2)
        grid.addWidget(QtWidgets.QLabel("y"), 1, 3)
        grid.addWidget(self.cen_y, 1, 4)
        grid.addWidget(self.pick_center_btn, 1, 5)
        grid.addWidget(auto, 1, 6)
        cr = QtWidgets.QHBoxLayout()
        cr.addWidget(QtWidgets.QLabel("radius"))
        cr.addWidget(self.cen_radius)
        cr.addWidget(self.cen_show)
        grid.addLayout(cr, 1, 7)
        # scale
        self.det_mm = dspin(0.01, 1e4, 40, 2, " mm", "Physical diameter of the detector "
                            "(active circle)")
        self.det_px = dspin(0, 1e5, 0, 1, " px", "Diameter of the active circle in native "
                            "pixels (filled in by the circle fit)")
        for s in (self.det_mm, self.det_px):
            s.valueChanged.connect(lambda _: self._scale_changed())
        fitc = QtWidgets.QPushButton("Fit circle")
        fitc.setToolTip("Fit the active circle on the std of the loaded frames")
        fitc.clicked.connect(self._fit_detector_circle)
        self.scale_label = QtWidgets.QLabel()
        grid.addWidget(QtWidgets.QLabel("<b>Scale</b>"), 2, 0)
        grid.addWidget(QtWidgets.QLabel("detector ⌀"), 2, 1)
        grid.addWidget(self.det_mm, 2, 2)
        grid.addWidget(QtWidgets.QLabel("= ⌀"), 2, 3)
        grid.addWidget(self.det_px, 2, 4)
        grid.addWidget(fitc, 2, 5)
        grid.addWidget(self.scale_label, 2, 6, 1, 2)
        v.addLayout(grid)
        self.hits_info = QtWidgets.QLabel("No hits extracted yet.")
        self.hits_info.setWordWrap(True)
        self.hits_info.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        v.addWidget(self.hits_info)

        # markers on the image: hits of the current frame, centre, detector circle
        self.hit_markers = pg.ScatterPlotItem(symbol="o", size=14, brush=None,
                                              pen=pg.mkPen((0, 255, 120), width=1.5))
        self.center_marker = pg.ScatterPlotItem(symbol="+", size=22, brush=pg.mkBrush(0, 200, 255),
                                                pen=pg.mkPen((0, 200, 255), width=2))
        for it in (self.hit_markers, self.center_marker):
            it.setZValue(30)
            it.setAcceptedMouseButtons(QtCore.Qt.NoButton)
            self.imv.view.addItem(it)
        self.circle = None      # (cx, cy, R) of the active area, native px
        self.hits = None        # results of find_all_hits
        self._hits_text = ""
        self._mask_version = 0  # bumped on every mask edit
        self._scale_changed()
        return w

    def _mm_per_px(self):
        d = self.det_px.value()
        return self.det_mm.value() / d if d > 0 else np.nan

    def _scale_changed(self):
        s = self._mm_per_px()
        self.scale_label.setText(f"{s:.5g} mm/px" if np.isfinite(s) else
                                 "scale unknown: fit the circle or enter ⌀ in px")
        self._update_hit_map()

    def set_circle(self, c):
        """Active-area circle (cx, cy, R) in native px, e.g. from the auto-mask fit."""
        self.circle = c
        if c is not None:
            self.det_px.setValue(2 * c[2])

    def _fit_detector_circle(self):
        if self.frames is None:
            return
        c = fit_active_circle(self._stat_image(VIEW_STD, all_frames=True)[0])
        if c is None:
            QtWidgets.QMessageBox.information(self, "Fit circle", "No circular edge found; "
                                              "enter the diameter in px by hand.")
            return
        self.set_circle(c)
        self.status.showMessage(f"Active circle: centre ({c[0]:.1f}, {c[1]:.1f}), "
                                f"R = {c[2]:.1f} px")

    def _set_center(self, x, y, user=True):
        """Set the centre; user=False for the provisional default (no crosshair)."""
        for s, val in ((self.cen_x, x), (self.cen_y, y)):
            s.blockSignals(True)
            s.setValue(val)
            s.blockSignals(False)
        self._center_changed(user=user)

    def _center_changed(self, user=False):
        if user:
            self._center_set = True
        on = self.cen_show.isChecked() and self.frames is not None and self._center_set
        self.center_marker.setData([self.cen_x.value()], [self.cen_y.value()]) if on else \
            self.center_marker.setData([], [])
        self._update_hit_map()

    def _pick_center_toggled(self, on):
        if on:
            self.edit_btn.setChecked(False)  # clicks set the centre, not the mask
            self.tabs.setCurrentIndex(0)
            self.status.showMessage("Click on the image to set the centre")

    def _auto_center(self):
        if self.frames is None:
            return
        # always calibrated on every loaded frame (all selected runs), not the stats selection
        mean = self._stat_image(VIEW_MEAN, all_frames=True)[0]
        x, y = find_center(mean, self._active_mask(), self.cen_radius.value())
        self._set_center(x, y)
        self.status.showMessage(f"Auto centre from the mean of all {len(self.frames)} loaded "
                                f"frames: ({x:.2f}, {y:.2f}) px")

    def _frame_hits(self, i):
        """Hits of frame i with the current settings (binned pixel indices)."""
        img = self.frames[i].astype(np.float32)
        dark = self._dark()
        if dark is not None:
            img -= dark
        return find_hits(self._display(img, use_log=False), self.hit_thr.value(),
                         self.hit_min.value())

    def _update_hit_markers(self, view):
        if not self.hit_show.isChecked() or view != VIEW_FRAME or self.frames is None:
            self.hit_markers.setData([], [])
            return
        h = self._frame_hits(self.current_index())
        b = self._bin
        self.hit_markers.setData((h["ix"] + 0.5) * b, (h["iy"] + 0.5) * b)
        self.status.showMessage(f"{len(h['ix'])} hits in this frame (threshold ≥ "
                                f"{self.hit_thr.value():g}, min size {self.hit_min.value()})")

    def find_all_hits(self):
        """Run the hit finder over every loaded frame with the current settings."""
        if self.frames is None:
            return
        dark = self._dark()
        n = len(self.frames)
        res = {k: [] for k in ("ix", "iy", "peak", "size", "sum")}
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            chunk = 64
            for c in range(0, n, chunk):
                self._hist_progress(c, n)
                st = self.frames[c:c + chunk].astype(np.float32)
                if dark is not None:
                    st -= dark
                st = self._display(st, use_log=False)
                for img in st:
                    h = find_hits(img, self.hit_thr.value(), self.hit_min.value())
                    for k in res:
                        res[k].append(h[k])
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.progress.setVisible(False)
        mask = self._active_mask()
        self.hits = dict(res, bin=self._bin, bin_mode=self._bin_mode(),
                         threshold=self.hit_thr.value(), min_size=self.hit_min.value(),
                         dark=self.dark_combo.currentText() if dark is not None else "",
                         n_masked=int(mask.sum()) if mask is not None else 0,
                         mask=None if mask is None else mask.copy(),
                         files=list(self.files), runs=list(self.loaded_runs),
                         run_of=self.run_of.copy(), settings=self._hit_settings())
        counts = np.array([len(x) for x in res["ix"]])
        self.hpf_curve.setData(np.arange(n), counts)
        self._update_hit_map()
        self._hits_text = (
            f"{counts.sum()} hits in {n} frames ({(counts > 0).sum()} frames with hits, "
            f"mean {counts.mean():.2f}, max {counts.max()} per frame) — bin ×{self._bin} "
            f"{self._bin_mode()}, threshold ≥ {self.hit_thr.value():g}, min size "
            f"{self.hit_min.value()}, {self.hits['n_masked']} px masked"
            + (f", dark: {self.hits['dark']}" if dark is not None else "")
            + ". Centre and scale can still be changed before exporting.")
        self._update_hits_info()

    def _hit_settings(self):
        """Everything that changes which hits are found (not centre/scale)."""
        return (self._bin, self._bin_mode(), self.hit_thr.value(), self.hit_min.value(),
                self.dark_combo.currentText() if self._dark() is not None else "",
                self.apply_mask_chk.isChecked(), self._mask_version,
                self.bin_thresh.value(), self.rescale_chk.isChecked())

    def _hits_stale(self):
        return self.hits is not None and self.hits["settings"] != self._hit_settings()

    def _update_hits_info(self):
        if self.hits is None:
            return
        stale = self._hits_stale()
        self.hits_info.setText(
            self._hits_text + ("<br><b style='color:#d9822b'>Out of date: binning, mask, "
                               "threshold or dark changed since. Press 'Find hits in all "
                               "frames' (Export does this automatically).</b>"
                               if stale else ""))

    def _hit_positions(self):
        """All hits as (frame, x_rel_mm, y_rel_mm, x_rel_px, y_rel_px), native px coords."""
        h = self.hits
        b = h["bin"]
        cx, cy = self.cen_x.value(), self.cen_y.value()
        f = np.concatenate([np.full(len(x), i) for i, x in enumerate(h["ix"])]) \
            if h["ix"] else np.zeros(0)
        x = (np.concatenate(h["ix"]) + 0.5) * b - cx if f.size else np.zeros(0)
        y = (np.concatenate(h["iy"]) + 0.5) * b - cy if f.size else np.zeros(0)
        s = self._mm_per_px()
        return f, x * s, y * s, x, y

    def _update_hit_map(self):
        if getattr(self, "hits", None) is None:
            self.hit_map_img.clear()
            return
        _, xm, ym, xp, yp = self._hit_positions()
        s = self._mm_per_px()
        use_mm = np.isfinite(s)
        x, y = (xm, ym) if use_mm else (xp, yp)
        unit = "mm" if use_mm else "px"
        self.hit_map_plot.setLabel("bottom", f"x − centre ({unit})")
        self.hit_map_plot.setLabel("left", f"y − centre ({unit})")
        if not x.size:
            self.hit_map_img.clear()
            return
        # one map pixel per binned pixel, edges on the pixel grid (hit positions are
        # bin centres, so unaligned edges would leave empty stripes)
        b = self.hits["bin"]
        f = s if use_mm else 1.0
        ix = np.concatenate(self.hits["ix"])
        iy = np.concatenate(self.hits["iy"])
        ex = (np.arange(ix.min(), ix.max() + 2) * b - self.cen_x.value()) * f
        ey = (np.arange(iy.min(), iy.max() + 2) * b - self.cen_y.value()) * f
        H, xe, ye = np.histogram2d(x, y, bins=[ex, ey])
        self.hit_map_img.setImage(H.T, autoLevels=True)
        self.hit_map_img.setRect(QtCore.QRectF(xe[0], ye[0], xe[-1] - xe[0], ye[-1] - ye[0]))
        if use_mm:  # detector edge, if the circle centre is known
            t = np.linspace(0, 2 * np.pi, 361)
            ox, oy = ((self.circle[0] - self.cen_x.value()) * s,
                      (self.circle[1] - self.cen_y.value()) * s) if self.circle else (0, 0)
            r = self.det_mm.value() / 2
            self.hit_map_circle.setData(ox + r * np.cos(t), oy + r * np.sin(t))
        else:
            self.hit_map_circle.setData([], [])

    def _export_hits(self):
        if self.frames is None:
            return
        path, _ =QtWidgets.QFileDialog.getSaveFileName(self, "Export hits", "hits.h5",
                                                        "HDF5 (*.h5)")
        if not path:
            return
        if not path.endswith((".h5", ".hdf5")):
            path += ".h5"
        self.export_hits(path)
        self.status.showMessage(f"Exported hits to {path}")

    def export_hits(self, path):
        """Write the extracted hits (see HITS_CONVENTION) to an HDF5 file.

        Hits are (re)computed first if they are missing or were found with other
        settings, so the file always matches the currently selected binning, mask,
        threshold and dark subtraction."""
        if self.hits is None or self._hits_stale():
            self.find_all_hits()
        h = self.hits
        n = len(h["ix"])
        counts = np.array([len(x) for x in h["ix"]], dtype=np.int32)
        hmax = int(counts.max()) if n else 0
        b = h["bin"]
        cx, cy = self.cen_x.value(), self.cen_y.value()
        s = self._mm_per_px()
        hits = np.full((n, hmax, 2, 2), np.nan, dtype=np.float64)
        rel_px = np.full((n, hmax, 2), np.nan)
        extra = {k: np.full((n, hmax), np.nan) for k in ("peak", "size", "sum")}
        for i in range(n):
            k = counts[i]
            if not k:
                continue
            x = (h["ix"][i] + 0.5) * b - cx
            y = (h["iy"][i] + 0.5) * b - cy
            rel_px[i, :k] = np.c_[x, y]
            hits[i, :k, 0] = np.c_[x, y] * s
            hits[i, :k, 1] = np.c_[h["ix"][i], h["iy"][i]]
            for key in extra:
                extra[key][i, :k] = h[key][i]
        str_dt = h5py.string_dtype()
        with h5py.File(path, "w") as f:
            d = f.create_dataset("hits", data=hits, compression="gzip")
            d.attrs["convention"] = HITS_CONVENTION
            d.attrs["axes"] = "frame, hit, (relative position mm | binned pixel index), (x, y)"
            f.create_dataset("n_hits", data=counts)
            f.create_dataset("hits_rel_px", data=rel_px, compression="gzip").attrs[
                "description"] = "(x, y) relative to the centre in native pixels"
            for key, desc in (("peak", "value of the brightest binned pixel"),
                              ("size", "number of binned pixels in the hit"),
                              ("sum", "sum of the hit's binned pixel values")):
                f.create_dataset(f"hit_{key}", data=extra[key], compression="gzip").attrs[
                    "description"] = desc
            f.create_dataset("files", data=np.array(h["files"], dtype=object), dtype=str_dt)
            f.create_dataset("run_index", data=h["run_of"])
            f.create_dataset("runs", data=np.array(h["runs"], dtype=object), dtype=str_dt)
            if h["mask"] is not None:
                f.create_dataset("mask", data=h["mask"].astype(np.uint8),
                                 compression="gzip").attrs["convention"] = MASK_CONVENTION
            f.attrs.update(dict(
                center_x_px=cx, center_y_px=cy, mm_per_px=s,
                detector_diameter_mm=self.det_mm.value(),
                detector_diameter_px=self.det_px.value(),
                bin_factor=b, bin_mode=h["bin_mode"], threshold=h["threshold"],
                min_size=h["min_size"], n_masked_px=h["n_masked"], dark_run=h["dark"]))
            if self.circle is not None:
                f.attrs["detector_circle_px"] = np.asarray(self.circle, dtype=float)

    # ---------------------------------------------------------------- levels
    def auto_levels(self):
        if self.frames is None or self.shown is None:
            return
        finite = self.shown[np.isfinite(self.shown)]
        if not finite.size:
            return
        lo, hi = np.percentile(finite, [self.plo.value(), self.phi.value()])
        if hi <= lo:
            hi = lo + 1
        self._set_levels(float(lo), float(hi))

    def _set_levels(self, lo, hi):
        for s, v in ((self.vmin, lo), (self.vmax, hi)):
            s.blockSignals(True)
            s.setValue(v)
            s.blockSignals(False)
        self.imv.setLevels(lo, hi)

    def _levels_from_spins(self, _):
        lo, hi = self.vmin.value(), self.vmax.value()
        if hi > lo:
            self.imv.setLevels(lo, hi)

    def _levels_from_hist(self, item):
        lo, hi = item.getLevels()
        for s, v in ((self.vmin, lo), (self.vmax, hi)):
            s.blockSignals(True)
            s.setValue(v)
            s.blockSignals(False)
        self._place_hist_lines(lo, hi)

    def _set_cmap(self, name):
        cmap = pg.colormap.get("CET-L1") if name == "grey" else pg.colormap.get(name)
        self.imv.setColorMap(cmap)

    def _hover(self, ev):
        if ev.isExit() or self.shown is None:
            return
        p = ev.pos()
        x, y = int(np.floor(p.x())), int(np.floor(p.y()))  # indices into the binned image
        h, w = self.shown.shape
        b = self._bin
        if 0 <= x < w and 0 <= y < h:
            where = f"x={x}  y={y}" if b == 1 else \
                f"bin ({x}, {y})  x={x * b}–{x * b + b - 1}  y={y * b}–{y * b + b - 1}"
            msg = f"{where}  value={self.rendered[y, x]:.3f}"
            if self.mask is not None:
                nm = int(self.mask[y * b:(y + 1) * b, x * b:(x + 1) * b].sum())
                if nm:
                    msg += f"  [masked {nm}/{b * b}]"
            self.status.showMessage(msg)


def main():
    global FRAME_KEY
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("roots", nargs="*", help="folders to scan for runs (can also be added "
                                             "from the GUI)")
    ap.add_argument("--key", default=None, help="HDF5 dataset path of the frame "
                                                "(default: first 2D dataset in the file)")
    args = ap.parse_args()
    FRAME_KEY = args.key
    app = pg.mkQApp("PXI-viewer")
    win = PatternViewer([os.path.abspath(r) for r in args.roots])
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
