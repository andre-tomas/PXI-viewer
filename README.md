# PXI-viewer

An interactive viewer for 2D detector frames stored as HDF5 files, one frame per file
(for example an MCP/phosphor screen imaged by a camera). It lets you flip through the
frames, sort them by total intensity, look at mean/median/std patterns, bin, mask
bad or background regions, compare pixel-value histograms and extract hit positions.

Everything lives in a single script, `pxi_viewer.py`.

## Requirements

- Python ≥ 3.8
- numpy, h5py, scipy
- pyqtgraph (≥ 0.13) with PyQt5

```bash
pip install numpy h5py scipy pyqtgraph PyQt5
```

## Starting

```bash
python pxi_viewer.py [DATA_DIR ...] [--key DATASET]
```

- Every directory below a `DATA_DIR` that contains `*.h5` / `*.hdf5` files is a **run**.
  Without arguments the viewer starts empty; use **Add folder…** to scan a folder.
- The frame is the first 2D numeric dataset in each file, or the one named with
  `--key` (e.g. `--key entry/data/image`). All frames of a run must have the same shape.
- Files in a run are ordered by natural sort of their names (`img2` before `img10`).
- Scalar attributes of the frame dataset (exposure, timestamps, …) are shown per frame.
- Directories with `dark` in their name are offered as dark runs.

## Window overview

The left side has three tabs (**Image**, **Histograms**, **Hits**). On the right is a
side panel whose width you can change with the splitter handle. The mouse wheel
scrolls the panel and never changes a setting.

### Data

- **Run list:** type in the filter box to narrow it down (all words must match).
  Ctrl/Shift-click selects several runs, and **Load selected** loads them together as
  one set of frames. Double-click loads a single run. Each frame remembers which run it
  came from; the Info box shows it, and in file order dashed lines in the intensity plot
  mark where each run starts.
- **Subtract mean dark:** subtracts the mean frame of the chosen dark run. It
  defaults to a dark run in the same folder as the selected run.

### Browse and show

- **Sort by:** file order, or total intensity (high → low / low → high). The total
  intensity is a plain sum over native pixels, minus the dark and excluding masked
  pixels if the mask is applied. The plot under the image shows it; click or drag the
  orange line to jump.
- **Frame / Mean / Median / Std:** show a single frame or a statistic. **Stats over first
  N** restricts mean/median/std to the first N frames of the current order (e.g. the N
  brightest); 0 means all frames.

### Colour scale

Drag the handles on the histogram next to the image, type Min/Max, or press
**Auto (percentile)**. **Lock limits while browsing** keeps them fixed while you flip
through frames. There is also a colormap choice and an optional `log10(1 + I)` display.

### Binning

- **Bin factor** ×1 (native) … ×64 and **Mode**:
  - **Sum** adds the b×b pixels of a bin.
  - **Max** takes the brightest pixel of the bin.
- Edge rows/columns that don't fill a whole bin are cropped. Axes stay in native pixels.

### Mask

The mask is stored at native resolution (True/1 = excluded pixel) and is kept when you
load other runs.

- **Paint mask (E):**
  - Left-click/drag masks; right-click or Shift unmasks.
  - The brush paints whole *displayed* pixels, i.e. b×b blocks when binned. You can mask
    coarsely at a high bin factor and refine at lower ones.
- **Mask ≥ value:** masks native pixels of the current image (frame, mean, median or
  std) at or above a value, e.g. hot pixels in the std image.
- **Auto mask…:** a first-guess mask from the mean/std of a background run.
  - **Active area:** a circle fitted to the edge of the region that ever records
    signal. The inside level comes from the dim part of that region, so bright signal
    does not pull the edge inwards. **Margin** shrinks the circle to also cut the soft
    rim. The dialog warns when a run shows no clear edge (e.g. long exposures or empty
    runs), and you can enter the circle by hand.
  - **Background / hot pixels:** masks pixels whose smoothed mean and/or std exceeds
    the median by more than k robust σ (from the MAD), then grows the masked regions by
    a few pixels.
  - A live preview is drawn on the image. **Replace mask** or **Add to mask** applies it.
    Applying with the circle ticked also sets the detector scale for the Hits tab.
- **Undo** (Ctrl+Z), **Invert**, **Clear**, **Save…** / **Load…** (`.h5` or `.npy`).
- **Apply mask:** masked pixels are excluded from the image, the totals/sorting, the
  histograms and hit finding.
- **Overlay:** drawn as an outline (red = hidden bin, dashed orange = partially
  masked) with the real values still visible, or as a fill.
- **When binned:**
  - A bin uses only its unmasked pixels.
  - It is hidden once its masked fraction reaches **Hide bin at ≥** (default 50 %).
  - With sum binning, partially masked bins can be rescaled by b²/n_valid.

### Histograms tab

Histograms of pixel values *as displayed*, i.e. after dark subtraction, mask, binning
and log.

- **Source:** the current view (updates live), all frames of the loaded runs, or the
  stats selection (first N). The frame sets are computed on **Compute**.
- **Axes:** log x (with log-spaced bins; values ≤ 0 are left out and counted) and
  log y. **Normalize** divides by the number of values.
- **Binning:** the number of bins can be set, with an automatic or manual range. For
  raw counts the bins are centred on whole numbers.
- **Pin** keeps a curve for comparison (other frames, runs, bin factors).
- **Zoom:** tick or untick **zoom x / zoom y** to zoom only one axis; **Fit** resets.
  Scrolling over an axis also zooms only that axis.
- Dashed lines mark the current colour limits.

### Hits tab

A **hit** is a connected group (diagonals included) of unmasked pixels at or above
the **threshold**, in the frame as displayed: dark-subtracted, masked, and binned with
the *current* bin factor and mode (the log display is ignored). **Min size** removes
smaller groups. A hit's position is the **centre of its brightest binned pixel**.

- **Threshold units:** with sum binning the threshold applies to the summed bin value
  (≈ b² × a single pixel). With max binning it stays in single-pixel units.
- **Centre:** type x/y (native px), click **Pick on image** and then click the image, or
  press **Auto centre**. Auto centre takes the brightest region of the masked mean of
  **all** loaded frames, refined with a local centroid; **radius** sets the smoothing
  size. The cross is only drawn once a centre has been set.
- **Scale:** mm/px = detector diameter (default 40 mm) / diameter of the fitted
  active circle in px. The px diameter is filled in by **Fit circle** or by applying an
  auto mask with the circle ticked, and can be typed.
- **Find hits in all frames:** runs over every loaded frame and shows a 2D map of all
  hit positions relative to the centre (with the detector edge) and the hits per frame.
  **Show hits on image** (off by default) circles the hits of the current frame.
- **Out of date:** if the binning, mask, threshold, min size or dark subtraction change
  afterwards, the tab says so. **Export hits…** re-runs the search first in that case,
  so an export always matches the current settings. Changing the centre or scale never
  needs a re-run.

## File formats

### Mask (`.h5` / `.npy`)

- `.h5`: dataset `mask`, uint8, shape (H, W), 1 = masked (excluded), 0 = good pixel,
  native resolution. Its `convention` attribute says the same.
- `.npy`: the same array.

When loading, a `.h5` file without a `mask` dataset uses the first 2D dataset. If a
file uses 1 = good, press **Invert** after loading.

### Hits export (`.h5`)

| Dataset | Shape | Content |
|---|---|---|
| `hits` | (F, Hmax, 2, 2) | `[f, h, 0, :]` = (x, y) relative to the centre in **mm**; `[f, h, 1, :]` = (ix, iy) index of the hit's brightest **binned** pixel |
| `n_hits` | (F,) | number of hits in frame f; slots `h ≥ n_hits[f]` are NaN |
| `hits_rel_px` | (F, Hmax, 2) | (x, y) relative to the centre in native pixels |
| `hit_peak`, `hit_size`, `hit_sum` | (F, Hmax) | brightest binned value, size in binned pixels, summed value |
| `files` | (F,) | source file of each frame |
| `run_index`, `runs` | (F,), (n_runs,) | which loaded run each frame came from |
| `mask` | (H, W) | the mask that was applied (if any) |

File attributes: `center_x_px`, `center_y_px`, `mm_per_px`, `detector_diameter_mm`,
`detector_diameter_px`, `detector_circle_px` (cx, cy, R, if fitted), `bin_factor`,
`bin_mode`, `threshold`, `min_size`, `n_masked_px`, `dark_run`.

Conventions:
- Frames are in file order (not the browsing sort order).
- x points right and y points down, as in the image.
- Native pixel i spans [i, i+1), so binned pixel j (bin factor b) has its centre at
  (j + 0.5)·b.
- Relative position = bin centre − centre.

Reading it back:

```python
import h5py, numpy as np
with h5py.File("hits.h5") as f:
    hits, n = f["hits"][()], f["n_hits"][()]
    xy_mm = hits[:, :, 0, :]          # (F, Hmax, 2), NaN-padded
    all_xy = xy_mm[~np.isnan(xy_mm[..., 0])]   # (total hits, 2)
    r_mm = np.hypot(*all_xy.T)
```

## Processing order

Every display, histogram and hit search applies the same steps:

raw frame → dark subtraction (optional) → mask → binning (sum/max) → log (display and
histograms only; not used for hit finding).

The total intensity used for sorting is always computed at native resolution, with the
dark and mask applied.

## Keyboard shortcuts

| Key | Action |
|---|---|
| ← / → | previous / next frame |
| PgUp / PgDn | jump 10 frames |
| F / M / D / S | show frame / mean / median / std |
| A | auto colour limits (percentile) |
| L | toggle log display |
| + / − | coarser / finer binning |
| B | toggle sum / max binning |
| E | toggle mask painting |
| Ctrl+Z | undo last mask edit |

## Typical workflow

1. Load a **background run** (no signal of interest) and use **Auto mask…** →
   **Replace mask**. This masks everything outside the active circle and the background
   structures, and sets the detector scale. Fine-tune with the brush and save the mask.
2. Select and load the **signal runs** (several at once if you like). The mask stays in
   place.
3. Browse, sort by intensity and check histograms to choose a hit threshold and bin
   factor.
4. In the **Hits** tab set the centre (auto or manual), check the scale, press **Find
   hits in all frames**, then **Export hits…**.

## Notes and limitations

- All frames of the selected runs are held in memory (e.g. about 3.5 GB for 11,000
  frames of 492×658 uint8).
- Median and std of large runs take a few seconds to tens of seconds; they are cached
  until the selection, sort order or dark subtraction changes.
- Frame-set histograms and hit finding run over every frame; a progress bar is shown.
- The circle finder needs a run in which the active area is visible (short exposure,
  enough counts). For empty or long-exposure runs it reports that no usable edge was
  found.
# PXI-viewer
