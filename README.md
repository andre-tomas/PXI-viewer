# PXI-viewer

Acquisition and viewing software for 2D detector frames, for example an MCP/phosphor
screen imaged by a camera. It has two programs:

| file | what it does |
|---|---|
| `Mako_v14.py` | Controls an Allied Vision **Mako G-032B** GigE camera: live view, camera settings, and saving one HDF5 file per trigger (hardware or software). |
| `pxi-viewer.py` | Interactive viewer for the saved frames: browse, sort by intensity, mean/median/std, binning, masks, histograms, hit finding and export, and a live tab that follows the camera while it records. |
| `requirements.txt` | Pinned package versions for both programs (Python 3.11). |
| `run_mako.bat`, `run_pxi_viewer.bat` | Windows launchers using a `.venv` next to the scripts. |
| `vimba_viewer.bat` | Starts Allied Vision's Vimba X Viewer with only Allied Vision's own transport layers (see below). |

The data flow:

```
camera ──Mako_v14.py──▶ data_camera/YY-MM-DD/<run>/<run>_<timestamp>_trigger000123.h5
                              │
                              ├─▶ pxi-viewer.py, Live tab (while recording)
                              └─▶ pxi-viewer.py, browse runs ─▶ mask, threshold ─▶ hits.h5
```

Both programs default to the `data_camera` folder next to the scripts. The camera
program writes there, and `run_pxi_viewer.bat` and the Live tab's **Today** button read
from there.

## Installation

The beamtime PC runs Windows with Python 3.11 in a virtual environment `.venv` next to
the scripts:

```bat
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

The viewer on its own runs on any OS with Python ≥ 3.8:

```bash
pip install numpy h5py scipy pyqtgraph PyQt5
```

For an **offline PC**, download the wheels on a machine with internet first
(`pip download -r requirements.txt --platform win_amd64 --python-version 3.11
--only-binary=:all: -d wheels`), copy them over, and install with
`pip install --no-index --find-links wheels -r requirements.txt`.

### Vimba X SDK (camera only)

`vmbpy` is only the Python wrapper. The camera driver and GigE filter driver come from
Allied Vision's **Vimba X** installer (alliedvision.com → Support → Software Downloads),
which is not on PyPI. During setup, include the **GigE Transport Layer** / filter driver,
which lets Windows receive the camera's data over Ethernet at full speed. Without it, frames can be dropped.

If other camera SDKs are installed (Daheng Galaxy, Spinnaker), Vimba X Viewer may open
the Mako through their GenTL producers, and full access then fails. `vimba_viewer.bat`
sets `GENICAM_GENTL64_PATH` to Allied Vision's own `cti` folder only. The Vimba X Viewer is
useful to check that the camera is visible before starting the script.

## Camera program (`Mako_v14.py`)

### Hardware

- **Power and data:** The camera is powered over Ethernet through a PoE injector
  (e.g. Digitus DN-95102-1): camera → injector "LAN+PoE" (out), injector "LAN" (in) → PC.
  The injector needs no driver.
- **Trigger** goes into the camera's 8-pin I/O socket (Hirose HR25-7TR-8PA; plug
  HR25-7TP-8S; cable with open ends: Allied Vision K7600503):

  | pin | signal | use |
  |---|---|---|
  | 1 | Out 1 | optional: "Exposing" signal to the scope |
  | 4 | In 1 | **trigger (+)**, BNC centre |
  | 5 | Isolated In GND | **trigger ground**, BNC shield |
  | 6 | Isolated Out Power | only for the outputs (3–30 V supply) |
  | 7 | Camera Power | leave unconnected (camera is powered via PoE) |
  | 8 | Camera GND | leave unconnected |

  Wire colours differ between cable versions, so check pins 4 and 5 with a multimeter.
- **Trigger levels:** high 3–24 V, low 0–1 V. A 5 V TTL pulse from the laser sync or a
  delay generator works (50 Ω outputs give about twice the set voltage into this
  high-impedance input, which is fine up to 24 V). The script triggers on the
  **rising edge**. A negative-going pulse needs `TriggerActivation` changed to falling edge.
- **Timing check (optional):** Out 1 (SyncOut1) is set to "Exposing". The outputs are open
  emitter: put a 5–12 V supply on pin 6 and ~1 kΩ from pin 1 to that supply's ground, and
  measure across the resistor. Compare with the laser pulse on a second channel and adjust
  the trigger delay / exposure until the pulse falls inside the exposure.

### Using it

Start with `python Mako_v14.py` (or `run_mako.bat`) and press **Connect & Start
Streaming**. It connects to the first camera named "Mako G-032B" (else the first camera
found), sets Mono8, turns auto exposure/gain off and negotiates the GigE packet size.

- **Live View:** free-running preview. Mouse wheel zooms, drag moves, double-click resets
  (display only). **Live view on hardware trigger (nothing is saved)** updates the
  image only on trigger pulses, for aligning and setting the trigger delay. The status bar
  shows the FPS and, in hardware mode, `SyncIn1=HIGH/LOW` (the level at the trigger input).
- **Camera Settings:** exposure (slider to 100 ms, typed up to 60 s), gain (0–30 dB),
  gamma, black level, binning H/V (1, 2, 4). The boxes show the value the camera actually
  uses (it rounds, e.g. gain to whole dB). A binning change restarts the stream and resets
  the image to the full binned sensor.
- **Acquisition:**
  - **Save location:** `<folder>/YY-MM-DD/<base filename><N>/`, with runs numbered within
    the day (`camera_test_run1`, `camera_test_run2`, …).
  - **Trigger mode:**
    - *Hardware Line1 (GPIO)*: every frame the camera delivers is one trigger. Settings:
      **Debounce time (ns)**, a glitch filter (pulses shorter than this are ignored; max
      50000 ns; not a delay), and **Trigger delay (µs)**, the wait from the edge to the start
      of exposure (can only shift later). The delay is sent at once, also during an
      acquisition, and stored with every frame. `TriggerOverlap=Off` makes the camera ignore
      edges during readout (prevents lockups on GigE).
    - *Software (manual button)*: **⚡ Software Trigger** saves the next *Frames per trigger*
      frames of the free-running stream into one file.
  - **Start Acquisition**: each trigger increments "Triggers received" and writes one
    `.h5` (and optionally one PNG per frame). No pulse means nothing is saved. On stop the
    camera returns to the live-view mode at full frame rate.
  - **Dump All Camera Features to Console**: prints every GenICam feature, for finding
    feature names on other firmware.

### Output files

`<run>_<YYYYmmdd_HHMMSS_ffffff>_trigger<NNNNNN>.h5`, one per trigger, written as `.part`
and renamed when complete (so the viewer's Live tab never reads a half-written file):

```
frames/                       attrs: trigger_index, exposure_frames, saved_time_utc
frames/frame_0000             uint8 (H, W), gzip; one dataset per frame of the trigger
    attrs: trigger_mode, trigger_delay_us, frame_index, trigger_index,
           wall_time_utc_s, wall_time_iso, camera_timestamp_ns, exposure_us, gain_dB,
           gamma, black_level, binning_h, binning_v, width, height, pixel_format
```

Full sensor is 658 × 492 px (W × H) at binning 1. PNGs go to
`<run>/<run>_<start time>_frames/`.

### Camera notes and limitations

- **First 0.5 s:** In hardware mode, frames arriving in the first 0.5 s after Start are discarded,
  because free-run frames already buffered before `TriggerMode=On` took effect would otherwise
  count as triggers. Real shots in that window are lost too.
- **Timestamps:** `wall_time_utc_s` is when the PC received the frame, not the shot time.
  `camera_timestamp_ns` is the camera's own clock.
- **Settings metadata:** Exposure, gain, etc. in the metadata are the GUI values (which are read back from the
  camera when changed).
- **No triggers or dark image:** "No cameras found": check the Vimba X install and the network adapter's IP.
  "Triggers received" stays 0: check pin 4/5 wiring and polarity, that the pulse is above
  3 V and longer than the debounce time, and the trigger mode. A black live view in
  hardware mode is normal without pulses.

## PXI-viewer (`pxi-viewer.py`)

It lets you flip through the frames, sort them by total intensity, look at
mean/median/std patterns, bin, mask bad or background regions, compare pixel-value
histograms and extract hit positions. Everything lives in the single script
`pxi-viewer.py`.

### Starting

```bash
python pxi-viewer.py [DATA_DIR ...] [--key DATASET]
```

- Every directory below a `DATA_DIR` that contains `*.h5` / `*.hdf5` files is a **run**.
  Without arguments the viewer scans the `Mesh_data` folder next to the script
  (`DEFAULT_ROOTS` at the top of the script) if it exists, otherwise it starts empty;
  use **Add folder…** to scan more folders.
- The frame is the first 2D numeric dataset in each file, or the one named with
  `--key` (e.g. `--key entry/data/image`). All frames of a run must have the same shape.
- Files in a run are ordered by natural sort of their names (`img2` before `img10`).
- Scalar attributes of the frame dataset (exposure, timestamps, …) are shown per frame.
- Directories with `dark` in their name are offered as dark runs.

### Window overview

The left side has three tabs (**Image**, **Histograms**, **Hits**). On the right is a
side panel whose width you can change with the splitter handle. The mouse wheel
scrolls the panel and never changes a setting.

#### Data

- **Run list:** type in the filter box to narrow it down (all words must match).
  Ctrl/Shift-click selects several runs, and **Load selected** loads them together as
  one set of frames. Double-click loads a single run. Each frame remembers which run it
  came from; the Info box shows it, and in file order dashed lines in the intensity plot
  mark where each run starts.
- **Subtract mean dark:** subtracts the mean frame of the chosen dark run. It
  defaults to a dark run in the same folder as the selected run.

#### Browse and show

- **Sort by:** file order, or total intensity (high → low / low → high). The total
  intensity is a plain sum over native pixels, minus the dark and excluding masked
  pixels if the mask is applied. The plot under the image shows it; click or drag the
  orange line to jump.
- **Frame / Mean / Median / Std:** show a single frame or a statistic. **Stats over first
  N** restricts mean/median/std to the first N frames of the current order (e.g. the N
  brightest); 0 means all frames.

#### Colour scale

Drag the handles on the histogram next to the image, type Min/Max, or press
**Auto (percentile)**. **Lock limits while browsing** keeps them fixed while you flip
through frames. There is also a colormap choice and an optional `log10(1 + I)` display.

#### Binning

- **Bin factor** ×1 (native) … ×64 and **Mode**:
  - **Sum** adds the b×b pixels of a bin.
  - **Max** takes the brightest pixel of the bin.
- Edge rows/columns that don't fill a whole bin are cropped. Axes stay in native pixels.

#### Mask

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

#### Histograms tab

Histograms of pixel values *as displayed*, i.e. after dark subtraction, mask, binning
and log.

- **Source:** the current view (updates live), all frames of the loaded runs, or the
  stats selection (first N). The frame sets are computed on **Compute**.
- **Axes:** log x (with log-spaced bins; values ≤ 0 are left out and counted) and
  log y. **Normalize** divides by the number of values.
- **Binning:** the number of bins can be set, with an automatic or manual range. For
  raw counts the bins are centred on whole numbers.
- **Histogram of:** *Pixel values* (above) or *Total intensity per frame*: one value per
  frame, the same total as in the Image tab (dark subtracted, mask applied, native
  pixels; the binning does not apply). Source *Current view* or *All frames in run* then
  means all loaded frames, *Stats selection* the first N of the current order. It updates
  at once, without **Compute**.
- **Pin** keeps a curve for comparison (other frames, runs, bin factors). Pinned curves
  are listed on the right, in their colour: untick one to hide it, select one or more
  (Ctrl/Shift-click) and press **Remove selected**, or **Clear all**.
- **Zoom:** tick or untick **zoom x / zoom y** to zoom only one axis; **Fit** resets.
  Scrolling over an axis also zooms only that axis.
- Dashed lines mark the current colour limits.

#### Hits tab

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

#### Live tab

Watches a folder while the camera records and shows the data as it comes in. It only
reads the files the camera script writes, so it can run at the same time as the camera
program.

- **Folder:** type it, **Browse…**, or **Today** (`data_camera/YY-MM-DD` next to the
  script, where the camera script saves today's data). Subfolders are watched too. The
  folder may not exist yet; it is picked up as soon as the first acquisition creates it.
- **Start watching:** checks the folder twice a second. Normally only frames that arrive
  after Start count; tick **include files already there** to also read the existing ones.
- **Every new frame** adds a point to the **total intensity vs. acquisition time** plot.
  The time is the frame's `wall_time_utc_s` attribute, else the file's modification time.
  A file holding a burst of several frames gives one point per frame.
- **Threshold:** frames with a total intensity at or above it are drawn orange and shown
  in the image. Two modes:
  - **factor × mean** (default): factor (default 1.2) × the mean total intensity of the
    N frames *before* the frame, with N = **Plot last** (default 200; 0 = all earlier
    frames). At the start fewer frames are used; the very first frame is always shown.
    The threshold follows slow drifts and is drawn as a dashed curve. A lasting jump in
    intensity becomes the new normal after about N frames.
  - **fixed value:** a fixed total intensity; type it or drag the dashed line.
- **for at least … s** (default 5 s): a shown frame stays on screen at least this long.
  Frames above the threshold that arrive meanwhile are not queued; when the time is up,
  the newest of them is shown ("next frame waiting" in the status line). 0 shows every
  frame at once. The plot keeps getting all points regardless.
- **Plot last N** (default 200): the plot shows only the last N frames and its axes follow
  new frames automatically, also after you zoomed. 0 = all frames. Earlier frames stay
  stored and reappear when N is raised. The same N is used for the mean of the
  factor × mean threshold.
- **Click a point** to show that frame. **Clear plot** starts the plot afresh.
- **Processing:** the total intensity and the image use the side panel's dark, mask,
  binning, log and colormap. A dark or mask whose shape differs from the frame is ignored
  for that frame. Totals are computed when the frame arrives, so changing the dark or mask
  later does not recompute earlier points.
- **Frame / Mean / Median / Std:** *Frame* shows frames at or above the threshold as they
  arrive. *Mean*, *Median* and *Std* show that statistic of the last N frames (N = **Plot
  last**; at most 2000 frames are kept in memory, also for N = 0), updated as frames
  arrive. **only frames ≥ threshold** restricts them to those of the N frames at or above
  the threshold. Dark is subtracted from mean and median (not std); mask, binning and log
  apply as usual. After a binning change only frames of the new size are used. Updates are
  limited to every 0.5 s, or every 2× the computing time if longer (the median of 200 full
  frames takes about 0.5 s). Clicking a point in the plot switches back to *Frame*.
- **Pause** freezes the image and the plot so you can look at one frame. New frames are
  still read in the background; **Resume** redraws the plot and shows the newest frame
  above the threshold. Clicking a point works while paused.
- **Colour limits:** only the very first frame gets automatic limits; after that the
  limits stay where you put them on the histogram next to the image. **Auto limits** sets
  percentile limits (side panel) for the shown frame once. Tick **auto limits every
  frame** to have them recomputed for each new frame.

### File formats

#### Mask (`.h5` / `.npy`)

- `.h5`: dataset `mask`, uint8, shape (H, W), 1 = masked (excluded), 0 = good pixel,
  native resolution. Its `convention` attribute says the same.
- `.npy`: the same array.

When loading, a `.h5` file without a `mask` dataset uses the first 2D dataset. If a
file uses 1 = good, press **Invert** after loading.

#### Hits export (`.h5`)

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

### Processing order

Every display, histogram and hit search applies the same steps:

raw frame → dark subtraction (optional) → mask → binning (sum/max) → log (display and
histograms only; not used for hit finding).

The total intensity used for sorting is always computed at native resolution, with the
dark and mask applied.

### Keyboard shortcuts

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

### Typical workflow

1. Load a **background run** (no signal of interest) and use **Auto mask…** →
   **Replace mask**. This masks everything outside the active circle and the background
   structures, and sets the detector scale. Fine-tune with the brush and save the mask.
2. Select and load the **signal runs** (several at once if you like). The mask stays in
   place.
3. Browse, sort by intensity and check histograms to choose a hit threshold and bin
   factor.
4. In the **Hits** tab set the centre (auto or manual), check the scale, press **Find
   hits in all frames**, then **Export hits…**.

### Notes and limitations

- All frames of the selected runs are held in memory (e.g. about 3.5 GB for 11,000
  frames of 492×658 uint8).
- Median and std of large runs take a few seconds to tens of seconds; they are cached
  until the selection, sort order or dark subtraction changes.
- Frame-set histograms and hit finding run over every frame; a progress bar is shown.
- The circle finder needs a run in which the active area is visible (short exposure,
  enough counts). For empty or long-exposure runs it reports that no usable edge was
  found.
- Browsing reads only the **first** 2D dataset of each file (or `--key`), so of a
  software-trigger burst (several `frames/frame_NNNN` in one file) only the first frame is
  loaded. The Live tab reads every frame of a file.
- The hits export has no flight distance: turning hit positions into ion directions
  needs the detector distance from the setup, and image y points down.
