#!/usr/bin/env python
"""Convert a PXI-viewer hits export into ion directions (unit vectors), one set per shot.

Usage:
    python hits_to_directions.py hits.h5 [-o directions.h5] [--L MM] [--mirror]
                                 [--min-hits N] [--nside 64] [--plot quicklook.png]

Geometry. The detector is a flat plate at flight distance L from the interaction point,
and the hit centre of the export (the "centre" in the viewer's Hits tab) is taken as the
foot of the perpendicular from the interaction point onto the plate. A hit at (x, y) mm
relative to that centre (image axes: x right, y down) is the direction

    u = normalize(L, s·x, −y)        s = +1, or −1 with --mirror

in the lab frame used by OREO and explode_tools: +x is the detector normal pointing from
the interaction point to the detector, and a hit lands at (L, L·u_y/u_x, L·u_z/u_x), as in
explode_tools.invert_H_to_r. z is "up" in the image.

Handedness. Without --mirror the image is taken as seen from behind the detector,
looking back towards the interaction point. That is the PXI setup: the camera sits
outside the chamber, views the back of the phosphor screen through the window, and the
optical path has no mirror (2026-11). The mapping is then a proper rotation of the lab
frame, with the same handedness as the simulated detector in explode_tools. Use --mirror
only if the optics change: a mirror in the path (an odd number of them), the camera's
ReverseX or ReverseY switched on, or a view of the ion-facing side of the screen.
Orientations do not depend on this, but the chirality of a reconstructed shape does.

Field of view. The detectable directions are those the hit finder could report: a hit is
the centre of a binned pixel, so a direction counts if it lands in a bin that the viewer
used (masked fraction below its "Hide bin at" threshold, edge rows/columns that do not
fill a bin excluded) whose centre lies inside the active circle (if fitted). This is
stored as a native-resolution mask (fov/mask); fov_contains() and fov_healpix_mask()
rebuild the field of view from it for any directions or HEALPix resolution. Hits outside
it are dropped and counted (hits_outside_fov). Without a mask in the export the circle,
or else the detector diameter around the centre, is used. The HEALPix mask (RING ordering)
is also written when healpy is installed.

Output (HDF5):
    directions   (F, Hmax, 3)  unit vectors, NaN-padded (like the hits export)
    n_hits       (F,)          hits per shot
    frame_index  (F,)          row of each shot in the hits export
    fov_mask     (npix,)       HEALPix field of view at `nside` (if healpy is available)
    fov/...                    native-pixel mask, circle, centre, scale: see fov_contains
    attrs: flight_distance_mm, mirror, convention, source, nside, n_hits_total, …

Needs numpy and h5py; healpy for the HEALPix mask, matplotlib for --plot.
"""
import argparse
import os
import sys

import h5py
import numpy as np

CONVENTION = (
    "u = normalize(L, s*x, -y) for a hit at (x, y) mm relative to the centre (image axes: "
    "x right, y down), s = -1 if mirror else +1. Lab frame as in OREO/explode_tools: +x is "
    "the detector normal from the interaction point to the detector; a direction u lands at "
    "(L, L*u_y/u_x, L*u_z/u_x). Unused slots are NaN.")


# ------------------------------------------------------------------ geometry
def hits_to_directions(xy_mm, L, mirror=False):
    """(n, 2) hit positions in mm (image axes, relative to the centre) -> (n, 3) unit vectors."""
    xy = np.asarray(xy_mm, dtype=float).reshape(-1, 2)
    s = -1.0 if mirror else 1.0
    v = np.column_stack([np.full(len(xy), float(L)), s * xy[:, 0], -xy[:, 1]])
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def directions_to_hits(u, L, mirror=False):
    """Inverse of hits_to_directions: (n, 3) directions -> (n, 2) mm in image axes.
    Directions with u_x <= 0 never reach the detector and give NaN."""
    u = np.asarray(u, dtype=float).reshape(-1, 3)
    s = -1.0 if mirror else 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        x = s * L * u[:, 1] / u[:, 0]
        y = -L * u[:, 2] / u[:, 0]
    bad = u[:, 0] <= 0
    x[bad] = y[bad] = np.nan
    return np.column_stack([x, y])


def effective_mask(mask, b=1, hide_threshold=0.5, circle=None):
    """Native-resolution mask of what the hit finder could see (True = not detectable).

    A b×b bin is usable if its masked fraction is below hide_threshold (as in the viewer)
    and, if circle (cx, cy, R) is given, its centre lies inside the circle. Edge rows and
    columns that do not fill a whole bin are not usable.
    """
    h, w = mask.shape
    b = max(int(b), 1)
    hb, wb = h // b, w // b
    frac = mask[:hb * b, :wb * b].reshape(hb, b, wb, b).mean(axis=(1, 3))
    valid = frac < hide_threshold if mask.any() else np.ones((hb, wb), bool)
    if circle is not None:
        cx, cy, R = circle
        yc, xc = (np.indices((hb, wb)) + 0.5) * b
        valid &= np.hypot(xc - cx, yc - cy) <= R
    out = np.ones((h, w), bool)
    out[:hb * b, :wb * b] = ~np.repeat(np.repeat(valid, b, axis=0), b, axis=1)
    return out


def fov_contains(u, fov):
    """Boolean per direction: can the detector (as the hit finder saw it) record it?

    fov: dict with L, mirror, mm_per_px, center_px (x, y), and one of: mask (H, W; True =
    not detectable, native px; see effective_mask), circle_px (cx, cy, R, native px), or
    radius_mm (a circle around the centre).
    """
    xy = directions_to_hits(u, fov["L"], fov["mirror"])
    ok = np.isfinite(xy[:, 0])
    s = fov["mm_per_px"]
    px = xy[:, 0] / s + fov["center_px"][0]   # native pixel coordinates (pixel i = [i, i+1))
    py = xy[:, 1] / s + fov["center_px"][1]
    mask = fov.get("mask")
    if mask is None and fov.get("circle_px") is not None:
        cx, cy, R = fov["circle_px"]
        ok &= np.hypot(px - cx, py - cy) <= R
    if mask is not None:
        h, w = mask.shape
        ix = np.floor(np.where(ok, px, -1)).astype(np.int64)
        iy = np.floor(np.where(ok, py, -1)).astype(np.int64)
        inside = ok & (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        good = np.zeros(len(u), bool)
        good[inside] = ~mask[iy[inside], ix[inside]].astype(bool)
        ok = good
    if fov.get("circle_px") is None and mask is None:
        if fov.get("radius_mm") is None:
            raise ValueError("field of view unknown: no mask, circle or radius")
        ok &= np.hypot(xy[:, 0], xy[:, 1]) <= fov["radius_mm"]
    return ok


def fov_healpix_mask(fov, nside=64):
    """Field of view as a boolean HEALPix map (RING ordering); pixel centres are tested."""
    import healpy as hp
    v = np.array(hp.pix2vec(nside, np.arange(hp.nside2npix(nside)))).T
    return fov_contains(v, fov)


def fov_half_angle_deg(fov):
    """Largest angle from the detector normal that can be detected (rough, from a grid)."""
    t = np.radians(np.linspace(0, 89.5, 360))
    p = np.radians(np.arange(0, 360, 1.0))
    T, P = np.meshgrid(t, p)
    u = np.column_stack([np.cos(T).ravel(), (np.sin(T) * np.cos(P)).ravel(),
                         (np.sin(T) * np.sin(P)).ravel()])
    ok = fov_contains(u, fov)
    return float(np.degrees(T.ravel()[ok].max())) if ok.any() else float("nan")


# ------------------------------------------------------------------ reading / writing
def read_hits_export(path, L=None, mm_per_px=None):
    """Hits, flight distance and field of view of a PXI-viewer hits export."""
    with h5py.File(path, "r") as f:
        hits = f["hits"][()]
        n_hits = f["n_hits"][()]
        a = dict(f.attrs)
        mask = f["mask"][()].astype(bool) if "mask" in f else None
        L_frames = f["flight_distance_mm"][()] if "flight_distance_mm" in f else None
    if L is None:
        if L_frames is not None:
            vals = np.unique(L_frames[np.isfinite(L_frames)])
            raise ValueError(f"the frames have different flight distances {vals} mm; pass "
                             "--L, or export the runs of one distance separately")
        L = a.get("flight_distance_mm", np.nan)
        if not np.isfinite(L):
            raise ValueError("no flight distance in the export (fill in Setup Notes in the "
                             "camera program, or pass --L in mm)")
    s = mm_per_px if mm_per_px is not None else a.get("mm_per_px", np.nan)
    if not np.isfinite(s):
        raise ValueError("no mm/px scale in the export (fit the circle in the viewer's Hits "
                         "tab, or pass --mm-per-px)")
    circle = a.get("detector_circle_px")
    circle = None if circle is None else tuple(float(c) for c in circle)
    if mask is not None:   # what the hit finder saw: bins, hide threshold, circle
        mask = effective_mask(mask, int(a.get("bin_factor", 1)),
                              float(a.get("bin_hide_threshold", 0.5)), circle)
    fov = dict(L=float(L), mm_per_px=float(s),
               center_px=(float(a["center_x_px"]), float(a["center_y_px"])), mask=mask,
               circle_px=circle,
               radius_mm=float(a["detector_diameter_mm"]) / 2
               if "detector_diameter_mm" in a else None)
    if mm_per_px is not None:   # positions in mm were made with the export's scale
        hits = hits.copy()
        hits[:, :, 0, :] *= mm_per_px / a["mm_per_px"]
    return hits[:, :, 0, :], n_hits, fov, a


def convert(hits_path, out_path, L=None, mirror=False, min_hits=0, nside=64,
            mm_per_px=None):
    xy, n_hits, fov, src_attrs = read_hits_export(hits_path, L, mm_per_px)
    fov["mirror"] = bool(mirror)
    rays, outside = [], 0
    for f in range(len(n_hits)):   # directions of the hits inside the field of view
        u = hits_to_directions(xy[f, :int(n_hits[f])], fov["L"], mirror)
        ok = fov_contains(u, fov)
        outside += int((~ok).sum())
        rays.append(u[ok])
    n_kept = np.array([len(r) for r in rays])
    keep = np.flatnonzero(n_kept >= min_hits)
    hmax = int(n_kept[keep].max()) if keep.size else 0
    dirs = np.full((len(keep), hmax, 3), np.nan)
    for row, f in enumerate(keep):
        dirs[row, :n_kept[f]] = rays[f]
    n_hits = n_kept
    try:
        hp_mask = fov_healpix_mask(fov, nside)
    except ImportError:
        hp_mask = None
    with h5py.File(out_path, "w") as f:
        d = f.create_dataset("directions", data=dirs, compression="gzip")
        d.attrs["convention"] = CONVENTION
        f.create_dataset("n_hits", data=n_hits[keep].astype(np.int32))
        f.create_dataset("frame_index", data=keep.astype(np.int64)).attrs["description"] = \
            "row of each shot in the hits export (frames, files, run_index there)"
        if hp_mask is not None:
            f.create_dataset("fov_mask", data=hp_mask.astype(np.uint8)).attrs.update(
                dict(nside=nside, ordering="RING",
                     description="1 = detectable direction (pixel centre on an unmasked "
                                 "pixel inside the active circle)"))
        g = f.create_group("fov")
        g.attrs["mm_per_px"] = fov["mm_per_px"]
        g.attrs["center_px"] = np.asarray(fov["center_px"])
        if fov["circle_px"] is not None:   # informative when a mask is present
            g.attrs["circle_px"] = np.asarray(fov["circle_px"])
        if fov["radius_mm"] is not None:
            g.attrs["radius_mm"] = fov["radius_mm"]
        if fov["mask"] is not None:
            g.create_dataset("mask", data=fov["mask"].astype(np.uint8), compression="gzip"
                             ).attrs["description"] = ("1 = not detectable (native px): masked "
                                                       "or hidden bins, outside the circle")
        half = fov_half_angle_deg(fov)
        f.attrs.update(dict(
            flight_distance_mm=fov["L"], mirror=int(bool(mirror)), convention=CONVENTION,
            source=os.path.abspath(hits_path), nside=nside, min_hits=min_hits,
            n_shots=len(keep), n_shots_source=len(n_hits),
            n_hits_total=int(n_hits[keep].sum()), fov_max_angle_deg=half,
            hits_outside_fov=outside))
        for k, v in src_attrs.items():   # the viewer's settings, for the record
            f.attrs["hits_" + k] = v
    return dict(n_shots=len(keep), n_source=len(n_hits), n_hits=int(n_hits[keep].sum()),
                L=fov["L"], fov_max_angle_deg=half, outside=outside,
                fov_fraction=None if hp_mask is None else float(hp_mask.mean()))


def load_directions(path):
    """Read a directions file: dict with rays (list of (n_i, 3) arrays, one per shot),
    frame_index, fov (for fov_contains), fov_mask (HEALPix, or None) and attrs."""
    with h5py.File(path, "r") as f:
        dirs, n = f["directions"][()], f["n_hits"][()]
        g = f["fov"]
        a = dict(f.attrs)
        fov = dict(L=float(a["flight_distance_mm"]), mirror=bool(a["mirror"]),
                   mm_per_px=float(g.attrs["mm_per_px"]),
                   center_px=tuple(g.attrs["center_px"]),
                   circle_px=tuple(g.attrs["circle_px"]) if "circle_px" in g.attrs else None,
                   radius_mm=float(g.attrs["radius_mm"]) if "radius_mm" in g.attrs else None,
                   mask=g["mask"][()].astype(bool) if "mask" in g else None)
        out = dict(rays=[dirs[i, :n[i]] for i in range(len(n))], n_hits=n,
                   frame_index=f["frame_index"][()], fov=fov, attrs=a,
                   fov_mask=f["fov_mask"][()].astype(bool) if "fov_mask" in f else None)
    return out


# ------------------------------------------------------------------ quick look
def quicklook(path, png):
    """Hits on the detector, all directions around the detector normal, hits per shot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = load_directions(path)
    fov, a = d["fov"], d["attrs"]
    u = np.concatenate(d["rays"]) if d["rays"] else np.zeros((0, 3))
    sparse = len(u) < 20000          # dots for few hits, densities for many
    fig = plt.figure(figsize=(15, 5))

    ax = fig.add_subplot(1, 3, 1)
    xy = directions_to_hits(u, fov["L"], fov["mirror"])
    if sparse:
        ax.plot(xy[:, 0], xy[:, 1], ".", ms=3, color="tab:blue")
    else:
        hh = ax.hist2d(xy[:, 0], xy[:, 1], bins=150, cmap="viridis", cmin=1)
        fig.colorbar(hh[3], ax=ax, label="hits per bin")
    if fov["circle_px"] is not None:
        cx, cy, R = fov["circle_px"]
        s = fov["mm_per_px"]
        t = np.linspace(0, 2 * np.pi, 361)
        ex = (cx - fov["center_px"][0] + R * np.cos(t)) * s
        ey = (cy - fov["center_px"][1] + R * np.sin(t)) * s
        ax.plot(ex, ey, "--", color="tab:green", lw=1, label="active circle")
        pad = 0.05 * R * s
        ax.set_xlim(ex.min() - pad, ex.max() + pad)
        ax.set_ylim(ey.min() - pad, ey.max() + pad)
    ax.plot([0], [0], "+", color="tab:red", ms=12, label="centre (normal)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x (mm, image right)")
    ax.set_ylabel("y (mm, image down)")
    ax.set_title(f"{len(u)} hits on the detector")

    # directions in polar coordinates around +x: radius = angle from the detector normal
    ax = fig.add_subplot(1, 3, 2, projection="polar")
    theta = np.degrees(np.arccos(np.clip(u[:, 0], -1, 1)))
    phi = np.arctan2(u[:, 2], u[:, 1])    # 0° = u_y, 90° = u_z ("up" in the image)
    tmax = a["fov_max_angle_deg"] if np.isfinite(a["fov_max_angle_deg"]) else 90
    if sparse:
        ax.plot(phi, theta, ".", ms=3, color="tab:blue")
    else:   # counts per solid angle
        H, pe, te = np.histogram2d(phi, theta, bins=[72, 40],
                                   range=[[-np.pi, np.pi], [0, tmax * 1.05]])
        sa = np.outer(np.diff(pe), np.diff(-np.cos(np.radians(te))))
        PE, TE = np.meshgrid(pe, te, indexing="ij")
        with np.errstate(invalid="ignore", divide="ignore"):
            pm = ax.pcolormesh(PE, TE, np.where(H > 0, H / sa, np.nan), cmap="viridis",
                               shading="flat")
        fig.colorbar(pm, ax=ax, label="hits per steradian", pad=0.1)
    ax.set_rlim(0, tmax * 1.05)
    ax.set_rlabel_position(112)
    # field-of-view boundary
    pp = np.linspace(-np.pi, np.pi, 361)
    tt = np.radians(np.linspace(0, tmax * 1.05, 400))
    edge = []
    for p in pp:
        v = np.column_stack([np.cos(tt), np.sin(tt) * np.cos(p), np.sin(tt) * np.sin(p)])
        ok = fov_contains(v, fov)
        edge.append(np.degrees(tt[ok].max()) if ok.any() else np.nan)
    ax.plot(pp, edge, "--", color="tab:green", lw=1)
    ax.set_title("ion directions: radius = angle from the detector normal (°),\n"
                 f"0° = u_y, 90° = u_z (image {'left' if fov['mirror'] else 'right'} / up); "
                 f"L = {fov['L']:g} mm", fontsize=9)

    ax = fig.add_subplot(1, 3, 3)
    n = d["n_hits"]
    ax.hist(n, bins=np.arange(n.max() + 2) - 0.5 if n.size else 10, color="tab:blue")
    ax.set_xlabel("hits per shot")
    ax.set_ylabel("shots")
    ax.set_title(f"{len(n)} shots, mean {n.mean() if n.size else 0:.2f} hits/shot")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(png, dpi=120)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("hits", help="hits export of the PXI-viewer (.h5)")
    ap.add_argument("-o", "--out", help="output file (default: <hits>_directions.h5)")
    ap.add_argument("--L", type=float, help="flight distance in mm (default: from the export, "
                                           "i.e. the camera program's Setup Notes)")
    ap.add_argument("--mm-per-px", type=float, help="override the export's scale")
    ap.add_argument("--mirror", action="store_true", help="flip image x (see Handedness)")
    ap.add_argument("--min-hits", type=int, default=0, help="drop shots with fewer hits")
    ap.add_argument("--nside", type=int, default=64, help="HEALPix nside of fov_mask")
    ap.add_argument("--plot", help="write a quick-look PNG")
    args = ap.parse_args(argv)
    out = args.out or os.path.splitext(args.hits)[0] + "_directions.h5"
    try:
        r = convert(args.hits, out, args.L, args.mirror, args.min_hits, args.nside,
                    args.mm_per_px)
    except (ValueError, KeyError) as e:
        sys.exit(f"error: {e}")
    print(f"{r['n_shots']} of {r['n_source']} shots, {r['n_hits']} hits, L = {r['L']:g} mm, "
          f"field of view up to {r['fov_max_angle_deg']:.1f}° from the normal"
          + (f", {100 * r['fov_fraction']:.2f} % of the sphere" if r["fov_fraction"] else "")
          + f" -> {out}")
    if r["outside"]:
        print(f"note: {r['outside']} hits outside the field of view were dropped (outside "
              "the active circle, or the mask changed after the hit search)")
    if args.plot:
        quicklook(out, args.plot)
        print(f"quick look -> {args.plot}")


if __name__ == "__main__":
    main()
