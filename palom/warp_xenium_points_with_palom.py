#!/usr/bin/env python3

import os
import re
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import tifffile
import matplotlib.pyplot as plt


def _read_scalar_npz_value(data, key, default=None):
    if key not in data.files:
        return default

    value = data[key]
    if np.asarray(value).shape == ():
        value = value.item()
    return value


def get_transform_shapes_and_scale(data):
    """
    Return full-resolution and registration-level shapes from old or new NPZs.

    Legacy registration files saved ref_shape/moving_shape at the registration
    level. The cleaned registration script saves full-resolution shapes plus
    ref_shape_registration/moving_shape_registration. This helper makes the
    warper explicit about which convention it detected.
    """
    level = _read_scalar_npz_value(data, "level", default=None)
    scale = _read_scalar_npz_value(data, "scale", default=None)

    if scale is None or float(scale) <= 0:
        if level is None:
            raise ValueError(
                "Transform NPZ has missing/non-positive scale and no level key. "
                "Expected scale=2**level."
            )
        scale = 2 ** int(level)
        print(
            f"[WARNING] Transform scale was missing or non-positive; "
            f"using scale=2**level={scale}",
            flush=True,
        )

    scale = float(scale)
    ref_shape = np.asarray(data["ref_shape"], dtype=float)
    moving_shape = np.asarray(data["moving_shape"], dtype=float)

    if "ref_shape_registration" in data.files and "moving_shape_registration" in data.files:
        shape_convention = "new_fullres_with_registration_shapes"
        ref_shape_fullres = ref_shape
        moving_shape_fullres = moving_shape
        ref_shape_level = np.asarray(data["ref_shape_registration"], dtype=float)
        moving_shape_level = np.asarray(data["moving_shape_registration"], dtype=float)
    elif scale == 1:
        shape_convention = "scale_1_shapes"
        ref_shape_fullres = ref_shape
        moving_shape_fullres = moving_shape
        ref_shape_level = ref_shape
        moving_shape_level = moving_shape
    else:
        shape_convention = "legacy_registration_level_shapes"
        ref_shape_level = ref_shape
        moving_shape_level = moving_shape
        ref_shape_fullres = ref_shape * scale
        moving_shape_fullres = moving_shape * scale
        print(
            "[WARNING] Legacy transform detected: ref_shape/moving_shape look "
            "like registration-level shapes. Using those for PALOM block lookup "
            "and multiplying by scale for full-resolution bounds.",
            flush=True,
        )

    return {
        "scale": scale,
        "level": level,
        "shape_convention": shape_convention,
        "ref_shape_fullres": ref_shape_fullres,
        "moving_shape_fullres": moving_shape_fullres,
        "ref_shape_level": ref_shape_level,
        "moving_shape_level": moving_shape_level,
    }


def _coerce_downsample_value(value):
    """Convert PALOM/tifffile downsample metadata to one numeric value."""
    if isinstance(value, dict):
        numeric_values = []
        for nested in value.values():
            try:
                numeric_values.append(_coerce_downsample_value(nested))
            except (TypeError, ValueError):
                pass
        if numeric_values:
            return float(np.nanmean(numeric_values))
        raise TypeError(f"No numeric downsample value in {value!r}")

    if isinstance(value, (list, tuple, np.ndarray)) and not np.isscalar(value):
        numeric_values = []
        for nested in value:
            try:
                numeric_values.append(_coerce_downsample_value(nested))
            except (TypeError, ValueError):
                pass
        if numeric_values:
            return float(np.nanmean(numeric_values))
        raise TypeError(f"No numeric downsample value in {value!r}")

    return float(value)


def _coerce_level_downsamples(level_downsamples):
    """
    PALOM versions differ: level_downsamples may be numeric, a list/array, or
    dictionaries containing x/y values. Normalize all of those to a 1D array.
    """
    if level_downsamples is None:
        return None

    if isinstance(level_downsamples, dict):
        keys = {str(k).lower() for k in level_downsamples}
        if keys.intersection({"x", "y", "width", "height", "downsample", "scale"}):
            return np.asarray([_coerce_downsample_value(level_downsamples)], dtype=float)

        def sort_key(key):
            try:
                return int(key)
            except (TypeError, ValueError):
                return str(key)

        return np.asarray(
            [
                _coerce_downsample_value(level_downsamples[key])
                for key in sorted(level_downsamples, key=sort_key)
            ],
            dtype=float,
        )

    if np.isscalar(level_downsamples):
        return np.asarray([float(level_downsamples)], dtype=float)

    return np.asarray(
        [_coerce_downsample_value(value) for value in level_downsamples],
        dtype=float,
    )


def _yx_shape(shape, axes=None):
    """Return image Y/X shape from tifffile-style shape and axes metadata."""
    shape = tuple(int(v) for v in shape)
    if len(shape) < 2:
        raise ValueError(f"Cannot infer Y/X dimensions from shape {shape}")

    if axes and "Y" in axes and "X" in axes:
        return np.asarray([shape[axes.index("Y")], shape[axes.index("X")]], dtype=float)

    if len(shape) == 2:
        return np.asarray(shape, dtype=float)

    # Heuristic fallback for RGB/RGBA arrays without axes metadata.
    if shape[-1] <= 4 and len(shape) >= 3:
        return np.asarray([shape[-3], shape[-2]], dtype=float)

    return np.asarray(shape[-2:], dtype=float)


def _extract_channel(img, channel=0):
    """Reduce common TIFF layouts to a 2D plane for overlay plotting."""
    img = np.asarray(img)

    while img.ndim > 3:
        img = img[0]

    if img.ndim == 3:
        if img.shape[-1] <= 4 and img.shape[0] > 4:
            channel = min(channel, img.shape[-1] - 1)
            img = img[..., channel]
        elif img.shape[0] <= 16:
            channel = min(channel, img.shape[0] - 1)
            img = img[channel]
        else:
            channel = min(channel, img.shape[-1] - 1)
            img = img[..., channel]

    if img.ndim != 2:
        raise ValueError(f"Expected a 2D image plane after channel selection, got {img.shape}")

    return img


def _normalize_for_plot(img):
    if img.dtype == np.uint8:
        return img

    # Use a sample for percentiles so plotting does not spend ages on large arrays.
    sample = img
    if sample.size > 4_000_000:
        step = int(np.ceil(np.sqrt(sample.size / 4_000_000)))
        sample = sample[::step, ::step]

    p1, p99 = np.percentile(sample, [1, 99])
    return np.clip((img - p1) / (p99 - p1 + 1e-9), 0, 1)


def _read_tiff_zarr_decimated(tif, series_index, level_index, channel, step):
    """
    Last-resort reader for non-pyramidal TIFFs. It avoids materializing the full
    plane, but requires zarr and may still be slow for very large compressed TIFFs.
    """
    try:
        import zarr
    except ImportError as exc:
        raise RuntimeError(
            "No usable TIFF pyramid level was found and zarr is not installed, "
            "so the full-resolution image cannot be safely downsampled for plotting."
        ) from exc

    store = tif.aszarr(series=series_index, level=level_index)
    try:
        arr = zarr.open(store, mode="r")
        source = tif.series[series_index].levels[level_index]
        axes = getattr(source, "axes", None)

        if axes and len(axes) == arr.ndim:
            key = []
            for axis, size in zip(axes, arr.shape):
                if axis in {"Y", "X"}:
                    key.append(slice(None, None, step))
                elif axis in {"C", "S"}:
                    key.append(min(channel, size - 1))
                else:
                    key.append(0)
            return np.asarray(arr[tuple(key)])

        if arr.ndim == 2:
            return np.asarray(arr[::step, ::step])
        if arr.ndim == 3 and arr.shape[-1] <= 4:
            return np.asarray(arr[::step, ::step, min(channel, arr.shape[-1] - 1)])
        if arr.ndim == 3 and arr.shape[0] <= 16:
            return np.asarray(arr[min(channel, arr.shape[0] - 1), ::step, ::step])

        raise ValueError(f"Cannot infer zarr slice for array shape {arr.shape}")
    finally:
        close = getattr(store, "close", None)
        if close is not None:
            close()


def load_palom_transform_npz(transform_npz):
    data = np.load(transform_npz)

    required = ["block_affine_matrices", "scale", "ref_shape", "moving_shape"]
    for key in required:
        if key not in data.files:
            raise KeyError(f"Missing key '{key}' in {transform_npz}. Found keys: {data.files}")

    mxs = np.asarray(data["block_affine_matrices"], dtype=float)
    meta = get_transform_shapes_and_scale(data)
    scale = meta["scale"]
    ref_shape = meta["ref_shape_fullres"]
    moving_shape = meta["moving_shape_fullres"]

    if mxs.ndim == 4:
        n_block_y, n_block_x, a, b = mxs.shape
        if (a, b) != (3, 3):
            raise ValueError(f"Expected last dimensions of mxs to be 3x3, got {mxs.shape}")
        layout = "4d"

    elif mxs.ndim == 2:
        if mxs.shape[0] % 3 != 0 or mxs.shape[1] % 3 != 0:
            raise ValueError(
                f"2D block_affine_matrices must have shape divisible by 3, got {mxs.shape}"
            )

        n_block_y = mxs.shape[0] // 3
        n_block_x = mxs.shape[1] // 3
        layout = "2d_tiled"

    else:
        raise ValueError(f"Unexpected block_affine_matrices ndim={mxs.ndim}, shape={mxs.shape}")

    print(f"[INFO] Loaded PALOM transform: {transform_npz}")
    print(f"[INFO] block_affine_matrices shape: {mxs.shape}")
    print(f"[INFO] inferred block grid: {n_block_y} x {n_block_x}")
    print(f"[INFO] scale: {scale}")
    print(f"[INFO] ref_shape full-res YX: {ref_shape}")
    print(f"[INFO] moving_shape full-res YX: {moving_shape}")
    print(f"[INFO] shape convention: {meta['shape_convention']}")

    return {
        "mxs": mxs,
        "scale": scale,
        "ref_shape": ref_shape,
        "moving_shape": moving_shape,
        "n_block_y": n_block_y,
        "n_block_x": n_block_x,
        "layout": layout,
    }


def get_local_affine_matrix(mxs, by, bx, layout):
    if layout == "4d":
        M = mxs[by, bx]

    elif layout == "2d_tiled":
        M = mxs[
            by * 3 : by * 3 + 3,
            bx * 3 : bx * 3 + 3,
        ]

    else:
        raise ValueError(f"Unknown PALOM matrix layout: {layout}")

    M = np.asarray(M, dtype=float)

    if M.shape != (3, 3):
        raise ValueError(f"Expected local affine matrix shape (3, 3), got {M.shape}")

    return M


def warp_points_with_palom_transform(
    xy_pixels,
    transform_npz,
    direction="moving_to_ref",
    output_fullres=True,
    post_shift_xy=(0, 0),
    warp_mode="local",
):
    data = np.load(transform_npz)

    mxs = np.asarray(data["block_affine_matrices"], dtype=float)
    meta = get_transform_shapes_and_scale(data)
    scale = meta["scale"]
    ref_shape = meta["ref_shape_fullres"]
    moving_shape = meta["moving_shape_fullres"]
    ref_shape_level = meta["ref_shape_level"]

    xy = np.asarray(xy_pixels, dtype=float)
    xy_level = xy / scale

    if warp_mode == "affine_only":
        if "affine_matrix" in data.files:
            mxs = np.asarray(data["affine_matrix"], dtype=float)
            print("[INFO] Using affine_matrix only, ignoring local block shifts", flush=True)
        elif "coarse_affine_matrix" in data.files:
            mxs = np.asarray(data["coarse_affine_matrix"], dtype=float)
            print("[INFO] Using coarse_affine_matrix only, ignoring local block shifts", flush=True)
        elif mxs.ndim == 4:
            mxs = np.asarray(mxs[0, 0], dtype=float)
            print(
                "[WARNING] affine_matrix not found; using first local block matrix "
                "for affine_only test",
                flush=True,
            )
        elif mxs.ndim == 2 and mxs.shape[0] >= 3 and mxs.shape[1] >= 3:
            mxs = np.asarray(mxs[:3, :3], dtype=float)
            print(
                "[WARNING] affine_matrix not found; using first tiled block matrix "
                "for affine_only test",
                flush=True,
            )
        else:
            raise ValueError("Cannot infer affine-only matrix from transform NPZ")

    if warp_mode == "affine_only":
        n_block_y = 1
        n_block_x = 1
        layout = "single"
    elif mxs.ndim == 2:
        if mxs.shape[0] % 3 != 0 or mxs.shape[1] % 3 != 0:
            raise ValueError(f"Unexpected block_affine_matrices shape: {mxs.shape}")

        n_block_y = mxs.shape[0] // 3
        n_block_x = mxs.shape[1] // 3
        layout = "2d_tiled"

    elif mxs.ndim == 4:
        n_block_y, n_block_x, a, b = mxs.shape
        if (a, b) != (3, 3):
            raise ValueError(f"Unexpected block_affine_matrices shape: {mxs.shape}")
        layout = "4d"

    else:
        raise ValueError(f"Unexpected block_affine_matrices shape: {mxs.shape}")

    print(f"[INFO] block_affine_matrices shape: {mxs.shape}")
    print(f"[INFO] inferred PALOM block grid: {n_block_y} x {n_block_x}")
    print(f"[INFO] ref_shape: {ref_shape}")
    print(f"[INFO] moving_shape: {moving_shape}")
    print(f"[INFO] ref_shape_registration: {meta['ref_shape_level']}")
    print(f"[INFO] moving_shape_registration: {meta['moving_shape_level']}")
    print(f"[INFO] scale: {scale}")
    print(f"[INFO] shape convention: {meta['shape_convention']}")
    print(f"[INFO] warp mode: {warp_mode}")

    linear_mxs = np.asarray(mxs, dtype=float)
    if warp_mode == "affine_only":
        flat_mxs = linear_mxs.reshape(1, 3, 3)
    elif layout == "4d":
        flat_mxs = linear_mxs.reshape(-1, 3, 3)
    else:
        flat_mxs = (
            linear_mxs.reshape(n_block_y, 3, n_block_x, 3)
            .transpose(0, 2, 1, 3)
            .reshape(-1, 3, 3)
        )
    dets = np.linalg.det(flat_mxs[:, :2, :2])
    print(
        "[INFO] Transform determinant percentiles: "
        f"{np.nanpercentile(dets, [0, 1, 50, 99, 100])}",
        flush=True,
    )

    # PALOM's block grid is defined in registration-level reference/output
    # coordinates. Full-resolution points are divided by scale before lookup.
    ref_y, ref_x = ref_shape_level[:2]
    block_h = ref_y / n_block_y
    block_w = ref_x / n_block_x
    print(
        f"[INFO] PALOM block size at registration level: "
        f"{block_h:.2f} x {block_w:.2f} pixels",
        flush=True,
    )

    warped = np.full_like(xy_level, np.nan, dtype=float)

    for i, (x, y) in enumerate(xy_level):
        if not np.isfinite(x) or not np.isfinite(y):
            continue

        bx = int(np.floor(x / block_w))
        by = int(np.floor(y / block_h))

        bx = np.clip(bx, 0, n_block_x - 1)
        by = np.clip(by, 0, n_block_y - 1)

        if warp_mode == "affine_only":
            M = mxs
        else:
            M = get_local_affine_matrix(mxs, by, bx, layout)

        if direction == "moving_to_ref":
            T = M
        elif direction == "ref_to_moving":
            T = np.linalg.inv(M)
        else:
            raise ValueError("direction must be 'moving_to_ref' or 'ref_to_moving'")

        p = T @ np.array([x, y, 1.0])
        warped[i] = p[:2]

    if output_fullres:
        warped *= scale

    warped += np.asarray(post_shift_xy, dtype=float)

    finite = np.isfinite(warped).all(axis=1)
    if np.any(finite):
        target_shape = moving_shape if direction == "ref_to_moving" else ref_shape
        target_y, target_x = target_shape[:2]
        warped_finite = warped[finite]
        in_bounds = (
            (warped_finite[:, 0] >= 0)
            & (warped_finite[:, 0] < target_x)
            & (warped_finite[:, 1] >= 0)
            & (warped_finite[:, 1] < target_y)
        )
        x_min, x_max = np.nanpercentile(warped_finite[:, 0], [0, 100])
        y_min, y_max = np.nanpercentile(warped_finite[:, 1], [0, 100])
        x_p1, x_p99 = np.nanpercentile(warped_finite[:, 0], [1, 99])
        y_p1, y_p99 = np.nanpercentile(warped_finite[:, 1], [1, 99])
        print(
            "[INFO] Warped full-resolution coordinate range: "
            f"x={x_min:.1f}..{x_max:.1f} "
            f"y={y_min:.1f}..{y_max:.1f}",
            flush=True,
        )
        print(
            "[INFO] Warped full-resolution 1-99 percentiles: "
            f"x={x_p1:.1f}..{x_p99:.1f} "
            f"y={y_p1:.1f}..{y_p99:.1f}",
            flush=True,
        )
        print(
            "[INFO] Target image shape for this direction: "
            f"Y={target_y:.1f}, X={target_x:.1f}; "
            f"in-bounds points={100 * np.mean(in_bounds):.2f}%",
            flush=True,
        )

    return warped


def read_xenium_bundle_as_adata(bundle_path):
    h5_path = os.path.join(bundle_path, "cell_feature_matrix.h5")
    cells_path = os.path.join(bundle_path, "cells.csv.gz")

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Cannot find {h5_path}")

    if not os.path.exists(cells_path):
        raise FileNotFoundError(f"Cannot find {cells_path}")

    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()

    cells = pd.read_csv(cells_path)

    if "cell_id" not in cells.columns:
        raise ValueError("cells.csv.gz must contain column 'cell_id'")

    cells = cells.set_index("cell_id")

    common = adata.obs_names.intersection(cells.index)
    adata = adata[common].copy()
    cells = cells.loc[common]

    adata.obs = adata.obs.join(cells)

    if {"x_centroid", "y_centroid"}.issubset(cells.columns):
        adata.obsm["spatial_microns"] = cells[["x_centroid", "y_centroid"]].to_numpy()
    else:
        raise ValueError("cells.csv.gz must contain x_centroid and y_centroid")

    return adata


def prepare_image_for_plot(
    image_path,
    downscale=10,
    channel=0,
    reader="tifffile",
    max_plot_pixels=16_000_000,
):
    """
    Read a safely downsampled plane for plotting.

    Default to tifffile pyramid/zarr reading. PALOM is useful for registration,
    but for plotting it can spend a long time computing without printing
    anything. The important rule here is to choose a low-resolution source
    before reading pixels into memory.
    """
    downscale = max(1, int(downscale))

    if reader == "palom":
        try:
            import palom

            print("[INFO] Plot reader: PALOM", flush=True)
            reader_obj = palom.reader.OmePyramidReader(image_path)
            downsamples = _coerce_level_downsamples(
                getattr(reader_obj, "level_downsamples", None)
            )

            if downsamples is not None and downsamples.size:
                usable = np.where(downsamples >= downscale)[0]
                if usable.size:
                    level = int(usable[np.argmin(downsamples[usable])])
                else:
                    level = int(np.argmax(downsamples))
                actual_downscale = float(downsamples[level])
            else:
                level = 0
                actual_downscale = 1.0

            print(
                f"[INFO] PALOM plot level={level} "
                f"level_downscale={actual_downscale:.3g}",
                flush=True,
            )

            try:
                img = reader_obj.read_level_channels(level, channel)
            except TypeError:
                img = reader_obj.read_level_channels(level, [channel])

            if hasattr(img, "compute"):
                img = img.compute()

            img = _extract_channel(img, channel=channel)
            extra_downscale = max(1, int(np.ceil(downscale / max(actual_downscale, 1e-9))))
            img = img[::extra_downscale, ::extra_downscale]
            effective_downscale = actual_downscale * extra_downscale

            return _normalize_for_plot(img), effective_downscale

        except Exception as e:
            print(f"[WARNING] PALOM reader failed for plotting: {e}", flush=True)
            print("[WARNING] Falling back to tifffile pyramid reading.", flush=True)

    print("[INFO] Plot reader: tifffile pyramid/zarr", flush=True)

    with tifffile.TiffFile(image_path) as tif:
        base_series = tif.series[0]
        base_yx = _yx_shape(base_series.shape, getattr(base_series, "axes", None))

        candidates = []

        for level_i, level in enumerate(getattr(base_series, "levels", [base_series])):
            level_yx = _yx_shape(level.shape, getattr(level, "axes", None))
            level_down = float(np.nanmean(base_yx / level_yx))
            candidates.append(
                {
                    "series_i": 0,
                    "level_i": level_i,
                    "source": level,
                    "downsample": level_down,
                    "shape": level_yx,
                    "label": f"series[0].levels[{level_i}]",
                }
            )

        for series_i, series in enumerate(tif.series[1:], start=1):
            try:
                series_yx = _yx_shape(series.shape, getattr(series, "axes", None))
            except ValueError:
                continue
            series_down = float(np.nanmean(base_yx / series_yx))
            candidates.append(
                {
                    "series_i": series_i,
                    "level_i": 0,
                    "source": series,
                    "downsample": series_down,
                    "shape": series_yx,
                    "label": f"series[{series_i}]",
                }
            )

        if not candidates:
            raise ValueError(f"No readable image series found in {image_path}")

        at_or_above = [c for c in candidates if c["downsample"] >= downscale]
        if at_or_above:
            selected = min(at_or_above, key=lambda c: c["downsample"])
        else:
            selected = max(candidates, key=lambda c: c["downsample"])

        actual_downscale = max(float(selected["downsample"]), 1.0)
        extra_downscale = max(1, int(np.ceil(downscale / actual_downscale)))

        output_yx = selected["shape"] / extra_downscale
        output_pixels = float(np.prod(output_yx))
        if max_plot_pixels and output_pixels > max_plot_pixels:
            extra_factor = int(np.ceil(np.sqrt(output_pixels / max_plot_pixels)))
            extra_downscale *= max(1, extra_factor)

        effective_downscale = actual_downscale * extra_downscale

        print(
            "[INFO] Plot image source: "
            f"{selected['label']} shape={tuple(selected['shape'].astype(int))} "
            f"level_downscale={actual_downscale:.3g} "
            f"extra_downscale={extra_downscale} "
            f"effective_downscale={effective_downscale:.3g}",
            flush=True,
        )

        if extra_downscale > 1:
            try:
                print("[INFO] Reading decimated image through zarr", flush=True)
                img = _read_tiff_zarr_decimated(
                    tif,
                    series_index=selected["series_i"],
                    level_index=selected["level_i"],
                    channel=channel,
                    step=extra_downscale,
                )
            except Exception:
                if selected["downsample"] <= 1.01:
                    raise
                print(
                    "[WARNING] zarr decimated read failed; "
                    "reading selected pyramid level directly",
                    flush=True,
                )
                img = selected["source"].asarray()
                img = _extract_channel(img, channel=channel)
                img = img[::extra_downscale, ::extra_downscale]
        else:
            print("[INFO] Reading selected pyramid level directly", flush=True)
            img = selected["source"].asarray()
            img = _extract_channel(img, channel=channel)

    print(f"[INFO] Plot image loaded at shape={img.shape}", flush=True)
    return _normalize_for_plot(img), effective_downscale


def plot_warped_points(
    image_path,
    xy_pixels,
    out_png,
    downscale=10,
    point_size=0.2,
    channel=0,
    max_plot_points=0,
    plot_reader="tifffile",
    max_plot_pixels=16_000_000,
    extent_shape_yx=None,
):
    print("[INFO] Preparing plot image", flush=True)
    img, effective_downscale = prepare_image_for_plot(
        image_path,
        downscale=downscale,
        channel=channel,
        reader=plot_reader,
        max_plot_pixels=max_plot_pixels,
    )

    if extent_shape_yx is None:
        xy_plot = xy_pixels / effective_downscale
        image_extent = None
        coord_label = "downsampled"
    else:
        extent_y, extent_x = np.asarray(extent_shape_yx, dtype=float)[:2]
        xy_plot = xy_pixels
        image_extent = (0, extent_x, extent_y, 0)
        coord_label = "full-resolution"

    valid = np.isfinite(xy_plot).all(axis=1)
    xy_plot = xy_plot[valid]
    if xy_plot.shape[0]:
        print(
            f"[INFO] Plot coordinate range ({coord_label}): "
            f"x={np.nanmin(xy_plot[:, 0]):.1f}..{np.nanmax(xy_plot[:, 0]):.1f} "
            f"y={np.nanmin(xy_plot[:, 1]):.1f}..{np.nanmax(xy_plot[:, 1]):.1f}; "
            f"image shape={img.shape}; "
            f"effective_downscale={effective_downscale:.3g}; "
            f"image_extent={image_extent}",
            flush=True,
        )

    if max_plot_points and xy_plot.shape[0] > max_plot_points:
        print(
            f"[INFO] Subsampling plotted cells: {xy_plot.shape[0]} -> {max_plot_points}",
            flush=True,
        )
        rng = np.random.default_rng(0)
        keep = rng.choice(xy_plot.shape[0], size=max_plot_points, replace=False)
        xy_plot = xy_plot[keep]

    print(f"[INFO] Rendering plot with {xy_plot.shape[0]} points", flush=True)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img, cmap="gray", extent=image_extent)
    if xy_plot.shape[0] > 500_000:
        ax.plot(
            xy_plot[:, 0],
            xy_plot[:, 1],
            linestyle="None",
            marker=",",
            alpha=0.7,
        )
    else:
        ax.scatter(
            xy_plot[:, 0],
            xy_plot[:, 1],
            s=point_size,
            alpha=0.7,
            linewidths=0,
        )
    ax.axis("off")
    plt.tight_layout()
    print(f"[INFO] Saving plot: {out_png}", flush=True)
    fig.savefig(out_png, dpi=200)
    plt.close()


def safe_filename(x):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x))


def process_one_sample(row, args):
    sample = row[args.sample_col]
    bundle_path = row[args.bundle_col]
    transform_npz = row[args.transform_col]

    print(f"[INFO] Processing {sample}")

    adata = read_xenium_bundle_as_adata(bundle_path)

    if args.pixel_size_um_col in row and not pd.isna(row[args.pixel_size_um_col]):
        pixel_size_um = float(row[args.pixel_size_um_col])
    else:
        pixel_size_um = float(args.default_pixel_size_um)

    xy_microns = adata.obsm["spatial_microns"]
    xy_pixels = xy_microns / pixel_size_um

    adata.obsm["spatial_pixels"] = xy_pixels

    warped_pixels = warp_points_with_palom_transform(
        xy_pixels=xy_pixels,
        transform_npz=transform_npz,
        direction=args.direction,
        output_fullres=True,
        post_shift_xy=(args.post_shift_x, args.post_shift_y),
        warp_mode=args.warp_mode,
    )
    warped_microns = warped_pixels * pixel_size_um

    adata.obsm[args.out_pixels_key] = warped_pixels
    adata.obsm[args.out_microns_key] = warped_microns

    os.makedirs(args.out_dir, exist_ok=True)

    sample_safe = safe_filename(sample)

    out_h5ad = os.path.join(args.out_dir, f"{sample_safe}_palom_warped.h5ad")
    adata.write_h5ad(out_h5ad)
    print(f"[INFO] Saved AnnData: {out_h5ad}")

    if args.skip_plots:
        return

    if args.image_col in row and isinstance(row[args.image_col], str) and row[args.image_col]:
        image_path = row[args.image_col]
        out_png = os.path.join(args.out_dir, f"{sample_safe}_warped_points_overlay.png")

        try:
            with np.load(transform_npz) as transform_data:
                transform_meta = get_transform_shapes_and_scale(transform_data)

            if args.direction == "ref_to_moving":
                plot_extent_shape = transform_meta["moving_shape_fullres"]
            else:
                plot_extent_shape = transform_meta["ref_shape_fullres"]

            print(f"[INFO] Starting verification plot: {out_png}", flush=True)
            plot_warped_points(
                image_path=image_path,
                xy_pixels=warped_pixels,
                out_png=out_png,
                downscale=args.plot_downscale,
                point_size=args.point_size,
                channel=args.plot_channel,
                max_plot_points=args.max_plot_points,
                plot_reader=args.plot_reader,
                max_plot_pixels=args.max_plot_pixels,
                extent_shape_yx=plot_extent_shape,
            )

            print(f"[INFO] Saved plot: {out_png}")

        except Exception as e:
            print(f"[WARNING] AnnData was saved, but plotting failed for {sample}: {e}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--sample_col", default="Name")
    parser.add_argument("--bundle_col", default="xenium_bundle")
    parser.add_argument("--transform_col", default="palom_transform_npz")
    parser.add_argument("--image_col", default="image_path")
    parser.add_argument("--pixel_size_um_col", default="pixel_size_um")

    parser.add_argument("--default_pixel_size_um", type=float, default=0.2125)

    parser.add_argument(
        "--direction",
        default="moving_to_ref",
        choices=["moving_to_ref", "ref_to_moving"],
    )
    parser.add_argument(
        "--warp_mode",
        default="local",
        choices=["local", "affine_only"],
        help="Use PALOM local block matrices, or only the global affine for debugging.",
    )

    parser.add_argument("--out_pixels_key", default="spatial_palom_pixels")
    parser.add_argument("--out_microns_key", default="spatial_palom_microns")

    parser.add_argument("--plot_downscale", type=int, default=10)
    parser.add_argument("--plot_channel", type=int, default=0)
    parser.add_argument("--point_size", type=float, default=0.2)
    parser.add_argument(
        "--plot_reader",
        choices=["tifffile", "palom"],
        default="tifffile",
        help="Reader used only for verification plots. tifffile uses pyramid/zarr.",
    )
    parser.add_argument(
        "--max_plot_pixels",
        type=int,
        default=16_000_000,
        help="Maximum image pixels in the verification plot after downsampling.",
    )
    parser.add_argument(
        "--max_plot_points",
        type=int,
        default=250_000,
        help="Plot a reproducible random subset of cells; 0 plots all cells.",
    )
    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="Save warped AnnData only and skip verification plot generation.",
    )

    parser.add_argument("--post_shift_x", type=float, default=0.0)
    parser.add_argument("--post_shift_y", type=float, default=0.0)

    args = parser.parse_args()

    table = pd.read_csv(args.csv)

    for _, row in table.iterrows():
        process_one_sample(row, args)


if __name__ == "__main__":
    main()
