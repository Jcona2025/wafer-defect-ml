"""Classical features for wafer map failure-pattern classification.

Feature groups follow Wu et al. (2015): regional fail densities, radon-transform
projection statistics, and geometry of the largest failing region. Together they
capture where failures sit on the wafer, their directional structure (scratches,
rings), and their contiguity (clusters vs. random) without any learned parameters.
"""

import numpy as np
from scipy import ndimage

N_REGION_RINGS = 5


def fail_mask(wafer_map: np.ndarray) -> np.ndarray:
    return wafer_map == 2


def on_wafer(wafer_map: np.ndarray) -> np.ndarray:
    return wafer_map > 0


def _radial_ring_densities(wafer_map: np.ndarray, n_rings: int = N_REGION_RINGS) -> list[float]:
    """Fail density in concentric rings from wafer center to edge."""
    h, w = wafer_map.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - (h - 1) / 2, xx - (w - 1) / 2)
    r_max = r[on_wafer(wafer_map)].max() + 1e-9
    fails, valid = fail_mask(wafer_map), on_wafer(wafer_map)
    out = []
    for i in range(n_rings):
        band = valid & (r >= r_max * i / n_rings) & (r < r_max * (i + 1) / n_rings)
        out.append(fails[band].mean() if band.any() else 0.0)
    return out


def _quadrant_densities(wafer_map: np.ndarray) -> list[float]:
    h, w = wafer_map.shape
    out = []
    for rows in (slice(0, h // 2), slice(h // 2, h)):
        for cols in (slice(0, w // 2), slice(w // 2, w)):
            sub, valid = wafer_map[rows, cols], on_wafer(wafer_map[rows, cols])
            out.append((sub[valid] == 2).mean() if valid.any() else 0.0)
    return out


def _radon_stats(wafer_map: np.ndarray, n_angles: int = 20) -> list[float]:
    """Mean/std of projection profiles at several angles.

    A scratch produces one angle with a sharp, high-variance projection; a ring
    is rotationally symmetric (low variance across angles)."""
    fails = fail_mask(wafer_map).astype(float)
    profiles = []
    for angle in np.linspace(0, 180, n_angles, endpoint=False):
        rotated = ndimage.rotate(fails, angle, reshape=False, order=0)
        profiles.append(rotated.sum(axis=0))
    proj = np.array(profiles)
    per_angle_std = proj.std(axis=1)
    return [
        float(per_angle_std.mean()), float(per_angle_std.std()),
        float(per_angle_std.max()), float(proj.max()),
    ]


def _geometry_stats(wafer_map: np.ndarray) -> list[float]:
    """Size/shape of the largest connected failing region."""
    fails = fail_mask(wafer_map)
    n_fail = int(fails.sum())
    if n_fail == 0:
        return [0.0] * 5
    labels, n_regions = ndimage.label(fails)
    sizes = ndimage.sum_labels(fails, labels, index=range(1, n_regions + 1))
    largest = int(sizes.max())
    slc = ndimage.find_objects(labels == np.argmax(sizes) + 1)[0]
    bbox_h, bbox_w = slc[0].stop - slc[0].start, slc[1].stop - slc[1].start
    aspect = max(bbox_h, bbox_w) / max(1, min(bbox_h, bbox_w))
    return [
        float(n_regions),
        largest / n_fail,                    # dominance of biggest cluster
        largest / fails.size,
        float(aspect),                       # elongation (high for scratches)
        n_fail / max(1, int(on_wafer(wafer_map).sum())),  # overall fail rate
    ]


FEATURE_NAMES = (
    [f"ring_density_{i}" for i in range(N_REGION_RINGS)]
    + [f"quadrant_density_{i}" for i in range(4)]
    + ["radon_std_mean", "radon_std_std", "radon_std_max", "radon_proj_max"]
    + ["n_regions", "largest_region_frac", "largest_region_area",
       "largest_region_aspect", "fail_rate"]
)


def extract_features(wafer_map: np.ndarray) -> np.ndarray:
    return np.array(
        _radial_ring_densities(wafer_map)
        + _quadrant_densities(wafer_map)
        + _radon_stats(wafer_map)
        + _geometry_stats(wafer_map),
        dtype=np.float32,
    )
