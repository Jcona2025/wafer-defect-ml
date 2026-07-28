"""Loading and preparation for the WM-811K (LSWMD) wafer map dataset.

Wafer map encoding: 0 = outside wafer, 1 = passing die, 2 = failing die.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_PKL = DATA_DIR / "raw" / "MIR-WM811K" / "Python" / "WM811K.pkl"
PROCESSED = DATA_DIR / "processed" / "labeled.npz"

CLASSES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc",
    "Near-full", "Random", "Scratch", "none",
]


def _unwrap(cell):
    """Labels are nested arrays like [['none']]; unlabeled wafers hold '0'.

    Flatten to str, mapping unlabeled to None."""
    arr = np.asarray(cell).ravel()
    if not arr.size or str(arr[0]) == "0":
        return None
    return str(arr[0])


def load_raw(path: Path = RAW_PKL) -> pd.DataFrame:
    df = pd.read_pickle(path)
    df["failureType"] = df["failureType"].map(_unwrap)
    df["trainTestLabel"] = df["trainTestLabel"].map(_unwrap)
    df["mapShape"] = df["waferMap"].map(lambda m: m.shape)
    return df


def resize_map(wafer_map: np.ndarray, size: int = 64) -> np.ndarray:
    """Nearest-neighbour resize to size x size, preserving the 0/1/2 encoding."""
    h, w = wafer_map.shape
    rows = (np.arange(size) * h / size).astype(int)
    cols = (np.arange(size) * w / size).astype(int)
    return wafer_map[np.ix_(rows, cols)]


def build_labeled_arrays(df: pd.DataFrame, size: int = 64,
                         out: Path = PROCESSED) -> tuple[np.ndarray, np.ndarray]:
    """Filter to labeled maps, resize, and cache as a compressed npz."""
    labeled = df[df["failureType"].notna()]
    X = np.stack([resize_map(m, size) for m in labeled["waferMap"]]).astype(np.uint8)
    y = labeled["failureType"].map(CLASSES.index).to_numpy(dtype=np.int64)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=y)
    return X, y


def load_labeled(path: Path = PROCESSED) -> tuple[np.ndarray, np.ndarray]:
    npz = np.load(path)
    return npz["X"], npz["y"]
