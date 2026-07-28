"""One-time preparation: load raw LSWMD.pkl, cache labeled maps as npz.

Usage: python -m src.prepare_data
"""

from . import data


def main():
    print(f"Loading {data.RAW_PKL} ...")
    df = data.load_raw()
    print(f"{len(df):,} wafer maps, {df['failureType'].notna().sum():,} labeled")
    X, y = data.build_labeled_arrays(df)
    print(f"Cached {X.shape} maps -> {data.PROCESSED}")


if __name__ == "__main__":
    main()
