import numpy as np
import pandas as pd
import sys
import os

def npy_to_csv(npy_path: str, csv_path: str, add_headers: bool = True) -> None:
    arr = np.load(npy_path)

    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, but got {arr.ndim}D with shape {arr.shape}")

    if add_headers:
        columns = [f"feature_{i}" for i in range(arr.shape[1])]
        df = pd.DataFrame(arr, columns=columns)
    else:
        df = pd.DataFrame(arr)

    df.to_csv(csv_path, index=False)
    print(f"✅ Saved {csv_path} with shape {arr.shape}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python npy_to_csv.py <input.npy> <output.csv>")
        sys.exit(1)

    npy_file = sys.argv[1]
    csv_file = sys.argv[2]

    # run conversion
    npy_to_csv(npy_file, csv_file, add_headers=True)
