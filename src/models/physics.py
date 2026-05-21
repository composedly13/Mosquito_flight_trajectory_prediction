from pathlib import Path
import numpy as np
import pandas as pd

R_HIT = 0.01


def constant_velocity_predict(sample_path: Path) -> np.ndarray:
    df = pd.read_csv(sample_path)
    prev_xyz = df.loc[df.index[-2], ["x", "y", "z"]].to_numpy(dtype=float)
    last_xyz = df.loc[df.index[-1], ["x", "y", "z"]].to_numpy(dtype=float)
    return last_xyz + 2.0 * (last_xyz - prev_xyz)


def build_prediction_df(sample_files) -> pd.DataFrame:
    rows = []
    for sample_path in sample_files:
        pred_xyz = constant_velocity_predict(sample_path)
        rows.append({
            "id": Path(sample_path).stem,
            "x": pred_xyz[0],
            "y": pred_xyz[1],
            "z": pred_xyz[2],
        })
    return pd.DataFrame(rows)
