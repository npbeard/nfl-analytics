"""
Data loader for NFL datasets.

In production this would pull from S3 (batch) or Kinesis (real-time).
Locally it reads the CSV files provided in Appendix 3 of the RFP.
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2]


def load_injury_records(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA_DIR / "InjuryRecord.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    expected = {"PlayerKey", "GameID", "PlayKey", "BodyPart", "Surface",
                "DM_M1", "DM_M7", "DM_M28", "DM_M42"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"InjuryRecord missing columns: {missing}")

    for col in ["DM_M1", "DM_M7", "DM_M28", "DM_M42"]:
        df[col] = df[col].astype(int)

    return df


def load_play_list(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA_DIR / "PlayList.csv"
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")

    # Normalise free-text stadium/weather fields
    df["StadiumType"] = df["StadiumType"].str.strip().str.title()
    df["Weather"] = df["Weather"].str.strip()

    return df


def load_player_tracking(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA_DIR / "PlayerTrackData_43540.csv"
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # Parse the compound PlayKey into player / game / play IDs
    parts = df["PlayKey"].str.split("-", expand=True)
    df["PlayerKey"] = parts[0].astype(int)
    df["GameNum"] = parts[1].astype(int)
    df["PlayNum"] = parts[2].astype(int)

    df["event"] = df["event"].fillna("")

    return df


def load_all() -> dict[str, pd.DataFrame]:
    return {
        "injuries": load_injury_records(),
        "plays": load_play_list(),
        "tracking": load_player_tracking(),
    }


if __name__ == "__main__":
    data = load_all()
    for name, df in data.items():
        print(f"{name}: {df.shape[0]:,} rows × {df.shape[1]} cols")
