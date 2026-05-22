"""
Joins the three datasets and engineers features for the injury prediction model.

On AWS this step would run as a Glue job, with outputs written to S3/Redshift.
"""

import pandas as pd
import numpy as np


def build_injury_feature_set(plays: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join PlayList onto InjuryRecord so every injured play gets its
    context features (weather, surface, position, workload, etc.).
    """
    df = injuries.merge(
        plays[[
            "PlayKey", "RosterPosition", "PositionGroup",
            "StadiumType", "FieldType", "Temperature",
            "Weather", "PlayType", "PlayerDay",
            "PlayerGame", "PlayerGamePlay",
        ]],
        on="PlayKey",
        how="left",
    )

    df = _encode_surface(df)
    df = _encode_field_type(df)
    df = _encode_stadium_type(df)
    df = _encode_play_type(df)
    df = _encode_body_part(df)
    df = _encode_position_group(df)
    df = _impute_temperature(df)
    df = _add_workload_features(df)

    return df


def _encode_surface(df: pd.DataFrame) -> pd.DataFrame:
    df["is_synthetic"] = (df["Surface"] == "Synthetic").astype(int)
    return df


def _encode_field_type(df: pd.DataFrame) -> pd.DataFrame:
    # FieldType comes from PlayList; Surface from InjuryRecord — keep both
    if "FieldType" in df.columns:
        df["field_synthetic"] = (df["FieldType"] == "Synthetic").astype(int)
    return df


def _encode_stadium_type(df: pd.DataFrame) -> pd.DataFrame:
    if "StadiumType" in df.columns:
        df["is_indoor"] = df["StadiumType"].str.lower().str.contains("indoor|dome|closed", na=False).astype(int)
    return df


def _encode_play_type(df: pd.DataFrame) -> pd.DataFrame:
    if "PlayType" in df.columns:
        dummies = pd.get_dummies(df["PlayType"], prefix="play", drop_first=False)
        df = pd.concat([df, dummies], axis=1)
    return df


def _encode_body_part(df: pd.DataFrame) -> pd.DataFrame:
    dummies = pd.get_dummies(df["BodyPart"], prefix="body", drop_first=False)
    return pd.concat([df, dummies], axis=1)


def _encode_position_group(df: pd.DataFrame) -> pd.DataFrame:
    if "PositionGroup" in df.columns:
        valid = df["PositionGroup"].replace("Missing Data", np.nan)
        dummies = pd.get_dummies(valid, prefix="pos", drop_first=False)
        df = pd.concat([df, dummies], axis=1)
    return df


def _impute_temperature(df: pd.DataFrame) -> pd.DataFrame:
    if "Temperature" in df.columns:
        # Closed-dome stadiums have no meaningful outdoor temp — fill with median
        median_temp = df.loc[df["is_indoor"] == 0, "Temperature"].median()
        df["Temperature"] = df["Temperature"].fillna(median_temp)
    return df


def _add_workload_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Player workload proxies: days into the season and plays in the current game.
    Higher workload is expected to correlate with injury severity.
    """
    if "PlayerDay" in df.columns:
        df["PlayerDay"] = pd.to_numeric(df["PlayerDay"], errors="coerce").fillna(0)
    if "PlayerGamePlay" in df.columns:
        df["PlayerGamePlay"] = pd.to_numeric(df["PlayerGamePlay"], errors="coerce").fillna(0)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Returns the final numeric feature column names."""
    exclude = {
        "PlayerKey", "GameID", "PlayKey",
        "BodyPart", "Surface", "FieldType",
        "StadiumType", "Weather", "PlayType",
        "RosterPosition", "Position", "PositionGroup",
        "PlayerGame",
        "DM_M1", "DM_M7", "DM_M28", "DM_M42",
    }
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


if __name__ == "__main__":
    from src.ingestion.data_loader import load_play_list, load_injury_records

    plays = load_play_list()
    injuries = load_injury_records()
    features = build_injury_feature_set(plays, injuries)

    print(f"Feature set shape: {features.shape}")
    print("Feature columns:", get_feature_columns(features))
