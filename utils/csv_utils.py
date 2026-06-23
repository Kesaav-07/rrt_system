"""
utils/csv_utils.py
------------------
CSV read/write helpers with safe atomic writes and schema enforcement.
"""

from __future__ import annotations

import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def safe_read_csv(path: str, **kwargs) -> pd.DataFrame:
    """
    Read a CSV, returning an empty DataFrame on any error.

    Args:
        path:   File path.
        kwargs: Extra arguments forwarded to pd.read_csv.

    Returns:
        DataFrame or empty DataFrame.
    """
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as exc:
        logger.warning(f"safe_read_csv failed for {path}: {exc}")
        return pd.DataFrame()


def atomic_write_csv(df: pd.DataFrame, path: str) -> None:
    """
    Write a DataFrame to CSV atomically via a temp file.

    Args:
        df:   DataFrame to save.
        path: Target file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    except Exception as exc:
        logger.error(f"atomic_write_csv failed for {path}: {exc}")
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def append_rows_csv(rows: list[dict], path: str, columns: list[str]) -> None:
    """
    Append rows to a CSV, creating it with headers if it doesn't exist.

    Args:
        rows:    List of row dicts.
        path:    Target CSV path.
        columns: Column order / schema.
    """
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new_df = pd.DataFrame(rows, columns=columns)
    header = not os.path.exists(path)
    try:
        new_df.to_csv(path, mode="a", index=False, header=header)
    except Exception as exc:
        logger.error(f"append_rows_csv failed for {path}: {exc}")
        raise
