import pandas as pd


def fix_text(series: pd.Series) -> pd.Series:
    """清理字符串字段"""
    return series.astype(str).str.strip().str.replace(r"[^\u4e00-\u9fa5a-zA-Z0-9]", "", regex=True)


def fix_numeric(series: pd.Series) -> pd.Series:
    """处理数值字段，非数字转为 NaN"""
    return pd.to_numeric(series, errors="coerce")


def fix_datetime(series: pd.Series) -> pd.Series:
    """处理时间字段，非法时间转为 NaT"""
    return pd.to_datetime(series, errors="coerce")


def fill_missing(series: pd.Series) -> pd.Series:
    """填补缺失值，数值字段填中位数，其它填 '未知'"""
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(series.median())
    return series.fillna("未知")