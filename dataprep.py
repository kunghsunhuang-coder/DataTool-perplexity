import tempfile
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils import get_selected_df, safe_to_list, extract_values

def data_precheck(file, sheet, cols, material_col, missing_strategy, outlier_strategy, scaler_type):
    sheet = extract_values(sheet)
    if not file or not sheet or not cols:
        return "請選擇檔案、分頁、欄位！", None, None, None

    df = get_selected_df(file, sheet)
    cols = safe_to_list(cols)
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        return "請至少選擇一個數值欄位！", None, None, None

    report_lines = []
    stat_before = df[cols].agg(['min', 'max', 'mean', 'std']).round(4)

    # 缺值處理
    if df[cols].isna().sum().sum() > 0:
        if missing_strategy == "刪除含缺值之列":
            df = df.dropna(subset=cols)
            report_lines.append("缺值已刪除")
        elif missing_strategy == "以欄均值插補":
            df[cols] = df[cols].fillna(df[cols].mean())
            report_lines.append("缺值以均值補足")
        elif missing_strategy == "以欄中位數插補":
            df[cols] = df[cols].fillna(df[cols].median())
            report_lines.append("缺值以中位數補足")

    # 離群值處理
    z = (df[cols] - df[cols].mean()) / df[cols].std(ddof=0)
    if outlier_strategy == "自動剔除":
        df = df[(z.abs() <= 3).all(axis=1)]
        report_lines.append("離群值已剔除")
    elif outlier_strategy == "標註但保留":
        df["outlier_flag"] = z.abs().max(axis=1) > 3
        report_lines.append("已標註outlier_flag")

    # 標準化
    if scaler_type == "Z-score標準化":
        scaler = StandardScaler()
        df[cols] = scaler.fit_transform(df[cols])
        report_lines.append("Z-score標準化完成")
    elif scaler_type == "MinMax歸一化":
        scaler = MinMaxScaler()
        df[cols] = scaler.fit_transform(df[cols])
        report_lines.append("MinMax歸一化完成")

    stat_after = df[cols].agg(['min', 'max', 'mean', 'std']).round(4)

    tmp_path = None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        df.to_excel(tmp, index=False)
        tmp_path = tmp.name

    return "<br>".join(report_lines), df.head(10), tmp_path, tmp_path
