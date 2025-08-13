import os, re
import pandas as pd
import numpy as np
import plotly.colors as pc

def safe_to_list(cols):
    if cols is None: return []
    if isinstance(cols, str): return [cols]
    if isinstance(cols, (tuple, np.ndarray)): cols = list(cols)
    if len(cols) == 1 and isinstance(cols[0], (list, tuple, np.ndarray)): cols = list(cols[0])
    return list(cols)

def extract_values(x):
    if hasattr(x, "value"): return x.value
    if isinstance(x, (list, tuple)): return list(x)
    if x is None: return []
    return [x]

def clean_filename(name):
    if not name: return ""
    return re.sub(r'[\\/*?:"<>|]', '_', str(name)).strip()

def get_file_path(file):
    if file is None: return None
    if hasattr(file, "name"): return file.name
    elif isinstance(file, dict) and "name" in file: return file["name"]
    elif isinstance(file, str): return file
    return None

def read_data(file):
    path = get_file_path(file)
    if path is None: return None, None, []
    ext = os.path.splitext(path)[-1].lower()
    try:
        if ext == ".csv":
            try:
                df = pd.read_csv(path, encoding="utf-8")
            except UnicodeDecodeError:
                try: df = pd.read_csv(path, encoding="big5")
                except Exception: df = pd.read_csv(path, encoding="cp950")
            return df, None, ["CSV檔"]
        elif ext in [".xls", ".xlsx"]:
            xl = pd.ExcelFile(path)
            return None, xl, xl.sheet_names
        else:
            raise ValueError("僅支援 CSV, XLS, XLSX 檔案")
    except:
        return None, None, []

def get_selected_df(file, sheet):
    sheet = extract_values(sheet)
    df, xl, sheet_names = read_data(file)
    if df is not None:
        return df
    elif xl is not None and sheet and sheet[0] in xl.sheet_names:
        return xl.parse(sheet[0])
    return pd.DataFrame()

def get_auto_palette(n):
    palette = pc.qualitative.Set1 + pc.qualitative.Pastel1 + pc.qualitative.Plotly
    return palette[:n] if n <= len(palette) else [palette[i % len(palette)] for i in range(n)]
