import pandas as pd

def class_vs_numeric_stat(df, class_col, num_cols):
    result = {}
    for c in num_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            stats = df.groupby(class_col)[c].agg(['mean','std','min','max','count']).round(4)
            result[c] = stats
    return result
