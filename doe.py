import numpy as np
import pandas as pd
from itertools import product
from pyDOE2 import lhs, bbdesign, ccdesign, fracfact, pbdesign
from scipy.stats.qmc import Sobol, Halton

def snap_to_step(val, min_val, step):
    return round((val - min_val) / step) * step + min_val if step > 0 else val

def gen_design(method, param_names, mins, maxs, steps=None, n_sample=None, mixture_tot=None, seed=None):
    n_var = len(param_names)
    steps = steps or [3] * n_var
    mins, maxs = np.array(mins), np.array(maxs)
    X_real = None

    def snap_array(X):
        Xs = np.copy(X)
        for i in range(X.shape[1]):
            if steps[i] > 0:
                Xs[:, i] = [snap_to_step(v, mins[i], steps[i]) for v in Xs[:, i]]
        return Xs

    if method == "Latin Hypercube":
        if seed is not None: np.random.seed(seed)
        X = lhs(n_var, samples=n_sample)
        X_real = X * (maxs - mins) + mins
    elif method == "Sobol":
        sobol_engine = Sobol(d=n_var, scramble=True, seed=seed)
        X_real = sobol_engine.random(n_sample) * (maxs - mins) + mins
    elif method == "Halton":
        halton_engine = Halton(d=n_var, scramble=True, seed=seed)
        X_real = halton_engine.random(n_sample) * (maxs - mins) + mins
    elif method == "Full Factorial":
        grid_axes = [np.linspace(mins[i], maxs[i], int(steps[i])) for i in range(n_var)]
        X_real = np.array(list(product(*grid_axes)))
    elif method == "Fractional Factorial":
        fact_str = ' '.join([chr(ord('a') + i) for i in range(n_var)])
        X = fracfact(fact_str)
        X_real = (X + 1) / 2 * (maxs - mins) + mins
    elif method == "Plackett-Burman":
        min_n = n_var + 1
        valid_runs = [4, 8, 12, 16, 20, 24, 28, 32]
        n_runs = next((v for v in valid_runs if v >= min_n), None)
        pb_mat = pbdesign(n_runs)
        X_real = (pb_mat[:, :n_var] + 1) / 2 * (maxs - mins) + mins
    elif method == "Box-Behnken":
        X = bbdesign(n_var)
        X_real = (X + 1) / 2 * (maxs - mins) + mins
    elif method == "Central Composite":
        X = ccdesign(n_var)
        X_real = (X + 1) / 2 * (maxs - mins) + mins

    if X_real is not None and method != "Mixture Design":
        X_real = snap_array(X_real)

    return pd.DataFrame(X_real, columns=param_names)

def taguchi_array(levels, l='L9'):
    """
    回傳指定 Taguchi 直交表的設計點
    levels: 每個因子的水準數（通常全2或全3）
    l: 指定 Taguchi 表型號, 如 'L4','L8','L9','L12'...
    目前僅舉例 L4/L8/L9
    """
    import pandas as pd
    taguchi_tables = {
        'L4': [[1,1],[1,2],[2,1],[2,2]],
        'L8': [
            [1,1,1],[1,2,2],[1,1,2],[1,2,1],
            [2,1,1],[2,2,2],[2,1,2],[2,2,1]
        ],
        'L9': [
            [1,1,1],[1,2,2],[1,3,3],
            [2,1,2],[2,2,3],[2,3,1],
            [3,1,3],[3,2,1],[3,3,2]
        ]
        # 更多請自行查表擴充
    }
    arr = taguchi_tables.get(l)
    if not arr:
        raise ValueError(f"暫不支援 {l} 直交表")
    cols = [f'因子{i+1}' for i in range(len(arr[0]))]
    df = pd.DataFrame(arr, columns=cols)
    # 可選用 levels 做水平值對應
    return df
