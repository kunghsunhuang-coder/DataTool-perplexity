import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


def train_and_predict_all2(df, feature_cols, target_col, model_type="RF", test_size=0.2, seed=None):
    from sklearn.model_selection import train_test_split
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    if model_type == "RF":
        model = RandomForestRegressor(n_estimators=100, random_state=seed)
    elif model_type == "LR":
        model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5   # 不傳 squared
    r2 = r2_score(y_test, y_pred)
    return {"rmse": rmse, "r2": r2, "model": model}



def train_and_predict_with_surface(df, features, target, model_name="RF", test_size=0.2, seed=42):
    # 支援多種模型
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    if model_name == "RF":
        model = RandomForestRegressor(n_estimators=100, random_state=seed)
    elif model_name == "LR":
        model = LinearRegression()
    elif model_name == "MLP":
        model = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=seed)
    elif model_name == "XGBoost":
        try:
            from xgboost import XGBRegressor
            model = XGBRegressor(random_state=seed)
        except ImportError:
            return {"error": "xgboost 未安裝"}
    elif model_name == "LightGBM":
        try:
            from lightgbm import LGBMRegressor
            model = LGBMRegressor(random_state=seed)
        except ImportError:
            return {"error": "lightgbm 未安裝"}
    else:
        return {"error": "未知模型"}
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    out = {
        "rmse": rmse,
        "r2": r2,
        "model": model,
    }
    # 3D/2D 視覺化：限 features=2
    if len(features) == 2:
        f1, f2 = features
        x1_grid = np.linspace(X[f1].min(), X[f1].max(), 50)
        x2_grid = np.linspace(X[f2].min(), X[f2].max(), 50)
        X1, X2 = np.meshgrid(x1_grid, x2_grid)
        pred_in = pd.DataFrame({f1: X1.ravel(), f2: X2.ravel()})
        grid_pred = model.predict(pred_in).reshape(X1.shape)
        # 3D surface
        import plotly.graph_objects as go
        surface_fig = go.Figure(data=[go.Surface(z=grid_pred, x=X1, y=X2)])
        surface_fig.update_layout(title="3D預測曲面", autosize=True, width=800, height=500)
        # 2D contour
        contour_fig = go.Figure(data=[go.Contour(z=grid_pred, x=x1_grid, y=x2_grid)])
        contour_fig.update_layout(title="2D預測等高線", autosize=True, width=800, height=500)
        out["surface_fig"] = surface_fig
        out["contour_fig"] = contour_fig
    return out
