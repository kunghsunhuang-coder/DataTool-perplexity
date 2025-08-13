import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px

def analyze_sheet(df, cols, n_cluster=3):
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df[cols])
    km = KMeans(n_clusters=n_cluster, n_init="auto", random_state=42)
    groups = km.fit_predict(X_pca)
    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df_pca["group"] = groups
    fig = px.scatter(df_pca, x="PC1", y="PC2", color="group", title="PCA + KMeans Clustering")
    return fig, df_pca
