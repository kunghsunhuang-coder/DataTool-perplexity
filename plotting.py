import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_scatter(df, x, y, color=None, size=None, title=None):
    fig = px.scatter(df, x=x, y=y, color=color, size=size, title=title)
    return fig

def plot_heatmap(df, title=None):
    fig = px.imshow(df.corr(), text_auto=True, aspect="auto", title=title)
    return fig

def plot_surface(df, x_col, y_col, z_col, title=None):
    df_pivot = df.pivot_table(index=y_col, columns=x_col, values=z_col)
    fig = go.Figure(data=go.Surface(
        z=df_pivot.values, 
        x=df_pivot.columns, 
        y=df_pivot.index
    ))
    fig.update_layout(title=title)
    return fig

def plot_box(df, x, y, color=None, title=None):
    fig = px.box(df, x=x, y=y, color=color, title=title)
    return fig

def pca_3d_plot(df, features, color_col=None):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    components = pca.fit_transform(df[features])
    pca_df = pd.DataFrame(components, columns=['PC1', 'PC2', 'PC3'])
    if color_col and color_col in df.columns:
        pca_df[color_col] = df[color_col]
    fig = px.scatter_3d(
        pca_df,
        x='PC1', y='PC2', z='PC3',
        color=color_col if color_col else None,
        title='3D PCA Plot'
    )
    fig.update_layout(scene_aspectmode='cube')
    return fig

def plot_contour(x, y, z, title="2D等高線圖"):
    """
    x, y: 一維座標陣列 (如 numpy.linspace)
    z: 二維網格 (shape: (len(y), len(x)))
    title: plot 標題
    """
    fig = go.Figure(data=[go.Contour(z=z, x=x, y=y)])
    fig.update_layout(title=title, autosize=True, width=800, height=500)
    return fig
