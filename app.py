import streamlit as st
import pandas as pd
from utils import read_data, get_selected_df, safe_to_list
from dataprep import data_precheck
from doe import gen_design, taguchi_array
from plotting import plot_scatter, plot_heatmap, pca_3d_plot
from clustering import analyze_sheet
from statistics import class_vs_numeric_stat
from ml_models import train_and_predict_with_surface

st.set_page_config(page_title="ITRI Data/AI 平台", layout="wide")

# LOGO
st.image("itri_CEL_C.png", width=120)
st.markdown("# ITRI Data/AI 科技平台 (Streamlit 版)")

# 初始化儲存用的 session_state
if "df" not in st.session_state:
    st.session_state['df'] = None

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1️⃣ 預處理", "2️⃣ DOE / Taguchi", "3️⃣ 視覺化分析", 
    "4️⃣ PCA 分群", "5️⃣ 類別統計", "6️⃣ 機器學習"])

with tab1:
    st.markdown("### 資料預處理")
    file = st.file_uploader("上傳檔案 (CSV/XLSX)")
    if file:
        df, xl, sheets = read_data(file)
        sheet = st.selectbox("選擇工作表", sheets)
        df = get_selected_df(file, sheet)
        # 更新 session state 以供下游使用
        st.session_state['df'] = df
        st.dataframe(df)

with tab2:
    st.markdown("### 實驗設計")
    method = st.selectbox("設計法", [
        "Latin Hypercube","Sobol","Halton","Full Factorial","Fractional Factorial",
        "Plackett-Burman","Box-Behnken","Central Composite","Taguchi"])
    factors = st.text_input("因子名稱（逗號分隔）", "A,B,C")
    if method == "Taguchi":
        taguchi_type = st.selectbox("Taguchi 表型號", ["L4", "L8", "L9"])
        taguchi_levels = st.number_input("因子水平數", 3)
        if st.button("產生田口設計"):
            df = taguchi_array([int(taguchi_levels)]*3, l=taguchi_type)
            st.dataframe(df)
    else:
        mins = st.text_input("最小值(逗號分隔)", "0,0,0")
        maxs = st.text_input("最大值(逗號分隔)", "1,1,1")
        steps = st.text_input("步長/水平數(逗號分隔)", "3,3,3")
        n_sample = st.number_input("樣本數", 10)
        seed = st.number_input("隨機種子", 42)
        if st.button("產生DOE設計"):
            names = safe_to_list(factors.split(","))
            mins = [float(x) for x in safe_to_list(mins.split(","))]
            maxs = [float(x) for x in safe_to_list(maxs.split(","))]
            steps = [float(x) for x in safe_to_list(steps.split(","))]
            df = gen_design(method, names, mins, maxs, steps, int(n_sample), int(seed))
            st.dataframe(df)
            st.download_button("下載DOE表", df.to_csv(index=False), "doe.csv")

with tab3:
    st.markdown("### 視覺化分析")
    df = st.session_state.get('df')
    if df is not None:
        num_cols = df.select_dtypes("number").columns.tolist()
        cat_cols = [c for c in df.columns if df[c].dtype == object or df[c].nunique() <= 20]
        st.subheader("散點圖")
        x_axis = st.selectbox("X軸", num_cols)
        y_axis = st.selectbox("Y軸", num_cols, index=1 if len(num_cols)>1 else 0)
        color_axis = st.selectbox("顏色(類別)", cat_cols) if cat_cols else None
        if st.button("繪製散點圖"):
            st.plotly_chart(plot_scatter(df, x_axis, y_axis, color_axis))
        st.subheader("熱力圖")
        if st.button("產生熱力圖"):
            st.plotly_chart(plot_heatmap(df))

# tab4, tab5, tab6 ... 依此模式類推

st.info("本頁僅為 Gradio → Streamlit 移植主要 flow 範例，詳細可根據原功能分頁和你的自定 callback 填入相同對應元件。")
