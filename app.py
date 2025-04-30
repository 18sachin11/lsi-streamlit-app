import streamlit as st
import geopandas as gpd
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import tempfile, os, zipfile
from rasterio.enums import Resampling

# ─── UI SETUP ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="LSI Mapping App", layout="wide")
st.title("🌍 Landslide Susceptibility Mapping")

st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Upload your raster layers (GeoTIFF).
2. Upload zipped shapefiles for landslide & non-landslide points.
3. (Optional) Preview rasters.
4. Select which ML models to run.
5. Train, evaluate, view SHAP & correlation.
6. Generate susceptibility maps & download.
""")

# ─── UPLOAD ────────────────────────────────────────────────────────────────
st.header("1️⃣ Upload Data")
num_layers = st.number_input("Number of raster layers:", 1, 20, 5)
layers_in = st.file_uploader("GeoTIFF layers:", type="tif", accept_multiple_files=True)
zip_ls = st.file_uploader("Zipped Landslide .shp/.dbf/.shx/.prj:", type="zip")
zip_nls = st.file_uploader("Zipped Non-Landslide:", type="zip")
preview = st.checkbox("Preview rasters")

@st.cache_data
def unzip_shp(zf):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "u.zip")
        with open(path, "wb") as f:
            f.write(zf.getbuffer())
        with zipfile.ZipFile(path, "r") as z:
            z.extractall(tmp)
        shp = [os.path.join(tmp, f) for f in os.listdir(tmp) if f.endswith(".shp")]
        return gpd.read_file(shp[0]) if shp else None

# Load rasters
rasters, meta, raster_crs = {}, None, None
if layers_in and len(layers_in) == num_layers:
    for i, f in enumerate(layers_in, 1):
        tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        tmp.write(f.read()); tmp.close()
        with rasterio.open(tmp.name) as src:
            arr = src.read(1)
            rasters[f"f{i}"] = (arr, src.transform, src.crs)
            if meta is None:
                meta, raster_crs = src.meta.copy(), src.crs
        if preview:
            fig, ax = plt.subplots(figsize=(4, 3))
            rasterio.plot.show(arr, transform=src.transform, ax=ax, cmap="terrain")
            st.pyplot(fig)

# Load points (only after rasters loaded)
points = None
if zip_ls and zip_nls:
    if raster_crs is None:
        st.error("⚠️ Please upload all raster layers first, then upload your point shapefiles.")
    else:
        g_ls = unzip_shp(zip_ls)
        g_nls = unzip_shp(zip_nls)
        if g_ls is not None and g_nls is not None:
            g_ls["label"] = 1
            g_nls["label"] = 0
            points = gpd.GeoDataFrame(pd.concat([g_ls, g_nls], ignore_index=True))
            epsg_code = raster_crs.to_epsg()
            points = points.to_crs(epsg=epsg_code)

# ─── SAMPLE ────────────────────────────────────────────────────────────────
st.header("2️⃣ Sample Rasters at Points")
if points is not None and rasters:
    df = points.copy()
    feats = list(rasters.keys())
    for k, (arr, tr, crs) in rasters.items():
        vals = []
        for pt in df.geometry:
            try:
                r, c = ~tr * (pt.x, pt.y)
                vals.append(arr[int(r), int(c)])
            except:
                vals.append(np.nan)
        df[k] = vals
    df.dropna(subset=feats, inplace=True)
    st.write("Sampled points:", df.shape)
else:
    st.info("Upload both rasters & zipped shapefiles to sample.")

# ─── TRAIN/EVAL ───────────────────────────────────────────────────────────
st.header("3️⃣ Train & Evaluate Models")
if points is not None and rasters:
    X = df[feats].astype(float)
    y = df["label"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost":        xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss"),
        "LightGBM":       lgb.LGBMClassifier(random_state=42)
    }
    chosen = st.multiselect("Select models to run:", list(models.keys()), default=list(models.keys()))

    if st.button("Run Models"):
        results = {}
        for name in chosen:
            m = models[name]
            m.fit(Xtr, ytr)
            pred = m.predict(Xte)
            pr = m.predict_proba(Xte)[:, 1]
            results[name] = pr

            st.subheader(f"{name} Report")
            st.text(classification_report(yte, pred))

            cm = confusion_matrix(yte, pred)
            fig, ax = plt.subplots(figsize=(3, 2))
            sns.heatmap(cm, annot=True, fmt="d", ax=ax)
            ax.set_title(f"{name} Confusion")
            st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(5, 4))
        for name, pr in results.items():
            fpr, tpr, _ = roc_curve(yte, pr)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.2f})")
        ax.plot([0, 1], [0, 1], '--', color='gray')
        ax.legend()
        st.pyplot(fig)

        st.subheader("Feature Importance (SHAP)")
        expl = shap.TreeExplainer(models[chosen[0]])
        sv = expl.shap_values(Xtr)
        plt.figure(figsize=(8, 4))
        shap.summary_plot(sv, Xtr, plot_type="bar", show=False)
        st.pyplot(plt.gcf())

        st.subheader("Correlation Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(X.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        # ─── FULL-AREA SUSCEPTIBILITY MAPS ─────────────────────────────────
        st.header("4️⃣ Full-Area Susceptibility Maps")

        # Reference grid
        ref_arr, ref_transform, ref_crs = next(iter(rasters.values()))
        height, width = ref_arr.shape

        # Resample all to reference
        aligned = {}
        for k, (arr, transform, crs) in rasters.items():
            if arr.shape == (height, width) and transform == ref_transform and crs == ref_crs:
                aligned[k] = arr
            else:
                dst = np.empty((height, width), dtype=arr.dtype)
                from rasterio.warp import reproject
                reproject(
                    source=arr,
                    destination=dst,
                    src_transform=transform,
                    src_crs=crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.nearest
                )
                aligned[k] = dst

        # Build stack
        stack = np.column_stack([aligned[k].flatten() for k in feats])
        mask = np.any(np.isnan(stack), axis=1)
        valid = stack[~mask]

        out_meta = meta.copy()
        out_meta.update(dtype="float32", count=1, nodata=-9999.0)

        ensemble = []
        for name in chosen:
            m = models[name]
            probs = m.predict_proba(valid)[:, 1]

            full = np.full(stack.shape[0], np.nan, dtype="float32")
            full[~mask] = probs
            full = full.reshape((height, width))
            full[np.isnan(full)] = -9999.0

            path = f"{name.replace(' ', '_')}_LSI.tif"
            with rasterio.open(path, "w", **out_meta) as dst:
                dst.write(full, 1)

            ensemble.append(full)
            st.subheader(f"{name} Susceptibility Map")
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.imshow(full, cmap="RdYlBu", vmin=0, vmax=1)
            st.pyplot(fig)

        st.subheader("Ensemble Mean Map")
        avg = np.nanmean(np.stack(ensemble), axis=0)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(avg, cmap="RdYlBu", vmin=0, vmax=1)
        st.pyplot(fig)

        zip_name = "LSI_maps.zip"
        with zipfile.ZipFile(zip_name, "w") as zf:
            for name in chosen:
                tif = f"{name.replace(' ', '_')}_LSI.tif"
                zf.write(tif)
        with open(zip_name, "rb") as f:
            st.download_button("Download All Maps", f, file_name=zip_name)
