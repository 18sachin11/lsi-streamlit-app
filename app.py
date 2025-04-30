import streamlit as st
import geopandas as gpd
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import tempfile
import os
import zipfile
from pyproj import Transformer
from rasterio.enums import Resampling

st.set_page_config(page_title="LSI Mapping App", layout="wide")
st.title("🌍 Landslide Susceptibility Mapping")

st.markdown("""
Upload raster layers and zipped shapefiles (with .shp, .shx, .dbf, .prj), select ML models, and generate a Landslide Susceptibility Index (LSI) map.
""")

# --- Upload Inputs ---
st.markdown("### Step 1: Upload Raster Layers")
num_layers = st.number_input("Number of raster layers:", 1, 20, 6)
uploaded_layers = st.file_uploader("Upload GeoTIFF layers:", type="tif", accept_multiple_files=True)

st.markdown("### Step 2: Upload Landslide & Non-Landslide Shapefiles")
uploaded_ls = st.file_uploader("Landslide points (ZIP):", type="zip")
uploaded_nls = st.file_uploader("Non-landslide points (ZIP):", type="zip")

@st.cache_data
def unzip_shp(zf):
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "temp.zip")
        with open(p, "wb") as f: f.write(zf.getbuffer())
        with zipfile.ZipFile(p, "r") as zip_ref: zip_ref.extractall(tmp)
        shapes = [os.path.join(tmp, f) for f in os.listdir(tmp) if f.endswith(".shp")]
        return gpd.read_file(shapes[0]) if shapes else None

# load rasters
rasters = {}
meta = None
raster_crs = None
if uploaded_layers and len(uploaded_layers)==num_layers:
    for i, layer in enumerate(uploaded_layers,1):
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp.write(layer.read())
            with rasterio.open(tmp.name) as src:
                data = src.read(1)
                rasters[f"layer_{i}"] = (data, src.transform, src.crs)
                if meta is None:
                    meta = src.meta.copy()
                    raster_crs = src.crs

# load point shapefiles
points_df = None
if uploaded_ls and uploaded_nls:
    gdf_ls  = unzip_shp(uploaded_ls)
    gdf_nls = unzip_shp(uploaded_nls)
    if gdf_ls is not None and gdf_nls is not None:
        gdf_ls["label"] = 1
        gdf_nls["label"] = 0
        points_df = pd.concat([gdf_ls, gdf_nls], ignore_index=True)
        # reproject to raster CRS
        points_df = points_df.to_crs(raster_crs)

# extract training samples
if points_df is not None and rasters:
    feature_keys = list(rasters.keys())
    for key, (arr, tr, crs) in rasters.items():
        vals = []
        for geom in points_df.geometry:
            try:
                r,c = ~tr*(geom.x, geom.y)
                vals.append(arr[int(r),int(c)])
            except:
                vals.append(np.nan)
        points_df[key] = vals
    points_df.dropna(subset=feature_keys, inplace=True)
    X = points_df[feature_keys].astype(float)   # all numeric
    y = points_df["label"]

    # split—no scaling!
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    st.markdown("### Step 3: Select & Train Models")
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost":        xgb.XGBClassifier(random_state=42),
        "LightGBM":       lgb.LGBMClassifier(random_state=42),
    }
    selected = st.multiselect("Choose models:", list(models.keys()), default=list(models.keys()))

    if st.button("Generate LSI Maps") and selected:
        # prepare map export
        h,w = list(rasters.values())[0][0].shape
        transform = meta["transform"]
        out_meta = meta.copy()
        out_meta.update(dtype="float32", count=1, nodata=-9999.0)

        outputs = []
        for name in selected:
            m = models[name]
            m.fit(X_train, y_train)
            prob_test = m.predict_proba(X_test)[:,1]
            fpr,tpr,_ = roc_curve(y_test, prob_test)
            auc_score = auc(fpr,tpr)
            st.write(f"**{name} AUC:** {auc_score:.3f}")

            # predict full map
            stack = np.column_stack([rasters[k][0].flatten() for k in feature_keys])
            mask = np.any(np.isnan(stack), axis=1)
            valid = stack[~mask]
            preds = m.predict_proba(valid)[:,1]

            full = np.full(stack.shape[0], np.nan, dtype="float32")
            full[~mask] = preds
            full = full.reshape((h,w))
            full[np.isnan(full)] = -9999.0

            out_path = f"{name.replace(' ','_')}_LSI.tif"
            with rasterio.open(out_path, "w", **out_meta) as dst:
                dst.write(full, 1)
            outputs.append(out_path)

            # show
            fig,ax = plt.subplots(figsize=(5,4))
            im = ax.imshow(np.ma.masked_equal(full, -9999.0), cmap="RdYlBu", vmin=0, vmax=1)
            fig.colorbar(im, ax=ax, label="Susceptibility")
            ax.set_title(f"{name} LSI")
            st.pyplot(fig)

        # zip & download
        with zipfile.ZipFile("LSI_outputs.zip","w") as zf:
            for f in outputs: zf.write(f)
        with open("LSI_outputs.zip","rb") as f:
            st.download_button("Download ZIP", f, "LSI_outputs.zip")
