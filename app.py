import streamlit as st
import geopandas as gpd
import rasterio
import rasterio.plot
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rasterio.transform import array_bounds
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from pyproj import Transformer
from sklearn.feature_selection import VarianceThreshold
import tempfile, os, zipfile

# ─── UI SETUP ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="LSI Mapping App", layout="wide")
st.title("🌍 Landslide Susceptibility Mapping")

st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Upload raster layers (GeoTIFF).  
2. Upload zipped shapefiles (landslide & non-landslide).  
3. (Optional) Preview rasters with legend & grid.  
4. Select ML models and train.  
5. View model reports, layer weights, SHAP importances.  
6. Generate full-area susceptibility maps & download.  
""")

# ─── 1️⃣ UPLOAD DATA ────────────────────────────────────────────────────────
st.header("1️⃣ Upload Data")
num_layers = st.number_input("Number of raster layers:", 1, 20, 5)
layers_in  = st.file_uploader("GeoTIFF layers:", type="tif", accept_multiple_files=True)
zip_ls     = st.file_uploader("Zipped Landslide (.zip):", type="zip")
zip_nls    = st.file_uploader("Zipped Non-Landslide (.zip):", type="zip")
preview    = st.checkbox("Preview rasters with legend & grid")

@st.cache_data
def unzip_shp(zf):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "upload.zip")
        with open(path, "wb") as f:
            f.write(zf.getbuffer())
        with zipfile.ZipFile(path, "r") as z:
            z.extractall(tmp)
        shp_files = [os.path.join(tmp, f) for f in os.listdir(tmp) if f.endswith(".shp")]
        return gpd.read_file(shp_files[0]) if shp_files else None

# Load rasters
rasters, meta, raster_crs = {}, None, None
if layers_in and len(layers_in) == num_layers:
    for i, up in enumerate(layers_in, 1):
        # Save to temp file, read, then delete
        tmpf = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        tmpf.write(up.read())
        tmpf.flush()
        tmpf.close()
        try:
            with rasterio.open(tmpf.name) as src:
                arr = src.read(1)
                rasters[f"layer_{i}"] = (arr, src.transform, src.crs)
                if meta is None:
                    meta, raster_crs = src.meta.copy(), src.crs

                if preview:
                    nod = src.nodata
                    data_ma = np.ma.masked_equal(arr, nod) if nod is not None else np.ma.masked_invalid(arr)
                    # Reproject to WGS84 for grid
                    if src.crs.to_epsg() != 4326:
                        t2, w2, h2 = calculate_default_transform(
                            src.crs, "EPSG:4326", src.width, src.height, *src.bounds
                        )
                        data2 = np.empty((h2, w2), dtype=data_ma.dtype)
                        reproject(
                            source=data_ma, destination=data2,
                            src_transform=src.transform, src_crs=src.crs,
                            dst_transform=t2, dst_crs="EPSG:4326",
                            resampling=Resampling.nearest
                        )
                        data_ma = data2
                        left, bottom, right, top = array_bounds(h2, w2, t2)
                    else:
                        left, bottom, right, top = src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top

                    fig, ax = plt.subplots(figsize=(4,3))
                    im = ax.imshow(
                        data_ma, cmap="terrain", origin="upper",
                        extent=[left, right, bottom, top], interpolation="none"
                    )
                    cbar = plt.colorbar(im, ax=ax, shrink=0.75)
                    cbar.set_label(f"Layer {i}")
                    xt = np.round(np.linspace(left, right, 5), 2)
                    yt = np.round(np.linspace(bottom, top, 5), 2)
                    ax.set_xticks(xt); ax.set_yticks(yt)
                    ax.set_xlabel("Longitude (°)")
                    ax.set_ylabel("Latitude (°)")
                    ax.grid(True, linestyle="--", alpha=0.5)
                    st.pyplot(fig)
                    plt.close(fig)
        finally:
            os.remove(tmpf.name)

# ─── 2️⃣ LOAD & REPROJECT POINTS ───────────────────────────────────────────
points = None
# Load & reproject point shapefiles
if zip_ls and zip_nls:
    if raster_crs is None:
        st.error("Upload rasters first, then shapefiles.")
    else:
        gls  = unzip_shp(zip_ls)
        gnls = unzip_shp(zip_nls)
        if gls is None or gnls is None:
            st.error("Could not read shapefiles.")
        else:
            # 1) assign labels
            gls["label"], gnls["label"] = 1, 0

            # 2) force both to the same CRS (use gls.crs here)
            if gls.crs != gnls.crs:
                gnls = gnls.to_crs(gls.crs)

            # 3) now concatenate safely
            merged = pd.concat([gls, gnls], ignore_index=True)
            points = gpd.GeoDataFrame(merged,
                                      geometry="geometry",
                                      crs=gls.crs)

            # 4) finally, reproject to your raster CRS
            points = points.to_crs(raster_crs)


# ─── 3️⃣ SAMPLE RASTERS AT POINTS ──────────────────────────────────────────
st.header("2️⃣ Sample Rasters at Points")
df = None
if points is not None and rasters:
    df = points.copy()
    feats = list(rasters.keys())
    for key, (arr, tr, crs) in rasters.items():
        vals = []
        for pt in df.geometry:
            try:
                r, c = ~tr * (pt.x, pt.y)
                vals.append(arr[int(r), int(c)])
            except:
                vals.append(np.nan)
        df[key] = vals

    # Drop any rows with NaN in any feature
    df.dropna(subset=feats, inplace=True)
    st.write(f"Sampled points: {df.shape[0]}")
    if df.shape[0] < 2:
        st.error("Need at least 2 valid samples.")
else:
    st.info("Upload rasters & shapefiles to sample.")

# ─── 4️⃣ TRAIN, EVALUATE & WEIGHTS ─────────────────────────────────────────
st.header("3️⃣ Train, Evaluate & View Weights")
if df is not None and df.shape[0] >= 2:
    X = df[feats].astype(float)
    y = df["label"]

    # Drop zero-variance features
    vt = VarianceThreshold(threshold=0.0)
    X = pd.DataFrame(vt.fit_transform(X), columns=[f for f,c in zip(feats, vt.get_support()) if c])
    feats = list(X.columns)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost":        xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss"),
        "LightGBM":       lgb.LGBMClassifier(random_state=42)
    }
    chosen = st.multiselect("Select models:", list(models.keys()), default=list(models.keys()))

    if st.button("Run Models"):
        results = {}
        for name in chosen:
            m = models[name]
            m.fit(Xtr, ytr)
            pred = m.predict(Xte)
            pr   = m.predict_proba(Xte)[:,1]
            results[name] = pr

            st.subheader(f"{name} Report")
            st.text(classification_report(yte, pred))

            cm = confusion_matrix(yte, pred)
            fig, ax = plt.subplots(figsize=(3,2))
            sns.heatmap(cm, annot=True, fmt="d", ax=ax)
            ax.set_title(f"{name} Confusion")
            st.pyplot(fig)
            plt.close(fig)

        # ROC curves
        fig, ax = plt.subplots(figsize=(5,4))
        for name, pr in results.items():
            fpr, tpr, _ = roc_curve(yte, pr)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.2f})")
        ax.plot([0,1],[0,1],'--',color='gray')
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        # Layer weights
        st.subheader("🌟 Layer Weights")
        imp_frames = []
        for name in chosen:
            m = models[name]
            fi = m.feature_importances_
            fi = fi / fi.sum()
            imp_frames.append(
                pd.DataFrame({"layer": feats, "importance": fi, "model": name})
            )
        all_imp = pd.concat(imp_frames)
        pivot  = all_imp.pivot(index="layer", columns="model", values="importance").fillna(0)
        st.dataframe(pivot.style.format("{:.3f}"))

        for name in chosen:
            sub = all_imp[all_imp["model"]==name].sort_values("importance", ascending=False)
            fig, ax = plt.subplots()
            ax.barh(sub["layer"], sub["importance"])
            ax.set_title(f"{name} Feature Importances")
            ax.set_xlabel("Normalized Importance")
            st.pyplot(fig)
            plt.close(fig)

        # SHAP mean|value|
        st.subheader("🔍 SHAP Mean |Value| Importance")
        expl = shap.TreeExplainer(models[chosen[0]])
        sv   = expl.shap_values(Xtr)
        if isinstance(sv, list):
            sv = sv[1]
        mean_abs = np.abs(sv).mean(axis=0)
        pairs   = list(zip(feats, mean_abs.tolist()))
        shap_df = pd.DataFrame(pairs, columns=["layer","mean_abs_shap"]).sort_values("mean_abs_shap", ascending=False)
        # only format the numeric column
        st.dataframe(shap_df.style.format({"mean_abs_shap":"{:.3f}"}))

        fig, ax = plt.subplots()
        ax.barh(shap_df["layer"], shap_df["mean_abs_shap"])
        ax.set_title(f"SHAP Importances ({chosen[0]})")
        ax.set_xlabel("Mean |SHAP value|")
        st.pyplot(fig)
        plt.close(fig)

        # ─── 5️⃣ FULL-AREA SUSCEPTIBILITY MAPS ─────────────────────────────────
        st.header("4️⃣ Full-Area Susceptibility Maps")

        ref_arr, ref_tr, ref_crs = next(iter(rasters.values()))
        h, w = ref_arr.shape

        # compute lon/lat ticks
        left, bottom, right, top = array_bounds(h, w, ref_tr)
        trans = Transformer.from_crs(ref_crs, "EPSG:4326", always_xy=True)
        lon_min, lat_min = trans.transform(left, bottom)
        lon_max, lat_max = trans.transform(right, top)
        xt = np.round(np.linspace(lon_min, lon_max, 6), 2)
        yt = np.round(np.linspace(lat_min, lat_max, 6), 2)

        # align rasters
        aligned = {}
        for k, (arr, tr, crs) in rasters.items():
            if arr.shape==(h,w) and tr==ref_tr and crs==ref_crs:
                aligned[k] = arr
            else:
                dst = np.empty((h,w), dtype=arr.dtype)
                reproject(
                    source=arr, destination=dst,
                    src_transform=tr, src_crs=crs,
                    dst_transform=ref_tr, dst_crs=ref_crs,
                    resampling=Resampling.nearest
                )
                aligned[k] = dst

        stack = np.column_stack([aligned[k].flatten() for k in feats])
        mask  = np.any(np.isnan(stack), axis=1)
        valid = stack[~mask]
        valid_df = pd.DataFrame(valid, columns=feats)

        out_meta = meta.copy()
        out_meta.update(dtype="float32", count=1, nodata=-9999.0)

        ensemble = []
        for name in chosen:
            m     = models[name]
            probs = m.predict_proba(valid_df)[:,1]

            full  = np.full(stack.shape[0], np.nan, dtype="float32")
            full[~mask] = probs
            full = full.reshape((h,w))
            full[np.isnan(full)] = -9999.0

            path = f"{name.replace(' ','_')}_LSI.tif"
            with rasterio.open(path, "w", **out_meta) as dst:
                dst.write(full, 1)
            ensemble.append(full)

            st.subheader(f"{name} Susceptibility Map")
            fig, ax = plt.subplots(figsize=(5,4))
            img = ax.imshow(full, cmap="RdYlBu", vmin=0, vmax=1,
                            extent=[lon_min, lon_max, lat_min, lat_max],
                            origin="upper")
            cbar = plt.colorbar(img, ax=ax, shrink=0.75)
            cbar.set_label("Susceptibility")
            ax.set_xticks(xt); ax.set_yticks(yt)
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
            ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
            ax.grid(True, linestyle="--", alpha=0.5)
            st.pyplot(fig)
            plt.close(fig)

        # ensemble mean
        st.subheader("🧮 Ensemble Mean Map")
        mean_map = np.nanmean(np.stack(ensemble), axis=0)
        fig, ax = plt.subplots(figsize=(5,4))
        img = ax.imshow(mean_map, cmap="RdYlBu", vmin=0, vmax=1,
                        extent=[lon_min, lon_max, lat_min, lat_max],
                        origin="upper")
        plt.colorbar(img, ax=ax, shrink=0.75).set_label("Susceptibility")
        ax.set_xticks(xt); ax.set_yticks(yt)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
        ax.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig)
        plt.close(fig)

        # download ZIP
        zip_name = "LSI_maps.zip"
        with zipfile.ZipFile(zip_name, "w") as zf:
            for name in chosen:
                tif = f"{name.replace(' ','_')}_LSI.tif"
                zf.write(tif)
        with open(zip_name, "rb") as f:
            st.download_button("📥 Download All Maps", f, file_name=zip_name)
        os.remove(zip_name)

else:
    st.info("Need at least 2 valid samples before training.")
