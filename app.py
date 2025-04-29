import streamlit as st
import geopandas as gpd
import rasterio
import rasterio.plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
import tempfile
import os
import zipfile
from rasterio.enums import Resampling
from pyproj import Transformer

# Streamlit configuration
st.set_page_config(page_title="Landslide Susceptibility Mapping App", layout="wide")
st.title("🌍 Landslide Susceptibility Mapping using Machine Learning")

st.markdown("""
Upload raster layers and zipped shapefiles (with .shp, .shx, .dbf, .prj), select ML models, and generate a Landslide Susceptibility Index (LSI) map.
""")

# Step 1: Upload raster layers
num_layers = st.number_input("Step 1: Enter number of raster layers:", min_value=1, max_value=20, value=5)
uploaded_layers = st.file_uploader("Upload raster layers (GeoTIFFs):", type=['tif'], accept_multiple_files=True)

# Optionally preview uploaded maps
show_input_maps = st.checkbox("Optionally show preview of uploaded raster layers")

# Step 2: Upload zipped shapefiles for landslide and non-landslide points
uploaded_landslide_zip = st.file_uploader("Upload zipped Landslide Shapefile:", type=['zip'])
uploaded_nonlandslide_zip = st.file_uploader("Upload zipped Non-Landslide Shapefile:", type=['zip'])

rasters = {}
points_df = None

if uploaded_layers and len(uploaded_layers) == num_layers:
    st.success(f"{len(uploaded_layers)} raster layers uploaded.")
    for i, file in enumerate(uploaded_layers):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
            with rasterio.open(tmp_path) as src:
                data = src.read(1)
                bounds = src.bounds
                extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
                rasters[f"layer_{i+1}"] = (data, src.transform, src.crs)

                if show_input_maps:
                    fig, ax = plt.subplots(figsize=(4, 3))
                    rasterio.plot.show(src, ax=ax, title=f"Layer {i+1}")

                    if src.crs.to_epsg() != 4326:
                        transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
                        lon_min, lat_min = transformer.transform(bounds.left, bounds.bottom)
                        lon_max, lat_max = transformer.transform(bounds.right, bounds.top)
                        xticks = np.round(np.linspace(lon_min, lon_max, 5), 2)
                        yticks = np.round(np.linspace(lat_min, lat_max, 5), 2)
                        ax.set_xticks(xticks)
                        ax.set_yticks(yticks)
                    else:
                        ax.set_xticks(np.round(np.linspace(extent[0], extent[1], 5), 2))
                        ax.set_yticks(np.round(np.linspace(extent[2], extent[3], 5), 2))

                    ax.set_xlabel("Longitude (°)")
                    ax.set_ylabel("Latitude (°)")
                    ax.grid(True, linestyle='--', alpha=0.5)
                    st.pyplot(fig)

@st.cache_data
def unzip_shapefile(uploaded_zip):
    with tempfile.TemporaryDirectory() as tmpdirname:
        zpath = os.path.join(tmpdirname, "uploaded.zip")
        with open(zpath, "wb") as f:
            f.write(uploaded_zip.getbuffer())
        with zipfile.ZipFile(zpath, 'r') as zip_ref:
            zip_ref.extractall(tmpdirname)
        shp_files = [os.path.join(tmpdirname, f) for f in os.listdir(tmpdirname) if f.endswith('.shp')]
        if shp_files:
            return gpd.read_file(shp_files[0])
    return None

if uploaded_landslide_zip and uploaded_nonlandslide_zip:
    gdf_ls = unzip_shapefile(uploaded_landslide_zip)
    gdf_nls = unzip_shapefile(uploaded_nonlandslide_zip)
    if gdf_ls is not None and gdf_nls is not None:
        gdf_ls['label'] = 1
        gdf_nls['label'] = 0
        points_df = pd.concat([gdf_ls, gdf_nls], ignore_index=True)
        points_df = points_df.to_crs(epsg=4326)
        points_df['longitude'] = points_df.geometry.x
        points_df['latitude'] = points_df.geometry.y
        st.map(points_df[['latitude', 'longitude']])
        st.success("Points loaded successfully.")

# Step 3: Select ML Models
st.markdown("### Step 3: Select Machine Learning Models")
model_options = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Support Vector Machine": SVC(probability=True),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": lgb.LGBMClassifier()
}

selected_models = st.multiselect("Select models:", list(model_options.keys()))

# Step 4: Optional Grid Display
show_grid = st.checkbox("Optionally show grid in LSI output maps")

# Step 5: Run model and generate maps
if st.button("Generate LSI Maps") and selected_models:
    if rasters and points_df is not None:
        feature_names = list(rasters.keys())

        for i, (layer_key, (data, transform, crs)) in enumerate(rasters.items()):
            values = []
            for geom in points_df.geometry:
                try:
                    row, col = ~transform * (geom.x, geom.y)
                    row, col = int(row), int(col)
                    values.append(data[row, col])
                except Exception:
                    values.append(np.nan)
            points_df[layer_key] = values

        points_df = points_df.dropna()
        X = points_df[feature_names]
        y = points_df['label']

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y)

        output_files = []
        height, width = list(rasters.values())[0][0].shape
        transform = list(rasters.values())[0][1]
        crs = list(rasters.values())[0][2]

        for model_name in selected_models:
            model = model_options[model_name]
            model.fit(X_train, y_train)
            y_score = model.predict_proba(X_test)[:, 1]

            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            st.markdown(f"**{model_name} AUC: {roc_auc:.3f}**")

            flat_layers = np.column_stack([rasters[k][0].flatten() for k in feature_names])
            flat_layers = np.where(flat_layers < -1e10, np.nan, flat_layers)
            valid_mask = ~np.any(np.isnan(flat_layers), axis=1)
            flat_layers_clean = flat_layers[valid_mask]
            flat_layers_scaled = scaler.transform(flat_layers_clean)

            prob_map = np.full(flat_layers.shape[0], np.nan)
            prob_map[valid_mask] = model.predict_proba(flat_layers_scaled)[:, 1]
            prob_map = prob_map.reshape((height, width))

            out_path = f"{model_name.replace(' ', '_')}_LSI.tif"
            output_files.append(out_path)
            with rasterio.open(out_path, 'w', driver='GTiff', height=height, width=width,
                               count=1, dtype='float32', crs=crs,
                               transform=transform, nodata=-9999.0) as dst:
                dst.write(np.where(np.isnan(prob_map), -9999.0, prob_map).astype(np.float32), 1)

            st.markdown(f"**{model_name} LSI Map**")
            fig, ax = plt.subplots(figsize=(5, 4))
            img = ax.imshow(np.ma.masked_where(np.isnan(prob_map), prob_map), cmap='RdYlBu', vmin=0, vmax=1)
            plt.colorbar(img, ax=ax, label='Susceptibility')

            if show_grid:
                bounds = rasterio.transform.array_bounds(height, width, transform)
                if crs.to_epsg() != 4326:
                    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                    lon_min, lat_min = transformer.transform(bounds[0], bounds[1])
                    lon_max, lat_max = transformer.transform(bounds[2], bounds[3])
                    ax.set_xticks(np.round(np.linspace(lon_min, lon_max, 6), 2))
                    ax.set_yticks(np.round(np.linspace(lat_min, lat_max, 6), 2))
                else:
                    ax.set_xticks(np.round(np.linspace(bounds[0], bounds[2], 6), 2))
                    ax.set_yticks(np.round(np.linspace(bounds[1], bounds[3], 6), 2))

                ax.set_xlabel("Longitude (°)")
                ax.set_ylabel("Latitude (°)")
                ax.grid(True, linestyle='--', alpha=0.5)

            st.pyplot(fig)

        with zipfile.ZipFile("LSI_outputs.zip", 'w') as zipf:
            for file in output_files:
                zipf.write(file)

        with open("LSI_outputs.zip", "rb") as f:
            st.download_button("Download All LSI Maps (ZIP)", f, file_name="LSI_outputs.zip")

# Step 6: Deployment help
st.markdown("---")
st.markdown("### 🚀 How to Deploy This App to GitHub")
st.markdown("""
1. Create a GitHub repository.
2. Upload your files (`app.py`, `requirements.txt`, `setup.sh`).
3. Deploy via [Streamlit Cloud](https://streamlit.io/cloud) or [Render.com](https://render.com).
""")
