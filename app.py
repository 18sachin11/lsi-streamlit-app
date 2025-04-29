import streamlit as st
import geopandas as gpd
import rasterio
import rasterio.plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc, classification_report
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
from rasterio.transform import from_origin

st.set_page_config(page_title="Landslide Susceptibility Mapping App", layout="wide")
st.title("🌍 Landslide Susceptibility Mapping using Machine Learning")

st.markdown("""
This web app allows you to upload raster layers and landslide/non-landslide points,
select machine learning models, and generate a Landslide Susceptibility Index (LSI) map.
""")

# Step 1: User Input - Number of Layers
num_layers = st.number_input("Step 1: Enter the number of input layers:", min_value=1, max_value=20, value=5)

# Step 2: Upload raster layers
uploaded_layers = st.file_uploader("Step 2: Upload raster layers (GeoTIFF format):", type=['tif'], accept_multiple_files=True)

# Step 3: Upload landslide and non-landslide shapefiles
uploaded_landslide = st.file_uploader("Upload Landslide Points (Shapefile .shp):", type=['shp'])
uploaded_nonlandslide = st.file_uploader("Upload Non-Landslide Points (Shapefile .shp):", type=['shp'])

# Placeholder for loaded rasters and points
rasters = {}
points_df = None
layer_paths = []

if uploaded_layers and len(uploaded_layers) == num_layers:
    st.success(f"{len(uploaded_layers)} raster layers uploaded.")
    for i, file in enumerate(uploaded_layers):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
            layer_paths.append(tmp_path)
            with rasterio.open(tmp_path) as src:
                data = src.read(1)
                fig, ax = plt.subplots()
                rasterio.plot.show(src, ax=ax, title=f"Layer {i+1}")
                st.pyplot(fig)
                rasters[f"layer_{i+1}"] = (data, src.transform, src.crs)

if uploaded_landslide and uploaded_nonlandslide:
    with tempfile.TemporaryDirectory() as tmpdir:
        shp_path_ls = os.path.join(tmpdir, "landslide.shp")
        shp_path_nls = os.path.join(tmpdir, "nonlandslide.shp")
        with open(shp_path_ls, 'wb') as f:
            f.write(uploaded_landslide.read())
        with open(shp_path_nls, 'wb') as f:
            f.write(uploaded_nonlandslide.read())

        gdf_ls = gpd.read_file(shp_path_ls)
        gdf_nls = gpd.read_file(shp_path_nls)
        gdf_ls['label'] = 1
        gdf_nls['label'] = 0
        points_df = pd.concat([gdf_ls, gdf_nls])

        st.map(points_df)
        st.success("Points uploaded and visualized.")

# Step 4: Select Machine Learning Models
st.markdown("### Step 4: Select Machine Learning Models")
model_options = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Support Vector Machine": SVC(probability=True),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": lgb.LGBMClassifier()
}

selected_models = st.multiselect("Select models to use:", list(model_options.keys()))

# Step 5: Generate LSI and Export
if st.button("Step 5: Generate Final LSI Maps") and selected_models:
    if rasters and points_df is not None:
        feature_names = list(rasters.keys())

        # Extract features at point locations
        for i, (layer_key, (data, transform, crs)) in enumerate(rasters.items()):
            values = []
            for geom in points_df.geometry:
                row, col = ~transform * (geom.x, geom.y)
                row, col = int(row), int(col)
                try:
                    values.append(data[row, col])
                except IndexError:
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

            # Save GeoTIFF
            out_path = f"{model_name.replace(' ', '_')}_LSI.tif"
            output_files.append(out_path)
            with rasterio.open(out_path, 'w', driver='GTiff', height=height, width=width,
                               count=1, dtype='float32', crs=list(rasters.values())[0][2],
                               transform=transform, nodata=-9999.0) as dst:
                dst.write(np.where(np.isnan(prob_map), -9999.0, prob_map).astype(np.float32), 1)

            # Display
            st.markdown(f"**{model_name} LSI Map**")
            fig, ax = plt.subplots(figsize=(6, 5))
            img = ax.imshow(np.ma.masked_where(np.isnan(prob_map), prob_map), cmap='RdYlBu', vmin=0, vmax=1)
            plt.colorbar(img, ax=ax, label='Susceptibility')
            st.pyplot(fig)

        # Create zip
        with zipfile.ZipFile("LSI_outputs.zip", 'w') as zipf:
            for file in output_files:
                zipf.write(file)

        with open("LSI_outputs.zip", "rb") as f:
            st.download_button("Download All LSI Maps (ZIP)", f, file_name="LSI_outputs.zip")

# Step 6: Help Section for GitHub Deployment
st.markdown("---")
st.markdown("### 🚀 How to Deploy This App to GitHub")
st.markdown("""
1. Create a new GitHub repository.
2. Upload all your code files (`app.py`, `requirements.txt`, etc.)
3. In your repository, go to [Streamlit Community Cloud](https://streamlit.io/cloud) and sign in.
4. Click "New app" and connect your GitHub repo.
5. Select `app.py` as the entry point and click **Deploy**.
6. Your app will be live at `https://yourname.streamlit.app` 🌐
""")
