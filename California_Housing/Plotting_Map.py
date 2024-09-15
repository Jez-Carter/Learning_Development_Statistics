# %% Importing Packages
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx

from matplotlib import cm
from matplotlib.colors import Normalize

# %% Load a sample dataset
from sklearn.datasets import fetch_california_housing
california = fetch_california_housing()
data = pd.DataFrame(california.data, columns=california.feature_names)
data['MedHouseVal'] = california.target

# %% Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.Longitude, data.Latitude))

# %% Set the CRS
gdf = gdf.set_crs(epsg=4326)  # WGS84
gdf = gdf.to_crs(epsg=3857)   # Web Mercator for compatibility with contextily

# %% Normalize price values for color mapping
norm = Normalize(vmin=gdf['MedHouseVal'].min(), vmax=gdf['MedHouseVal'].max())
cmap = cm.viridis  # You can choose other colormaps like 'plasma', 'cividis', etc.
gdf['color'] = gdf['MedHouseVal'].apply(lambda x: cmap(norm(x)))

# %% Plot the data
fig, ax = plt.subplots(figsize=(10, 10))

# Plot the points
gdf.plot(ax=ax, 
         alpha=0.3,
         color=gdf['color'])

# Add a basemap

ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron)

# Set plot title and labels
ax.set_title('California Housing Prices')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Create a colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Only needed for the colorbar
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('House Price ($)')

plt.show()

