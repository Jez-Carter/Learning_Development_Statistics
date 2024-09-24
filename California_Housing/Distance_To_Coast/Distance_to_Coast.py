# %% Loading Packages
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from math import cos, sin, asin, sqrt, radians

import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib import cm
from matplotlib.colors import Normalize

# %% Load a sample dataset
from sklearn.datasets import fetch_california_housing
california = fetch_california_housing()
data = pd.DataFrame(california.data, columns=california.feature_names)
data['MedHouseVal'] = california.target

# %% Load Coastline shapefile
coastline_data = gpd.read_file('~/Machine_Learning_Revision/California_Housing/Auxiliary_Data/ne_110m_coastline/ne_110m_coastline.shp')
coastline = gpd.GeoSeries(coastline_data.geometry.unary_union)

# %% Distance to coastline calculators
def calc_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees):
    from: https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points/4913653#4913653
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2]) # convert decimal degrees to radians
    dlon = lon2 - lon1 
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2  #haversine formula
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    return km

def calc_distance_to_coastline(longitude,latitude ):
    target_coordinate=Point(longitude,latitude )
    return coastline.distance(target_coordinate).values[0]

def distance_degrees_to_kilometers(distance,coord=[0,0]):
    coord_plus=[c+distance for c in coord]
    coord_minus=[c-distance for c in coord]
    return (calc_distance(*coord,*coord_plus)+calc_distance(*coord,*coord_minus))*0.5

def calc_distance_to_coastline_km(longitude,latitude ):
    target_coordinate=Point(longitude,latitude )
    return distance_degrees_to_kilometers(coastline.distance(target_coordinate).values[0],[longitude,latitude])

# %% Computing the distance
data['DTC'] = data.apply(
    lambda x: calc_distance_to_coastline_km(x.Longitude, x.Latitude), axis=1)

# %% Saving Data
data.to_pickle('~/Machine_Learning_Revision/California_Housing/Auxiliary_Data/california_housing_dtc.pkl')

# %% Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.Longitude, data.Latitude))

# %% Set the CRS
gdf = gdf.set_crs(epsg=4326)  # WGS84
gdf = gdf.to_crs(epsg=3857)   # Web Mercator for compatibility with contextily

# %% Normalize price values for color mapping
norm = Normalize(vmin=gdf['DTC'].min(), vmax=gdf['DTC'].max())
cmap = cm.viridis  # You can choose other colormaps like 'plasma', 'cividis', etc.
gdf['color'] = gdf['DTC'].apply(lambda x: cmap(norm(x)))

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
cbar.set_label('Distance to Coast (km)')

plt.show()
# %%
