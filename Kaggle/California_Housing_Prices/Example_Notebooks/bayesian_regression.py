# Bayesian Regression | House Price Prediction (https://www.kaggle.com/code/shtrausslearning/bayesian-regression-house-price-prediction)

# This script uses a Bayesian regression model to predict house prices in California. The model is trained on a dataset of house prices and various features such as location, number of rooms, etc. The script uses the ______ library to perform Bayesian regression and predict house prices.

# Topic Information:
# - Factors that influence house prices include location, size, number of rooms, etc.
# - Housing market is volatile (supply&demand, interest rates and inflation) and so it's difficult to predict variation over time.
# - Most datasets have limited number of features available.
# - Missing data exists, which may need to be imputed.
# - Outliers may exist, which may influence the model.

# Aim is to predict the median house value given a set of features. 

# %% Loading relevant libraries
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import contextily as ctx
from matplotlib import cm
from matplotlib.colors import Normalize, LogNorm

# %% Initial data exploration
folder_path = 'C:/Users/jercar/OneDrive - UKCEH/VSCode/Learning_Development_Statistics/Kaggle/California_Housing_Prices/'

df = pd.read_csv(f'{folder_path}/Data/housing.csv')

df.info()
display(df)
display(df.describe())
display(df[df.isnull().any(axis=1)])

# Note some of the nulls on the example notebook are no longer present in the dataset. e.g. df.iloc[[20566]]

# %% Further data exploration - Marginal Histograms
df.hist(bins=60, figsize=(15,9),color='tab:blue',edgecolor='black')
plt.show()

# Outliers are present in the data, which may need to be addressed. Obvious at the upper end for median age and median house value.
# Range of X axis for total rooms, total bedrooms, population and households are very broad indicating the presence of outliers.
# Unclear why there are regularly spaced gaps in the median age histogram.
# The distributions are either skewed or multi-modal.
# Population, total bedrooms and total rooms in a block are very connected things and have similar distributions.

# %%
test = df.select_dtypes(include=[np.number]).corr().round(2)
np.zeros_like(test,dtype=bool)

# %% Further data exploration - Correlation Matrix

def corrMat(df,id=False):
    ''' Function to plot correlation of features '''
    
    corr_mat = df.corr().round(2)
    fig, axs = plt.subplots(figsize=(6,6))
    sns.heatmap(corr_mat,vmin=-1,vmax=1,center=0, 
                cmap='plasma',square=False,lw=2,annot=True,cbar=False)
    plt.show()

corrMat(df.select_dtypes(include=[np.number]))

# Median income is clearly the most correlated feature with median house value.
# Low correlation for latitude and longitude with median house value suggests that either location is not an important feature (counter to intuition) or that the data is very spread and the relationship with house price very non-linear (agrees with intuition). Simple models may not be able to capture this relationship and so would be worth dropping these features.

# %% Further data exploration - Pairplot
''' Draw a Bivariate Seaborn Pairgrid /w KDE density w/ '''
def snsPairGrid(df):

    ''' Plots a Seaborn Pairgrid w/ KDE & scatter plot of df features'''
    g = sns.PairGrid(df,diag_sharey=False)
    g.figure.set_size_inches(14,13)
    g.map_diag(sns.kdeplot, lw=2) # draw kde approximation on the diagonal
    g.map_lower(sns.scatterplot,s=15,edgecolor="k",linewidth=1,alpha=0.4) # scattered plot on lower half
    g.map_lower(sns.kdeplot,cmap='plasma',n_levels=10) # kde approximation on lower half
    plt.tight_layout()

pairlist = ['median_house_value','median_income','total_rooms','housing_median_age','latitude','population']
snsPairGrid(df[pairlist].sample(n=5000)) 

# Median income and median house value appear linearly related with Gaussian noise. Although, there's an artificial upper limit of around $150,000 for median income.

# Median house age and median house value have a very spread out and weak relationship. There again appears to be artifical upper limits for both the median house value and median house age.

# Total rooms and population have a strong linear relationship. The data of both is largely concentrated at lower values. 

# %% Further data exploration - Spatial Map

def spatialMap(df,column,ax):
    ''' Plot the spatial distribution of the data '''
    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
    # Set the CRS
    gdf = gdf.set_crs(epsg=4326)  # WGS84
    gdf = gdf.to_crs(epsg=3857) 

    # Normalize price values for color mapping
    # norm = Normalize(vmin=gdf[column].min(), vmax=gdf[column].max())
    norm = LogNorm(vmin=gdf[column].min(), vmax=gdf[column].max())
    cmap = cm.viridis  # You can choose other colormaps like 'plasma', 'cividis', etc.
    gdf['color'] = gdf[column].apply(lambda x: cmap(norm(x)))
    
    # Plot the points
    gdf.plot(ax=ax,
            markersize=1, 
            alpha=0.8,
            color=gdf['color'])
    # Add a basemap
    ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron)
    # Set plot title and labels
    ax.set_title(f'{column}')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    # Create a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Only needed for the colorbar
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(f'{column}')

fig, axs = plt.subplots(2,2,figsize=(10, 10))
spatialMap(df,'median_house_value',axs[0,0])
spatialMap(df,'median_income',axs[0,1])
spatialMap(df,'housing_median_age',axs[1,0])
spatialMap(df,'population',axs[1,1])
plt.show()

# Median house value shows two main clusters, with higher prices nearer the clusters.

# Population, income and median house age all show some correlation with distance from these major cities. The relationship is weaker though.


# %%

