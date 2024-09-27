# Regression | House Price Prediction (https://www.kaggle.com/code/shtrausslearning/bayesian-regression-house-price-prediction)

# This script uses regression models to predict house prices in California. The model is trained on a dataset of house prices and various features such as location, number of rooms, etc. The script shows a basic data processing pipeline.

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
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, PowerTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import contextily as ctx
from matplotlib import cm
from matplotlib.colors import Normalize, LogNorm
# from mllibs.bl_regressor import BR
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

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

# %% Removing Outliers
df = df[df['median_house_value']<500000] 
df = df[df['housing_median_age']<52] 

df.hist(bins=60, figsize=(15,9),color='tab:blue',edgecolor='black')
plt.show()

# %% Imputing Missing Data

def impute_knn(df):
    
    ''' inputs: pandas df containing feature matrix '''
    ''' outputs: dataframe with NaN imputed '''
    # imputation with KNN unsupervised method

    # separate dataframe into numerical/categorical
    ldf = df.select_dtypes(include=[np.number]) # numerical columns
    ldf_putaside = df.select_dtypes(exclude=[np.number]) # categorical columns
    # define columns w/ and w/o missing data
    cols_nan = ldf.columns[ldf.isna().any()].tolist() # columns with missing data
    cols_no_nan = ldf.columns.difference(cols_nan).values # columns without missing data

    for col in cols_nan:                
        imp_test = ldf[ldf[col].isna()] # rows with missing data - test set
        imp_train = ldf.dropna() # rows with no missing data - train set 
        model = KNeighborsRegressor(n_neighbors=5)  # KNR Unsupervised Approach
        knr = model.fit(imp_train[cols_no_nan], imp_train[col])
        ldf.loc[df[col].isna(), col] = knr.predict(imp_test[cols_no_nan])
    
    return pd.concat([ldf,ldf_putaside],axis=1)

df_imputed = impute_knn(df)
df_imputed.info()

# %% Feature Engineering
# Feature engineering is the process of transforming raw data into features that better represent the underlying problem to the predictive models, resulting in improved model accuracy on unseen data.
# Features with very high correlation teach a model similar things, multiple times, maybe consider combing them and dropping the others.
df_imputed['diag_coord'] = (df_imputed['longitude'] + df_imputed['latitude']) 
df_imputed['bedroom_per_room'] = df_imputed['total_bedrooms']/df_imputed['total_rooms'] 
df_imputed['pop_per_house'] = df_imputed['population']/df_imputed['households']

corrMat(df_imputed.select_dtypes(include=[np.number]))

# %% Filtering Features

filter_list = ['longitude',
               'latitude',
               'total_rooms',
               'total_bedrooms',
               'population',
               'households',
               'pop_per_house',
               'housing_median_age',
               'ocean_proximity']

df_imputed_filtered = df_imputed.drop(filter_list,axis=1)

# %% Scaling the Data

scaler = StandardScaler()
df_imputed_filtered_scaled = pd.DataFrame(scaler.fit_transform(df_imputed_filtered),columns = df_imputed_filtered.columns)
df_imputed_filtered_scaled.describe()

# %% Splitting into Training and Testing Data 
df_imputed_filtered_scaled_train,df_imputed_filtered_scaled_test = train_test_split(df_imputed_filtered_scaled,test_size=0.3,random_state=43) # 70-30 split

# %% Model Evaluation Function with Cross Validation

def modelEval(df,model):
    ''' Standard Cross Validation '''
    cv_score = np.sqrt(-cross_val_score(
        model,
        df.drop('median_house_value',axis=1),
        df['median_house_value'],
        cv=5,
        scoring='neg_mean_squared_error'))
    print("Scores:",cv_score)
    print("Mean:", cv_score.mean())
    print("std:", cv_score.std())


# %% Model Evaluation
print("Dummy Model")
modelEval(df_imputed_filtered_scaled,DummyRegressor())

print("\n Linear Regression Model")
modelEval(df_imputed_filtered_scaled,LinearRegression())

print("\n Random Forest Model")
modelEval(df_imputed_filtered_scaled,RandomForestRegressor(n_estimators=10,random_state=10))

# %% Pipeline Version of Model Evaluation

# Pipelines are a very neat way to group multiple steps of a model's preparation process, from feature matrix adjustment to the actual model evaluation step. Pipelines are also used to prevent data leakage.

# We'll create a simple pipeline that uses PolynomialFeatures() and see if that helps the model adjust to the nonlinear nature of a lot of our data.

# Model Evaluation Function w/ Pipelines
def modelEval_pipeline(df,model):
    pipe = Pipeline(steps=[('poly',PolynomialFeatures(2)),
                        #    ('power',PowerTransformer(method='yeo-johnson')), # Removing skew 
                                   ('model',model)])
    # Note I could put scaling in the pipeline also
    ''' Standard Cross Validation '''
    cv_score = np.sqrt(-cross_val_score(
        pipe,
        df.drop('median_house_value',axis=1),
        df['median_house_value'],
        cv=5,
        scoring='neg_mean_squared_error'))
    print("Scores:",cv_score)
    print("Mean:", cv_score.mean())
    print("std:", cv_score.std())


# %% Model Evaluation with Polynomial Features Included
print("Dummy Model")
modelEval_pipeline(df_imputed_filtered_scaled,DummyRegressor())

print("\n Linear Regression Model")
modelEval_pipeline(df_imputed_filtered_scaled,LinearRegression())

print("\n Random Forest Model")
modelEval_pipeline(df_imputed_filtered_scaled,RandomForestRegressor(n_estimators=10,random_state=10))
# %%
