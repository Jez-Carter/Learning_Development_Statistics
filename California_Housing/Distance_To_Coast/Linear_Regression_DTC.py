# %% Importing Packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# %% Load a sample dataset

data = pd.read_pickle('~/Machine_Learning_Revision/California_Housing/Auxiliary_Data/california_housing_dtc.pkl')

data = data.sample(n=len(data))

scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data),columns = data.columns)

# %% Exploring data relationships
# sns.pairplot(data[:100],alpha=0.5)
sns.pairplot(data[:1000],
             kind='reg',
             plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.1}})


# %% Split the data into training and testing sets
# X = data_scaled.drop('MedHouseVal', axis=1)
X = data_scaled.drop(['Latitude','Longitude','MedHouseVal'], axis=1)

y =  data_scaled['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% Stats Models
    
model = sm.OLS(y_train, X_train)
results = model.fit()
print(results.summary())

# %% Make predictions
y_pred = results.predict(X_test)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# %%
