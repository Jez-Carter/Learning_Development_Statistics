# %% Importing Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from tinygp import kernels, GaussianProcess
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

import jaxopt
from functools import partial

jax.config.update("jax_enable_x64", True)

# %% Load a sample dataset
from sklearn.datasets import fetch_california_housing
california = fetch_california_housing()
data = pd.DataFrame(california.data, columns=california.feature_names)
data['MedHouseVal'] = california.target

data = data.sample(n=len(data))

scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data),columns = data.columns)

# %% Exploring data relationships
sns.pairplot(data[:100])

# %% Split the data into training and testing sets
X = data_scaled.drop('MedHouseVal', axis=1)
y =  data_scaled['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% Converting to Jax Arrays
X_jnp = jnp.array(X[['Latitude','Longitude']].values)
y_jnp = jnp.array(y.values)

X_train_jnp = jnp.array(X_train[['Latitude','Longitude']].values)
y_train_jnp = jnp.array(y_train.values)

X_test_jnp = jnp.array(X_test[['Latitude','Longitude']].values)
y_test_jnp = jnp.array(y_test.values)

# %%
class GreatCircleDistance(kernels.stationary.Distance):
    def distance(self, X1, X2):
        lat1,lat2 = X1[0]*jnp.pi/180,X2[0]*jnp.pi/180
        lon1,lon2 = X1[1]*jnp.pi/180,X2[1]*jnp.pi/180
        
        a = jnp.sin((lat1-lat2)/2)**2+jnp.cos(lat1)*jnp.cos(lat2)*jnp.sin((lon1-lon2)/2)**2
        c = 2 * jnp.arctan2(jnp.sqrt(a),jnp.sqrt(1-a))
        d = 6371 * c
        return d

def build_gp(params,X):
    kernel = jnp.exp(params["log_amp"]) * kernels.Matern52(
        jnp.exp(params["log_scale"]), distance=GreatCircleDistance()
    )
    return GaussianProcess(
        kernel,
        X,
        diag=jnp.exp(2 * params["log_sigma"]),
        mean=params["mean"],
    )

@jax.jit
def neg_log_likelihood(params, X, y):
    gp = build_gp(params, X)
    return -gp.log_probability(y)

params = {
    "log_amp": np.zeros(()),
    "log_scale": np.zeros(()),
    "log_sigma": np.zeros(()),
    "mean": np.zeros(()),
}

# %%
X_mean = X.drop(['Latitude','Longitude'], axis=1)
X_mean_train = X_train.drop(['Latitude','Longitude'], axis=1)
X_mean_test = X_test.drop(['Latitude','Longitude'], axis=1)

X_jnp_mean = jnp.array(X_mean.values)
X_mean_train_jnp = jnp.array(X_mean_train.values)
X_mean_test_jnp = jnp.array(X_mean_test.values)

# %%
def mean_function(params, X):
    b0 = params["b0"]
    weights = params["weights"]
    return(b0+jnp.sum(weights*X))


# %%
mean_params = {
    "b0": np.zeros(()),
    "weights": jnp.zeros(6),
}

# %%
mean_function(mean_params,X_mean_train_jnp[0])

# %%
def build_gp(params,X,X_mean):
    kernel = jnp.exp(params["log_amp"]) * kernels.Matern52(
        jnp.exp(params["log_scale"]), distance=GreatCircleDistance()
    )
    return GaussianProcess(
        kernel,
        X,
        diag=jnp.exp(2 * params["log_sigma"]),
        mean=partial(mean_function, params)(X_mean),
    )

@jax.jit
def neg_log_likelihood(params, X,X_mean, y):
    gp = build_gp(params, X, X_mean)
    return -gp.log_probability(y)

mean_params = {
    "b0": np.zeros(()),
    "weights": jnp.zeros(6),
}

params = {
    "log_amp": np.zeros(()),
    "log_scale": np.zeros(()),
    "log_sigma": np.zeros(()),
    **mean_params,
}

# %%
jnp.array(X.values)[:,-2:].shape

# %%


# %%


def build_gp(params,X):
    kernel = jnp.exp(params["log_amp"]) * kernels.Matern52(
        jnp.exp(params["log_scale"]), distance=GreatCircleDistance()
    )
    return GaussianProcess(
        kernel,
        X[:,-2:],
        diag=jnp.exp(2 * params["log_sigma"]),
        mean=partial(mean_function, params)(X[:,:-2])
    )

@jax.jit
def neg_log_likelihood(params, X, y):
    gp = build_gp(params, X)
    return -gp.log_probability(y)

mean_params = {
    "b0": np.zeros(()),
    "weights": jnp.zeros(6),
}

params = {
    "log_amp": np.zeros(()),
    "log_scale": np.zeros(()),
    "log_sigma": np.zeros(()),
    **mean_params,
}

# %%
X_train_jnp = jnp.array(X_train.values)
X_test_jnp = jnp.array(X_test.values)

# %%
X_train_jnp[:1000].shape

# %% Fitting the GP

solver = jaxopt.ScipyMinimize(fun=neg_log_likelihood)
soln = solver.run(
    params,
    X=X_train_jnp[:1000],
    y=y_train_jnp[:1000])
print(f"Final negative log likelihood: {soln.state.fun_val}")
print(f"Parameter Estimates: {soln.params}")

# %% Fitting the GP

solver = jaxopt.ScipyMinimize(fun=neg_log_likelihood)
soln = solver.run(
    params,
    X=X_train_jnp[:1000],
    X_mean=X_mean_train_jnp[:1000],
    y=y_train_jnp[:1000])
print(f"Final negative log likelihood: {soln.state.fun_val}")
print(f"Parameter Estimates: {soln.params}")

# %%
X_test_jnp.shape

# %%
X_train_jnp[:6000].shape

# %%
y_train_jnp[:6000].shape

# %%
gp = build_gp(soln.params, X_train_jnp[:6000])
y_pred = gp.condition(y_train_jnp[:6000])
# %%
y_pred = gp.condition(y_train_jnp[:6000], X_test_jnp).gp.loc

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


# %%
soln.params

# %% Fitting the GP

solver = jaxopt.ScipyMinimize(fun=neg_log_likelihood)
soln = solver.run(params, X=X_train_jnp[:1000], y=y_train_jnp[:1000])
print(f"Final negative log likelihood: {soln.state.fun_val}")
print(f"Parameter Estimates: {soln.params}")

# %%
gp = build_gp(soln.params, X_train_jnp[:6000])
y_pred = gp.condition(y_train_jnp[:6000], X_test_jnp).gp.loc

# %%
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")




