# %% Importing Packages
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.diagnostics import hpdi
import jax.numpy as jnp
from jax import random, vmap

from scipy.stats import norm, halfnorm
import matplotlib.pyplot as plt
from jax import random
import arviz as az

import seaborn as sns

from sklearn.preprocessing import StandardScaler

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

# %% Bayesian Linear Regression
################# Bayesian Linear Regression
# Define the model
def bayesian_model(X=None, y=None):
    # Priors
    beta = numpyro.sample('beta', dist.Normal(jnp.zeros(X.shape[1]), jnp.ones(X.shape[1])))
    sigma = numpyro.sample('sigma', dist.HalfNormal(10.))
    # Likelihood
    mu = jnp.dot(X, beta)
    numpyro.sample('y', dist.Normal(mu, sigma), obs=y)

# Create the data
X_train_np = jnp.array(X_train.values)#[:2000]
y_train_np = jnp.array(y_train.values)#[:2000]

X_test_np = jnp.array(X_test.values)
y_test_np = jnp.array(y_test.values)

# %% Plotting the Prior Distributions

rng_key = random.PRNGKey(3)
rng_key, rng_key_ = random.split(rng_key)

fig, axs = plt.subplots(2)
#Beta
axs[0].set_title('Beta')
xs = np.linspace(-3,3,100)
ys = norm.pdf(xs, 0, 1)
axs[0].plot(xs, ys, lw=2, label='Scipy')
axs[0].hist(dist.Normal(0.0,1.0).sample(rng_key,(10000,)),density=True,bins=100,label='Numpyro')
axs[0].legend()
#Sigma
axs[1].set_title('Sigma')
xs = np.linspace(-1,30,100)
ys = halfnorm.pdf(xs, 0.0, 1.0)
axs[1].plot(xs, ys, lw=2, label='Scipy')
axs[1].hist(dist.HalfNormal(1.).sample(rng_key,(10000,)),density=True,bins=100,label='Numpyro')
axs[1].legend()

plt.tight_layout()

# %% Prior Predictive Test

rng_key, rng_key_ = random.split(rng_key)
prior_predictive = Predictive(bayesian_model, num_samples=100)
prior_predictions = prior_predictive(rng_key_, X=X_train_np)["y"]

print(pd.DataFrame(prior_predictions.T).describe().iloc[:, : 5])
print(pd.DataFrame(y_train_np).describe())

# %% Train the model

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
nuts_kernel = NUTS(bayesian_model)
mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=2000, num_chains=4)
mcmc.run(rng_key_, X_train_np, y_train_np)

# Convert the MCMC samples to an ArviZ InferenceData object
posterior_samples = mcmc.get_samples()
inference_data = az.from_numpyro(mcmc)
inference_data.posterior = inference_data.posterior.assign_coords(beta_dim_0=X.columns)
summary = az.summary(inference_data)
print(summary)

# %% Posterior Predictives

rng_key, rng_key_ = random.split(rng_key)
samples = mcmc.get_samples()
predictive = Predictive(bayesian_model, samples)
predictions = predictive(rng_key_, X=X_test_np)["y"]

# %% Mean Squared Error
mean_postpred = jnp.mean(predictions, axis=0)
mse_bayesian = mean_squared_error(y_test, mean_postpred)
print(f"Bayesian Regression Mean Squared Error: {mse_bayesian}")

# %% Converting Numpryo Inference Data to ArviZ
inference_data = az.from_numpyro(
    mcmc,
    prior=prior_predictive(rng_key_, X=X_train_np),
    posterior_predictive=predictive(rng_key_, X=X_test_np)
    )

# %% Plotting Posterior Predictive Distribution
az.plot_ppc(inference_data, data_pairs={"y":"y"})

# %% Forest Plot
plt.figure(figsize=(10,5))
az.plot_forest(inference_data,
               ess=True,
               r_hat=True,)
plt.tight_layout()
plt.show()

# %%
