
# Kaggle Example for R-INLA Package 
# https://www.kaggle.com/code/martiningram/spatial-inla-ca-housing-prices

# %% Loading the data 
file_path <- normalizePath('Kaggle/California_Housing_Prices/Data/housing.csv')
df <- read.csv(file_path, stringsAsFactors = TRUE)

# %% Showing the first few rows of the dataframe
head(df)
summary(df)

# Considerations:
# Log Transforms - For all positive columns it can be useful to consider log transformations.
# N/As in total_bedrooms could be imputed with the mean/median value.
# Median house value is capped at $500,000, which could be an issue for the model.
    # Strategy 1 is to ignore this and a downward bias will be introduced.
    # Strategy 2 is to remove the capped values.
    # Strategy 3 is to fit censored regression, which models these points as at least 500,000.
# Feature engineering - divide total rooms and total bedrooms by number of households.
# Filtering out observations on an island. 

# %% Data Preprocessing
# Drop Island level
df <- df[df$ocean_proximity != 'ISLAND', ]
df$ocean_proximity <- droplevels(df$ocean_proximity)

# Impute using median
df$total_bedrooms[is.na(df$total_bedrooms)] = median(df$total_bedrooms , na.rm = TRUE)

# Normalise per household and log-transform
df$log_rooms_per_household <- log(df$total_rooms / df$households)
df$log_bedrooms_per_household <- log(df$total_bedrooms / df$households)

# Log-transform the other all-positive quantities
df$log_median_income <- log(df$median_income)
df$log_households <- log(df$households)
df$log_population <- log(df$population)
df$log_median_house_value <- log(df$median_house_value)
df$log_housing_median_age <- log(df$housing_median_age)

# %% Plotting the covariates
library(ggplot2)
library(reshape2)
ggplot(data = melt(df), mapping = aes(x = value)) +
  geom_histogram(bins = 30) + facet_wrap(~variable, scales = "free_x") +
  theme_minimal()

# %% Plotting on a map
library(sf)
library("rnaturalearth")
library("rnaturalearthdata")
library("maps")

spat_df <- st_as_sf(df, coords = c("longitude", "latitude"), crs = 4326)
spat_df$log_median_house_value <- log10(spat_df$median_house_value)

states <- st_as_sf(map("state", plot = FALSE, fill = TRUE))

library(ggplot2)

ggplot(data = states[4, ]) +
  geom_sf() +
  geom_sf(aes(colour = log_median_house_value), data = spat_df) +
  theme_classic()

# %% Train and Test Split
set.seed(2)
# We'll use 90% of the data in our training set and the rest in the test set
is_train <- sample(c(TRUE, FALSE), size = nrow(df), replace = TRUE, prob = c(0.9, 0.1))

train_df <- df[is_train, ]
test_df <- df[!is_train, ]

# %% Examine Test Set
spat_df$is_train <- factor(is_train, levels=c(TRUE, FALSE))

ggplot(data = states[4, ]) +
  geom_sf() +
  geom_sf(aes(colour=is_train), data = spat_df, alpha = 0.4) +
  theme_classic()

# %% Fitting a Linear Model
library(INLA)

# For prediction, treat the test set outcomes as missing data:
test_df_pred <- test_df
test_df_pred$median_house_value <- NA

# Put these together:
inla_data_pred <- rbind(train_df, test_df_pred)

# Let's fit a linear model:
m0 <- inla(log_median_house_value ~ log_median_income + ocean_proximity + log_rooms_per_household
            + log_bedrooms_per_household + log_households + log_housing_median_age + log_population, 
           data = inla_data_pred,
           control.predictor = list(compute = TRUE, link = 1), 
           control.compute=list(return.marginals.predictor=TRUE), family = 'gaussian')


# %% Summary of covariates
summary(m0)

# It makes sense that the median income is the most important predictor.
# The 'INLAND' category is also important with a negative effect.

# %% Predictions for Test Set

n_train <- nrow(train_df)
n_test <- nrow(test_df)
start_test <- n_train + 1
end_test <- n_train + n_test

# INLA does have some support for computing posterior distributions, but unfortunately they don't include the observation noise. So I have to do a bit of maths here to add on the observation noise. We can then use the fact that our outcome has a log-normal distribution and compute its mean, and also plot the posterior predictive distribution.

posterior_predictive_log_normal <- function(model, marginal_summaries) {
  
  # Work out the approximate observation standard deviation
  approx_obs_sd <- sqrt(1 / model$summary.hyperpar['Precision for the Gaussian observations', 'mean'])
  
  # Add on the observation noise
  total_sd <- marginal_summaries$sd + approx_obs_sd
  
  # Put these together with the means
  marginals <- cbind(mean=marginal_summaries$mean, sd=total_sd)
  
  # Compute log-normal distribution mean:
  pred_means = exp(marginals[, 'mean'] + total_sd^2 / 2)
  
  list(marginals = marginals, pred_means = pred_means)
  
}

rel_marginal_summaries <- m0$summary.linear.predictor[start_test:end_test, ]
preds <- posterior_predictive_log_normal(m0, rel_marginal_summaries)

marginals <- preds$marginals
pred_means <- preds$pred_means

# %% Plotting a Prediction

i <- 3
example <- marginals[i, ]
min_val <- exp(example['mean'] - 6 * example['sd'])
max_val <- exp(example['mean'] + 6 * example['sd'])
to_plot <- seq(min_val, max_val, length.out=100)
y_plot <- dlnorm(to_plot, example['mean'], example['sd'])
actual_value <- test_df$median_house_value[i]
plot(to_plot, y_plot, type='l')
abline(v=actual_value, col='red')

# %% Limiting Predictions to 500,000

pred_means_censored <- ifelse(pred_means > 500000, 500000, pred_means)
plot(pred_means_censored, test_df$median_house_value, asp=1)

# %% Computing the RMSE

square_errors <- ((pred_means_censored - test_df$median_house_value)^2)
mse <- mean(square_errors)

sqrt(mse)

# %% SPDE Approach - MESH
coords <- as.matrix(inla_data_pred[, c('longitude', 'latitude')])

# Find boundary from the points
pts.bound <- inla.nonconvex.hull(coords, 0.3, 0.3)

# Define mesh (parameters from Parana example)
mesh <- inla.mesh.2d(
  loc=coords, boundary = pts.bound, max.edge=c(0.3, 1), offset = c(1e-5, 1.5), cutoff = 0.1
  # Some other ones I played with:
  # loc=coords, boundary = pts.bound, max.edge=c(0.1, 1), offset = c(1e-2, 1.5), cutoff = 0.13
  # loc=coords, boundary = boundary_alt, max.edge=c(0.3, 1), offset = c(1e-5, 1.5), cutoff = 0.1
)

plot(mesh)

# %% Define the projector matrix
A <- inla.spde.make.A(mesh, loc = coords)

# %% Set Priors
# Define priors using PC prior
spde = inla.spde2.pcmatern(mesh=mesh,
                           prior.range = c(0.05, 0.01), # P(practic.range < 0.05) = 0.01
                           prior.sigma = c(1., 0.01) # P(sigma > 1) = 0.01
)

# Stack the data. This puts everything together for INLA to fit.
stk.dat <- inla.stack(
  data = list(y = inla_data_pred$log_median_house_value),
  A = list(A, 1),
  effects = list(list(s = 1:spde$n.spde),
                 # Covariates go here:
                 data.frame(Intercept = 1,
                            log_income = log(inla_data_pred$median_income),
                            ocean_dist = inla_data_pred$ocean_proximity,
                            log_bedrooms = inla_data_pred$log_bedrooms_per_household,
                            log_rooms = inla_data_pred$log_rooms_per_household,
                            log_households = log(inla_data_pred$households),
                            log_population = log(inla_data_pred$population),
                            log_median_age = log(inla_data_pred$housing_median_age))
                 ),
  tag = 'dat'
)

# %% Fit the model

f.spat <- y ~ 0 + log_income + ocean_dist + f(s, model = spde) + log_bedrooms + log_rooms +
              log_households + log_population + log_median_age

r.spat <- inla(f.spat, family = 'gaussian', data = inla.stack.data(stk.dat), 
               control.predictor = list(A = inla.stack.A(stk.dat), compute = TRUE, link = 1),
                control.inla = list(int.strategy = "eb"),
               control.compute=list(return.marginals.predictor = TRUE))

# %% Summary of the covariates
summary(r.spat)

# The differences between the different ocean distances are now quite small, so the spatial term has probably made them unnecessary. Some other things are interesting; for example, the log_bedrooms coefficient has switched signs from before.

# %% Predictions for the Test Set

rel_marginals <- r.spat$summary.fitted.values[start_test:end_test, ]

predictions <- posterior_predictive_log_normal(r.spat, rel_marginals)

# We can plot one:
# I'll show the actual observed value in red.
i <- 28
example <- predictions$marginals[i, ]
min_val <- exp(example['mean'] - 6 * example['sd'])
max_val <- exp(example['mean'] + 6 * example['sd'])
to_plot <- seq(min_val, max_val, length.out=100)
y_plot <- dlnorm(to_plot, example['mean'], example['sd'])
actual_value <- test_df$median_house_value[i]
plot(to_plot, y_plot, type='l')
abline(v=actual_value, col='red')

# %% Limiting Predictions to 500,000
pred_means_censored <- ifelse(predictions$pred_means > 500000, 500000, predictions$pred_means)
pred_means <- predictions$pred_means

plot(pred_means_censored, test_df$median_house_value)

# %% Computing the RMSE
square_errors <- ((pred_means - test_df$median_house_value)^2)
mse <- mean(square_errors)
sqrt(mse)

square_errors_censored <- ((pred_means_censored - test_df$median_house_value)^2)
mse_censored <- mean(square_errors_censored)
sqrt(mse_censored)

# %% Plotting the Predictions on a Map
# Set the grid for prediction:
stepsize <- 1 / 111

x.range <- diff(range(df$longitude))
y.range <- diff(range(df$latitude))
nxy <- round(c(x.range, y.range) / stepsize)

# This is the grid size in pixels:
nxy

# INLA code to make the grid
projgrid <- inla.mesh.projector(mesh, xlim = range(df$longitude), ylim = range(df$latitude), dims = nxy)

# INLA code to get mean and sd of the spatial prediction
xmean <- inla.mesh.project(projgrid, r.spat$summary.random$s$mean)
xsd <- inla.mesh.project(projgrid, r.spat$summary.random$s$sd)

# Some plotting tools:
library(lattice)
library(viridisLite)
library(sf)

# I downloaded a state boundary to only show values where needed:
boundary_path <- normalizePath('Kaggle/California_Housing_Prices/Data/CA_State_TIGER2016.shp')
boundary <- st_read(boundary_path)
boundary_other_coords <- st_transform(boundary, crs = 4326)

# Set points outside CA to NA
library(splancs)

pts <- st_as_sf(data.frame(projgrid$lattice$loc), coords = c('X1', 'X2'), crs = 4326)
xy.in <- st_within(pts, boundary_other_coords, sparse = FALSE)

xmean_sparse <- xmean
xmean_sparse[!xy.in] <- NA
xsd_sparse <- xsd
xsd_sparse[!xy.in] <- NA

# Turn these into rasters:
library(raster)

z <- as.vector(xmean_sparse)
xyz <- cbind(projgrid$lattice$loc, z)
xyz_sd <- cbind(projgrid$lattice$loc, as.vector(xsd_sparse))

pred_raster_mean <- rasterFromXYZ(xyz)
pred_raster_sd <- rasterFromXYZ(xyz_sd)

crs(pred_raster_mean) <- "+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0" 
crs(pred_raster_sd) <- "+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"

plot(pred_raster_mean)

# %% Interactive Map

library(mapview)

mapview(pred_raster_mean)

# %% Comparison to Random Forest

library(randomForest)

cov_names <- c('longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households',
               'median_income', 'log_rooms_per_household', 'log_bedrooms_per_household', 'ocean_proximity')

X <- train_df[, cov_names]
y <- train_df$median_house_value

m <- randomForest(x=X, y=y)

rf_preds <- predict(m, test_df[, cov_names])

rf_mse <- sqrt(mean((rf_preds - test_df$median_house_value)^2))

rf_mse

# %% Discussion

# The random forest does perform well and has about the same RMSE as the spatial model (~49,500 for the RF vs ~49,300 for the spatial model). So the random forest is not actually more accurate, despite being much less interpretable. In detail, I think the advantages of the statistical model are:

# We can make a map of the spatial effect in isolation. That would be impossible in the random forest, because everything potentially interacts with everything else, so the map for one value of the median income, say, could be quite different from that for another.
# We can also look at the effect of individual covariates by looking at coefficients.
# With the statistical model, we also get a full distribution of values, rather than a single most likely guess.