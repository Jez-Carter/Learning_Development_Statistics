
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
is_train <- sample(c(TRUE, FALSE), size=nrow(df), replace=TRUE, prob=c(0.9, 0.1))

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


