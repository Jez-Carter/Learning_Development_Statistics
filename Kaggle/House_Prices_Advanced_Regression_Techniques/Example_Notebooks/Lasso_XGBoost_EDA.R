
# https://www.kaggle.com/code/erikbruin/house-prices-lasso-xgboost-and-a-detailed-eda

# %% Importing relevant libraries

library(knitr)
library(ggplot2)
library(plyr)
library(dplyr)
library(corrplot)
library(caret)
library(gridExtra)
library(scales)
library(Rmisc)
library(ggrepel)
library(randomForest)
library(psych)
library(xgboost)
library(Ckmeans.1d.dp) #required for ggplot clustering

# %% Reading the data

file_path_test <- normalizePath('Kaggle/House_Prices_Advanced_Regression_Techniques/Data/test.csv')
file_path_train <- normalizePath('Kaggle/House_Prices_Advanced_Regression_Techniques/Data/train.csv')

test <- read.csv(file_path_test, stringsAsFactors = TRUE)
train <- read.csv(file_path_train, stringsAsFactors = TRUE)

# %% Glimpse of the data
dim(train)
str(train[,c(1:10, 81)])

dim(test)
str(test[,c(1:10, 80)])

# %% Removing ID column
test$Id <- NULL
train$Id <- NULL

test$SalePrice <- NA
all <- rbind(train, test)

# %% Exploring important variables

ggplot(data=all[!is.na(all$SalePrice),], aes(x=SalePrice)) +
    geom_histogram(fill="blue", binwidth = 10000) +
    scale_x_continuous(breaks= seq(0, 800000, by=100000), labels = comma)

summary(all$SalePrice)

# %% Correlations with response variable
numericVars <- which(sapply(all, is.numeric)) #index vector numeric variables
numericVarNames <- names(numericVars) #saving names vector for use later on
cat('There are', length(numericVars), 'numeric variables')

all_numVar <- all[, numericVars]
cor_numVar <- cor(all_numVar, use="pairwise.complete.obs") #correlations of all numeric variables

#sort on decreasing correlations with SalePrice
cor_sorted <- as.matrix(sort(cor_numVar[,'SalePrice'], decreasing = TRUE))
 #select only high corelations
CorHigh <- names(which(apply(cor_sorted, 1, function(x) abs(x)>0.5)))
cor_numVar <- cor_numVar[CorHigh, CorHigh]

corrplot.mixed(cor_numVar, tl.col="black", tl.pos = "lt")

# %% Examining most important predictors in detail (OverallQual)

ggplot(data=all[!is.na(all$SalePrice),], aes(x=factor(OverallQual), y=SalePrice))+
        geom_boxplot(col='blue') + labs(x='Overall Quality') +
        scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma)

# %% Examining most important predictors in detail (GrLivArea)
ggplot(data=all[!is.na(all$SalePrice),], aes(x=GrLivArea, y=SalePrice))+
geom_point(col='blue') + geom_smooth(method = "lm", se=FALSE, color="black", aes(group=1)) +
scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) +
geom_text_repel(aes(label = ifelse(all$GrLivArea[!is.na(all$SalePrice)]>4500, rownames(all), '')))

# %% Checking for missing values
NAcol <- which(colSums(is.na(all)) > 0)
sort(colSums(sapply(all[NAcol], is.na)), decreasing = TRUE)

# %% Imputing missing values

# Pool Variables
levels(all$PoolQC) <- c(levels(all$PoolQC), 'None')
all$PoolQC[is.na(all$PoolQC)] <- 'None'
Qualities <- c('None' = 0, 'Po' = 1, 'Fa' = 2, 'TA' = 3, 'Gd' = 4, 'Ex' = 5)
all$PoolQC <- as.integer(as.character(revalue(all$PoolQC, Qualities)))

all[all$PoolArea>0 & all$PoolQC==0, c('PoolArea', 'PoolQC', 'OverallQual')]

all$PoolQC[2421] <- 2
all$PoolQC[2504] <- 3
all$PoolQC[2600] <- 2

# MiscFeature

levels(all$MiscFeature) <- c(levels(all$MiscFeature), 'None')
all$MiscFeature[is.na(all$MiscFeature)] <- 'None'

# Alley
levels(all$Alley) <- c(levels(all$Alley), 'None')
all$Alley[is.na(all$Alley)] <- 'None'

# Fence
levels(all$Fence) <- c(levels(all$Fence), 'None')
all$Fence[is.na(all$Fence)] <- 'None'

# FireplaceQu
levels(all$FireplaceQu) <- c(levels(all$FireplaceQu), 'None')
all$FireplaceQu[is.na(all$FireplaceQu)] <- 'None'
all$FireplaceQu <- as.integer(as.character(revalue(all$FireplaceQu, Qualities)))

# LotFrontage
for (i in 1:nrow(all)){
        if(is.na(all$LotFrontage[i])){
               all$LotFrontage[i] <- as.integer(median(all$LotFrontage[all$Neighborhood==all$Neighborhood[i]], na.rm=TRUE)) 
        }
}

# LotShape
all$LotShape<-as.integer(as.character(revalue(all$LotShape, c('IR3'=0, 'IR2'=1, 'IR1'=2, 'Reg'=3))))

# GarageYrBlt
all$GarageYrBlt[is.na(all$GarageYrBlt)] <- all$YearBuilt[is.na(all$GarageYrBlt)]

# GarageType
all$GarageCond[2127] <- names(sort(-table(all$GarageCond)))[1]
all$GarageQual[2127] <- names(sort(-table(all$GarageQual)))[1]
all$GarageFinish[2127] <- names(sort(-table(all$GarageFinish)))[1]

# Garage Cars and Area
all$GarageCars[2577] <- 0
all$GarageArea[2577] <- 0
all$GarageType[2577] <- NA

# Garage Type NA's 
levels(all$GarageType) <- c(levels(all$GarageType), 'None')
all$GarageType[is.na(all$GarageType)] <- 'None'

# GarageFinish
levels(all$GarageFinish) <- c(levels(all$GarageFinish), 'None')
all$GarageFinish[is.na(all$GarageFinish)] <- 'None'
Finish <- c('None'=0, 'Unf'=1, 'RFn'=2, 'Fin'=3)
all$GarageFinish <- as.integer(as.character(revalue(all$GarageFinish, Finish)))

# GarageQual
levels(all$GarageQual) <- c(levels(all$GarageQual), 'None')
all$GarageQual[is.na(all$GarageQual)] <- 'None'
all$GarageQual <- as.integer(as.character(revalue(all$GarageQual, Qualities)))

# GarageCond
levels(all$GarageCond) <- c(levels(all$GarageCond), 'None')
all$GarageCond[is.na(all$GarageCond)] <- 'None'
all$GarageCond <- as.integer(as.character(revalue(all$GarageCond, Qualities)))

# Basement Vars
all$BsmtFinType2[333] <- names(sort(-table(all$BsmtFinType2)))[1]
all$BsmtExposure[c(949, 1488, 2349)] <- names(sort(-table(all$BsmtExposure)))[1]
all$BsmtCond[c(2041, 2186, 2525)] <- names(sort(-table(all$BsmtCond)))[1]
all$BsmtQual[c(2218, 2219)] <- names(sort(-table(all$BsmtQual)))[1]

# BsmtQual
levels(all$BsmtQual) <- c(levels(all$BsmtQual), 'None')
all$BsmtQual[is.na(all$BsmtQual)] <- 'None'
all$BsmtQual <- as.integer(as.character(revalue(all$BsmtQual, Qualities)))

# BsmtCond
levels(all$BsmtCond) <- c(levels(all$BsmtCond), 'None')
all$BsmtCond[is.na(all$BsmtCond)] <- 'None'
all$BsmtCond <- as.integer(as.character(revalue(all$BsmtCond, Qualities)))

# GarageFinish
levels(all$BsmtExposure) <- c(levels(all$BsmtExposure), 'None')
all$BsmtExposure[is.na(all$BsmtExposure)] <- 'None'
Exposure <- c('None'=0, 'No'=1, 'Mn'=2, 'Av'=3, 'Gd'=4)
all$BsmtExposure <- as.integer(as.character(revalue(all$BsmtExposure, Exposure)))

# BsmtFinType1
levels(all$BsmtFinType1) <- c(levels(all$BsmtFinType1), 'None')
all$BsmtFinType1[is.na(all$BsmtFinType1)] <- 'None'
FinType <- c('None'=0, 'Unf'=1, 'LwQ'=2, 'Rec'=3, 'BLQ'=4, 'ALQ'=5, 'GLQ'=6)
all$BsmtFinType1 <- as.integer(as.character(revalue(all$BsmtFinType1, FinType)))

# BsmtFinType2
levels(all$BsmtFinType2) <- c(levels(all$BsmtFinType2), 'None')
all$BsmtFinType2[is.na(all$BsmtFinType2)] <- 'None'
all$BsmtFinType2 <- as.integer(as.character(revalue(all$BsmtFinType2, FinType)))

# BsmtFullBath, BsmtHalfBath, BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF
all$BsmtFullBath[is.na(all$BsmtFullBath)] <-0
all$BsmtHalfBath[is.na(all$BsmtHalfBath)] <-0
all$BsmtFinSF1[is.na(all$BsmtFinSF1)] <-0
all$BsmtFinSF2[is.na(all$BsmtFinSF2)] <-0
all$BsmtUnfSF[is.na(all$BsmtUnfSF)] <-0
all$TotalBsmtSF[is.na(all$TotalBsmtSF)] <-0

# MasVnrType
all$MasVnrType[2611] <- names(sort(-table(all$MasVnrType)))[2]

# MasVnrType
levels(all$MasVnrType) <- c(levels(all$MasVnrType), 'None')
all$MasVnrType[is.na(all$MasVnrType)] <- 'None'
Masonry <- c('None'=0, 'BrkCmn'=0, 'BrkFace'=1, 'Stone'=2)
all$MasVnrType <- as.integer(as.character(revalue(all$MasVnrType, Masonry)))

# MasVnrArea
all$MasVnrArea[is.na(all$MasVnrArea)] <-0

# Zoning
all$MSZoning[is.na(all$MSZoning)] <- names(sort(-table(all$MSZoning)))[1]

# KitchenQual
all$KitchenQual[is.na(all$KitchenQual)] <- 'TA'
all$KitchenQual <- as.integer(as.character(revalue(all$KitchenQual, Qualities)))

# Utilities
all$Utilities <- NULL

# Functional
all$Functional[is.na(all$Functional)] <- names(sort(-table(all$Functional)))[1]
Functional <- c('Sal'=0, 'Sev'=1, 'Maj2'=2, 'Maj1'=3, 'Mod'=4, 'Min2'=5, 'Min1'=6, 'Typ'=7)
all$Functional <- as.integer(as.character(revalue(all$Functional, Functional)))

# Exterior1st
all$Exterior1st[is.na(all$Exterior1st)] <- names(sort(-table(all$Exterior1st)))[1]

# Exterior2nd
all$Exterior2nd[is.na(all$Exterior2nd)] <- names(sort(-table(all$Exterior2nd)))[1]

# ExterQual
all$ExterQual <- as.integer(as.character(revalue(all$ExterQual, Qualities)))

# ExterCond
all$ExterCond <- as.integer(as.character(revalue(all$ExterCond, Qualities)))

# Electrical
all$Electrical[is.na(all$Electrical)] <- names(sort(-table(all$Electrical)))[1]

# SaleType
all$SaleType[is.na(all$SaleType)] <- names(sort(-table(all$SaleType)))[1]

# %% Variables without missing values

# HeatingQC
all$HeatingQC <- as.integer(as.character(revalue(all$HeatingQC, Qualities)))

# CentralAir
all$CentralAir <- as.integer(as.character(revalue(all$CentralAir, c('N'=0, 'Y'=1))))

# LandSlope
all$LandSlope <- as.integer(as.character(revalue(all$LandSlope, c('Sev'=0, 'Mod'=1, 'Gtl'=2))))

# Street
all$Street <- as.integer(as.character(revalue(all$Street, c('Grvl'=0, 'Pave'=1))))

# PavedDrive
all$PavedDrive <- as.integer(as.character(revalue(all$PavedDrive, c('N'=0, 'P'=1, 'Y'=2))))

# %% Changing numeric variables to factors (year and month sold)
all$MoSold <- as.factor(all$MoSold)

all$MSSubClass <- as.factor(all$MSSubClass)
dwellings <- c('20'='1 story 1946+', '30'='1 story 1945-', '40'='1 story unf attic', '45'='1,5 story unf', '50'='1,5 story fin', '60'='2 story 1946+', '70'='2 story 1945-', '75'='2,5 story all ages', '80'='split/multi level', '85'='split foyer', '90'='duplex all style/age', '120'='1 story PUD 1946+', '150'='1,5 story PUD all', '160'='2 story PUD 1946+', '180'='PUD multilevel', '190'='2 family conversion')
all$MSSubClass<-revalue(all$MSSubClass,dwellings)

# %% Counting types of Variables
numericVars <- which(sapply(all, is.numeric)) #index vector numeric variables
factorVars <- which(sapply(all, is.factor)) #index vector factor variables
cat('There are', length(numericVars), 'numeric variables, and', length(factorVars), 'categoric variables')

# %% Re-examining Correlations for Numeric Variables
all_numVar <- all[, numericVars]
cor_numVar <- cor(all_numVar, use="pairwise.complete.obs") #correlations of all numeric variables

#sort on decreasing correlations with SalePrice
cor_sorted <- as.matrix(sort(cor_numVar[,'SalePrice'], decreasing = TRUE))
 #select only high corelations
CorHigh <- names(which(apply(cor_sorted, 1, function(x) abs(x)>0.5)))
cor_numVar <- cor_numVar[CorHigh, CorHigh]

corrplot.mixed(cor_numVar, tl.col="black", tl.pos = "lt", tl.cex = 0.7,cl.cex = .7, number.cex=.7)

# %% Quick Random Forest Model to check for important variables
set.seed(2018)
quick_RF <- randomForest(x=all[1:1460,-79], y=all$SalePrice[1:1460], ntree=100,importance=TRUE)
imp_RF <- importance(quick_RF)
imp_DF <- data.frame(Variables = row.names(imp_RF), MSE = imp_RF[,1])
imp_DF <- imp_DF[order(imp_DF$MSE, decreasing = TRUE),]

ggplot(imp_DF[1:20,], aes(x=reorder(Variables, MSE), y=MSE, fill=MSE)) + geom_bar(stat = 'identity') + labs(x = 'Variables', y= '% increase MSE if variable is randomly permuted') + coord_flip() + theme(legend.position="none")

# %% Examining Distributions for most important variables

s1 <- ggplot(data= all, aes(x=GrLivArea)) +
        geom_density() + labs(x='Square feet living area')
s2 <- ggplot(data=all, aes(x=as.factor(TotRmsAbvGrd))) +
        geom_histogram(stat='count') + labs(x='Rooms above Ground')
s3 <- ggplot(data= all, aes(x=X1stFlrSF)) +
        geom_density() + labs(x='Square feet first floor')
s4 <- ggplot(data= all, aes(x=X2ndFlrSF)) +
        geom_density() + labs(x='Square feet second floor')
s5 <- ggplot(data= all, aes(x=TotalBsmtSF)) +
        geom_density() + labs(x='Square feet basement')
s6 <- ggplot(data= all[all$LotArea<100000,], aes(x=LotArea)) +
        geom_density() + labs(x='Square feet lot')
s7 <- ggplot(data= all, aes(x=LotFrontage)) +
        geom_density() + labs(x='Linear feet lot frontage')
s8 <- ggplot(data= all, aes(x=LowQualFinSF)) +
        geom_histogram() + labs(x='Low quality square feet 1st & 2nd')

layout <- matrix(c(1,2,5,3,4,8,6,7),4,2,byrow=TRUE)
multiplot(s1, s2, s3, s4, s5, s6, s7, s8, layout=layout)

# %% Exploring the most important categorical variable Neighborhood

n1 <- ggplot(all[!is.na(all$SalePrice),], aes(x=Neighborhood, y=SalePrice)) +
        geom_bar(stat='summary', fun.y = "median", fill='blue') +
        theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
        scale_y_continuous(breaks= seq(0, 800000, by=50000), labels = comma) +
        geom_label(stat = "count", aes(label = ..count.., y = ..count..), size=3) +
        geom_hline(yintercept=163000, linetype="dashed", color = "red") #dashed line is median SalePrice
n2 <- ggplot(data=all, aes(x=Neighborhood)) +
        geom_histogram(stat='count')+
        geom_label(stat = "count", aes(label = ..count.., y = ..count..), size=3)+
        theme(axis.text.x = element_text(angle = 45, hjust = 1))
grid.arrange(n1, n2)

# %% Examining quality variables
q1 <- ggplot(data=all, aes(x=as.factor(OverallQual))) +
        geom_histogram(stat='count')
q2 <- ggplot(data=all, aes(x=as.factor(ExterQual))) +
        geom_histogram(stat='count')
q3 <- ggplot(data=all, aes(x=as.factor(BsmtQual))) +
        geom_histogram(stat='count')
q4 <- ggplot(data=all, aes(x=as.factor(KitchenQual))) +
        geom_histogram(stat='count')
q5 <- ggplot(data=all, aes(x=as.factor(GarageQual))) +
        geom_histogram(stat='count')
q6 <- ggplot(data=all, aes(x=as.factor(FireplaceQu))) +
        geom_histogram(stat='count')
q7 <- ggplot(data=all, aes(x=as.factor(PoolQC))) +
        geom_histogram(stat='count')

layout <- matrix(c(1,2,8,3,4,8,5,6,7),3,3,byrow=TRUE)
multiplot(q1, q2, q3, q4, q5, q6, q7, layout=layout)


# %% Feature Engineering

# Total number of bathrooms

all$TotBathrooms <- all$FullBath + (all$HalfBath*0.5) + all$BsmtFullBath + (all$BsmtHalfBath*0.5)

all$Remod <- ifelse(all$YearBuilt==all$YearRemodAdd, 0, 1) #0=No Remodeling, 1=Remodeling
all$Age <- as.numeric(all$YrSold)-all$YearRemodAdd
all$IsNew <- ifelse(all$YrSold==all$YearBuilt, 1, 0)

all$YrSold <- as.factor(all$YrSold) #the numeric version is now not needed anymore

all$NeighRich[all$Neighborhood %in% c('StoneBr', 'NridgHt', 'NoRidge')] <- 2
all$NeighRich[!all$Neighborhood %in% c('MeadowV', 'IDOTRR', 'BrDale', 'StoneBr', 'NridgHt', 'NoRidge')] <- 1
all$NeighRich[all$Neighborhood %in% c('MeadowV', 'IDOTRR', 'BrDale')] <- 0

all$TotalSqFeet <- all$GrLivArea + all$TotalBsmtSF

all$TotalPorchSF <- all$OpenPorchSF + all$EnclosedPorch + all$X3SsnPorch + all$ScreenPorch

# %% Preparing data for modeling

# Dropping highly correlated variables
dropVars <- c('YearRemodAdd', 'GarageYrBlt', 'GarageArea', 'GarageCond', 'TotalBsmtSF', 'TotalRmsAbvGrd', 'BsmtFinSF1')

all <- all[,!(names(all) %in% dropVars)]

# Removing outliers
all <- all[-c(524, 1299),]

# %% Preprocessing predictor variables
numericVarNames <- numericVarNames[!(numericVarNames %in% c('MSSubClass', 'MoSold', 'YrSold', 'SalePrice', 'OverallQual', 'OverallCond'))] #numericVarNames was created before having done anything
numericVarNames <- append(numericVarNames, c('Age', 'TotalPorchSF', 'TotBathrooms', 'TotalSqFeet'))

DFnumeric <- all[, names(all) %in% numericVarNames]

DFfactors <- all[, !(names(all) %in% numericVarNames)]
DFfactors <- DFfactors[, names(DFfactors) != 'SalePrice']

cat('There are', length(DFnumeric), 'numeric variables, and', length(DFfactors), 'factor variables')

# Skewness
for(i in 1:ncol(DFnumeric)){
        if (abs(skew(DFnumeric[,i]))>0.8){
                DFnumeric[,i] <- log(DFnumeric[,i] +1)
        }
}

# Normalizing
PreNum <- preProcess(DFnumeric, method=c("center", "scale"))
print(PreNum)

DFnorm <- predict(PreNum, DFnumeric)
dim(DFnorm)

# One hot encoding
DFdummies <- as.data.frame(model.matrix(~.-1, DFfactors))
dim(DFdummies)

# Removing dummy variable predictors with values absent in the train/test set and less than 10 ones in the train set.

ZerocolTest <- which(colSums(DFdummies[(nrow(all[!is.na(all$SalePrice),])+1):nrow(all),])==0)
ZerocolTrain <- which(colSums(DFdummies[1:nrow(all[!is.na(all$SalePrice),]),])==0)

DFdummies <- DFdummies[,-ZerocolTest] #removing predictors
DFdummies <- DFdummies[,-ZerocolTrain] #removing predictor

fewOnes <- which(colSums(DFdummies[1:nrow(all[!is.na(all$SalePrice),]),])<10)
DFdummies <- DFdummies[,-fewOnes] #removing predictors

combined <- cbind(DFnorm, DFdummies) #combining all (now numeric) predictors into one dataframe 

# %% Skewness of Response Variable

all$SalePrice <- log(all$SalePrice) #default is the natural logarithm, "+1" is not necessary as there are no 0's

# %% Composing test and train sets
train1 <- combined[!is.na(all$SalePrice),]
test1 <- combined[is.na(all$SalePrice),]

# %% Lasso Regression

set.seed(27042018)
my_control <-trainControl(method="cv", number=5)
lassoGrid <- expand.grid(alpha = 1, lambda = seq(0.001,0.1,by = 0.0005))

lasso_mod <- train(x=train1, y=all$SalePrice[!is.na(all$SalePrice)], method='glmnet', trControl= my_control, tuneGrid=lassoGrid) 
lasso_mod$bestTune

min(lasso_mod$results$RMSE)

# %% Variables used and not used
lassoVarImp <- varImp(lasso_mod,scale=F)
lassoImportance <- lassoVarImp$importance

varsSelected <- length(which(lassoImportance$Overall!=0))
varsNotSelected <- length(which(lassoImportance$Overall==0))

cat('Lasso uses', varsSelected, 'variables in its model, and did not select', varsNotSelected, 'variables.')

# %% Predictions from Lasso

LassoPred <- predict(lasso_mod, test1)
predictions_lasso <- exp(LassoPred) #need to reverse the log to the real values
head(predictions_lasso)

# %% XGBoost

xgb_grid = expand.grid(
nrounds = 1000,
eta = c(0.1, 0.05, 0.01),
max_depth = c(2, 3, 4, 5, 6),
gamma = 0,
colsample_bytree=1,
min_child_weight=c(1, 2, 3, 4 ,5),
subsample=1
)

#xgb_caret <- train(x=train1, y=all$SalePrice[!is.na(all$SalePrice)], method='xgbTree', trControl= my_control, tuneGrid=xgb_grid) 
#xgb_caret$bestTune

label_train <- all$SalePrice[!is.na(all$SalePrice)]

# put our testing & training data into two seperate Dmatrixs objects
dtrain <- xgb.DMatrix(data = as.matrix(train1), label= label_train)
dtest <- xgb.DMatrix(data = as.matrix(test1))

default_param<-list(
        objective = "reg:linear",
        booster = "gbtree",
        eta=0.05, #default = 0.3
        gamma=0,
        max_depth=3, #default=6
        min_child_weight=4, #default=1
        subsample=1,
        colsample_bytree=1
)

xgbcv <- xgb.cv( params = default_param, data = dtrain, nrounds = 500, nfold = 5, showsd = T, stratified = T, print_every_n = 40, early_stopping_rounds = 10, maximize = F)

# %% Train the model using the best iteration found by cross validation

xgb_mod <- xgb.train(data = dtrain, params=default_param, nrounds = 454)

XGBpred <- predict(xgb_mod, dtest)
predictions_XGB <- exp(XGBpred) #need to reverse the log to the real values
head(predictions_XGB)

# %% View variable importance
mat <- xgb.importance (feature_names = colnames(train1),model = xgb_mod)
xgb.ggplot.importance(importance_matrix = mat[1:20], rel_to_first = TRUE)

# %% Averaging Lasso and XGBoost predictions

# sub_avg <- data.frame(Id = test_labels, SalePrice = (predictions_XGB+2*predictions_lasso)/3)
# head(sub_avg)
# write.csv(sub_avg, file = 'average.csv', row.names = F)
