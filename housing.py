import numpy as np 
import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score


# Load The Data 
housing = pd.read_csv("housing.csv")

# Create a Startified test Set based on column icome cat
housing["income_cat"] = pd.cut(housing["median_income"],bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], labels = [1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_set , test_set in split.split(housing,housing["income_cat"]):
    strat_train_set = housing.loc[train_set].drop("income_cat",axis=1)
    strat_test_set = housing.loc[test_set].drop("income_cat",axis=1)

# Work on training data
housing= strat_train_set.copy()


#saparate predictors and labels 
housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value",axis =1)

# saparate numerical and categorical values
num_attribs = housing.drop("ocean_proximity",axis=1).columns.tolist()
cat_attribs = ["ocean_proximity"]

#Pipeline

#numerical
num_pipline = Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler()),
])

#categorical Pipeline
cat_pipline = Pipeline([
    ("onehot",OneHotEncoder(handle_unknown="ignore"))
])

# Full Pipeline
full_pipeline = ColumnTransformer([
    ("num",num_pipline,num_attribs),
    ("cat",cat_pipline,cat_attribs),
 ])

# Transform the data
housing_prepared = full_pipeline.fit_transform(housing)

# housing_Prepared is now a numy array ready for training
# print(housing_prepared.shape)

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)


# Decision Tree
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared,housing_labels)

# Random Forest
forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(housing_prepared,housing_labels)

# Predict using Training data 
lin_pred = lin_reg.predict(housing_prepared)
tree_pred = tree_reg.predict(housing_prepared)
forest_pred = forest_reg.predict(housing_prepared)

 
# calculate rmse
# lin_rmse = root_mean_squared_error(housing_labels,lin_pred)
# tree_rmse = root_mean_squared_error(housing_labels,tree_pred)
# forest_rmse = root_mean_squared_error(housing_labels,forest_pred) 

# print(f"The rmse for linear reg is {lin_rmse}")
# print(f"The rmse for decision tree is {tree_rmse}")
# print(f"The rmse for random forest is {forest_rmse}")


# Cross validation ---- use cross validation because rmse overfits the training data -- results may be bais

# Desicion tree rmses
tree_rmses = - cross_val_score(
    tree_reg,housing_prepared,housing_labels,scoring="neg_root_mean_squared_error", cv=10
)

print(f"The decision tree error is {pd.Series(tree_rmses).describe()}")

# Linear reg rmses
lin_rmses = - cross_val_score(
    lin_reg,housing_prepared,housing_labels,scoring="neg_root_mean_squared_error",cv=10
)

print(f"The linear regression error is {pd.Series(lin_rmses).describe()}")

# Random forest rmses
forest_rmses = - cross_val_score(
    forest_reg,housing_prepared,housing_labels,scoring="neg_root_mean_squared_error",cv=10
)

print(f"The linear regression error is {pd.Series(forest_rmses).describe()}")