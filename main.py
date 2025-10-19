import numpy as np 
import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer

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
print(housing_prepared.shape)






