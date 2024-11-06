# %% read data
import pandas as pd

train = pd.read_csv(
    "house-prices-advanced-regression-techniques/train.csv"
)
test = pd.read_csv(
    "house-prices-advanced-regression-techniques/test.csv"
)


# %% checkout out first few rows
train.head(3)


# %% checkout out dataframe info
train.info()


# %% describe the dataframe
train.describe(include="all")


# %% SalePrice distribution
import seaborn as sns

sns.distplot(train["SalePrice"])


# %% SalePrice distribution w.r.t CentralAir / OverallQual / BldgType / etc
import matplotlib.pyplot as plt

train['LotArea_scaled'] = train['LotArea'] / 1000
categorical_vars = ["LotArea_scaled","CentralAir","YearBuilt","Neighborhood", "HouseStyle", "OverallQual", "BldgType","YrSold"]
for var in categorical_vars:
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=train, x=var, y="SalePrice")
    plt.title(f'SalePrice Distribution with respect to {var}')
    
    plt.xticks(rotation=90, fontsize=10)  
    plt.tight_layout()  # Adjust layout to fit label
    
    plt.show()

# %% lote area
top_10_lot_area = train.nlargest(10, 'LotArea_scaled')
bottom_10_lot_area = train.nsmallest(10, 'LotArea_scaled')

top_bottom_lot_area = pd.concat([top_10_lot_area, bottom_10_lot_area])
plt.figure(figsize=(15, 8))
sns.boxplot(data=top_bottom_lot_area, x='LotArea_scaled', y='SalePrice')
plt.title('SalePrice Distribution for Top 10 and Bottom 10 LotArea Scaled')
plt.xlabel('LotArea (in thousands)')
plt.ylabel('SalePrice')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# %% SalePrice distribution w.r.t YearBuilt / Neighborhood

train['LotArea_scaled'] = train['LotArea'] / 1000
categorical_vars = ["YearBuilt","Neighborhood"]
for var in categorical_vars:
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=train, x=var, y="SalePrice")
    plt.title(f'SalePrice Distribution with respect to {var}')
    
    plt.xticks(rotation=90, fontsize=10)  
    plt.tight_layout()  # Adjust layout to fit label
    
    plt.show()

# %%
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_log_error
import numpy as np


def evaluate(reg, x, y):
    pred = reg.predict(x)
    result = np.sqrt(mean_squared_log_error(y, pred))
    return f"RMSLE score: {result:.3f}"


dummy_reg = DummyRegressor()

dummy_selected_columns = ["MSSubClass"]
dummy_train_x = train[dummy_selected_columns]
dummy_train_y = train["SalePrice"]

dummy_reg.fit(dummy_train_x, dummy_train_y)
print("Training Set Performance")
print(evaluate(dummy_reg, dummy_train_x, dummy_train_y))

truth = pd.read_csv("truth_house_prices.csv")
dummy_test_x = test[dummy_selected_columns]
dummy_test_y = truth["SalePrice"]

print("Test Set Performance")
print(evaluate(dummy_reg, dummy_test_x, dummy_test_y))

print("Can you do better than a dummy regressor?")


# %% your solution to the regression problem

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

# Defining Training and testing
target = "SalePrice"
selected_features = ["LotArea", "OverallQual", "YearBuilt", "CentralAir", "Neighborhood", "HouseStyle", "BldgType"]
X = train[selected_features]
y = train[target]

X = pd.get_dummies(X, columns=["CentralAir", "Neighborhood", "HouseStyle", "BldgType"], drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Call Learner
lr = LinearRegression()
lr.fit(X_train, y_train)


def evaluate(reg, x, y):
    pred = reg.predict(x)
    result = np.sqrt(mean_squared_log_error(y, np.maximum(pred, 0))) 
    return f"RMSLE score: {result:.3f}"

# Training and validation performance
print("Training Set Performance")
print(evaluate(lr, X_train, y_train))

print("Validation Set Performance")
print(evaluate(lr, X_test, y_test))