# %% read data
import pandas as pd

train = pd.read_csv("titanic/train.csv")
test = pd.read_csv("titanic/test.csv")


# %% checkout out first few rows
train.head()


# %% checkout out dataframe info
train.info()


# %% describe the dataframe
train.describe(include="all")


# %% visualize the dataset, starting with the Survied distribution
import seaborn as sns

sns.countplot(x="Survived", data=train)


# %% Survived w.r.t Pclass / Sex / Embarked ?
import matplotlib.pyplot as plt

sns.countplot(data=train, x="Pclass", hue="Survived")

plt.title("Survival Rate by Passenger Class")
plt.show()

# %%
sns.countplot(data=train, x="Sex", hue="Survived")

plt.title("Survival Rate by Sex")
plt.show()

#%%

sns.countplot(data=train, x="Embarked", hue="Survived")

plt.title("Survival Rate by Embarking status")
plt.show()

# %% Age distribution ?

sns.histplot(data=train, x="Age")
plt.title("Age Distribuiton")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# %% Survived w.r.t Age distribution ?

sns.histplot(data=train, x="Age",hue="Survived")
plt.title("Age Distribuiton Survival")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()


# %% SibSp / Parch distribution ?
sns.countplot(data=train, x="SibSp")
plt.title("Distribution of SibSp")
plt.xlabel("Number of Siblings/Spouses")
plt.ylabel("Count")
plt.show()

# %% Survived w.r.t SibSp / Parch  ?

sns.countplot(data=train, x="SibSp", hue="Survived")
plt.title("Distribution of SibSp Survived")
plt.xlabel("Number of Siblings/Spouses")
plt.ylabel("Count")
plt.show()

# %% Dummy Classifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score


def evaluate(clf, x, y):
    pred = clf.predict(x)
    result = f1_score(y, pred)
    return f"F1 score: {result:.3f}"


dummy_clf = DummyClassifier(random_state=2020)

dummy_selected_columns = ["Pclass"]
dummy_train_x = train[dummy_selected_columns]
dummy_train_y = train["Survived"]

dummy_clf.fit(dummy_train_x, dummy_train_y)
print("Training Set Performance")
print(evaluate(dummy_clf, dummy_train_x, dummy_train_y))

truth = pd.read_csv("truth_titanic.csv")
dummy_test_x = test[dummy_selected_columns]
dummy_test_y = truth["Survived"]

print("Test Set Performance")
print(evaluate(dummy_clf, dummy_test_x, dummy_test_y))

print("Can you do better than a dummy classifier?")


# %% Your solution to this classification problem

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import pandas as pd


train = pd.read_csv("titanic/train.csv")
test = pd.read_csv("titanic/test.csv")
truth = pd.read_csv("truth_titanic.csv")


selected_columns = ["Pclass", "Sex", "Age", "Embarked"]
train_x = train[selected_columns]
train_y = train["Survived"]
test_x = test[selected_columns]
test_y = truth["Survived"]

#%% 

#Cleaning Train 

train_x.loc[:, "Pclass"] = train_x["Pclass"].fillna(train_x["Pclass"].mode()[0])
train_x.loc[:, "Sex"] = train_x["Sex"].fillna(train_x["Sex"].mode()[0])
train_x.loc[:, "Age"] = train_x["Age"].fillna(train_x["Age"].median())
train_x.loc[:, "Embarked"] = train_x["Embarked"].fillna(train_x["Embarked"].mode()[0])

#Cleaning test
test_x.loc[:, "Pclass"] = test_x["Pclass"].fillna(test_x["Pclass"].mode()[0])
test_x.loc[:, "Sex"] = test_x["Sex"].fillna(test_x["Sex"].mode()[0])
test_x.loc[:, "Age"] = test_x["Age"].fillna(test_x["Age"].median())
test_x.loc[:, "Embarked"] = test_x["Embarked"].fillna(test_x["Embarked"].mode()[0])

print("Columns in train_x:", train_x.columns)

#%% converting categorical data in dummy
from sklearn.preprocessing import StandardScaler

train_x = pd.get_dummies(train_x, columns=["Pclass","Sex", "Embarked"], drop_first=True)
test_x = pd.get_dummies(test_x, columns=["Pclass","Sex", "Embarked"], drop_first=True)

#%%#Managing scales in data
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

#%%

X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.3, random_state=2020)

knn_clf = KNeighborsClassifier(n_neighbors=3) 
knn_clf.fit(X_train, y_train)

prediction= knn_clf.predict(X_test)
f1= f1_score(y_test, prediction)
print(f'KNN vallidation f1 Score:{f1:.3f}')
