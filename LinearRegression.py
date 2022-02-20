#Imports required libraries

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


#Set pandas DataFrame Object from WineQT.csv file.
DataSet = pd.read_csv("WineQT.csv", sep=",", index_col=False)

# Explore the data

# Columns:
DataSet.columns

#Id columns is not used so it will be dropped
DataSet.drop(DataSet[['Id']], axis=1, inplace=True)

# Identify to empty lines if there are
DataSet.isnull().sum()


# Look at general and quantitative information
DataSet.info
DataSet.describe()


# Print the correlation between quality and others to select for linear regression 
DataSet.corr().quality

sns.heatmap(DataSet.corr())
plt.show()

"""According to correlations, alcohol has the highest corelation between quality of wine, which is .484866."""
 


#Prepare the dataset for linear regression by identifying feature and target values
features = DataSet.drop(DataSet[['quality']], axis=1).values
target = DataSet['quality'].values

train, test, train_label, test_label = train_test_split(features, target, test_size=0.4, random_state=52)

# Regression:
regression = LinearRegression(fit_intercept=True)

# Fitting:

model = regression.fit(train, train_label)


# Prediction:
predict = model.predict(test)

mean_squared_error(test_label, predict)

mean_absolute_error(test_label, predict)

r2_score(test_label, predict)

model.score(test, test_label)


