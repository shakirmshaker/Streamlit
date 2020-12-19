# Data
import pandas as pd
import numpy as np

# Preprocessing
from sklearn.preprocessing import StandardScaler

# Model_selection
from sklearn.model_selection import train_test_split

# Machine learning
from sklearn.ensemble import RandomForestRegressor

import pickle

df = pd.read_csv('FinalData.csv')

df.drop(['Degrees_C', 'Wind_Speed'], axis = 1, inplace = True)

df = pd.get_dummies(df, columns = ['date_month', 'date_day', 'date_hour', 'date_dayofweek'])

X = df.drop('TotalCount', axis = 1)

y = df['TotalCount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(max_depth = 20)

model.fit(X_train, y_train)

print(model.score(X_test, y_test))

pickle.dump(model, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))