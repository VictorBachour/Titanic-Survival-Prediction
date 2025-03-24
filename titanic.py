import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv('Titanic-Dataset.csv')
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
data.drop(columns='Cabin', inplace=True)

x = data.drop(columns=['Survived', 'Name'])
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= .2, random_state=42)
model = LogisticRegression(solver='lbfgs', max_iter=200, C=1.0, penalty='l2')
model.fit(X_train, y_train)

y_pred = model.predict(y_test)
print(y_pred)