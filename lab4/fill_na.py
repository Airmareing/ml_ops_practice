import pandas as pd

dataset = pd.read_csv('data/titanic_initial.csv')
# Заполняем пропущенные значения в поле Age
dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)
dataset.to_csv('data/titanic_filled.csv', index=False)