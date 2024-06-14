import pandas as pd
from catboost.datasets import titanic

train, _ = titanic()
train.to_csv('data/titanic_full.csv', index=False)
dataset = train[['Pclass', 'Sex', 'Age']]
dataset.to_csv('data/titanic_initial.csv', index=False)