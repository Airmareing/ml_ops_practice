import pandas as pd

dataset = pd.read_csv('data/titanic_filled.csv')
# Применим one-hot-encoding для столбца Sex
dataset = pd.get_dummies(dataset, columns=['Sex'], drop_first=True)
dataset.to_csv('data/titanic_encoded.csv', index=False)