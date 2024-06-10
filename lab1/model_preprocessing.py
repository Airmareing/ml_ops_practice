import pandas as pd
from sklearn.preprocessing import StandardScaler

# Загрузка данных
train_data = pd.read_csv('data/train/train_data.csv')
test_data = pd.read_csv('data/test/test_data.csv')

# Предобработка данных
scaler = StandardScaler()
train_data[['Car Horsepower', 'Car Price']] = scaler.fit_transform(train_data[['Car Horsepower', 'Car Price']])
test_data[['Car Horsepower', 'Car Price']] = scaler.transform(test_data[['Car Horsepower', 'Car Price']])

# Сохранение предобработанных данных
train_data.to_csv('data/train/train_data_preprocessed.csv', index=False)
test_data.to_csv('data/test/test_data_preprocessed.csv', index=False)
