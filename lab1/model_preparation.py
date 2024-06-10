import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Загрузка и подготовка данных
train_data = pd.read_csv('data/train/train_data_preprocessed.csv')
X_train = train_data[['Car Horsepower']]
y_train = train_data['Car Price']

# Обучение модели
model = LinearRegression()
model.fit(X_train, y_train)

# Сохранение модели
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
