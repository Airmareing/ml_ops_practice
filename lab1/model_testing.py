import pandas as pd
from sklearn.metrics import mean_squared_error
import pickle

# Загрузка данных и модели
test_data = pd.read_csv('data/test/test_data_preprocessed.csv')
X_test = test_data[['Car Horsepower']]
y_test = test_data['Car Price']

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Оценка модели
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')
