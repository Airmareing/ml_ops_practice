import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Создание директорий для данных при их остутствии 
os.makedirs('data/train', exist_ok=True)
os.makedirs('data/test', exist_ok=True)

# Загрузка данных
data = pd.read_csv('car_data.csv')

# Разделение данных на тренировочные и тестовые
train_data, test_data = train_test_split(data, test_size=0.33, random_state=42)

# Добавление шума и аномалий
def add_noise_and_anomalies(df, noise_level=0.1, anomaly=False):
    df['Car Price'] += noise_level * np.random.randn(len(df)) * 10000
    if anomaly:
        df['Car Price'][::10] += 50000
    return df

train_data = add_noise_and_anomalies(train_data, noise_level=0.2)
test_data = add_noise_and_anomalies(test_data, noise_level=0.2, anomaly=True)

# Сохранение данных
train_data.to_csv('data/train/train_data.csv', index=False)
test_data.to_csv('data/test/test_data.csv', index=False)
