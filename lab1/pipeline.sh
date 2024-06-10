#!/bin/bash

# Установка необходимых библиотек
pip3 install pandas scikit-learn

# Запуск скриптов последовательно
python3 data_creation.py
python3 model_preprocessing.py
python3 model_preparation.py
python3 model_testing.py
