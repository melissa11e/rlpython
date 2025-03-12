import pandas as pd
import numpy as n
from sklearn.model_selection import train_test_split   ##Para fazer treino e teste
from sklearn.linear_model import LinearRegression  ##Para fazer a regressão
from sklearn.metrics import mean_absolute_error #análisa o valor de erro da suposição
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt  ##Faz os gráficos


psv = pd.read_csv('baseEnergia.csv')

x1=psv.tipo_construção
x2=psv.área
x3=psv.número_ocupantes
x4=psv.aparelhos
x5=psv.temperatura_média
x6=psv.dia_semana
y=psv.energia_consumida


print(psv)

print(x2)


