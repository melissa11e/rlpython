import pandas as pd
import numpy as n
from sklearn.model_selection import train_test_split   ##Para fazer treino e teste
from sklearn.linear_model import LinearRegression  ##Para fazer a regressão
from sklearn.metrics import mean_absolute_error #análisa o valor de erro da suposição
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt  ##Faz os gráficos
from sklearn.preprocessing import LabelEncoder  ##transforma os str em numérico

psv = pd.read_csv('baseEnergia.csv')

encoder_dia = LabelEncoder()
encoder_tipo = LabelEncoder()

psv['dia_semana'] = encoder_dia.fit_transform(psv['dia_semana'])
psv['tipo_construção'] = encoder_tipo.fit_transform(psv['tipo_construção'])


#x=psv[['número_ocupantes','área']]
#x=psv.drop('energia_consumida',axis=1)  ##esse une todas as variáveis, excluindo o y
x=psv[['tipo_construção','área']]
x1=psv.tipo_construção
x2=psv.área
x3=psv.número_ocupantes
x4=psv.aparelhos
x5=psv.temperatura_média
x6=psv.dia_semana
y=psv.energia_consumida

print(psv)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
reg = LinearRegression ().fit(x_train,y_train) ##Não precisa fazer o reshape porque tem duas colunas envolvidas.
print ("coeficientes",reg.coef_) #quanto maior o coeficiente, mais ele influência
print("intercepto",reg.intercept_)
y_pred =reg.predict(x_test)
print("\n                                ..  ...   ")
print(mean_squared_error(y_test,y_pred))
print(mean_absolute_error(y_test,y_pred)) 
##Analise de o mean error vai aumentar ou diminuir conforme você vai usando as colunas


fig,ax=plt.subplots()
ax.scatter(y_pred,y_test)
ax.plot([1000,7000],[1000,7000],'--r')   ##essa reta compara o valor previsto com o valor de teste(esperado)

plt.show()
##esse gráfico mostra em x os valores; previstos para o u - que é a energia consumida
## E no Y ele mostra os reais valores de x.  
## resumindo  -  no gráfico, o horizonte é valores previstos e o vertical é valores reais.
## A margem de erro mostra a margem de diferença entre o real e o gerado.
##Se os pontos estão agupados, encontrou bem o



