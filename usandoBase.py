import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split   ##Para fazer treino e teste
from sklearn.linear_model import LinearRegression  ##Para fazer a regressão
from sklearn.metrics import mean_absolute_error #análisa o valor de erro da suposição
from sklearn.metrics import mean_squared_error  #análisa o valor de erro da suposição
import matplotlib.pyplot as plt

dados= 'rl.csv'

df= pd.read_csv(dados)

da= df.to_numpy()

print("dataframe do pandas")
print(df)

#print("\nNumpy array")
#print(da)

x1= df.age
x2= df.experience
y=df.income

#print(x1)

x1_train, x1_test, y_train, y_test = train_test_split(x1, y, test_size=0.33, random_state=42)   ##ta dividindo os dados de treino e os de teste
##0,33 define quanto será destinado ao teste

reg = LinearRegression ().fit(x1_train.values.reshape(-1,1,),y_train)
y_pred =reg.predict(x1_test.values.reshape(-1,1))

print("\n                                ..  ...   ")
print(mean_squared_error(y_test,y_pred))
print(mean_absolute_error(y_test,y_pred))

fig,ax=plt.subplots()

ax.scatter(y_pred,y_test)
plt.show()   #os valores debaixo são os previstos, os do lado são os originais
#esse gráfico acima eu não entendi bulhufas, o que o chat me deu abaixo foi melhor.


plt.scatter(x1, y, color='blue', label='Dados reais')  # Pontos reais
plt.plot(x1_test, y_pred, color='red', label='Regressão Linear')  # Linha da regressão - Tendencia da renda subir com a idade
##ele usa os x de testes e exibe os y de predição
plt.xlabel("Idade")
plt.ylabel("Renda")
plt.title("Influência da Idade na Renda")
plt.legend() ##Mostra as legendas estabelecidas
plt.show()  ##Mostra os gráficos



idade_input = int(input("Digite a idade para prever a renda: "))  # Entrada do usuário
renda_prevista = reg.predict([[idade_input]])  # Faz a predição

print(f"Para a idade de {idade_input} anos, a renda prevista é: {renda_prevista[0]:.2f}")
##Precisa de [0] porque esse retorno é um array. então retornamos só a primeira posição
