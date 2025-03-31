import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split   ##Para fazer treino e teste
from sklearn.linear_model import LinearRegression  ##Para fazer a regressão
from sklearn.metrics import mean_squared_error, mean_absolute_error #análisa o valor de erro da suposição
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


#x1 - tipo de construção
x1_train, x1_test, y_train, y_test = train_test_split(x1, y, test_size=0.33, random_state=42)
reg1= LinearRegression().fit(x1_train.values.reshape(-1,1),y_train)
print("Coeficientes da regressão entre tipo de construção e energia consumida:  ", reg1.coef_)
print("Intercepto da regressão entre tipo de construção e energia consumida:  ", reg1.intercept_)
y_pred1 =reg1.predict(x1_test.values.reshape(-1,1))
print("\n.............. ...  ..................")
mae1= mean_absolute_error(y_test,y_pred1)
print("Erro absoluto", mean_absolute_error(y_test,y_pred1))
erro_percentual = (mae1 / y_test.mean()) * 100
print(f"Erro percentual médio: {erro_percentual:.2f}%")
if(erro_percentual<20):
    print("Erro aceitável")
else:
    print("Erro grave")


## vendo se preveu x1 bem - grafico de barras
df1 = pd.DataFrame({'Tipo de Construção': x1_test.values, 'Consumo Original': y_test, 'Consumo Previsto': y_pred1})
grupo1 = df1.groupby('Tipo de Construção').mean()
fig, ax1=plt.subplots()
indice1 = np.arange(len(grupo1))
width = 0.4  # Largura das barras
plt.bar(indice1 - width/2, grupo1['Consumo Original'], width, label='Consumo Original')
plt.bar(indice1 + width/2, grupo1['Consumo Previsto'], width, label='Consumo Previsto', alpha=0.7)
plt.xticks(indice1, grupo1.index)
plt.xlabel("Tipo de Construção")
plt.ylabel("Consumo Médio de Energia")
plt.title("Comparação entre Consumo Previsto e Original por Tipo de Construção")
plt.legend()
plt.show()

#Gráfico de análise
plt.scatter(x1, y, color='blue', label='Dados reais')  # Pontos reais
plt.plot(x1_test, y_pred1, color='red', label='Regressão Linear')  # Linha da regressão - Tendencia da renda subir com a idade
##ele usa os x de testes e exibe os y de predição
plt.xlabel("tipo de construção")
plt.ylabel("Consumo de energia")
plt.title("Influência do tipo de contrução no consumo de energia")
plt.legend() ##Mostra as legendas estabelecidas
plt.show()


#x2 - àrea
x2_train, x2_test, y_train, y_test = train_test_split(x2, y, test_size=0.33, random_state=42)
reg2= LinearRegression().fit(x2_train.values.reshape(-1,1),y_train)
print("Coeficientes da regressão entre área e energia consumida:  ", reg2.coef_)
print("Intercepto da regressão entre área e energia consumida:  ", reg2.intercept_)
y_pred2 =reg2.predict(x2_test.values.reshape(-1,1))
print("\n.............. ...  ..................")
print("Erro squared",mean_squared_error(y_test,y_pred2))
print("Erro absoluto",mean_absolute_error(y_test,y_pred2)) 
## vendo se preveu x2 bem
fig,ax2=plt.subplots()
ax2.scatter(y_pred2,y_test)
ax2.plot([1000,7000],[1000,7000],'--r')
ax2.set_title("Área x Energia consumida")
ax2.set_xlabel("Consumo previsto")
ax2.set_ylabel("Consumo original")
plt.show()
print("\n     ")
#Gráfico de análise
plt.scatter(x2, y, color='blue', label='Dados reais')  # Pontos reais
plt.plot(x2_test, y_pred2, color='red', label='Regressão Linear')  # Linha da regressão - Tendencia da renda subir com a idade
##ele usa os x de testes e exibe os y de predição
plt.xlabel("Área")
plt.ylabel("Consumo de energia")
plt.title("Influência da área no consumo de energia")
plt.legend() ##Mostra as legendas estabelecidas
plt.show()

#x3 - Número de ocupantes
x3_train, x3_test, y_train, y_test = train_test_split(x3, y, test_size=0.33, random_state=42)
reg3= LinearRegression().fit(x3_train.values.reshape(-1,1),y_train)
print("Coeficientes da regressão entre número de ocupantes e energia consumida:  ", reg3.coef_)
print("Intercepto da regressão entre número de ocupantes e energia consumida:  ", reg3.intercept_)
y_pred3=reg3.predict(x3_test.values.reshape(-1,1))
print("\n.............. ...  ..................")
print("Erro squared",mean_squared_error(y_test,y_pred3))
print("Erro absoluto",mean_absolute_error(y_test,y_pred3)) 
## vendo se preveu x3 bem
fig,ax3=plt.subplots()
ax3.scatter(y_pred3,y_test)
ax3.plot([1000,7000],[1000,7000],'--r')
ax3.set_title("Número de ocupantes x Energia consumida")
ax3.set_xlabel("Consumo previsto")
ax3.set_ylabel("Consumo original")
plt.show()
print("\n     ")
#Gráfico de análise
plt.scatter(x3, y, color='blue', label='Dados reais')  # Pontos reais
plt.plot(x3_test, y_pred3, color='red', label='Regressão Linear')  # Linha da regressão - Tendencia da renda subir com a idade
##ele usa os x de testes e exibe os y de predição
plt.xlabel("Número de ocupantes")
plt.ylabel("Consumo de energia")
plt.title("Influência do número de ocupantes no consumo de energia")
plt.legend() ##Mostra as legendas estabelecidas
plt.show()

#x4 - Aparelhos
x4_train, x4_test, y_train, y_test = train_test_split(x4, y, test_size=0.33, random_state=42)
reg4= LinearRegression().fit(x4_train.values.reshape(-1,1),y_train)
print("Coeficientes da regressão entre aparelhos e energia consumida:  ", reg4.coef_)
print("Intercepto da regressão entre aparelhos e energia consumida:  ", reg4.intercept_)
y_pred4 =reg4.predict(x4_test.values.reshape(-1,1))
print("\n.............. ...  ..................")
print("Erro squared",mean_squared_error(y_test,y_pred4))
print("Erro absoluto",mean_absolute_error(y_test,y_pred4)) 
## vendo se preveu x4 bem
fig,ax4=plt.subplots()
ax4.scatter(y_pred4,y_test)
ax4.plot([1000,7000],[1000,7000],'--r')
ax4.set_title("Aparelhos x Energia consumida")
ax4.set_xlabel("Consumo previsto")
ax4.set_ylabel("Consumo original")
plt.show()
print("\n     ")
#Gráfico de análise
plt.scatter(x4, y, color='blue', label='Dados reais')  # Pontos reais
plt.plot(x4_test, y_pred4, color='red', label='Regressão Linear')  # Linha da regressão - Tendencia da renda subir com a idade
##ele usa os x de testes e exibe os y de predição
plt.xlabel("Aparelhos")
plt.ylabel("Consumo de energia")
plt.title("Influência da quantidade de aparelhos no consumo de energia")
plt.legend() ##Mostra as legendas estabelecidas
plt.show()

#x5 - Temperatura média
x5_train, x5_test, y_train, y_test = train_test_split(x5, y, test_size=0.33, random_state=42)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x5_train_scaled = scaler.fit_transform(x5_train.values.reshape(-1,1))
x5_test_scaled = scaler.transform(x5_test.values.reshape(-1,1))
reg5 = LinearRegression().fit(x5_train_scaled, y_train)
y_pred5 = reg5.predict(x5_test_scaled)
print("Coeficientes da regressão entre temperatura média e energia consumida:  ", reg5.coef_)
print("Intercepto da regressão entre Temperatura média e energia consumida:  ", reg5.intercept_)
y_pred5=reg5.predict(x5_test.values.reshape(-1,1))
print("\n.............. ...  ..................")
print("Erro squared",mean_squared_error(y_test,y_pred5))
print("Erro absoluto",mean_absolute_error(y_test,y_pred5)) 
print(y_pred5[:10])  # Exibir os primeiros valores previstos
print(y_test[:10])  # Exibir os valores reais correspondentes
print(psv[['temperatura_média', 'energia_consumida']].corr())
## vendo se preveu x5 bem
fig,ax5=plt.subplots()
ax5.scatter(y_pred5,y_test)
ax5.plot([3000,4600],[3000,4600],'--r')
ax5.set_title("Temperatura média x Energia consumida")
ax5.set_xlabel("Consumo previsto")
ax5.set_ylabel("Consumo original")
plt.show()
print("\n     ")
#Gráfico de análise
plt.scatter(x5, y, color='blue', label='Dados reais')  # Pontos reais
plt.plot(x5_test, y_pred5, color='red', label='Regressão Linear')  # Linha da regressão - Tendencia da renda subir com a idade
##ele usa os x de testes e exibe os y de predição
plt.xlabel("temperatura média")
plt.ylabel("Consumo de energia")
plt.title("Influência da temperatura no consumo de energia")
plt.legend() ##Mostra as legendas estabelecidas
plt.show()

#x6 - Dia da semana
x6_train, x6_test, y_train, y_test = train_test_split(x6, y, test_size=0.33, random_state=42)
reg6= LinearRegression().fit(x6_train.values.reshape(-1,1),y_train)
print("Coeficientes da regressão entre dia da semana e energia consumida:  ", reg6.coef_)
print("Intercepto da regressão entre dia da semana e energia consumida:  ", reg6.intercept_)
y_pred6 =reg6.predict(x6_test.values.reshape(-1,1))
print("\n.............. ...  ..................")
print("Erro squared",mean_squared_error(y_test,y_pred6))
print("Erro absoluto",mean_absolute_error(y_test,y_pred6)) 
## vendo se preveu x6 bem
df6=pd.DataFrame({'Dia da Semana':x6_test.values,'Consumo Original':y_test, 'Consumo Previsto':y_pred6})
grupo6=df6.groupby('Dia da Semana').mean()
fig,ax6=plt.subplots()
indice6=np.arange(len(grupo6))
plt.bar(indice6 - width/2, grupo6['Consumo Original'], width, label='Consumo Original',alpha=0.7)
plt.bar(indice6 + width/2, grupo6['Consumo Previsto'], width, label='Consumo Previsto',alpha=0.7)
plt.xticks(indice6,grupo6.index)
plt.xlabel("Semana/Final de semana")
plt.ylabel("Consumo Médio de Energia")
plt.title("Comparação entre Consumo Previsto e Original por dia da semana")
plt.legend() 
plt.show()

print("\n     ")
#Gráfico de análise
plt.scatter(x6, y, color='blue', label='Dados reais')  # Pontos reais
plt.plot(x6_test, y_pred6, color='red', label='Regressão Linear')  # Linha da regressão - Tendencia da renda subir com a idade
##ele usa os x de testes e exibe os y de predição
plt.xlabel("Dia da semana")
plt.ylabel("Consumo de energia")
plt.title("Influência do dia da semana no consumo de energia")
plt.legend() ##Mostra as legendas estabelecidas
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
reg = LinearRegression ().fit(x_train,y_train) ##Não precisa fazer o reshape porque tem duas colunas envolvidas.
print ("coeficientes",reg.coef_) #quanto maior o coeficiente, mais ele influência
print("intercepto",reg.intercept_)
y_pred =reg.predict(x_test)
print("\n                                ..  ...   ")
print(mean_squared_error(y_test,y_pred))
print(mean_absolute_error(y_test,y_pred))


##Analise de o mean error vai aumentar ou diminuir conforme você vai usando as colunas


fig,ax10=plt.subplots()
ax10.scatter(y_pred,y_test)
ax10.plot([1000,7000],[1000,7000],'--r')   ##essa reta compara o valor previsto com o valor de teste(esperado)

#plt.show()
##esse gráfico mostra em x os valores; previstos para o u - que é a energia consumida
## E no Y ele mostra os reais valores de x.  
## resumindo  -  no gráfico, o horizonte é valores previstos e o vertical é valores reais.
## A margem de erro mostra a margem de diferença entre o real e o gerado.
##Se os pontos estão agupados, encontrou bem o
