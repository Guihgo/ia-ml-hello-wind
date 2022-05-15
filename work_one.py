# Carregando conjunto de dados:
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.tree import DecisionTreeRegressor

# Exercise 1
df = pd.read_csv('/data/wind_dataset.csv')

# Apagando a coluna DATE para conseguir fazer a analise de dados
df.drop('DATE', inplace=True, axis=1)

# Tratando valores vazios do dataset pelo metodo interpolate      
df['IND.1'] = df['IND.1'].interpolate()
df['T.MAX'] = df['T.MAX'].interpolate()
df['IND.2'] = df['IND.2'].interpolate()
df['T.MIN'] = df['T.MIN'].interpolate()
df['T.MIN.G'] = df['T.MIN.G'].interpolate()

df.head()

# Exercise 2
scatter_matrix(df, diagonal='hist')
plt.savefig('output/scatter_matrix.jpg')

# Exercise 3 
plt.show()

# Exercise 4

#Separando as variáveis entre preditoras (x) e variável alvo (y)
y = df['WIND']
x = df.drop('WIND', axis= 1)

#Criando os conjuntos de dados de teste e treino
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size =0.3, train_size=0.7)

# Criação do modelo
modelo_DT = DecisionTreeRegressor()

# Treina a regressão
modelo_DT.fit(x_treino, y_treino)
DT_pred = modelo_DT.predict(x_teste)

