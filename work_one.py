import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

# Exercise 1
df = pd.read_csv('data/wind_dataset.csv')

# Exercise 2
scatter_matrix(df, diagonal='hist')
plt.savefig('output/scatter_matrix.jpg')

# Exercise 3 
plt.show()

# Exercise 4: machine learning (KNN OR Random OR Forest OR Decision Tree)... @Ju

#Separando as variáveis entre preditoras (x) e variável alvo (y)
y = df['WIND']
x = df.drop('WIND', axis= 1)

#Criando os conjuntos de dados de teste e treino
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size =0.3, train_size=0.7)

#Criação do modelo
modelo = ExtraTreesClassifier()
modelo.fit(x_treino, y_treino)

#Imprimindo resultados
resultado = modelo.score(x_teste, y_teste)
print("Acurácia:", resultado)

