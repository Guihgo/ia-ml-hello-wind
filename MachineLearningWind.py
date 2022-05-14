import pandas as pd
#Carregando dataset
file = pd.read_csv('DataSet/wind_dataset.csv')
print(file)

# Replace values example
# file['IND'] = file['IND'].replace(0, 'teste')

# Separate collumn
# x = file['T.MAX']

#criando conjunto de dados de treino e teste:
#x_treino, x_teste=
