import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

# Exercise 1
df = pd.read_csv('data/wind_dataset.csv')

# Replace values example
# file['IND'] = file['IND'].replace(0, 'teste')

# Separate collumn
# x = file['T.MAX']


# Exercise 2
scatter_matrix(df, diagonal='hist')
plt.savefig('output/scatter_matrix.jpg')

# Exercise 3 
plt.show()



# Exercise 4: machine learning (KNN OR Random OR Forest OR Decision Tree)... @Ju

#criando conjunto de dados de treino e teste:
# x_treino, x_teste=
