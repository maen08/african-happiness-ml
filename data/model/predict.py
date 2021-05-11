import pandas as pd
from matplotlib import pyplot as plt
import seaborn




seaborn.set()


# load the file
data = pd.read_csv('africahappiness.csv')


# get the africa data only
africa_data = data['Regional Indicator'] == 'Africa'
africa = data[africa_data]


# if null is available, clean it
# data_check = africa.isnull().sum()
# africa = africa.dropna(axis=0)
# print(africa.describe(include='all'))


y = africa['Happiness Scores']
x = africa[['Generosity', 'Perceptions of Corruption','Social Support', 'Healthy Life Expectancy', 'Regional Indicator']]
    

# ys = y.value.reshape(-1,1)
x_matrix = x.iloc[:,0].values

plt.scatter(x_matrix,y)
plt.show()


# print(x)
# print(z)