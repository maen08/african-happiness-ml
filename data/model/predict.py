import pandas as pd
from matplotlib import pyplot as plt
import seaborn
from sklearn.feature_selection import f_regression




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
    
# 2d in 1d for plot
x_matrix = x.iloc[:,0].values

plt.scatter(x_matrix,y)
# plt.show()



Gen = africa.sort_values(by=['Generosity'], ascending=False).head()
# print(Gen)


X = x.values.reshape(-1,1)
p_values = f_regression(X, y)[1]

print(p_values)
# print(x)
# print(z)
