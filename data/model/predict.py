import pandas as pd
from matplotlib import pyplot as plt


# load the file
data = pd.read_csv('africahappiness.csv')


# get the africa data only
africa_data = data['Regional Indicator'] == 'Africa'
africa = data[africa_data]

''' check some important info

print(africa.describe(include='all'))
print(africa.isnull().sum())

'''


y = africa['Happiness Scores']
x = africa.drop['Happiness Scores']

print(x)