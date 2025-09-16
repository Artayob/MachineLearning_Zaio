import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

cars = pd.read_csv("C:/Users/sakit/Desktop/MachineLearning_Zaio/CAR DETAILS FROM CAR DEKHO.csv")
print(cars)
print(cars.describe())

plt.figure(figsize=(20, 8))
 
plt.subplot(1,2,1)
plt.title('Car price Distribution Plot')
sns.displot(cars['selling_price'])

plt.subplot(1,2,2)
plt.title("car Price spread")
sns.boxplot(y=cars['selling_price'])
plt.show()