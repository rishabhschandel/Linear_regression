import csv
from typing import Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data Import Methods
# Method 1(Generic File open method)

# X = []
# Y = []
# with open("data_1d.csv") as f:
#     for line in f:
#         X.append(float(line.split(',')[0].strip()))
#         Y.append(float(line.split(',')[1].strip()))

# X = np.array(X)
# Y = np.array(Y)

# Method 2(Using numpy loadtxt)

# data = np.loadtxt("data_1d.csv", delimiter=',')
# X = data[:, 0]
# Y = data[:, 1]

# Method 3(Using csv module)
# X = []
# Y = []
# with open("data_1d.csv", 'r') as csv_file:
#     # We can also use DictReader but then first values would become keys
#     csv_reader = csv.reader(csv_file, delimiter=",")
#     for line in csv_reader:
#         X.append(line[0])
#         Y.append(line[1])

# X = np.array([float(x) for x in X])
# Y = np.array([float(y) for y in Y])

# print(X)
# Method 4(Using DictReader)
# X = []
# Y = []

# with open("data_1d.csv", 'r') as csv_file:
# We can also use DictReader but then first values would become keys
#     csv_reader = csv.DictReader(csv_file, fieldnames=[0, 1])
#     for line in csv_reader:
#         X.append(line[0])
#         Y.append(line[1])

# X = np.array([float(x) for x in X])
# Y = np.array([float(y) for y in Y])

# Method 5(Using Pandas)

df = pd.read_csv('datasets\data_1d.csv', header=None)
# df.plot.scatter(df.loc[:, 0], df.loc[:, 1])
X = df.loc[:, 0].values
Y = df.loc[:, 1].values

denominator = np.mean(X**2)-(np.mean(X)**2)
a = (np.mean(X*Y) - np.mean(X)*np.mean(Y))/denominator
b = (np.mean(X**2)*np.mean(Y) - np.mean(X)*np.mean(X*Y))/denominator
r_squared = 1 - (np.sum((Y-(a*X+b))**2)/np.sum((Y-np.mean(Y))**2))
print(r_squared)
plt.scatter(X, Y)
plt.plot(X, (a*X+b), c='red')
plt.show()
