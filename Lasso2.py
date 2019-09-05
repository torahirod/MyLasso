import pandas as pd
import numpy as np

df = pd.read_csv("https://raw.githubusercontent.com/satopirka/Lasso/master/Boston.csv")
y = df.iloc[:,-1]
df = (df - df.mean())/df.std()
X = df.iloc[:,1:-1]
X = np.column_stack((np.ones(len(X)),X))

n = X.shape[0]
d = X.shape[1]
w = np.zeros(d)
r = 1.0

for _ in range(1000) :
    w[0] = (y - np.dot(X[:,1:],w[1:])).sum() / n
    for k in range(1,d) :
        w[k] = 0
        a = np.dot((y - np.dot(X, w)),X[:,k]).sum()
        b = (X[:,k] ** 2).sum()
        w[k] = (np.sign(a) * np.maximum(abs(a) - n * r,0)) / b

print(w[0])
print(w[1:])
