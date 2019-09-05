import pandas as pd
import numpy as np

df = pd.read_csv("https://raw.githubusercontent.com/satopirka/Lasso/master/Boston.csv")

# 目的変数を抽出 ※ 目的変数は標準化前に抽出している点に注意
y = df.iloc[:,-1]
# データの標準化
df = (df - df.mean())/df.std()
# 説明変数を抽出 ※ 1列目はただの行番号なので無視
X = df.iloc[:,1:-1]
# Xにバイアス（w0）用の値が1のダミー列を追加
X = np.column_stack((np.ones(len(X)),X))

n = X.shape[0] # 行数
d = X.shape[1] # 次元数（列数）
w = np.zeros(d) # 重み
r = 1.0 # ハイパーパラメータ ※ 正則化の強弱を調整する

for _ in range(1000) : # 以下の重み更新を1000回繰り返し
    for k in range(d) : # 重みの数だけ繰り返し（w0含む）
        if k == 0 :
            # バイアスの重みを更新
            w[0] = (y - np.dot(X[:,1:],w[1:])).sum() / n
        else :
            # バイアス、更新対象の重み 以外の添え字
            _k = [i for i in range(d) if i not in [0,k]]
            # wk更新式の分子部分
            a = np.dot((y - np.dot(X[:,_k], w[_k]) - w[0]),X[:,k]).sum()
            # wk更新式の分母部分
            b = (X[:,k] ** 2).sum()

            if a > n * r : # wkが正となるケース
                w[k] = (a - n * r) / b
            elif a < -r * n : # wkが負となるケース
                w[k] = (a + n * r) / b
            else : # それ以外のケース
                w[k] = 0

print('----------- MyLasso1 ------------')
print(w[0])  # バイアス
print(w[1:]) # 重み

import sklearn.linear_model as lm
lasso = lm.Lasso(alpha=1.0, max_iter=1000, tol=0.0)
lasso.fit(X, y)
print("---------- sklearn Lasso ------------")
print(model.intercept_)
print(model.coef_)
