import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import Lasso

# サンプルデータ数
m = 50

# 乱数のシード値を指定することで，再現性を保つ
np.random.seed(seed=2021)

# 「-3」から「3」の間で等間隔にm個のデータを作成
X = np.linspace(-3, 3, m)

# 後のグラフ描画用途に，100倍細かいグリッドを準備しておく
X_plot = np.linspace(-3, 3, m*100)

# 周期的なsin関数(第一項)に右上がり成分(第二項)と乱数(第三項)を加えたデータを作る
y = np.sinc(X) + 0.1 * X + 0.05 * np.random.normal(loc=0., scale=1., size=m)

# グラフ表示するため，各数列を1列の行列に変換
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)
X_plot = X_plot.reshape(-1,1)

# RBFカーネルを用いて、データを非線形写像する
kX = rbf_kernel(X, X)

# 写像したデータを用いて、Lasso回帰を学習する
clf = Lasso(alpha=0.001)
clf.fit(kX, y)
print(clf.coef_) 

# グラフ描画用途に，細かいグリッド点で計算
kX_plot = rbf_kernel(X_plot, X)
p_plot = clf.predict(kX_plot)

# グラフ化
plt.scatter(X, y, label="test data")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(X_plot, p_plot, label="model")
plt.show()