# %%
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)


def data_generation1(n=100):
    return np.concatenate([np.random.randn(n, 1) * 2, np.random.randn(n, 1)], axis=1)


def data_generation2(n=100):
    return np.concatenate(
        [
            np.random.randn(n, 1) * 2,
            2 * np.round(np.random.rand(n, 1)) - 1 + np.random.randn(n, 1) / 3.0,
        ],
        axis=1,
    )


# %%
def calc_weight_matrix(x):
    W = []
    for x_i in x:
        W_i = []
        for x_j in x:
            W_i.append(np.exp(-np.dot(x_i - x_j, x_i - x_j)))
        W.append(W_i)
    return np.array(W)


def visualize(x, v):
    plt.clf()
    plt.axis("square")
    plt.xlim(-6.0, 6.0)
    plt.ylim(-6.0, 6.0)
    plt.plot(x[:, 0], x[:, 1], "rx")
    plt.plot(np.array([-v[:, 1], v[:, 1]]) * 9, np.array([-v[:, 0], v[:, 0]]) * 9)
    # plt.savefig('lecture9-h2_1.png')
    plt.savefig("lecture9-h2_2b.png")


# %%
n = 100
n_components = 1
x = data_generation1(n)
# x = data_generation2(n)
W = calc_weight_matrix(x)
D = np.diag(np.sum(W, axis=1))
L = D - W
# %%
w, v = np.linalg.eig(np.linalg.inv(x.T @ D @ x) @ x.T @ L @ x)
print(w)
max_w_index = np.argsort(w)[::-1][0]
print(1)
print(v)
print(2)
print(v[:, max_w_index : max_w_index + 1].T)
visualize(x, v[:, max_w_index : max_w_index + 1].T)
# %%
