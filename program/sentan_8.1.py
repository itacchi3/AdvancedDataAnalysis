import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(1)


def generate_data(n=200):
    x = np.linspace(0, np.pi, n // 2)
    u = np.stack([np.cos(x) + 0.5, -np.sin(x)], axis=1) * 10.0
    u += np.random.normal(size=u.shape)
    v = np.stack([np.cos(x) - 0.5, np.sin(x)], axis=1) * 10.0
    v += np.random.normal(size=v.shape)
    x = np.concatenate([u, v], axis=0)
    y = np.zeros(n)
    y[0] = 1
    y[-1] = -1
    return x, y


# def lrls(x, y, h, l, nu, k): #knn weight
def lrls(x, y, h, l, nu, h_w):
    """

    :param x: data points
    :param y: labels of data points
    :param h: width parameter of the Gaussian kernel
    :param l: weight decay
    :param nu: Laplace regularization
    :return:
    """

    K = calc_design_matrix(x, x, h)
    K_tilde = K[y != 0]
    yl = y[y != 0]

    # W = calc_weight_matrix(x, k) #knn weight
    W = calc_weight_matrix(x, h_w)
    D = np.diag(np.sum(W, axis=1))
    L = D - W

    A = K_tilde.T @ K_tilde + l * np.identity(x.shape[0]) + 2 * nu * K.T @ L @ K
    b = K_tilde.T @ yl

    return np.linalg.solve(A, b)


def calc_design_matrix(x, y, h):
    M = []
    for _x in x:
        M_i = []
        for _y in y:
            M_i.append(kernel(_x, _y, h))
        M.append(M_i)
    return np.array(M)


def kernel(x_i, x_j, h):
    return np.exp(-np.dot(x_i - x_j, x_i - x_j) / (2 * h ** 2))


"""
def calc_weight_matrix(x, k): #knn weight
    W = []
    for _x in x:
        W_i = []
        for _y in x:
            W_i.append(knn(_x, _y, x, k))
        W.append(W_i)
    return np.array(W)
"""


def calc_weight_matrix(x, h_w):
    W = []
    for _x in x:
        W_i = []
        for _y in x:
            W_i.append(kernel(_x, _y, h_w))
        W.append(W_i)
    return np.array(W)


def knn(x_i, x_j, x, k):
    D = [np.dot(_x - x_i, _x - x_i) for _x in x]
    if np.dot(x_j - x_i, x_j - x_i) in sorted(D)[: k + 1]:
        return 1.0
    return 0.0


def visualize(x, y, theta, h=1.0):
    plt.clf()
    plt.figure(figsize=(6, 6))
    plt.xlim(-20.0, 20.0)
    plt.ylim(-20.0, 20.0)
    grid_size = 100
    grid = np.linspace(-20.0, 20.0, grid_size)
    X, Y = np.meshgrid(grid, grid)
    mesh_grid = np.stack([np.ravel(X), np.ravel(Y)], axis=1)
    k = np.exp(
        -np.sum(
            (x.astype(np.float32)[:, None] - mesh_grid.astype(np.float32)[None]) ** 2,
            axis=2,
        ).astype(np.float64)
        / (2 * h ** 2)
    )
    plt.contourf(
        X,
        Y,
        np.reshape(np.sign(k.T.dot(theta)), (grid_size, grid_size)),
        alpha=0.4,
        cmap=plt.cm.coolwarm,
    )
    plt.scatter(x[y == 0][:, 0], x[y == 0][:, 1], marker="$.$", c="black")
    plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], marker="$X$", c="red")
    plt.scatter(x[y == -1][:, 0], x[y == -1][:, 1], marker="$O$", c="blue")
    # plt.savefig('lecture8-h1_homework.png') #knn weight
    plt.savefig("lecture8-h1_homework2.png")


x, y = generate_data(n=200)
# theta = lrls(x, y, h=1., l=1., nu=1., k=10) #knn weight
theta = lrls(x, y, h=1.0, l=1.0, nu=1.0, h_w=1.0)
visualize(x, y, theta)
