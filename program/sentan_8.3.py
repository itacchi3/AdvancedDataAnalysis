import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

np.random.seed(1)


def generate_data(n_total, n_positive):
    x = np.random.normal(size=(n_total, 2))
    x[:n_positive, 0] -= 2
    x[n_positive:, 0] += 2
    x[:, 1] *= 2.0
    y = np.empty(n_total, dtype=np.int64)
    y[:n_positive] = -1
    y[n_positive:] = 1
    return x, y


def cwls(train_x, train_y, eval_x):
    train_x = np.block([train_x, np.ones(100)[:, np.newaxis]])
    eval_x = np.block([eval_x, np.ones(100)[:, np.newaxis]])
    pi = calc_pi(train_x, train_y, eval_x)
    train_x_p = train_x[train_y == 1]
    train_x_n = train_x[train_y == -1]
    n_p = train_x_p.shape[0]
    n_n = train_x_n.shape[0]
    n = train_x.shape[0]
    W = np.diag(np.where(train_y == 1, pi / (n_p / n), (1 - pi) / (n_n / n)))
    return np.linalg.solve(train_x.T @ W @ train_x, train_x.T @ W @ train_y)
    # return np.linalg.solve(train_x.T@train_x, train_x.T@train_y) #unweighted


def calc_pi(train_x, train_y, eval_x):
    train_x_p = train_x[train_y == 1]
    train_x_n = train_x[train_y == -1]
    n_p = train_x_p.shape[0]
    n_n = train_x_n.shape[0]
    n = eval_x.shape[0]

    A_pp = calc_A_b(train_x_p, train_x_p, n_p, n_p)
    A_pn = calc_A_b(train_x_p, train_x_n, n_p, n_n)
    A_nn = calc_A_b(train_x_n, train_x_n, n_n, n_n)
    b_p = calc_A_b(train_x_p, eval_x, n_p, n)
    b_n = calc_A_b(train_x_n, eval_x, n_n, n)
    return (A_pn - A_nn - b_p + b_n) / (2 * A_pn - A_pp - A_nn)


def calc_A_b(x, y, n1, n2):
    A = 0
    for _x in x:
        for _y in y:
            A += np.sqrt(np.dot(_x - _y, _x - _y))
    A /= n1 * n2
    return A


def visualize(train_x, train_y, test_x, test_y, theta):
    for x, y, name in [(train_x, train_y, "train"), (test_x, test_y, "test")]:
        plt.clf()
        plt.figure(figsize=(6, 6))
        plt.xlim(-5.0, 5.0)
        plt.ylim(-7.0, 7.0)
        lin = np.array([-5.0, 5.0])
        plt.plot(lin, -(theta[2] + lin * theta[0]) / theta[1])
        plt.scatter(x[y == -1][:, 0], x[y == -1][:, 1], marker="$O$", c="blue")
        plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], marker="$X$", c="red")
        plt.savefig("lecture8-h3-{}_weighted.png".format(name))
        # plt.savefig('lecture8-h3-{}_unnweighted.png'.format(name)) #unweighted


train_x, train_y = generate_data(n_total=100, n_positive=90)
eval_x, eval_y = generate_data(n_total=100, n_positive=10)
theta = cwls(train_x, train_y, eval_x)
visualize(train_x, train_y, eval_x, eval_y, theta)
