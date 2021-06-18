
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use('Agg')

np.random.seed(1)


def generate_data(sample_size, n_class):
    x = (np.random.normal(size=(sample_size // n_class, n_class))
         + np.linspace(-3., 3., n_class)).flatten()
    y = np.broadcast_to(np.arange(n_class),
                        (sample_size // n_class, n_class)).flatten()
    return x, y


def calc_design_matrix(x, c, h):
    return np.exp(-(x[None] - c[:, None]) ** 2 / (2 * h ** 2))


def visualize(x, y, theta, h, fname):
    X = np.linspace(-5., 5., num=100)
    K = np.exp(-(x - X[:, None]) ** 2 / (2 * h ** 2))

    plt.clf()
    plt.xlim(-5, 5)
    plt.ylim(-.3, 1.8)
    prob = []
    for i in range(n_class):
        prob.append(K.dot(theta[i]))

    plt.plot(X, prob[0], c='blue')
    plt.plot(X, prob[1], c='red')
    plt.plot(X, prob[2], c='green')

    plt.scatter(x[y == 0], -.1 * np.ones(len(x) // 3), c='blue', marker='o')
    plt.scatter(x[y == 1], -.2 * np.ones(len(x) // 3), c='red', marker='x')
    plt.scatter(x[y == 2], -.1 * np.ones(len(x) // 3), c='green', marker='v')

    plt.savefig(fname)


sample_size = 90
n_class = 3
x, y = generate_data(sample_size, n_class)

l = 0.1
h = 0.3
k = calc_design_matrix(x, x, h)
theta = []
for i in range(n_class):
    theta.append(np.linalg.solve(
        k.T.dot(k) + l * np.identity(len(k)),
        k.T.dot(np.where(y == i, 1., 0.).T)))

fname = "homework_l" + str(l) + "_h" + str('{:.1f}'.format(h)) + ".png"
visualize(x, y, theta, h, fname)
