import numpy as np
import matplotlib
import random
import matplotlib.pyplot as plt

np.random.seed(0)

def generate_sample_data(xmin, xmax, sample_size):
    x = np.linspace(start=xmin, stop=xmax, num=sample_size)
    pix = np.pi * x
    target = np.sin(pix) / pix + 0.1 * x
    noise = 0.05 * np.random.normal(loc=0., scale=1., size=sample_size)
    return x, target + noise

def calc_design_matrix(x, c, h):
    return np.exp(-(x[None] - c[:, None]) ** 2 / (2 * h ** 2))

def omit_data_i(data, lnt, i, idx_int):
    data_len = len(data)
    group_len = int(data_len/lnt)
    cropper = idx_int[i:i + group_len]
    data_omitted = data[cropper]
    data_i = np.delete(data, cropper)
    return data_i, data_omitted

def calc_squre_err(theta, k, y):
    s_err = np.linalg.norm(np.dot(k.T, theta).T-y)/10
    return s_err

def validate(x, y, lamb, h, lnt, i, idx_int):
    x_i, x_omitted = omit_data_i(x, lnt, i, idx_int)
    y_i, y_omitted = omit_data_i(y, lnt, i, idx_int)
    k = calc_design_matrix(x_i, x_i, h)

    theta = np.linalg.solve(
        k.T.dot(k) + lamb * np.identity(len(k)),
        k.T.dot(y_i[:, None]))

    K_valid = calc_design_matrix(x_omitted, x_i, h)
    err = calc_squre_err(theta, K_valid, y_omitted)

    xmin, xmax = -3, 3
    X = np.linspace(start=xmin, stop=xmax, num=5000)
    K = calc_design_matrix(x_i, X, h)
    prediction = K.dot(theta)

    # visualization
    plt.clf()
    plt.scatter(x_i, y_i, c='green', marker='o')
    plt.plot(X, prediction)
    return err

def main():
    # create sample
    sample_size = 50
    xmin, xmax = -3, 3
    x, y = generate_sample_data(xmin=xmin, xmax=xmax, sample_size=sample_size)
    lamb = np.logspace(-3,3,7) # [0.0001, 0.01, 0.1, 1, 10, 100, 1000]
    h = np.logspace(-3,3,7) # [0.0001, 0.01, 0.1, 1, 10, 100, 1000]
    lnt = len(h)
    idx_int = np.random.randint(0,50,50)

    for _lamb in lamb:
        for _h in h:
            print('*'*20)
            print('lamb = ', _lamb)
            print('h = ', _h)
            err_array = np.array([])
            for i in np.arange(0, 50, 10):
                err = validate(x, y, _lamb, _h, lnt, i, idx_int)
                err_array = np.append(err_array, err)
                plt.show()
            print(err_array)
            print('mean = ',np.mean(err_array))

if __name__ == '__main__':
    main()