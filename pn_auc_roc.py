from scipy.optimize import curve_fit
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def main():
    # find_power_param()
    # run_tests(length=20000, iterations=250, a=0.01, scale=40000, true_labels=20)
    a = np.random.random_sample((6, 2))
    a[:, 0] *= 6000
    a[:, 0] += 4000
    a[:, 1] *= 1.25
    a[:, 1] += 1.25
    np.set_printoptions(suppress=True)
    print(a)
    plt.scatter(x=a[:, 0], y=a[:, 1])
    plt.show()


def find_power_param():
    y = np.load("data/bookkeeping/pops.npy")
    x = np.arange(len(y)) + 1
    power_pdf = lambda x, a, b, c: b * np.power(x, a-1) + c

    fitting_parameters, covariance = curve_fit(power_pdf, x, y, p0=(0.1, 40000, -100))
    a, b, c = fitting_parameters[0], fitting_parameters[1], fitting_parameters[2]
    print(a, b, c)
    plt.plot(power_pdf(x, a, b, c))
    plt.scatter(x, y)
    plt.yscale('log')
    plt.show()

    power_pdf = lambda x, a, b: b * np.power(x, a)
    y = np.load("data/bookkeeping/pops.npy")
    x = np.arange(len(y)) + 1
    fitting_parameters, covariance = curve_fit(power_pdf, x[5000:], y[5000:], p0=(.5, 100))
    a, b = fitting_parameters[0], fitting_parameters[1]
    print(f'a={a}, b={b}')

    plt.plot(x, power_pdf(x, a, b))
    plt.scatter(x, y, s=1)
    plt.yscale('log')
    plt.show()

    power_pdf = lambda x, a: 45000 * np.power(x, a)
    y = np.load("data/bookkeeping/pops.npy")
    x = np.arange(len(y)) + 100
    fitting_parameters, covariance = curve_fit(power_pdf, x, y, p0=-0.5)
    a = fitting_parameters[0]
    print(f'a={a}')

    plt.plot(x, power_pdf(x, a))
    plt.scatter(x, y, s=1)
    plt.yscale('log')
    plt.show()
    for i in range(10):
        print(power_pdf(x[i], a, b))

    i = 0
    while power_pdf(x[i], a, b) > 50000:
        i+=1
    print(i)


def pn_auc_roc(true, recs, pop_norms, y_scaling=False, x_scaling=True):
    # get the locations of the hidden true labels
    true_indices = true.nonzero()[0]
    rec_splits = np.array_split(recs, true_indices)

    # get the first value in every split but the first, which are the indices (locations) of the true labels
    get_first = lambda array: array[0]
    ordered_true_index_iter = map(get_first, rec_splits[1:])
    ordered_true_index_arr = np.fromiter(ordered_true_index_iter, dtype=np.int)

    if y_scaling:
        # get the weights of the true labels from the reciprocal of the norm for each label
        ordered_true_weights = np.reciprocal(pop_norms[ordered_true_index_arr])

        # calculate the height of each rectangle on the graph by adding the new height to the last height
        # first rectangle has height 0, so we concat in a zero
        # h is our vector of the heights of each rectangle
        h = np.add.accumulate(np.concatenate((np.zeros((1,)), ordered_true_weights)))

    else:
        # h = np.zeros(len(ordered_true_index_arr))
        # for i in range(len(ordered_true_index_arr) - 1):
        #     h[i] = ordered_true_index_arr[i+1] - ordered_true_index_arr[i]

        n = len(ordered_true_index_arr)
        h = np.arange(start=0, stop=n+1, step=1) / n

        if h[-1] != 1.0:
            print("Error, incorrect y_scaling, last value:", h[-1])

    # drop the first index from all the splits but the first, so that the true song doesn't add to the width
    drop_first = lambda array: array[1:]
    width_indices_iter = map(drop_first, rec_splits[1:])
    width_indices = [rec_splits[0]] + list(width_indices_iter)

    if x_scaling:
        # calculate the width of each rectangle based on the indices it contains
        get_norm_sum = lambda array: pop_norms[array].sum()
        w_iter = map(get_norm_sum, width_indices)

        # w is our vector of the widths of the rectangle
        w = np.fromiter(w_iter, dtype=np.float32)

    else:
        w = np.zeros((len(width_indices) - 1,))
        for i in range(len(width_indices) - 1):
            w[i] = len(width_indices[i+1]) - len(width_indices[i])

        w = w / np.sum(w)

        # if np.sum(w) != 1.0:
        #     print("Error, incorrect x_scaling, sum:", np.sum(w))

    # calculate the area scaling constant
    scale = np.sum(w) * h[-1]

    # dot product of w and h gives the sum of the areas of each rectangle, divide by scale to map it onto unit square
    score = w @ h / scale

    return score


def run_tests(length=2000000, iterations=250, a=0.001, scale=30000, true_labels=12):
    print(f'length={length}, iterations={iterations}, a={a}, scale={scale}, hidden={true_labels}')

    # generate our popularity distribution
    pops = np.random.power(a, length) * scale

    # convert to int, sort, and flip
    pops = np.sort(np.trunc(pops + 1))
    pops = np.flip(pops)

    # visually check distribution
    plt.plot(pops)
    plt.yscale('log')
    plt.show()

    # get popularity norms
    pop_norms = np.sqrt(pops)

    # get popularity recommendation vector
    pop_recs = np.arange(length).astype(dtype=int)

    # make copy to shuffle for random input testing
    shuffle_recs = np.copy(pop_recs)

    pop_scores = []
    rand_scores = []
    mask = np.zeros(length)

    for _ in tqdm(range(iterations)):

        # randomly generate true hidden number of true labels
        true = np.random.randint(0, length, true_labels)

        # create masked vector to evaluate
        mask[true] = np.reciprocal(pop_norms[true])

        # shuffle randomized recommendations
        np.random.shuffle(shuffle_recs)

        # score recommendations with pn_auc_roc
        pop_scores.append(pn_auc_roc(mask, pop_recs, pop_norms))
        rand_scores.append(pn_auc_roc(mask, shuffle_recs, pop_norms))

        # reset mask vector to 0s
        mask[:] = 0

    print("pop avg:", sum(pop_scores) / len(pop_scores))
    print("rand avg:", sum(rand_scores) / len(rand_scores))


main()
