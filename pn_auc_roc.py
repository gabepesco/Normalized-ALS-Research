from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def main():
    run_tests(length=1000000, iterations=1000, a=0.0001, scale=40000, true_labels=12)


def pn_auc_roc(true, recs, pop_norms):
    # get the locations of the hidden true labels
    true_indices = true.nonzero()[0]
    rec_splits = np.array_split(recs, true_indices)

    # get the first value in every split but the first, which are the indices (locations) of the true labels
    get_first = lambda array: array[0]
    ordered_true_index_iter = map(get_first, rec_splits[1:])
    ordered_true_index_arr = np.fromiter(ordered_true_index_iter, dtype=np.int)

    # get the weights of the true labels from the reciprocal of the norm for each label
    ordered_true_weights = np.reciprocal(pop_norms[ordered_true_index_arr])

    # calculate the height of each rectangle on the graph by adding the new height to the last height
    # first rectangle has height 0, so we concat in a zero
    # h is our vector of the heights of each rectangle
    h = np.add.accumulate(np.concatenate((np.zeros((1,)), ordered_true_weights)))

    # drop the first index from all the splits but the first, so that the true song doesn't add to the width
    drop_first = lambda array: array[1:]
    width_indices_iter = map(drop_first, rec_splits[1:])
    width_indices = [rec_splits[0]] + list(width_indices_iter)

    # calculate the width of each rectangle based on the indices it contains
    get_norm_sum = lambda array: pop_norms[array].sum()
    w_iter = map(get_norm_sum, width_indices)

    # w is our vector of the widths of the rectangle
    w = np.fromiter(w_iter, dtype=np.float32)

    # calculate the area scaling constant
    scale = np.sum(w) * h[-1]

    # dot product of w and h gives the sum of the areas of each rectangle, divide by scale to map it onto unit square
    score = w @ h / scale

    return score


def run_tests(length=2000000, iterations=250, a=0.0001, scale=30000, true_labels=12):
    print(f'length={length}, iterations={iterations}, a={a}, scale={scale}, hidden={true_labels}')

    # generate our popularity distribution
    pops = np.random.power(a, length) * scale

    # convert to int, sort, and flip
    pops = np.sort(np.trunc(pops + 1))
    pops = np.flip(pops)

    # visually check distribution
    # plt.plot(pops)
    # plt.show()

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
