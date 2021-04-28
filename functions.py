import math
import random
from tqdm import tqdm
import scipy.sparse as sp
import numpy as np
import implicit
from sklearn import metrics
from scipy import stats


def get_sh_mb(matrix):
    pops = np.array(matrix.sum(axis=0)).ravel()
    total = sum(pops)
    i = 0
    current_total = 0
    cutoff = total / 3
    while current_total < cutoff:
        current_total += pops[i]
        i += 1
    sh = i
    current_total = 0
    while current_total < cutoff:
        current_total += pops[i]
        i += 1
    mb = i
    return sh, mb


def get_train_test_masked(matrix: sp.csr_matrix, test_size=500, percent_mask=.4):
    m, n = np.shape(matrix)
    train = matrix[:m - test_size, :]
    test = matrix[m - test_size:, :]

    build_masked = sp.lil_matrix(np.shape(test))

    for i in range(test_size):
        # get all indices from the test playlist where there are 1s, make a list, get 20% of them =

        playlist = test[i, :]

        nonzero_indices = playlist.nonzero()[1]
        num_to_mask = math.ceil(len(nonzero_indices) * percent_mask)

        mask_indices = random.sample(nonzero_indices.tolist(), num_to_mask)

        # zero out the masked songs
        build_masked[i, mask_indices] = 1
        test[i, mask_indices] = 0

    masked = sp.csr_matrix(build_masked)

    return train, test, masked


def get_model(train, alpha, reg, factors=192):
    # creates and trains the model
    model = implicit.als.AlternatingLeastSquares(factors=factors, regularization=reg, calculate_training_loss=True)
    model.fit(train.T * alpha, show_progress=True)

    return model


def simple_score_model(model, test: sp.csr_matrix, masked):
    # test is playlists with some songs removed
    # masked are the songs that test is missing

    n = test.shape[1]
    all_indices = np.arange(n)

    auc = []
    sh_auc = []
    mb_auc = []
    lt_auc = []
    pop_gaps = []

    for i in tqdm(range(test.shape[0])):
        # indices where we have 1s and 0s in the playlist
        nonzero_indices = test[i, :].nonzero()[1]
        zero_indices = np.delete(all_indices, nonzero_indices)

        recommendations = model.recommend(userid=i,
                                          user_items=test,
                                          N=n - len(nonzero_indices),
                                          filter_items=nonzero_indices.tolist(),
                                          filter_already_liked_items=False,
                                          recalculate_user=True)

        indices, scores = zip(*recommendations)
        indices, scores = np.array(indices), np.array(scores)

        # get a vector of scores for each song
        recs = scores[np.argsort(indices)]

        # get a vector of the missing songs
        true_labels = masked[i, zero_indices].toarray().ravel()

        # locations where the missing songs are
        true_inds = masked[i, zero_indices].nonzero()[1]

        # add overall roc_auc_score
        auc.append(metrics.roc_auc_score(true_labels, recs))

    return sum(auc) / len(auc)


def score_model(model, test: sp.csr_matrix, masked, sh, mb):
    # test is playlists with some songs removed
    # masked are the songs that test is missing

    try:
        pops = np.load('data/bookkeeping/pops.npy')
    except FileNotFoundError:
        pref = sp.load_npz('data/matrices/pref_matrix.npz')
        pops = np.array(pref.sum(axis=0)).ravel()
        np.save('data/bookkeeping/pops.npy', pops, allow_pickle=True, fix_imports=False)
        del pref

    n = pops.shape[0]
    all_indices = np.arange(n)

    auc = []
    sh_auc = []
    mb_auc = []
    lt_auc = []
    pop_gaps = []

    for i in tqdm(range(1000)):
        # indices where we have 1s and 0s in the playlist
        nonzero_indices = test[i, :].nonzero()[1]
        zero_indices = np.delete(all_indices, nonzero_indices)

        recommendations = model.recommend(userid=i,
                                          user_items=test,
                                          N=n - len(nonzero_indices),
                                          filter_items=nonzero_indices.tolist(),
                                          filter_already_liked_items=False,
                                          recalculate_user=True)

        indices, scores = zip(*recommendations)
        indices, scores = np.array(indices), np.array(scores)

        # get average popularity of songs in playlist
        pl_avg_pop = pops[nonzero_indices].sum() / nonzero_indices.shape[0]
        # get average popularity of top 10 recommendations
        rec_avg_pop = pops[indices[:10]].sum() / 10
        # calculate the popularity gap
        pop_gaps.append((rec_avg_pop - pl_avg_pop) / pl_avg_pop)

        # get a vector of scores for each song
        recs = scores[np.argsort(indices)]

        # get a vector of the missing songs
        true_labels = masked[i, zero_indices].toarray().ravel()

        # locations where the missing songs are
        true_inds = masked[i, zero_indices].nonzero()[1]

        # add overall roc_auc_score
        auc.append(metrics.roc_auc_score(true_labels, recs))

        true_labels[true_inds[true_inds >= sh]] = 0
        try:
            sh_auc.append(metrics.roc_auc_score(true_labels, recs))
        except:
            pass
        true_labels[true_inds] = 1
        true_labels[true_inds[true_inds < sh]] = 0
        true_labels[true_inds[true_inds > mb]] = 0

        try:
            mb_auc.append(metrics.roc_auc_score(true_labels, recs))
        except:
            pass

        true_labels[true_inds] = 1
        true_labels[true_inds[true_inds < mb]] = 0
        try:
            lt_auc.append(metrics.roc_auc_score(true_labels, recs))
        except:
            pass

    avg = lambda l: sum(l) / len(l)
    print("pop_gaps:", avg(pop_gaps), len(pop_gaps))
    print('auc:', avg(auc), len(auc))
    print('sh_auc:', avg(sh_auc), len(sh_auc))
    print('mb_auc:', avg(mb_auc), len(mb_auc))
    print('lt_auc:', avg(lt_auc), len(lt_auc))

    return avg(pop_gaps), avg(auc), avg(sh_auc), avg(mb_auc), avg(lt_auc)


def iterate_model(model, test: sp.csr_matrix, masked, users=10, iterations=10):
    # test is playlists with some songs removed
    # masked are the songs that test is missing

    try:
        pops = np.load('data/bookkeeping/pops.npy')
    except FileNotFoundError:
        pref = sp.load_npz('data/matrices/pref_matrix.npz')
        pops = np.array(pref.sum(axis=0)).ravel()
        np.save('data/bookkeeping/pops.npy', pops, allow_pickle=True, fix_imports=False)
        del pref

    n = pops.shape[0]

    results = np.zeros((users, iterations + 1))
    for u in tqdm(range(users)):

        new_user = sp.lil_matrix(test[u, :] + masked[u, :])
        nonzero_indices = new_user.nonzero()[1]
        results[u, 0] = pops[nonzero_indices].sum() / nonzero_indices.shape[0]
        for i in range(iterations):
            # indices where we have 1s and 0s in the playlist
            nonzero_indices = list(new_user.nonzero()[1])
            recommendations = model.recommend(userid=0,
                                              user_items=new_user,
                                              N=10,
                                              filter_items=nonzero_indices,
                                              filter_already_liked_items=False,
                                              recalculate_user=True)

            indices, scores = zip(*recommendations)
            indices = np.array(indices)

            # get average popularity of recommendations
            rec_avg_pop = pops[indices].sum() / indices.shape[0]
            results[u, i+1] = rec_avg_pop
            # calculate the popularity gap
            #pop_gaps.append((rec_avg_pop - original_avg_pop) / original_avg_pop)

            # fitting_parameters, covariance = curve_fit(power_pdf, xdata=indices, ydata=pops[indices], p0=0.5, bounds=[0, 1])
            # alphas.append(fitting_parameters[0])

            new_user[:] = sp.lil_matrix(np.shape(new_user))
            new_user[0, indices] = 1

    return results


def gini_coefficient(prefs):
    unsorted_pops = np.array(prefs.sum(axis=0)).ravel()
    sort_indices = np.argsort(unsorted_pops)
    matrix = prefs[:, sort_indices]
    pops = matrix.sum(axis=0)
    pops = np.ravel(pops)
    vec = np.cumsum(pops)
    # vec MUST BE a sorted cumulative sum
    A = (len(vec) * vec[-1]) / 2.0
    B = np.sum(vec)
    return A / (A + B)


def get_playlist_lengths():
	mpd = sp.load_npz('data/matrices/pref_matrix.npz')
	playlist_lengths = np.sum(mpd, axis=1)
	a = np.ravel(playlist_lengths)
	a[a > 250] = 25
	return a


def f_dist_fit(data):
	# estimate parameters from data
	mean, var, skew, kurt = stats.f.fit(data)
	print(f'mean = {mean}, var = {var}, skew = {skew}, kurt = {kurt}')

	# Plot the PDF.
	plt.figure()
	plt.hist(data, bins=80, density=True, alpha=0.6, color='g')
	x = np.linspace(0, 250, 1000)
	p = stats.f.pdf(x, mean, var, skew, kurt)
	plt.plot(x, p, 'k', linewidth=2)
	plt.title("F-distribution")
	plt.show()


def get_playlist_length_samples(n):
	mean = 2.888281575475316
	var = 161.2872788656095
	skew = 2.997516420302752
	kurt = 61.73270382432051
	x = np.linspace(0, 250, 1000)

	pdf = stats.f.pdf(x, mean, var, skew, kurt)
	samples = np.rint(np.random.choice(x, size=n, p=pdf / np.sum(pdf))).astype(np.int32)

	return samples