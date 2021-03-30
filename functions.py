import math
import random
from tqdm import tqdm
import scipy.sparse as sp
import numpy as np
import implicit
from sklearn import metrics
from scipy.optimize import curve_fit


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


def auc_score(predictions, test):
    '''
    This simple function will output the area under the curve using sklearn's metrics.

    parameters:

    - predictions: your prediction output

    - test: the actual target result you are comparing to

    returns:

    - AUC (area under the Receiver Operating Characterisic curve)
    '''
    fpr, tpr, thresholds = metrics.roc_curve(test, predictions)
    return metrics.auc(fpr, tpr)


def calc_mean_auc(training_set, altered_users, predictions, test_set):
    '''
    This function will calculate the mean AUC by user for any user that had their user-item matrix altered.

    parameters:

    training_set - The training set resulting from make_train, where a certain percentage of the original
    user/item interactions are reset to zero to hide them from the model

    predictions - The matrix of your predicted ratings for each user/item pair as output from the implicit MF.
    These should be stored in a list, with user vectors as item zero and item vectors as item one.

    altered_users - The indices of the users where at least one user/item pair was altered from make_train function

    test_set - The test set constucted earlier from make_train function



    returns:

    The mean AUC (area under the Receiver Operator Characteristic curve) of the test set only on user-item interactions
    there were originally zero to test ranking ability in addition to the most popular items as a benchmark.
    '''

    store_auc = []  # An empty list to store the AUC for each user that had an item removed from the training set
    popularity_auc = []  # To store popular AUC scores
    pop_items = np.array(test_set.sum(axis=0)).reshape(-1)  # Get sum of item iteractions to find most popular
    item_vecs = predictions[1]
    for user in tqdm(altered_users):  # Iterate through each user that had an item altered
        training_row = training_set[user.nonzero(), :].toarray().reshape(-1)  # Get the training set row
        zero_inds = np.where(training_row == 0)  # Find where the interaction had not yet occurred
        # Get the predicted values based on our user/item vectors
        user_vec = predictions[0][user.nonzero(), :]
        pred = user_vec.dot(item_vecs).toarray()[0, zero_inds].reshape(-1)
        # Get only the items that were originally zero
        # Select all ratings from the MF prediction for this user that originally had no iteraction
        actual = test_set[user, :].toarray()[0, zero_inds].reshape(-1)
        # Select the binarized yes/no interaction pairs from the original full data
        # that align with the same pairs in training
        pop = pop_items[zero_inds]  # Get the item popularity for our chosen items
        store_auc.append(auc_score(pred, actual))  # Calculate AUC for the given user and store
        popularity_auc.append(auc_score(pop, actual))  # Calculate AUC using most popular and score
    # End users iteration

    return float('%.3f' % np.mean(store_auc)), float('%.3f' % np.mean(popularity_auc))
    # Return the mean AUC rounded to three decimal places for both test and popularity benchmark


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

    #n = pops.shape[0]
    #all_indices = np.arange(n)


    #alphas = []
    #avg_pops = []
    #pop_gaps = []

    #nonzero_indices = (test[0, :] + masked[0, :]).nonzero()[1]
    #original_avg_pop = pops[nonzero_indices].sum() / nonzero_indices.shape[0]
    #new_user = test[0, :] + masked[0, :]

    #power_pdf = lambda x, a: 45000 * a * np.power(x, a)
    results = np.zeros((users, iterations))
    print(np.sum(pops[:66])/66)
    for u in tqdm(range(users)):
        # nonzero_indices = (test[u, :] + masked[u, :]).nonzero()[1]
        # original_avg_pop = pops[nonzero_indices].sum() / nonzero_indices.shape[0]
        new_user = test[u, :] + masked[u, :]

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
            indices, scores = np.array(indices), np.array(scores)

            # get average popularity of recommendations
            rec_avg_pop = pops[indices].sum() / indices.shape[0]
            results[u, i] = rec_avg_pop
            # calculate the popularity gap
            #pop_gaps.append((rec_avg_pop - original_avg_pop) / original_avg_pop)

            # fitting_parameters, covariance = curve_fit(power_pdf, xdata=indices, ydata=pops[indices], p0=0.5, bounds=[0, 1])
            # alphas.append(fitting_parameters[0])

            new_user[:] = sp.csr_matrix(np.shape(new_user))
            new_user[0, indices] = 1

    averages = (np.sum(results, axis=0) / results.shape[0]).ravel()
    print(results.shape)
    print(np.shape(averages))

    return list(averages)
