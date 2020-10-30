import math
import random
from tqdm import tqdm
import scipy.sparse as sp
import numpy as np
import implicit
from sklearn import metrics


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


def get_train_test_masked(matrix: sp.csr_matrix, test_size=1000, percent_mask=.2):
    m, n = np.shape(matrix)
    train = matrix[:m - test_size, :]
    test = matrix[m - test_size:, :]

    build_masked = sp.lil_matrix(np.shape(test))

    for i in tqdm(range(test_size)):
        playlist = test[i, :].todense()

        one_indices = list(playlist.nonzero()[1])
        num_to_mask = math.ceil(len(one_indices) * percent_mask)

        mask_indices = random.sample(one_indices, num_to_mask)

        build_masked[i, mask_indices] = 1
        test[i, mask_indices] = 0

    masked = sp.csr_matrix(build_masked)

    return train, test, masked


def get_model(train, alpha, reg, factors=192):
    model = implicit.als.AlternatingLeastSquares(factors=factors, regularization=reg, calculate_training_loss=True)
    model.fit(train.T * alpha, show_progress=True)

    return model


def score_model(model, test: sp.csr_matrix, masked, sh, mb):
    # test has some songs replaced with 0s
    # masked are the songs that test is missing

    auc = []
    sh_auc = []
    mb_auc = []
    lt_auc = []

    for i in tqdm(range(1000)):
        # nonzero_indices = test[i, :].nonzero
        # print(np.shape(nonzero_indices))
        # zero_indices = np.where(test[i, :].todense() == 0)[1]
        # pl = masked[i, zero_indices].todense().ravel()
        # print("pl:", np.shape(pl))

        playlist = test[i, :].todense()

        # indices where we have 1s and 0s in the playlist
        zero_indices = np.where(playlist == 0)[1]
        nonzero_indices = np.where(playlist != 0)[1]

        # make a vector of true values to identify
        true_labels = np.ravel(masked[i, zero_indices].todense())
        indices, scores = zip(*model.recommend(userid=i,
                                               user_items=test,
                                               N=np.size(true_labels),
                                               filter_items=nonzero_indices.tolist(),
                                               filter_already_liked_items=False,
                                               recalculate_user=True))

        indices, scores = np.array(indices), np.array(scores)
        recs = scores[np.argsort(indices)]

        sh_labels = np.copy(true_labels)
        sh_labels[sh:] = 0

        mb_labels = np.copy(true_labels)
        mb_labels[:sh] = 0
        mb_labels[mb:] = 0

        lt_labels = np.copy(true_labels)
        lt_labels[:mb] = 0

        try:
            auc.append(metrics.roc_auc_score(true_labels, recs))
        except:
            pass

        try:
            sh_auc.append(metrics.roc_auc_score(sh_labels, recs))
        except:
            pass

        try:
            mb_auc.append(metrics.roc_auc_score(mb_labels, recs))
        except:
            pass

        try:
            lt_auc.append(metrics.roc_auc_score(lt_labels, recs))
        except:
            pass

    avg = lambda l: sum(l) / len(l)
    print('auc:', avg(auc), len(auc))
    print('sh_auc:', avg(sh_auc), len(sh_auc))
    print('mb_auc:', avg(mb_auc), len(mb_auc))
    print('lt_auc:', avg(lt_auc), len(lt_auc))

    return avg(auc), avg(sh_auc), avg(mb_auc), avg(lt_auc)
