import json
import math
import random
import time
from tqdm import tqdm
import scipy.sparse as sp
import numpy as np
import implicit
import multiprocessing
from sklearn import metrics


def get_filenames():
    get_slice_filename = lambda n: 'mpd/data/mpd.slice.' + str(n * 1000) + "-" + str(n * 1000 + 999) + '.json'
    filenames = [get_slice_filename(i) for i in range(1000)]
    return filenames


def get_dataset_info():
    filenames = get_filenames()

    unique_track_uris = {}
    track_col_titles = []
    track_col_uris = []
    playlist_names = []
    playlist_followers = []
    i = 0

    for filename in tqdm(filenames):
        with open(filename) as f:
            file = json.load(f)

        for playlist in file['playlists']:
            playlist_names.append(playlist['name'])
            playlist_followers.append(playlist['num_followers'])
            for track in playlist['tracks']:
                track_uri = track['track_uri']
                if track_uri not in unique_track_uris:
                    unique_track_uris[track_uri] = i
                    full_title = track['track_name'] + " by " + track['artist_name'] + " on " + track['album_name']
                    track_col_titles.append(full_title)
                    track_col_uris.append(track_uri)
                    i += 1

    pl_names_array = np.array(playlist_names).ravel()
    pl_followers_array = np.array(playlist_followers).ravel()
    track_col_uris_array = np.array(track_col_uris).ravel()
    track_col_titles_array = np.array(track_col_titles).ravel()

    return unique_track_uris, pl_names_array, pl_followers_array, track_col_uris_array, track_col_titles_array


def get_interaction_matrix(unique_track_uris):
    row, col = [], []
    filenames = get_filenames()

    for i in tqdm(range(len(filenames))):
        with open(filenames[i]) as f:
            file = json.load(f)

        for j in range(len(file['playlists'])):
            playlist = file['playlists'][j]
            for track in playlist['tracks']:
                m = i * 1000 + j
                n = unique_track_uris[track['track_uri']]
                row.append(m)
                col.append(n)

    build_matrix = sp.lil_matrix((len(filenames) * 1000, len(unique_track_uris)))
    build_matrix[row, col] = 1
    csc_matrix = sp.csc_matrix(build_matrix)

    return csc_matrix


def get_shuffled_data(interaction_matrix, pl_names_array, pl_followers_array):
    shuffle_inds = np.arange(np.shape(interaction_matrix)[0])
    np.random.shuffle(shuffle_inds)
    shuffled_pl_names_array = pl_names_array[shuffle_inds]
    shuffled_pl_followers_array = pl_followers_array[shuffle_inds]
    shuffled_interaction_matrix = sp.csr_matrix(interaction_matrix)[shuffle_inds, :]

    return shuffled_interaction_matrix, shuffled_pl_names_array, shuffled_pl_followers_array


def get_sorted_data(interaction_matrix, track_col_uris_array, track_col_titles_array):
    pops = np.array(interaction_matrix.sum(axis=0)).ravel()
    sortIndices = np.flip(np.argsort(pops))

    csr_matrix = sp.csr_matrix(interaction_matrix[:, sortIndices])
    sorted_uris = track_col_uris_array[sortIndices]
    sorted_titles = track_col_titles_array[sortIndices]
    return csr_matrix, sorted_uris, sorted_titles


def get_bm25_confidence_matrix(interaction_matrix: sp.csr_matrix):
    m, n = np.shape(interaction_matrix)

    popularity = np.array(interaction_matrix.sum(axis=0)).ravel()
    inv_doc_freq = np.log((m - popularity + 0.5) / popularity + 0.5)
    matrix = sp.csr_matrix(interaction_matrix.multiply(inv_doc_freq))
    return matrix


def get_length_normalized_confidence_matrix(interaction_matrix: sp.csr_matrix, conf_matrix: sp.csr_matrix):
    m, n = np.shape(interaction_matrix)

    playlist_lengths = np.array(interaction_matrix.sum(axis=1)).ravel()
    avg_playlist_length = np.sum(playlist_lengths) / m
    log_lengths = np.log(1 + avg_playlist_length / playlist_lengths)

    matrix = sp.casr_matrix(conf_matrix.T.multiply(log_lengths).T)
    return matrix


def get_optimal_normalized_confidence_matrix(conf_matrix: sp.csr_matrix, followers):
    log_followers = np.log(1 + followers).ravel()
    matrix = sp.csr_matrix(conf_matrix.T.multiply(log_followers).T)

    return matrix


def get_sh_mb(matrix):
    pops = np.array(matrix.sum(axis=0)).ravel()
    total = sum(pops)
    i = 0
    current_total = 0
    cutoff = total / 3
    while current_total < cutoff:
        current_total += pops[i]
    sh = i
    current_total = 0
    while current_total < cutoff:
        current_total += pops[i]
    mb = i
    return sh, mb


def get_train_test_masked(matrix: sp.csr_matrix, test_size=1000, percent_mask=.2):
    m, n = np.shape(matrix)
    train = matrix[:m - test_size, :]
    test = matrix[m - test_size:, :]

    build_masked = sp.lil_matrix(np.shape(test))

    for i in tqdm(range(test_size)):
        playlist = test[i, :].todense()

        one_inds = list(playlist.nonzero()[1])
        num_to_mask = math.ceil(len(one_inds) * percent_mask)

        mask_inds = random.sample(one_inds, num_to_mask)

        build_masked[i, mask_inds] = 1
        test[i, mask_inds] = 0

    masked = sp.csr_matrix(build_masked)

    return train, test, masked


def get_model(train, alpha, reg, factors=192):
    model = implicit.als.AlternatingLeastSquares(factors=factors, regularization=reg, calculate_training_loss=True)
    model.fit(train.T * alpha, show_progress=True)

    return model


def get_recs(model, test):
    sorted_recs = []
    m, n = np.shape(test)

    for i in tqdm(range(m)):
        recs = model.recommend(userid=i, user_items=test, N=n - test[i, :].count_nonzero())
        inds, scores = zip(*recs)
        inds, scores = np.array(inds), np.array(scores)
        sorted_recs.append(scores[np.argsort(inds)])

    return sorted_recs


def get_scores(masked, recs, sh, mb):
    auc = []
    sh_auc = []
    mb_auc = []
    lt_auc = []

    iters = np.shape(masked)[0]
    for i in tqdm(range(iters)):
        pl = np.array(masked[i, :]).ravel()

        sh_pl = np.copy(pl)
        sh_pl[sh:] = 0

        mb_pl = np.copy(pl)
        mb_pl[:sh] = 0
        mb_pl[mb:] = 0

        lt_pl = np.copy(pl)
        lt_pl[:mb] = 0

        try:
            auc.append(metrics.roc_auc_score(pl, recs[i]))
        except:
            pass

        try:
            sh_auc.append(metrics.roc_auc_score(sh_pl, recs[i]))
        except:
            pass

        try:
            mb_auc.append(metrics.roc_auc_score(mb_pl, recs[i]))
        except:
            pass

        try:
            lt_auc.append(metrics.roc_auc_score(lt_pl, recs[i]))
        except:
            pass
    avg = lambda l: sum(l)/len(l)
    print('auc:', avg(auc))
    print('sh_auc:', avg(sh_auc))
    print('mb_auc:', avg(mb_auc))
    print('lt_auc:', avg(lt_auc))

    return auc, sh_auc, mb_auc, lt_auc
