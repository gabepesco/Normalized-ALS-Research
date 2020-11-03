import json
from tqdm import tqdm
import scipy.sparse as sp
import numpy as np


def get_filenames():
    # generates list of filenames for files in the dataset
    get_slice_filename = lambda n: 'mpd/data/mpd.slice.' + str(n * 1000) + "-" + str(n * 1000 + 999) + '.json'
    filenames = [get_slice_filename(i) for i in range(1000)]
    return filenames


def get_dataset_info():
    filenames = get_filenames()

    unique_track_uris = {}
    track_titles = []
    track_uris = []
    pl_names = []
    pl_followers = []
    i = 0

    for filename in tqdm(filenames):
        with open(filename) as f:
            file = json.load(f)

        for playlist in file['playlists']:
            pl_names.append(playlist['name'])
            pl_followers.append(playlist['num_followers'])
            for track in playlist['tracks']:
                track_uri = track['track_uri']
                if track_uri not in unique_track_uris:
                    unique_track_uris[track_uri] = i
                    full_title = track['track_name'] + " by " + track['artist_name'] + " on " + track['album_name']
                    track_titles.append(full_title)
                    track_uris.append(track_uri)
                    i += 1

    pl_names_array = np.array(pl_names).ravel()
    pl_followers_array = np.array(pl_followers).ravel()
    track_uris_array = np.array(track_uris).ravel()
    track_titles_array = np.array(track_titles).ravel()

    return pl_names_array, pl_followers_array, track_uris_array, track_titles_array


def get_pref_matrix(track_uris):
    track_dict = {}
    for i in range(np.size(track_uris)):
        track_dict[track_uris[i]] = i

    row, col = [], []
    filenames = get_filenames()

    for i in tqdm(range(len(filenames))):
        with open(filenames[i]) as f:
            file = json.load(f)

        for j in range(len(file['playlists'])):
            playlist = file['playlists'][j]
            for track in playlist['tracks']:
                m = i * 1000 + j
                n = track_dict[track['track_uri']]
                row.append(m)
                col.append(n)

    build_matrix = sp.lil_matrix((len(filenames) * 1000, len(track_dict)))
    build_matrix[row, col] = 1
    matrix = sp.csr_matrix(build_matrix)

    return matrix


def get_shuffled_data(pref_matrix, pl_names_array, pl_followers_array):
    # shuffle order of playlists (rows), maintain in pl data arrays

    shuffle_indices = np.arange(np.shape(pref_matrix)[0])
    np.random.shuffle(shuffle_indices)

    shuffled_pl_names = pl_names_array[shuffle_indices]
    shuffled_pl_followers = pl_followers_array[shuffle_indices]
    shuffled_pref_matrix = sp.csr_matrix(pref_matrix)[shuffle_indices, :]

    return shuffled_pref_matrix, shuffled_pl_names, shuffled_pl_followers


def get_sorted_data(pref_matrix: sp.csr_matrix, track_uris, track_titles):
    # sort columns by track popularity, maintain in names and freqs

    pops = np.array(pref_matrix.sum(axis=0)).ravel()
    sort_indices = np.flip(np.argsort(pops))

    copy_matrix = pref_matrix.copy()
    matrix = sp.csr_matrix(copy_matrix[:, sort_indices])

    sorted_uris = track_uris[sort_indices]
    sorted_titles = track_titles[sort_indices]

    return matrix, sorted_uris, sorted_titles


def get_tfidf_conf_matrix(pref_matrix: sp.csr_matrix):
    m, n = np.shape(pref_matrix)
    popularity = np.array(pref_matrix.sum(axis=0)).ravel()
    inv_doc_freq = np.log(m / popularity)
    copy_matrix = pref_matrix.copy()
    matrix = sp.csr_matrix(copy_matrix.multiply(inv_doc_freq))

    return matrix


def get_bm25_conf_matrix(pref_matrix: sp.csr_matrix):
    m, n = np.shape(pref_matrix)

    popularity = np.array(pref_matrix.sum(axis=0)).ravel()
    inv_doc_freq = np.log((m - popularity + 0.5) / popularity + 0.5)

    matrix_copy = pref_matrix.copy()
    matrix = sp.csr_matrix(matrix_copy.multiply(inv_doc_freq))

    return matrix


def get_bm25_len_norm_conf_matrix(pref_matrix: sp.csr_matrix):
    m, n = np.shape(pref_matrix)

    popularity = np.array(pref_matrix.sum(axis=0)).ravel()
    inv_doc_freq = np.log((m - popularity + 0.5) / popularity + 0.5)

    playlist_lengths = np.array(pref_matrix.sum(axis=1)).ravel()
    avg_playlist_length = np.sum(playlist_lengths) / m
    log_lengths = np.log(1 + avg_playlist_length / playlist_lengths)

    matrix_copy = pref_matrix.copy()
    matrix = matrix_copy.multiply(inv_doc_freq).T.multiply(log_lengths).T

    return sp.csr_matrix(matrix)


def get_optimized_conf_matrix(pref_matrix: sp.csr_matrix, bm25_conf_matrix: sp.csr_matrix, followers):
    m, n = np.shape(pref_matrix)

    log_followers = np.log(1 + followers).ravel()

    playlist_lengths = np.array(pref_matrix.sum(axis=1)).ravel()
    avg_playlist_length = np.sum(playlist_lengths) / m
    log_lengths = np.log(1 + avg_playlist_length / playlist_lengths)

    copy_matrix = bm25_conf_matrix.copy()
    matrix = copy_matrix.T.multiply(log_followers).multiply(log_lengths).T

    return sp.csr_matrix(matrix)
