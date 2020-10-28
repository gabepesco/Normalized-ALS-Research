import json
from tqdm import tqdm
import scipy.sparse as sp
import numpy as np


def get_filenames():
    get_slice_filename = lambda n: 'mpd/data/mpd.slice.' + str(n * 1000) + "-" + str(n * 1000 + 999) + '.json'
    filenames = [get_slice_filename(i) for i in range(1000)]
    return filenames


def get_playlist_followers():
    filenames = get_filenames()
    followers = np.zeros(1000000)
    i = 0
    for filename in tqdm(filenames):
        with open(filename) as f:
            file = json.load(f)

        for playlist in file['playlists']:
            followers[i] = playlist['num_followers']
            i += 1

    return followers


def get_unique_track_uris():
    filenames = get_filenames()

    get_track_uri = lambda track_uri: track_uri[14:]
    unique_track_uris = {}
    i = 0

    for filename in tqdm(filenames):
        with open(filename) as f:
            file = json.load(f)

        for playlist in file['playlists']:
            for track in playlist['tracks']:
                track_uri = get_track_uri(track['track_uri'])
                if track_uri not in unique_track_uris:
                    unique_track_uris[track_uri] = i
                    i += 1

    return unique_track_uris


def get_interaction_matrix(unique_track_uris):
    row, col = [], []
    filenames = get_filenames()
    get_track_uri = lambda track_uri: track_uri[14:]

    for i in tqdm(range(len(filenames))):
        with open(filenames[i]) as f:
            file = json.load(f)

        for j in range(len(file['playlists'])):
            playlist = file['playlists'][j]
            for track in playlist['tracks']:
                m = i * 1000 + j
                n = unique_track_uris[get_track_uri(track['track_uri'])]
                row.append(m)
                col.append(n)

    interaction_matrix = sp.lil_matrix((len(filenames) * 1000, len(unique_track_uris)))
    interaction_matrix[row, col] = 1

    return sp.csr_matrix(interaction_matrix)


def get_bm25_confidence_matrix(interaction_matrix: sp.csr_matrix):
    m, n = np.shape(interaction_matrix)

    popularity = np.array(interaction_matrix.sum(axis=0)).ravel()
    inv_doc_freq = np.log((m - popularity + 0.5) / popularity + 0.5)

    return interaction_matrix.multiply(inv_doc_freq)


def get_length_normalized_confidence_matrix(interaction_matrix: sp.csr_matrix, conf_matrix: sp.csr_matrix):
    m, n = np.shape(interaction_matrix)

    playlist_lengths = np.array(interaction_matrix.sum(axis=1)).ravel()
    avg_playlist_length = np.sum(playlist_lengths) / m
    log_lengths = np.log(1 + avg_playlist_length / playlist_lengths)

    return conf_matrix.T.multiply(log_lengths).T


def get_optimal_normalized_confidence_matrix(conf_matrix: sp.csr_matrix, followers):
    log_followers = np.log(1 + followers).ravel()

    return conf_matrix.T.multiply(log_followers).T
