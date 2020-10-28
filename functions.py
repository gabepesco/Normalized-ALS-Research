import json
from tqdm import tqdm
import scipy.sparse as sp
import numpy as np

get_track_uri = lambda track_uri: track_uri[14:]
avg = lambda lst: sum(lst) / len(lst)


def get_unique_track_uris(filenames):
    unique_track_uris = {}
    i = 0
    for filename in tqdm(filenames):
        with open(filename) as f:
            slice = json.load(f)

        playlists = slice['playlists']
        for playlist in playlists:
            for track in playlist['tracks']:
                track_uri = get_track_uri(track['track_uri'])
                if track_uri not in unique_track_uris:
                    unique_track_uris[track_uri] = i
                    i += 1

    return unique_track_uris


def get_playlist_indices(filenames, unique_track_uris):
    playlist_indices = []
    for filename in tqdm(filenames):
        with open(filename) as f:
            slice = json.load(f)

        playlists = slice['playlists']
        for playlist in playlists:
            indices = [unique_track_uris[get_track_uri(track['track_uri'])] for track in playlist['tracks']]
            playlist_indices.append(indices)

    return playlist_indices


def get_interaction_matrix(playlist_indices, dimensions):
    matrix = sp.lil_matrix(dimensions)
    for row in tqdm(range(dimensions[0])):
        matrix[row, playlist_indices[row]] = 1
    return sp.csr_matrix(matrix)


def get_bm25_vector(interaction_matrix):
    track_popularity = np.sum(interaction_matrix, axis=0)
    print(np.shape(track_popularity))
    print(np.shape(interaction_matrix))

def get_bm25_confidence_matrix(interaction_matrix):
    c = sp.lil_matrix(interaction_matrix.dimensions)