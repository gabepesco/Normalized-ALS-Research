import json
import functions
import numpy as np
import scipy.sparse as sp

get_slice_filename = lambda n: 'mpd/data/mpd.slice.' + str(n * 1000) + "-" + str(n * 1000 + 999) + '.json'
filenames = [get_slice_filename(i) for i in range(1000)]

try:
    with open('data/unique_track_uris.json') as f:
        unique_track_uris = json.load(f)
    print('Loaded unique tracks.')
except FileNotFoundError:
    print('Getting unique tracks...')
    unique_track_uris = functions.get_unique_track_uris(filenames)
    with open('data/unique_track_uris.json', 'w') as w:
        json.dump(unique_track_uris, w)

try:
    with open('data/playlist_indices.json') as f:
        playlist_indices = json.load(f)
    print("Loaded playlist track indices.")
except FileNotFoundError:
    print('Getting playlist track indices...')
    playlist_indices = functions.get_playlist_indices(filenames, unique_track_uris)
    with open('data/playlist_indices.json', 'w') as w:
        json.dump(playlist_indices, w)

try:
    interaction_matrix = sp.load_npz('data/interaction_matrix.npz')
    print('Loaded interaction matrix.')
except FileNotFoundError:
    print('Getting interaction matrix...')
    interaction_matrix = functions.get_interaction_matrix(playlist_indices, (1000000, len(unique_track_uris)))
    sp.save_npz('data/interaction_matrix.npz', interaction_matrix, compressed=True)

functions.get_bm25_vector(interaction_matrix)
# try:
#     bm25_confidence_matrix = np.load('data/bm25_confidence_matrix')
#     print('Loaded BM25 confidence matrix.')
# except FileNotFoundError:
#     print('Generating BM25 confidence matrix...')
#     bm25_confidence_matrix = get_bm25_confidence_matrix(interaction_matrix)
#     np.save('data/bm25_confidence_matrix.npz', bm25_confidence_matrix, allow_pickle=True)

