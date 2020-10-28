import json
import functions
import numpy as np
import scipy.sparse as sp


def get_matrices():
    try:
        followers = np.load('data/followers.npy', allow_pickle=True)
    except FileNotFoundError:
        print('Getting followers...')
        followers = functions.get_playlist_followers()
        np.save('data/followers.npy', followers, allow_pickle=True, fix_imports=False)
    print('Loaded followers.')

    try:
        with open('data/unique_track_uris.json') as f:
            unique_track_uris = json.load(f)
    except FileNotFoundError:
        print('Getting unique tracks...')
        unique_track_uris = functions.get_unique_track_uris()
        with open('data/unique_track_uris.json', 'w') as w:
            json.dump(unique_track_uris, w)
    print('Loaded unique tracks.')


    try:
        interaction_matrix = sp.load_npz('data/interaction_matrix.npz')
    except FileNotFoundError:
        print('Generating interaction matrix...')
        interaction_matrix = functions.get_interaction_matrix(unique_track_uris)
        sp.save_npz('data/interaction_matrix.npz', interaction_matrix, compressed=True)
    print('Loaded interaction matrix.')

    try:
        bm25_confidence_matrix = sp.load_npz('data/bm25_confidence_matrix.npz')
    except FileNotFoundError:
        print('Generating bm_25_confidence_matrix...')
        bm25_confidence_matrix = functions.get_bm25_confidence_matrix(interaction_matrix)
        sp.save_npz('data/bm25_confidence_matrix.npz', bm25_confidence_matrix, compressed=True)
    print('Loaded bm_25_confidence_matrix.')

    try:
        length_normalized_confidence_matrix = sp.load_npz('data/length_normalized_confidence_matrix.npz')
    except FileNotFoundError:
        print('Generating length_normalized_confidence_matrix...')
        length_normalized_confidence_matrix = functions.get_length_normalized_confidence_matrix(interaction_matrix, bm25_confidence_matrix)
        sp.save_npz('data/length_normalized_confidence_matrix.npz', length_normalized_confidence_matrix, compressed=True)
    print('Loaded length_normalized_confidence_matrix.')

    try:
        optimal_confidence_matrix = sp.load_npz('data/optimal_confidence_matrix')
    except FileNotFoundError:
        print('Generating optimal_confidence_matrix...')
        optimal_confidence_matrix = functions.get_optimal_normalized_confidence_matrix(length_normalized_confidence_matrix, followers)
        sp.save_npz('data/optimal_confidence_matrix.npz', optimal_confidence_matrix, compressed=True)

    return interaction_matrix, bm25_confidence_matrix, length_normalized_confidence_matrix, optimal_confidence_matrix
