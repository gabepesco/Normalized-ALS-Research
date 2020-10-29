import json
import functions
import numpy as np
import scipy.sparse as sp


def build_files():
    try:
        with open('data/unique_track_uris.json') as f:
            unique_track_uris = json.load(f)
        pl_names_array = np.load('data/pl_names_array.npy', allow_pickle=True)
        pl_followers_array = np.load('data/pl_followers_array.npy', allow_pickle=True)
        track_col_uris_array = np.load('data/track_col_uris_array.npy', allow_pickle=True)
        track_col_titles_array = np.load('data/track_col_titles_array.npy', allow_pickle=True)

    except FileNotFoundError:
        print('Getting dataset info...')
        unique_track_uris, pl_names_array, pl_followers_array, track_col_uris_array, track_col_titles_array = \
            functions.get_dataset_info()

        with open('data/unique_track_uris.json', 'w') as w:
            json.dump(unique_track_uris, w)
        np.save('data/pl_names_array.npy', pl_names_array, allow_pickle=True, fix_imports=False)
        np.save('data/pl_followers_array.npy', pl_followers_array, allow_pickle=True, fix_imports=False)
        np.save('data/track_col_uris_array.npy', track_col_uris_array, allow_pickle=True, fix_imports=False)
        np.save('data/track_col_titles_array.npy', track_col_titles_array, allow_pickle=True, fix_imports=False)

    print('Loaded info.')

    try:
        interaction_matrix = sp.load_npz('data/interaction_matrix.npz')

    except FileNotFoundError:
        print('Generating interaction_matrix...')
        interaction_matrix = functions.get_interaction_matrix(unique_track_uris)

        sp.save_npz('data/interaction_matrix.npz', interaction_matrix, compressed=True)

    print('Loaded interaction_matrix.')

    try:
        shuffled_interaction_matrix = sp.load_npz('data/shuffled_interaction_matrix.npz')
        shuffled_pl_names_array = np.load('data/shuffled_pl_names_array', allow_pickle=True)
        shuffled_pl_followers_array = np.load('data/shuffled_pl_followers_array', allow_pickle=True)

    except FileNotFoundError:
        print('Generating shuffled data...')
        shuffled_interaction_matrix, shuffled_pl_names_array, shuffled_pl_followers_array = \
            functions.get_shuffled_data(interaction_matrix, pl_names_array, pl_followers_array)

        sp.save_npz('data/interaction_matrix.npz', interaction_matrix, compressed=True)
        np.save('data/shuffled_pl_names_array.npy', pl_names_array, allow_pickle=True, fix_imports=False)
        np.save('data/shuffled_pl_followers_array.npy', pl_names_array, allow_pickle=True, fix_imports=False)

    print('Loaded shuffled data.')

    try:
        sorted_interaction_matrix = sp.load_npz('data/sorted_interaction_matrix.npz')
        sorted_track_col_uris_array = np.load('data/sorted_track_col_uris_array', allow_pickle=True)
        sorted_track_col_titles_array = np.load('data/sorted_track_col_titles_array', allow_pickle=True)

    except FileNotFoundError:
        print("Generating sorted data...")
        sorted_interaction_matrix, sorted_track_col_uris_array, sorted_track_col_titles_array = \
            functions.get_sorted_data(shuffled_interaction_matrix, track_col_uris_array, track_col_titles_array)

        sp.save_npz('data/sorted_interaction_matrix.npz', sorted_interaction_matrix, compressed=True)
        np.save('data/sorted_track_col_uris_array.npy', sorted_track_col_uris_array, allow_pickle=True,
                fix_imports=False)
        np.save('data/sorted_track_col_titles_array.npy', sorted_track_col_titles_array, allow_pickle=True,
                fix_imports=False)

    print('Loaded sorted data.')

    try:
        bm25_confidence_matrix = sp.load_npz('data/bm25_confidence_matrix.npz')

    except FileNotFoundError:
        print('Generating bm_25_confidence_matrix...')
        bm25_confidence_matrix = functions.get_bm25_confidence_matrix(sorted_interaction_matrix)
        sp.save_npz('data/bm25_confidence_matrix.npz', bm25_confidence_matrix, compressed=True)

    print('Loaded bm_25_confidence_matrix.')

    try:
        length_normalized_confidence_matrix = sp.load_npz('data/length_normalized_confidence_matrix.npz')

    except FileNotFoundError:
        print('Generating length_normalized_confidence_matrix...')
        length_normalized_confidence_matrix = functions.get_length_normalized_confidence_matrix(
            sorted_interaction_matrix, bm25_confidence_matrix)

        sp.save_npz('data/length_normalized_confidence_matrix.npz', length_normalized_confidence_matrix,
                    compressed=True)

    print('Loaded length_normalized_confidence_matrix.')

    try:
        optimal_confidence_matrix = sp.load_npz('data/optimal_confidence_matrix')

    except FileNotFoundError:
        print('Generating optimal_confidence_matrix...')
        optimal_confidence_matrix = functions.get_optimal_normalized_confidence_matrix(
            length_normalized_confidence_matrix, shuffled_pl_followers_array)

        sp.save_npz('data/optimal_confidence_matrix.npz', optimal_confidence_matrix, compressed=True)

    print("All data generated.")
    return


build_files()
