import parsefunctions
import numpy as np
import scipy.sparse as sp


def build_files():

    # pl_names, pl_followers, track_uris, track_titles = parsefunctions.get_dataset_info()
    # print('Loaded info.')
    #
    # pref_matrix = parsefunctions.get_pref_matrix(track_uris)
    # print('Loaded pref_matrix.')
    #
    # print('Generating shuffled data...')
    # shuffled_pref_matrix, shuffled_pl_names, shuffled_pl_followers = parsefunctions.get_shuffled_data(pref_matrix, pl_names, pl_followers)
    #
    # np.save('data/pl_names.npy', shuffled_pl_names, allow_pickle=True, fix_imports=False)
    # np.save('data/pl_followers.npy', shuffled_pl_followers, allow_pickle=True, fix_imports=False)
    # print('Saved column data.')
    #
    # print("Generating sorted data...")
    # sorted_pref_matrix, sorted_track_uris, sorted_track_titles = parsefunctions.get_sorted_data(shuffled_pref_matrix, track_uris, track_titles)
    #
    # sp.save_npz('data/pref_matrix.npz', sorted_pref_matrix, compressed=True)
    # np.save('data/track_uris.npy', sorted_track_uris, allow_pickle=True, fix_imports=False)
    # np.save('data/track_titles.npy', sorted_track_titles, allow_pickle=True, fix_imports=False)
    # print('Saved sorted data.')
    #
    # print("Generating bm25_conf_matrix...")
    # bm25_conf_matrix = parsefunctions.get_bm25_conf_matrix(sorted_pref_matrix)
    # sp.save_npz('data/bm25_conf_matrix.npz', bm25_conf_matrix, compressed=True)
    # print('Saved bm25_conf_matrix.')
    #
    # print('Generating tfidf_conf_matrix...')
    # tfidf_conf_matrix = parsefunctions.get_tfidf_conf_matrix(sorted_pref_matrix)
    # sp.save_npz('data/tfidf_conf_matrix.npz', tfidf_conf_matrix, compressed=True)
    # print('Saved tfidf_conf_matrix.')

    sorted_pref_matrix = sp.load_npz('data/pref_matrix.npz')
    print('Generating bm25_len_norm_conf_matrix...')
    bm25_len_norm_conf_matrix = parsefunctions.get_bm25_len_norm_conf_matrix(sorted_pref_matrix)
    sp.save_npz('data/bm25_len_norm_conf_matrix.npz', bm25_len_norm_conf_matrix, compressed=True)
    print('Saved bm25_len_norm_conf_matrix.')

    sorted_pref_matrix = sp.load_npz('data/pref_matrix.npz')
    print('Generating bm25_len_norm_conf_matrix...')
    bm25_len_norm_conf_matrix = parsefunctions.get_bm25_len_norm_conf_matrix(sorted_pref_matrix)
    sp.save_npz('data/bm25_len_norm_conf_matrix.npz', bm25_len_norm_conf_matrix, compressed=True)
    print('Saved bm25_len_norm_conf_matrix.')

    print('Generating optimized_conf_matrix...')
    optimized_conf_matrix = parsefunctions.get_optimized_conf_matrix(shuffled_pref_matrix, bm25_conf_matrix, shuffled_pl_followers)
    sp.save_npz('data/optimized_conf_matrix.npz', optimized_conf_matrix, compressed=True)
    print('Saved optimized_conf_matrix.')

    print("All data generated.")
    return


build_files()
