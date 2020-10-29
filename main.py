import os
import scipy.sparse as sp
import functions
import pickle


def main():
    int_matrix = sp.load_npz('data/pref_matrix.npz')
    sh, mb = functions.get_sh_mb(int_matrix)
    del int_matrix

    filename = "pref_matrix"
    matrix = sp.load_npz('data/' + filename + '.npz')
    train, test, masked = functions.get_train_test_masked(matrix)

    os.environ['MKL_NUM_THREADS'] = '1'
    model = functions.get_model(train, alpha=512, reg=.25)
    with open(r"data/' + filename + '_model.pickle", "wb") as output_file:
        pickle.dump(model, output_file)

    # with open(r"data/bm25_model.pickle", "rb") as input_file:
    #     model = pickle.load(input_file)

    auc, sh_auc, mb_auc, lt_auc = functions.score_model(model, test, masked, sh, mb)
    with open(r"data/model_scores.pickle", "wb") as output_file:
        pickle.dump((auc, sh_auc, mb_auc, lt_auc), output_file)


main()
