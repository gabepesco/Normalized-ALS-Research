import os
import scipy.sparse as sp
import functions
import pickle


def main():
    int_matrix = sp.load_npz('data/pref_matrix.npz')
    sh, mb = functions.get_sh_mb(int_matrix)
    # total_interactions = int_matrix.sum()
    del int_matrix

    matrix_name = "pref"
    matrix_filename = f'data/{matrix_name}_matrix.npz'
    matrix = sp.load_npz(matrix_filename)
    train, test, masked = functions.get_train_test_masked(matrix)

    os.environ['MKL_NUM_THREADS'] = '1'
    a, r = 512, .25
    model_name = f'data/models/{matrix_name}_a{a}_r{r}_model.pickle'

    # model = functions.get_model(train, alpha=a, reg=r, factors=8)
    # with open(model_name, 'wb') as output_file:
    #     pickle.dump(model, output_file)

    with open(model_name, "rb") as input_file:
        model = pickle.load(input_file)

    auc, sh_auc, mb_auc, lt_auc = functions.score_model(model, test, masked, sh, mb)
    with open(r'data/models/' + str(filename) + '_model_scores.pickle', 'wb') as output_file:
        pickle.dump((auc, sh_auc, mb_auc, lt_auc), output_file)


main()
