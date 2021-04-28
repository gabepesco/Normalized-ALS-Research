import os
import scipy.sparse as sp
import functions
import pickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def main():
	os.environ['MKL_NUM_THREADS'] = '1'
	os.environ['MKL_DEBUG_CPU_TYPE'] = '5'
	mpd = sp.load_npz('data/matrices/pref_matrix.npz')
	current_pref = mpd[:100000, :]
	print(f'gini: {functions.gini_coefficient(current_pref)}')
	sp.save_npz(f'data/iteration/matrices/pref{0}.npz', current_pref, compressed=True)

	for i in range(5):
		model_name = f'data/iteration/models/model{i}.pkl'
		try:
			with open(model_name, "rb") as input_file:
				model = pickle.load(input_file)

		except FileNotFoundError:
			model = functions.get_model(train=current_pref, alpha=7900.0, reg=1.3, factors=128)

			with open(model_name, 'wb') as output_file:
				pickle.dump(model, output_file)

		# try:
		# 	current_pref = sp.load_npz(f'data/iteration/matrices/pref{i}.npz')
		# except FileNotFoundError or ValueError:
		# print(f'iter: 0, gini: {functions.gini_coefficient(current_pref)}')

		current_pref = get_next_pref(current_pref, model)
		print(f'gini: {functions.gini_coefficient(current_pref)}')
		sp.save_npz(f'data/iteration/matrices/pref{i+1}.npz', current_pref, compressed=True)


def get_next_pref(pref, model, batch_size=100):
	m, n = np.shape(pref)
	new_pref = sp.lil_matrix((m, n))
	lengths = functions.get_playlist_length_samples(n=m)
	num_batches = int(m / batch_size)

	for b in tqdm(range(num_batches)):
		vec = np.cumsum(lengths[b*batch_size: (b+1)*batch_size])
		x = np.zeros((vec[-1],))
		y = np.zeros((vec[-1],))
		for i in range(batch_size):
			idx = b*batch_size + i
			recs = model.recommend(userid=idx,
									user_items=pref,
									N=lengths[idx],
									filter_already_liked_items=True,
									recalculate_user=False)
			indices, scores = zip(*recs)
			if i == 0:
				start = 0
			else:
				start = vec[i-1]
			end = vec[i]
			x[start:end] = idx
			y[start:end] = indices
		new_pref[(x, y)] = 1
	return sp.csr_matrix(new_pref)


main()
# mpd = sp.load_npz('data/matrices/pref_matrix.npz')
# print(functions.gini_coefficient(mpd))
# for i in range(5):
# 	pref = sp.load_npz(f'data/iteration/matrices/pref{i}.npz')
# 	print(functions.gini_coefficient(pref))