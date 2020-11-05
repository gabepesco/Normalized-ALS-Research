import pickle
import numpy as np


def main():
    song = 'Signal'
    artist = 'Sylvan Esso'
    album = 'What Now'
    n = 20
    exclude_artist = False

    track_titles = np.load('data/bookkeeping/track_titles.npy')

    # search_term = "Magic In The Hamptons"
    # for i in range(len(track_titles)):
    #     if search_term in track_titles[i]:
    #         print(track_titles[i])
    # quit()

    track_title = f'{song} by {artist} on {album}'
    track_index = np.where(track_titles == track_title)[0][0]

    with open('data/models/bm25_len_norm_conf_a774.0_r1.19_f128_model.pickle', 'rb') as f:
        model = pickle.load(f)
    recs = model.similar_items(itemid=track_index, N=n)
    rec_indices, scores = zip(*recs)

    rec_list = [f'Top {n} songs like {track_title}\n', '\n']

    # First rec is always the original song, so slice it out

    for index in rec_indices[1:]:
        track = track_titles[index]
        if exclude_artist:
            if f'by {artist} on' not in track:
                # noinspection PyTypeChecker
                rec_list.append(track + '\n')
        else:
            # noinspection PyTypeChecker
            rec_list.append(track + '\n')

    with open(f'data/song_recs/{song}.txt', 'a') as f:
        f.writelines(rec_list)
        f.close()


main()
