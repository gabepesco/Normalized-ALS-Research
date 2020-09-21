# Normalized-ALS-Research

**NOTE: The required files to run this code are not publicly available. If you request the MPD from spotify, you can use this after generating your required matrices from the dataset.**

## Improving Long-Tail Recommendation in ALS using Item Normalized Interaction Matrix

Recommender systems are a crucial field of machine learning, used across many disciplines. Domain-free algorithms such as Alternating Least Squares (ALS) that are are especially crucial, as they can be applied to any scenario where one can construct an interaction matrix. One flaw of ALS is its tendency to recommend more popular items to users regardless of the quality of the interactions between two items, as models accumulate more loss on items with more interactions. This can cause items that are in reality more closely related to be less aligned in the recommendations from the model. In an effort to reduce this phenomenon, we demonstrate a simple and effective method for improving recommendation in the long tail with little overall performance impact.

More popular items tend to accrue many more interactions. For example, in the Spotify Million Playlist Dataset (MPD), the song added to the most playlists is Kendrick Lamar’s "Humble", which showed up on over 45 thousand playlists, reaching 4.5% presence on all playlists in the dataset. By comparison, 85% of the 2.2 million songs were on less than 10 playlists, with at most a .001% presence. This disparity leads to serious consequences in the recommendation of the long tail items, as the total error of the model is being contributed to at a much higher proportion by the most popular songs. This influences the weights in the factor matrices to be tuned so that they primarily minimize the error on those items.

One way to correct for this is to normalize all items by projecting them onto vectors of length 1. This is done on the sparse interaction matrix using the normalize function from sklearn.preprocessing. This significantly reduces the variance in the impact of each song onto the weighting system by changing the value of the interaction to be inversely proportional with the popularity of the song.

## Results

| Model          | Overall AUC | Short Head AUC | Mid Body AUC | Long Tail AUC |
|----------------|-------------|----------------|--------------|---------------|
| Popularity     | 95.03%      | 99.96%         | 99.61%       | 89.10%        |
| ALS            | 97.39%      | 99.27%         | 99.02%       | 93.32%        |
| Normalized ALS | 98.32%      | 99.43%         | 99.28%       | 96.13%        |

As we can see, both ALS models offer significantly higher performance than the popularity recommendation model. The normalized ALS scores better across the board, with especially large gains in the long tail. The normalized model reduced the remaining error by 35%, indicating that this method holds significant promise with respect to sparse datasets.

## Areas for Future Research

1. As the interaction values are significantly smaller when a vector in R1000000 is projected onto a unit sphere, a significantly higher learning rate is required to give appropriate weight to interactions compared to noninteractions. As the sparsity of the matrix is extremely high (99.997%), it is possible that the gridsearch of only scaling alpha up to 512 was insufficient, as is indicated by the fact that 512 was the highest performing model for both the normalized and non-normalized models, which was the highest value tested. 

2. Adding a popularity vector into the user and item factor update rule equations to more accurately be able to give every song the same weight in the loss function. 

3. Utilizing probabilistic models to determine the significance of each interaction with respect to the popularity of each song and the frequency of each interaction across the entire dataset.

4. Newer recommender systems that also utilize interaction matrices that could offer a better algorithm that is more easily modifiable than ALS to improve long tail recommendation.
