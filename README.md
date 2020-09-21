# ALS-Research
The summer work of Gabe Pesco and Douglass Turnbull, working on mimizing popularity bias in the ALS recommender system.

Improving Long-Tail Recommendation in ALS using Item Normalized Interaction Matrix

Recommender systems are a crucial field of machine learning, used across many disciplines. Algorithms such as Alternating Least Squares (ALS) that are domain-free are especially important, as they can be applied to any scenario where one can construct an interaction matrix. One flaw of ALS is its tendency to recommend more popular items to users, as models tend to accumulate more loss on items with more interactions. This can cause items that are more closely aligned in reality to be less aligned with respect to the recommendations from the model. In an effort to reduce this phenomenon, we demonstrate a simple and effective method for improving recommendation in the long tail with little overall performance impact.

More popular items tend to accrue many more interactions. For example, in the Spotify Million Playlist Dataset (MPD), the song added to the most playlists is Kendrick Lamar’s Humble, which showed up on over 45 thousand playlists, reaching 4.5% presence on all playlists in the dataset. By comparison, 85% of the 2.2 million songs were on less than 10 playlists, with at most a .001% presence. This disparity leads to serious consequences in the recommendation of the long tail items, as the total error of the model is being contributed to at a much higher proportion by the most popular songs. This influences the weights in the factor matrices to be tuned so that they primarily minimize the error on those items.

A simple workaround for this is to normalize all items by projecting them onto vectors of length 1. This is done on the sparse interaction matrix using the normalize function from sklearn.preprocessing. This significantly reduces the variance in the impact of each song onto the weighting system by changing the value of the interaction to be inversely related with the popularity of the song. To test, we validated this on a smaller portion of the dataset that excluded any song that was on less than 10 playlists.

| Model          | Overall AUC | Short Head AUC | Mid Body AUC | Long Tail AUC |
|----------------|-------------|----------------|--------------|---------------|
| Popularity     | 95.03%      | 99.96%         | 99.61%       | 89.10%        |
| ALS            | 97.39%      | 99.27%         | 99.02%       | 93.32%        |
| Normalized ALS | 98.32%      | 99.43%         | 99.28%       | 96.13%        |

As we can see, both ALS models offer significantly higher performance than the popularity recommendation model. Unfortunately, the differences in the models are all extremely small (>.5% for any segment), so it is not clear from the cutdown dataset if the normalization step is improving the accuracy of the recommender system. One possible source of error in this is a fundamental difference in how different hyperparameters, especially alpha, affect the system. Since the interaction values are significantly smaller when a vector in R1000000 is projected onto a unit sphere, a significantly higher learning rate is required to give appropriate weight to interactions compared to noninteractions. As the sparsity of the matrix is extremely high (99.997%), it is possible that the gridsearch of only scaling alpha up to 512 was insufficient, as is indicated by the fact that 512 was the highest performing model for both the normalized and non-normalized models, which was the highest value tested. 

These results are not extremely promising. Future work on improving long tail recommendation in ALS would likely come from adding a popularity vector into the user and item factor update rule equations to more accurately be able to give every song the same weight. Additionally, probabilistic models could be used to determine the significance of each interaction by its inverse relationship with popularity. Alternatively, there could be newer recommender systems that also utilize interaction matrices that could offer a better algorithm that is more easily modifiable than ALS to improve long tail recommendation. Lastly, the testing methodology might be flawed, with inadequate sample sizes or poor choices for how to divide up sections of the data making the results appear less accurate. Consider that splitting the dataset into density leaves roughly two thousand songs in the short head, twenty thousand songs in the mid body, and over three hundred thousand songs in the long tail. The variance introduced in having to perform AUC on what is on average 13 songs per playlist, split into 3 sections, means that we have roughly 4 songs in the long tail per playlist. Another test is being conducted with the full dataset to see if including songs with less than 10 plays more adequately shows the significance of the improvement in long tail recommendation, as this would increase the size of the long tail by approximately two million.

Upon running the test with the full dataset, it is clear how much more effective it is at true long tail recommendation.

| Model          | Overall AUC | Short Head AUC | Mid Body AUC | Long Tail AUC |
|----------------|-------------|----------------|--------------|---------------|
| Popularity     | 95.03%      | 99.96%         | 99.61%       | 89.10%        |
| ALS            | 97.39%      | 99.27%         | 99.02%       | 93.32%        |
| Normalized ALS | 98.32%      | 99.43%         | 99.28%       | 96.13%        |

These results demonstrate more accurately how the normalized model improves with respect to long tail recommendation, indicating that this method holds significant promise with respect to extremely sparse datasets.
