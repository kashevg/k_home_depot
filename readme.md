# Home Depot Product Search Relevance Kaggle competition<br>
Eugene Kashcheev

In this competition, I tried to achieve the best score in 80 hours.
The score is RMSE. Relevance was estimated by humans on the scale from 1 to 3.


Steps:
1. Baseline.ipynb<br>
Here I made a baseline model with a humble score around 0.493
Main ideas:
	- Use small amount of features from train.csv + product_product.description.csv
	- Few text processing before stemming
	- Stemmming for string attributes and counting occurrence ratio search_term in product title and description
	- Adding length of search_term and boolean feature for complete match of search term in title and description


2. lstm.ipynb <br>
Actually, the whole idea of using DL was the waste of time.<br>
Here I tried to solve the task by using LSTM for coding word sequences of search term and product_title + description
I worked with LSTM, BI-LSTM, Merge, Dense and Dropout layers in Keras. Some models started to overfit,
then I tried to use regularizers and dropout layers.<br>
Actually, then score on validation and test datasets did not match, and the difference was bigger than with GBDT
The best score was beyond the baseline, around 0.52.<br>
One of the reason was another language model in search term and descriptions, so pre-trained embeddings I used did not work properly.
Moreover, there were not enough data to train my own embeddings.<br>
Finally, there was no sense to use complex NN models when "simple" trees provide better results.<br>

3. stage2.ipynb
There I used more sophisticated text preprocessing, to deal with dimensions, numbers and some typos.<br>
I chose LGBMRegressor, the sklearn-way wrapped gradient boosting decision tree model of LightGBM library.  <br>
Also, I made several experiments with adding new features and adjusting LGBM hyperparameters.<br>
	- I added a length of a search term in number letters and words.
	- I added brand attribute and occurrence ratio of search term words in brand
	- I also added a feature for matching word including matching their Part-Of-Speech tag.
After that I added TfIdfVectorizer for an every stemmed text column and TruncatedSVD to lower the output dimensionality.<br>
I added this features and increase number of estimator in GB Regressor.<br>
That boosted my score to 0.481<br>
Then I continued to play with other features:<br>
	Minimal index of matching word divided by the length of field (description/product_title) (0.479)
After these steps, I checked feature importance and drop "complete_match_title" as it was no longer meaningful.<br>
I also enlarged output dimension of TruncatedSVD for search_term, description and product_title,
while their feature_importance_ stayed high and by adding them I improved the score.<br>

I also added a feature to describe the difference between the position of matched words in product text attributes.<br>
There were two features:<br>
	- Variance between first positions of occurred words
	- Difference between maximal and minimal positions of matched words
This stage I finished with the best score 0.47476<br>

