# Home Depot Product Search Relevance Kaggle competition <br>
Eugene Kashcheev

In this competition, I tried to achieve the best score in 80 hours.
The score is RMSE. Relevance was estimated by humans on the scale from 1 to 3.


# Steps:
## Baseline.ipynb <br>
Here I made a baseline model with a humble score around 0.493
Main ideas:
  + Use small amount of features from train.csv + product_product.description.csv
  + Few text processing before stemming
  + Stemmming for string attributes and counting occurrence ratio search_term in product title and description
  + Adding length of search_term and boolean feature for complete match of search term in title and description


## lstm.ipynb <br>
Actually, the whole idea of using DL was the waste of time.<br>
Here I tried to solve the task by using LSTM for coding word sequences of search term and product_title + description
I worked with LSTM, BI-LSTM, Merge, Dense and Dropout layers in Keras. Some models started to overfit,
then I tried to use regularizers and dropout layers.<br>
Actually, then score on validation and test datasets did not match, and the difference was bigger than with GBDT
The best score was beyond the baseline, around 0.52.<br>
One of the reason was another language model in search term and descriptions, so pre-trained embeddings I used did not work properly.
Moreover, there were not enough data to train my own embeddings.<br>
Finally, there was no sense to use complex NN models when "simple" trees provide better results.<br>

## stage2.ipynb 
There I used more sophisticated text preprocessing, to deal with dimensions, numbers and some typos.<br>
I chose LGBMRegressor, the sklearn-way wrapped gradient boosting decision tree model of LightGBM library.  <br>
Also, I made several experiments with adding new features and adjusting LGBM hyperparameters.<br>
  + I added a length of a search term in number letters and words.
  + I added brand attribute and occurrence ratio of search term words in brand
  + I also added a feature for matching word including matching their Part-Of-Speech tag.<br>

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
  * Variance between first positions of occurred words
  * Difference between maximal and minimal positions of matched words
  
This stage I finished with the best score 0.47476<br>

## stage3.ipynb
There I added different features and tuned hyperparameters.<br>
Features:<br>
  + words_std_title, word_std_descr variance of positions of first occurences words of search_term in product_title and description. Feature return 0 for one occurence and provide bad results for search_term with one word. Fixed in stage8
  + len_title, len_descr words count in description and product_title
  + last_word_title / last_word_descr (0 or 1) if the last word of search_term occures in title/descr (improved the score to 0.47395)
  + bad_pred - a wrong feature to add. I train a classifier for predicting outliers of regressoin prediction. the score fell to 0.48... <br>
I ended up this stage with the following parameters: colsample_bytree=0.5, learning_rate=0.03, max_depth=-1, n_estimators=700, num_leaves=80<br>

## stage4.ipynb
Update text cleaning procedures, added rules for fixing some typos, recalc all the features.<br>
Result 0.47135<br>

## stage5.ipynb
Install fuzzywuzzy and added next related features:
  + fuzzy_title, fuzzy_descr - fuzzy matching of searh_term string with title and description
  + fuzzy_orig_title, fuzzy_orig_descr - the same thing for original strings<br>
Result 0.46959 <br>
Then I replaced token_set_ratio matching to partial_token_set_ratio to get a matching of the closest words, insted of the whole text. It had not worked and the score dropped to 0.47003.<br>
I reloaded the files and add partial matching for the same fields as the distinct fields. And they improved the score to 0.46886<br>
  + match_any_brand - does the word in search_term match any of the brand (did not work)
  + first_word_title, first_word_descr - does the first word of the search_term match anything in title or description (improved the score a little bit)
  + match_numbers_title, match_numbers_descr - does the number from search term match anything in title or description (the first field worked, the second was worthless)
  + has_number_search_term, has_number_title, has_number_descr - does the search_term, product_title or description a number (worthless feature)
  + complete_match_title, complete_match_descr - tried to add these features one more time (no imporving)<br>
The stage ended with the score = 0.46747. Only ~0.001 to descirable bronze.<br>

## stage6.ipynb
Tried to add different fields, but with no success or improvement. 
  + match_other_attrs - partial fuzzy match of search_term and all other attributes from attributes.csv
  + spacy_similarity - to strong, but dirty feature, dropped the score to ~0.48<br>
  
## stage7.ipynb
Yet another feature engineering:
  + fuzzy_brand - fuzzy matching search_term with the brand. pointless
  + then long gridsearch for better hyperparameters. won there ~0.0001
  + color, bullets, material - values for the product attributes from attributes.csv 
  + match_color, match_bullets, match_material - the ration of matching words from search_term with the corresponding feature. worked only for match_bullets
  + is_kit, look_kit - does search_term contains the word kit, and does title contains the word kit. (worked fine)
  + typo_search_term - is there an unknown word in search_term<br>
Finished with the 0.46747<br>

## stage8.ipynb
Improved rules for text processing. Exclude all words waht is not digit and have only one letter.<br>
Implemented several replcements for typos. Recalculate all other features with the new stemmed features.<br>
Changed is_kit/look_kit to match any word from the list part|case|cover|tool|kit.<br>
Finished with the score about 0.4665<br>

## stage9.ipynb
Tuned hyperparametres. Lowered learning rate, added estimators, increased max_bin. THe final restuls is 0.46631. Bronze!!!<br>


## Conclusion:
In this competition, I've learned different tecniques of feature mining, adding them as a parametres in GBDT and model tunning. There is a space to improvement: fixing typos, finding deeper and more sophisticated features.<br>

### Thank you!
