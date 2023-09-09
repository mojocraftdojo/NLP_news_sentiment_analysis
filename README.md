# **(NLP) Sentiment Analysis on Financial News Headlines**

## **Objective**

This analysis aims at predicting sentiments from financial news headlines from the perspective of retail investors using various deep learning NLP and traditional supervised machine learning modeling techniques. 


## **Dataset**

The dataset composed of 4846 pieces of news headlines that has been given three sentiment ratings, which are "Positive"", "Negative" and "Neutral", see below break down:

![news](https://github.com/mojocraftdojo/NLP_news_sentiment_analysis/blob/main/news_samples.png "news")

![sentiment](https://github.com/mojocraftdojo/NLP_news_sentiment_analysis/blob/main/multi-classes_stats.png "multi-classes")

## **Methodologies for Modeling**

### **Natural Language Process (NLP): Text Preprocessing using TensorFlow/Keras**

### Two important NLP preprocessing steps are conduced before training models:

----------
###  ---   **Tokenization:** Vectorize text, by turning each text into either a sequence of integers or into a vector.
#### In this analysis, I set below parameter values:
>>Limit the data set to the top 10000 words.

>>Set the max number of words = 2 * average_text_length.

----------
###  ---   **Embedding:** Transforms tokenized text into numbers so that Deep learning models can understand. In this analysis, I either leveraged TensorFlow/Keras framework to generate embedding layer, all utilized pretrained embeddings from TensorFlow Hub, all with just a few lines of code


### Compare Deep Learning vs Traditional Supervised Learning Machine Learning approach

####  A) Multi-classes Classification ML models: through **scikit-learn** library and **NLP packages
####  B) Deep Learning: through **TensorFlow and Keras frameworks


More specifically, we're addressing the NLP problem by building the follow Text Classification models:

>### **Classic Supervised learning models( Scikit-learn)**
>> #### Model 0: Naive Bayes (baseline)
>> #### Model 1: Random Forest
>> #### Model 2: XGBoost
>> #### Model 3: Support Vector Machine 
>> #### Model 4: Logistic Regression 

>### **Deep Learning with NLP text preprocessing (TensorFlow/Keras)**
>>#### Model 5: RNNs (LSTM)
>>#### Model 6: TensorFlow Hub Pretrained Feature Extractor (Transfer Learning use USE)


## **Results Highlight**

#### here is an example of print out summary for model performance matrics

![performance_metrics](https://github.com/mojocraftdojo/NLP_news_sentiment_analysis/blob/main/performance_metrics.png "performance_metrics")

#### When comparing different models, based on F1 score:
>#### -- Majority of the models can achieve a F1 score in 0.7 - 0.75 range on this small dataset (4000+ news) out of box -- which is not bad but can be bettter, either through supplying more training data or tuning some of the hyper parameters. 
>#### -- In this analysis, a few classic ML methods such as XGBoost, linear kernel SVC, random forest robustly perform equally well with the RNN-LSTM method. 
>#### -- Tensorflow Hub Pretrained USE model performs somewhat better than RNN-LSTM model here. This is also in line with a seperate analysis I perfored with more complicated datasets, which indicates the pretrained model with sentence level embedding performs overall better than models with custom embedding at word level for this type of text classification.

>### **Top 3 models:** : F1 >= 0.72
>> --- XGboost  (scikit-learn)

>> --- SVC(linear kernel)      (scikit-learn)

>> --- TF-Hub Pretrained USE (TensorFlow Hub)


>### **followed by**:  0.7 <= F1 < 0.72
>> --- Random Forest (scikit-learn)

>> --- LogisticRegression  (scikit-learn)

>> --- RNN-LSTM  (TensorFlow/Keras)

>### **Significant lower performance**: F1 < 0.6

>> ---  Naive Bayes (scikit-learn)


![comparison](https://github.com/mojocraftdojo/NLP_news_sentiment_analysis/blob/main/Comparison_models.png "model-comparison")


