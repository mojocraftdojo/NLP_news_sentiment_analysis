## **Sentiment Analysis on Financial News Headlines(LLM/ML/DL NLP)**

## **Objective**

This analysis aims at predicting sentiments from financial news headlines using the latest large language models(transformers),  deep learning NLP and some traditional supervised machine learning modeling approaches. 

The large language model (LLM) will be compared side-by-side with all other models for their performance at the end of the analysis.

## **Dataset**

The dataset is composed of 4846 pieces of news headlines that have been given three sentiment ratings, which are "Positive"", "Negative" and "Neutral", see below break down:

![news](https://github.com/mojocraftdojo/NLP_news_sentiment_analysis/blob/main/news_samples.png "news")

![sentiment](https://github.com/mojocraftdojo/NLP_news_sentiment_analysis/blob/main/multi-classes_stats.png "multi-classes")

## **NLP Preprocessing**

### **Natural Language Process (NLP): Text Preprocessing using TensorFlow / Keras**

>#### Two important NLP preprocessing steps are conduced before training models:
----------
###  ---   **Tokenization:** Vectorize text, by turning each text into either a vector (or a sequence of integers).

In this analysis, I set below parameter values:

>>Limit the data set to the top 10000 words.

>>Set the max number of words = 2 * average_text_length.

----------
###  ---   **Embedding:** Transforms tokenized text into numbers so that Deep learning models can understand. 

 In this analysis, I used the TensorFlow/Keras framework to generate embedding layers, and also used pretrained embeddings from TensorFlow Hub



## **Modeling to Predict News Headlines sentiments (Multi-Labels)**


More specifically, we're addressing the NLP problem by building the follow Text Classification models:


> ### -- **Large Language Models (Transformers) (Hugging face)**
>> #### LLM model: Pre-trained transformers DistilBERT 


> ### -- **Classic Supervised learning models( Scikit-learn)**
>> ### Model 0: Naive Bayes 
>> ### Model 1: Random Forest
>> ### Model 2: XGBoost
>> ### Model 3: Support Vector Machine 
>> ### Model 4: Logistic Regression 


> ### -- **Deep Learning with NLP text preprocessing (TensorFlow/Keras)**
>> ### Model 5: RNNs (LSTM)
>> ### Model 6: TensorFlow Hub Pretrained Feature Extractor (Transfer Learning use USE)



## **example performance metrics**

#### here is an example of print out summary for model performance matrics

![performance_metrics](https://github.com/mojocraftdojo/NLP_news_sentiment_analysis/blob/main/performance_metrics.png "performance_metrics")


# **Insights:**

  Using the same dataset with sentiments associated to the news headlines (Multi-Labels: Positive/Negative/Neutral) (4000+ records), here is a side-by-side performance comparison of the below models based on F1 score :

 -- The obvious winner is the **pre-trained Transformers DistilBERT model** from **huggingface**. With a simple setup and a few epochs, it's the only model that achieved above F1 > 0.8,  which is **F1 = 0.84**.

 -- Majority of the rest of models can achieve a F1 score in 0.7 - 0.75 range out -of-the-box -- which can be potentially improved by tuning some of the hyper parameters.

-- In this analysis, a few classic ML methods such as **XGBoost, linear kernel SVC, random forest** robustly perform equally well with the  RNN-LSTM method.   

 -- **Tensorflow Hub Pretrained Universal Sentence Encoder** model performs somewhat better than **RNN-LSTM** model with the custom embeddings. This is also in line with a seperate analysis I perfored with more complicated datasets, which indicates the pretrained model with sentence level embedding performs overall better than models with custom embedding at word level for this type of text classification.However, its performance is lower than **Transformers**.



## **Top 1 model:** : F1 > 0.8

  --- Pre-trained LLM: DistilBERT Transformers (Huggingface)


### **Followed by the three models:** : F1 > 0.72

 --- XGboost  (scikit-learn)

 --- SVC(linear kernel)      (scikit-learn)

 --- TF-Hub Pretrained USE (TensorFlow Hub)


### **and then followed by these models**:  0.7 < F1 <= 0.72
 --- Random Forest (scikit-learn)

 --- LogisticRegression  (scikit-learn)

 --- RNN-LSTM  (TensorFlow/Keras)

### **Significant lower performance**: F1 < 0.6

 ---  Naive Bayes (scikit-learn)


![comparison](https://github.com/mojocraftdojo/NLP_news_sentiment_analysis/blob/main/Comparison_models.png "model-comparison")
![summary](https://github.com/mojocraftdojo/NLP_news_sentiment_analysis/blob/main/val_summary_.png "text-comparison")


