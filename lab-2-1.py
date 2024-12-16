#!/usr/bin/env python
# coding: utf-8

# Version: 02.14.2023

# # Lab 2.1: Applying ML to an NLP Problem
# 
# In this lab, you will use the built-in machine learning (ML) model in Amazon SageMaker, __LinearLearner__, to predict the __isPositive__ field of the review dataset.
# 
# ## Introducing the business scenario
# You work for an online retail store that wants to improve customer engagement for customers who have posted negative reviews. The company wants to detect negative reviews and assign these reviews to a customer service agent to address.
# 
# You are tasked with solving part of this problem by using ML to detect negative reviews. You were given access to a dataset that contains reviews, which have been classified as positive or negative. You will use this dataset to train an ML model to predict the sentiment of new reviews.
# 
# ## About this dataset
# The [AMAZON-REVIEW-DATA-CLASSIFICATION.csv](https://github.com/aws-samples/aws-machine-learning-university-accelerated-nlp/tree/master/data/examples) file contains actual reviews of products, and these reviews include both text data and numeric data. Each review is labeled as _positive (1)_ or _negative (0)_.
# 
# The dataset contains the following features:
# * __reviewText:__ Text of the review
# * __summary:__ Summary of the review
# * __verified:__ Whether the purchase was verified (True or False)
# * __time:__ UNIX timestamp for the review
# * __log_votes:__ Logarithm-adjusted votes log(1+votes)
# * __isPositive:__ Whether the review is positive or negative (1 or 0)
# 
# The dataset for this lab is being provided to you by permission of Amazon and is subject to the terms of the Amazon License and Access (available at https://www.amazon.com/gp/help/customer/display.html?nodeId=201909000). You are expressly prohibited from copying, modifying, selling, exporting, or using this dataset in any way other than for the purpose of completing this course.
# 
# ## Lab steps
# 
# To complete this lab, you will follow these steps:
# 
# 1. [Reading the dataset](#1.-Reading-the-dataset)
# 2. [Performing exploratory data analysis](#2.-Performing-exploratory-data-analysis)
# 3. [Text processing: Removing stopwords and stemming](#3.-Text-processing:-Removing-stopwords-and-stemming)
# 4. [Splitting training, validation, and test data](#4.-Splitting-training,-validation,-and-test-data)
# 5. [Processing data with pipelines and a ColumnTransformer](#5.-Processing-data-with-pipelines-and-a-ColumnTransformer)
# 6. [Training a classifier with a built-in SageMaker algorithm](#6.-Training-a-classifier-with-a-built-in-SageMaker-algorithm)
# 7. [Evaluating the model](#7.-Evaluating-the-model)
# 8. [Deploying the model to an endpoint](#8.-Deploying-the-model-to-an-endpoint)
# 9. [Testing the endpoint](#9.-Testing-the-endpoint)
# 10. [Cleaning up model artifacts](#10.-Cleaning-up-model-artifacts)
#     
# ## Submitting your work
# 
# 1. In the lab console, choose **Submit** to record your progress and when prompted, choose **Yes**.
# 
# 1. If the results don't display after a couple of minutes, return to the top of these instructions and choose **Grades**.
# 
#      **Tip**: You can submit your work multiple times. After you change your work, choose **Submit** again. Your last submission is what will be recorded for this lab.
# 
# 1. To find detailed feedback on your work, choose **Details** followed by **View Submission Report**.  

# Start by installing/upgrading pip, sagemaker, and scikit-learn.
# 
# [scikit-learn](https://scikit-learn.org/stable/) is an open source machine learning library. It provides various tools for model fitting, data preprocessing, model selection and evaluation and many other utilities.

# In[1]:


#Upgrade dependencies
get_ipython().system('pip install --upgrade pip')
get_ipython().system('pip install --upgrade scikit-learn')
get_ipython().system('pip install --upgrade sagemaker')
get_ipython().system('pip install --upgrade botocore')
get_ipython().system('pip install --upgrade awscli')


# ## 1. Reading the dataset
# ([Go to top](#Lab-2.1:-Applying-ML-to-an-NLP-Problem))
# 
# You will use the __pandas__ library to read the dataset. [Pandas](https://pandas.pydata.org/pandas-docs/stable/index.html) is a popular python library that is used for data analysis. It provides data manipulation, cleaning, and data wrangling features as well as visualizations.

# In[2]:


import pandas as pd

df = pd.read_csv('../data/AMAZON-REVIEW-DATA-CLASSIFICATION.csv')

print('The shape of the dataset is:', df.shape)


# Look at the first five rows in the dataset.

# In[3]:


df.head(5)


# You can change the options in the notebook to display more of the text data.

# In[4]:


pd.options.display.max_rows
pd.set_option('display.max_colwidth', None)
df.head()


# You can look at specific entries if needed.

# In[5]:


print(df.loc[[580]])


# It's good to know what data types you are dealing with. You can use `dtypes` on the dataframe to display the types.

# In[6]:


df.dtypes


# ## 2. Performing exploratory data analysis
# ([Go to top](#Lab-2.1:-Applying-ML-to-an-NLP-Problem))
# 
# You will now look at the target distribution for your dataset.

# In[7]:


df['isPositive'].value_counts()


# The business problem is concerned with finding the negative reviews (_0_). However, the model tuning for linear learner defaults to finding positive values (_1_). You can make this process run more smoothly by switching the negative values (_0_) and positive values (_1_). By doing so, you can tune the model more easily.

# In[8]:


df = df.replace({0:1, 1:0})
df['isPositive'].value_counts()


# Check the number of missing values:

# In[9]:


df.isna().sum()


# The text fields have missing values. Typically, you would decide what to do with these missing values. You could remove the data or fill it with some standard text. 

# ## 3. Text processing: Removing stopwords and stemming
# ([Go to top](#Lab-2.1:-Applying-ML-to-an-NLP-Problem))
# 
# In this task, you will remove some of the stopwords, and perform stemming on the text data. You are normalizing the data to reduce the amount of different information you have to deal with.
# 
# [nltk](https://www.nltk.org/) is a popular platform for working with human language data. It provides interfaces and functions for processing text for classification, tokenization, stemming, tagging, parsin, and semantic reasoning. 
# 
# Once imported, you can download only the functionality you need. In this example, you will use:
# 
# - **punkt** is a sentence tokenizer
# - **stopwords** provides a list of stopwords you can use.

# In[22]:


# Install the library and functions
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')


# You will create the processes for removing stopwords and cleaning the text in the following section. The Natural Language Toolkit (NLTK) library provides a list of common stopwords. You will use the list, but you will first remove some of the words from that list. The stopwords that you keep in the text are useful for determining sentiment.

# In[23]:


import nltk, re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

# Get a list of stopwords from the NLTK library
stop = stopwords.words('english')

# These words are important for your problem. You don't want to remove them.
excluding = ['against', 'not', 'don', 'don\'t','ain', 'are', 'aren\'t', 'could', 'couldn\'t',
             'did', 'didn\'t', 'does', 'doesn\'t', 'had', 'hadn\'t', 'has', 'hasn\'t', 
             'have', 'haven\'t', 'is', 'isn\'t', 'might', 'mightn\'t', 'must', 'mustn\'t',
             'need', 'needn\'t','should', 'shouldn\'t', 'was', 'wasn\'t', 'were', 
             'weren\'t', 'won\'t', 'would', 'wouldn\'t']

# New stopword list
stopwords = [word for word in stop if word not in excluding]




# The snowball stemmer will stem words. For example, 'walking' will be stemmed to 'walk'.

# In[24]:


snow = SnowballStemmer('english')


# You must perform a few other normalization tasks on the data. The following function will:
# 
# - Replace any missing values with an empty string
# - Convert the text to lowercase
# - Remove any leading or training whitespace
# - Remove any extra space and tabs
# - Remove any HTML markup
# 
# In the `for` loop, any words that are __NOT__ numeric, longer than 2 characters, and not part of the list of stop words will be kept and returned.

# In[25]:


def process_text(texts): 
    final_text_list=[]
    for sent in texts:
        
        # Check if the sentence is a missing value
        if isinstance(sent, str) == False:
            sent = ''
            
        filtered_sentence=[]
        
        sent = sent.lower() # Lowercase 
        sent = sent.strip() # Remove leading/trailing whitespace
        sent = re.sub('\s+', ' ', sent) # Remove extra space and tabs
        sent = re.compile('<.*?>').sub('', sent) # Remove HTML tags/markups:
        
        for w in word_tokenize(sent):
            # Applying some custom filtering here, feel free to try different things
            # Check if it is not numeric and its length>2 and not in stopwords
            if(not w.isnumeric()) and (len(w)>2) and (w not in stopwords):  
                # Stem and add to filtered list
                filtered_sentence.append(snow.stem(w))
        final_string = " ".join(filtered_sentence) # Final string of cleaned words
 
        final_text_list.append(final_string)
        
    return final_text_list


# ## 4. Splitting training, validation, and test data
# ([Go to top](#Lab-2.1:-Applying-ML-to-an-NLP-Problem))
# 
# In this step, you will split the dataset into training (80 percent), validation (10 percent), and test (10 percent) by using the sklearn [__train_test_split()__](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) function.
# 
# The training data will be used to train the model which is then tested with the test data. The validation set is used once the model has been trained to give you metrics on how the model might perform on real data. 

# In[26]:


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(df[['reviewText', 'summary', 'time', 'log_votes']],
                                                  df['isPositive'],
                                                  test_size=0.20,
                                                  shuffle=True,
                                                  random_state=324
                                                 )

X_val, X_test, y_val, y_test = train_test_split(X_val,
                                                y_val,
                                                test_size=0.5,
                                                shuffle=True,
                                                random_state=324)


# With the dataset split, you can now run the `process_text` function defined above on each of the text features in the training, test, and validation sets.

# In[27]:


print('Processing the reviewText fields')
X_train['reviewText'] = process_text(X_train['reviewText'].tolist())
X_val['reviewText'] = process_text(X_val['reviewText'].tolist())
X_test['reviewText'] = process_text(X_test['reviewText'].tolist())

print('Processing the summary fields')
X_train['summary'] = process_text(X_train['summary'].tolist())
X_val['summary'] = process_text(X_val['summary'].tolist())
X_test['summary'] = process_text(X_test['summary'].tolist())


# ## 5. Processing data with pipelines and a ColumnTransformer
# ([Go to top](#Lab-2.1:-Applying-ML-to-an-NLP-Problem))
# 
# You will often perform many tasks on data before you use it to train a model. These steps must also be done on any data that's used for inference after the model is deployed. A good way of organizing these steps is to define a _pipeline_. A pipeline is a collection of processing tasks that will be performed on the data. Different pipelines can be created to process different fields. Because you are working with both text and numeric data, you can define the following pipelines:
# 
#    * For the numerical features pipeline, the __numerical_processor__ uses a MinMaxScaler. (You don't need to scale features when you use decision trees, but it's a good idea to see how to use more data transforms.) If you want to perform different types of processing on different numerical features, you should build different pipelines, like the ones that are shown for the two text features.
#    * For the text features pipeline, the __text_processor__ uses `CountVectorizer()` for the text fields.
#    
# The selective preparations of the dataset features are then put together into a collective ColumnTransformer, which will be used with in a pipeline along with an estimator. This process ensures that the transforms are performed automatically on the raw data when you fit the model or make predictions. (For example, when you evaluate the model on a validation dataset via cross-validation, or when you make predictions on a test dataset in the future.)

# In[29]:


# Grab model features/inputs and target/output
numerical_features = ['time',
                      'log_votes']

text_features = ['summary',
                 'reviewText']

model_features = numerical_features + text_features
#print(model_features)
model_target = 'isPositive'


# In[30]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

### COLUMN_TRANSFORMER ###
##########################

# Preprocess the numerical features
numerical_processor = Pipeline([
    ('num_imputer', SimpleImputer(strategy='mean')),
    ('num_scaler', MinMaxScaler()) 
                                ])
# Preprocess 1st text feature
text_processor_0 = Pipeline([
    ('text_vect_0', CountVectorizer(binary=True, max_features=50))
                                ])

# Preprocess 2nd text feature (larger vocabulary)
text_precessor_1 = Pipeline([
    ('text_vect_1', CountVectorizer(binary=True, max_features=150))
                                ])

# Combine all data preprocessors from above (add more, if you choose to define more!)
# For each processor/step specify: a name, the actual process, and finally the features to be processed
data_preprocessor = ColumnTransformer([
    ('numerical_pre', numerical_processor, numerical_features),
    ('text_pre_0', text_processor_0, text_features[0]),
    ('text_pre_1', text_precessor_1, text_features[1])
                                    ]) 

### DATA PREPROCESSING ###
##########################

print('Datasets shapes before processing: ', X_train.shape, X_val.shape, X_test.shape)

X_train = data_preprocessor.fit_transform(X_train).toarray()
X_val = data_preprocessor.transform(X_val).toarray()
X_test = data_preprocessor.transform(X_test).toarray()

print('Datasets shapes after processing: ', X_train.shape, X_val.shape, X_test.shape)


# Note how the number of features in the datasets went from 4 to 202.

# In[31]:


print(X_train[0])


# ## 6. Training a classifier with a built-in SageMaker algorithm
# ([Go to top](#Lab-2.1:-Applying-ML-to-an-NLP-Problem))
# 
# In this step, you will call the Sagemaker `LinearLearner()` algorithm with the following options:
# * __Permissions -__ `role` is set to the AWS Identity and Access Management (IAM) role from the current environment.
# * __Compute power -__ You will use the `train_instance_count` parameter and the `train_instance_type` parameter. This example uses an `ml.m4.xlarge` resource for training. You can change the instance type depending on your needs. (For example, you could use GPUs for neural networks.) 
# * __Model type -__ `predictor_type` is set to __`binary_classifier`__ because you are working with a binary classification problem. You could use __`multiclass_classifier`__ if three or more classes are involved, or you could use __`regressor`__ for a regression problem.
# 

# In[32]:


import sagemaker

# Call the LinearLearner estimator object
linear_classifier = sagemaker.LinearLearner(role=sagemaker.get_execution_role(),
                                           instance_count=1,
                                           instance_type='ml.m4.xlarge',
                                           predictor_type='binary_classifier')


# To set the training, validation, and test parts of the estimator, you can use the `record_set()` function of the `binary_estimator`. 

# In[33]:


train_records = linear_classifier.record_set(X_train.astype('float32'),
                                            y_train.values.astype('float32'),
                                            channel='train')
val_records = linear_classifier.record_set(X_val.astype('float32'),
                                          y_val.values.astype('float32'),
                                          channel='validation')
test_records = linear_classifier.record_set(X_test.astype('float32'),
                                           y_test.values.astype('float32'),
                                           channel='test')


# The `fit()` function applies a distributed version of the Stochastic Gradient Descent (SGD) algorithm, and you are sending the data to it. The logs were disabled with `logs=False`. You can remove that parameter to see more details about the process. __This process takes about 3-4 minutes on an ml.m4.xlarge instance.__

# In[34]:


linear_classifier.fit([train_records,
                       val_records,
                       test_records],
                       logs=False)


# ## 7. Evaluating the model
# ([Go to top](#Lab-2.1:-Applying-ML-to-an-NLP-Problem))
# 
# You can use SageMaker analytics to get some performance metrics (of your choosing) on the test set. This process doesn't require you to deploy the model. 
# 
# Linear learner provides metrics that are computed during training. You can use these metrics when tuning the model. The available metrics for the validation set are:
# 
# - objective_loss - For a binary classification problem, this will be the mean value of the logistic loss for each epoch
# - binary_classification_accuracy - The accuracy of the final model on the dataset i.e. how many predictions did the model get right
# - precision - Quantifies the number of positive class predictions that are actually positive
# - recall - Quantifies the number of positive class predictions
# - binary_f_beta - The harmonic mean of the precision and recall metrics
# 
# For this example, you are interested in how many predictions were correct. Using the **binary_classification_accuracy** metric seems appropriate.

# In[35]:


sagemaker.analytics.TrainingJobAnalytics(linear_classifier._current_job_name, 
                                         metric_names = ['test:binary_classification_accuracy']
                                        ).dataframe()


# You should see a value of around 0.85. Your value will may be different, but should be around that value. This translates to the model accuractely predicting the correct answer 85% of the time. Depending upon the business case, you may need to tune the model further using a hyperparameter tuning job, or do some more feature engineering.

# ## 8. Deploying the model to an endpoint
# ([Go to top](#Lab-2.1:-Applying-ML-to-an-NLP-Problem))
# 
# In this last part of this exercise, you will deploy your model to another instance of your choice. You can use this model in a production environment. Deployed endpoints can be used with other AWS services, such as AWS Lambda and Amazon API Gateway. If you are interested in learning more, see the following walkthrough: [Call an Amazon SageMaker model endpoint using Amazon API Gateway and AWS Lambda](https://aws.amazon.com/blogs/machine-learning/call-an-amazon-sagemaker-model-endpoint-using-amazon-api-gateway-and-aws-lambda/).
# 
# To deploy the model, run the following cell. You can use different instance types, such as: _ml.t2.medium_, _ml.c4.xlarge_), and others. __This process will take some time to complete (approximately 7-8 minutes).__

# In[36]:


linear_classifier_predictor = linear_classifier.deploy(initial_instance_count = 1,
                                                       instance_type = 'ml.c5.large'
                                                      )


# ## 9. Testing the endpoint
# ([Go to top](#Lab-2.1:-Applying-ML-to-an-NLP-Problem))
# 
# Now that the endpoint is deployed, you will send the test data to it and get predictions from the data.

# In[37]:


import numpy as np

# Get test data in batch size of 25 and make predictions.
prediction_batches = [linear_classifier_predictor.predict(batch)
                      for batch in np.array_split(X_test.astype('float32'), 25)
                     ]

# Get a list of predictions
print([pred.label['score'].float32_tensor.values[0] for pred in prediction_batches[0]])


# ## 10. Cleaning up model artifacts
# ([Go to top](#Lab-2.1:-Applying-ML-to-an-NLP-Problem))
# 
# You can run the following to delete the endpoint after you are done using it. 
# 
# **Tip:** - Remember that when using your own account, you will accrue charges if you don't delete the endpoint and other resources.

# In[38]:


linear_classifier_predictor.delete_endpoint()


# # Congratulations!
# 
# In this lab, you looked at a very simple NLP problem. Using a labelled dataset, you used a simple tokenizer and encoder to generate the data required to train a linear learner model. You then deployed the model and performed some predictions. If you were doing this for real, you would likely need to obtain the data and label it for training. An alternative might be to use a pretrained algorithm or managed service. You would also likely tune the model further using a hyperparameter tuning job.
# 
# You have completed this lab, and you can now end the lab by following the lab guide instructions.

# ©2023 Amazon Web Services, Inc. or its affiliates. All rights reserved. This work may not be reproduced or redistributed, in whole or in part, without prior written permission from Amazon Web Services, Inc. Commercial copying, lending, or selling is prohibited. All trademarks are the property of their owners.

# In[ ]:




