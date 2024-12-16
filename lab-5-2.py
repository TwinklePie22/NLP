#!/usr/bin/env python
# coding: utf-8

# Version: 02.14.2023

# # Lab 5.2: Working with Entities
# 
# In this lab, you will use Amazon Comprehend to extract key phrases and entities from the [CMU Movie Summary Corpus](http://www.cs.cmu.edu/~ark/personas/). You will then feed the results into an OpenSearch cluster and build a Kibana dashboard to view and filter the results. You will also look at building word clouds and using ngrams to extract key phrases.
# 
# ## About the dataset
# 
# The CMU Movie Summary Corpus is a collection of 42,306 movie plot summaries and metadata at both the movie level (including box office revenue, genre, and date of release) and character level (including gender and estimated age).  This data supports work in the following paper:
# 
# David Bamman, Brendan O'Connor, and Noah Smith. "Learning Latent Personas of Film Characters." Presented at the Annual Meeting of the Association for Computational Linguistics (ACL 2013), Sofia, Bulgaria, August 2013. http://www.cs.cmu.edu/~dbamman/pubs/pdf/bamman+oconnor+smith.acl13.pdf.
# 
# You will use two datasets in this lab:
# 
# **plot_summaries.txt**
# 
# This dataset contains plot summaries of 42,306 movies, extracted from the November 2, 2012 dump of English-language Wikipedia. Each line contains the Wikipedia movie ID (which indexes into movie.metadata.tsv) followed by the summary.
# 
# **movie.metadata.tsv**
# 
# This dataset contains metadata for 81,741 movies, extracted from the November 4, 2012 dump of Freebase. The data is tab-separated and contains the following columns:
# 
# 1. Wikipedia movie ID
# 2. Freebase movie ID
# 3. Movie name
# 4. Movie release date
# 5. Movie box office revenue
# 6. Movie runtime
# 7. Movie languages (Freebase ID:name tuples)
# 8. Movie countries (Freebase ID:name tuples)
# 9. Movie genres (Freebase ID:name tuples)

# ## Lab steps
# 
# To complete this lab, you will follow these steps:
# 
# 1. [Installing the packages](#1.-Installing-the-packages)
# 2. [Reviewing the datasets](#2.-Reviewing-the-datasets)
# 3. [Normalizing the text](#3.-Normalizing-the-text)
# 4. [Starting the Amazon Comprehend jobs](#4.-Starting-the-Amazon-Comprehend-jobs)
# 5. [Creating the OpenSearch cluster](#5.-Creating-the-OpenSearch-cluster)
# 6. [Using word clouds and ngrams](#6.-Using-word-clouds-and-ngrams)
# 7. [Returning to Amazon Comprehend](#7.-Returning-to-Amazon-Comprehend)
# 8. [Creating the Kibana dashboard](#8.-Creating-the-Kibana-dashboard)
# 9. [Cleaning up](#9.-Cleaning-up)
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

# ## 1. Installing the packages
# ([Go to top](#Lab-5.2:-Working-with-Entities))
# 
# First, update and install the packages that you will use in the notebook. 
# 
# - **opensearch** - provides a package for interacting with the OpenSearch cluster
# - **opensearch-py** - provides a OpenSearch python client
# - **requests** - provides a package for making HTTP(S) requests
# - **aws4auth** - provides a simple wrapper for signing requests for AWS

# In[1]:


get_ipython().system('pip install --upgrade pip')
get_ipython().system('pip install opensearch')
get_ipython().system('pip install opensearch-py')
get_ipython().system('pip install requests')
get_ipython().system('pip install requests-aws4auth')


# Next, import the packages. You have used most of these packages in previous labs.

# In[2]:


import boto3
import os, io, struct, json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import uuid
from time import sleep
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer


# In the lab, you will upload files to Amazon Simple Storage Service (Amazon S3) to be processed, and you will download the results. The following bucket information should already be set for you.

# In[3]:


bucket = "c137242a3503185l8670479t1w177172075141-labbucket-901ffpjxjyoe"
prefix='lab52'


# ## 2. Reviewing the datasets
# ([Go to top](#Lab-5.2:-Working-with-Entities))
# 
# In this section, you will load the two datasets into pandas and explore the data.
# 
# First, load the plot_summaries.tsv data into a pandas DataFrame.
# 
# The file contains two columns: **movie_id** and **plot**. The data is tab-separated, and the '\t' escape sequence is used as the separator.
# 

# In[4]:


df = pd.read_csv('../data/plot_summaries.tsv', sep='\t', names=['movie_id','plot'])
df.shape


# For the rest of this lab we will be using the first 4000 records of the dataset.  This is done to streamline this lab and reduce processing time later.

# In[5]:


df = df.head(4000)
df.shape


# Review the first few rows of data to get an overview of how the data is structured.

# In[6]:


pd.options.display.max_rows
pd.set_option('display.max_colwidth', None)
df.head(5)


# To add a word count to the plot, use a lambda function. In the following cell, the lambda function splits the string, `x`, using a space and returns the number of items in the array. This will not be 100 percent accurate due to punctuation, but it will be accurate enough for this lab.

# In[7]:


df['word_count'] = df['plot'].apply(lambda x: len(str(x).split(" ")))
pd.set_option('display.max_colwidth', 150)
df.head()


# With an estimated word count, you can get an idea of how big the plot text is and get some information about the plot, including the max and min.

# In[8]:


df.word_count.describe()


# You can also find out what the most popular words are in your dataset. 

# In[9]:


freq = pd.Series(' '.join(df['plot']).split()).value_counts()[:20]


# In[10]:


freq


# As you can see, the most popular words are mostly stopwords.

# Now examine the metadata. The [dataset documentation](http://www.cs.cmu.edu/~ark/personas/data/README.txt) explains that the data contains nine fields. Load the data into a pandas DataFrame and specify the column names.

# In[11]:


movie_meta_df = pd.read_csv('../data/movie.metadata.tsv', sep='\t', names=['movie_id','freebase_id','name','release_date','box_office_revenue','runtime','languages','countries','genres'])
movie_meta_df.head()


# Set the index to **movie_id**, which will make it easier to merge this dataset with the plot.

# In[12]:


movie_meta_df.set_index('movie_id', inplace=True)


# Because you only need the movie name and something to link this metadata to the plot (**movie_id**), drop the remaining columns.

# In[13]:


movie_meta_df=movie_meta_df.drop(['freebase_id','release_date','box_office_revenue','runtime','languages','countries','genres'], axis=1)
movie_meta_df.head()


# Now that you have a general view of the dataset you can begin normalizing the text.

# ## 3. Normalizing the text
# ([Go to top](#Lab-5.2:-Working-with-Entities))
# 
# In this section, you will normalize the text. Normalization steps will remove:
#     - urls
#     - whitespace
#     - extra lines
#     - html tags

# In[14]:


def normalize_text(content):
    text = re.sub(r"http\S+", "", content ) # Remove URLs
    text = text.lower() # Lowercase all letters
    text = text.strip() # Remove leading/trailing whitespace
    text = re.sub('\s+', ' ', text) # Remove extra spaces and tabs
    text = re.sub('\n',' ',text) # Remove newlines
    text = re.compile('<.*?>').sub('', text) # Remove HTML tags/markup
    return text


# In[15]:


get_ipython().run_cell_magic('time', '', "df['plot_normalized'] = df['plot'].apply(normalize_text)\n")


# In[16]:


pd.set_option('display.max_colwidth', 150)
df.head()


# ## 4. Starting the Amazon Comprehend jobs
# ([Go to top](#Lab-5.2:-Working-with-Entities))
# 
# In this section, you will start the Amazon Comprehend jobs.

# The data needs to be uploaded to an S3 bucket to be available to Amazon Comprehend. The following function uploads the data to Amazon S3.
# 

# In[17]:


s3_resource = boto3.Session().resource('s3')

def upload_comprehend_s3_csv(filename, folder, dataframe):
    csv_buffer = io.StringIO()
    
    dataframe.to_csv(csv_buffer, header=False, index=False )
    s3_resource.Bucket(bucket).Object(os.path.join(prefix, folder, filename)).put(Body=csv_buffer.getvalue())    


# The data also needs to be formatted so that each line has a single document. The size of each line can be no more than 5,000 characters.

# In[18]:


comprehend_file = 'comprehend_input.csv'
upload_comprehend_s3_csv(comprehend_file, 'comprehend', df['plot_normalized'].str.slice(0,5000))
test_url = f's3://{bucket}/{prefix}/comprehend/{comprehend_file}'
print(f'Uploaded input to {test_url}')


# Next, you need to define the following job parameters:
# 
# - **job_data_access_role** - This is the Amazon Resource Number (ARN) for the AWS Identity and Access Management (IAM) role that has permissions to the S3 bucket. This has been provided for you.
# - **input_data_format** - Each line in the file is considered to be a separate document.
# - **job_uuid** - Use this to generate a unique job ID.
# - **job_name** - This uses the **job_uuid** to build a job name.
# - **input_data_s3_path** - This is the path to the input data.
# - **output_data_s3_path** - This is the path to where Amazon Comprehend will store the output.

# In[19]:


# Amazon Comprehend client information
comprehend_client = boto3.client(service_name="comprehend")

# IAM role with access to Amazon Comprehend and specified S3 buckets
job_data_access_role = 'arn:aws:iam::177172075141:role/service-role/c137242a3503185l8670479t1w-ComprehendDataAccessRole-A0gY0WqIHqS0'

# Other job parameters
input_data_format = 'ONE_DOC_PER_LINE'
job_uuid = uuid.uuid1()
job_name = f"kpe-job-{job_uuid}"
input_data_s3_path = test_url
output_data_s3_path = f's3://{bucket}/'


# Start the Amazon Comprehend job to detect key phrases. The job will take about 20 minutes, but you do not need to wait here for it to finish. Continue to the next step.

# In[20]:


# Begin the inference job
kpe_response = comprehend_client.start_key_phrases_detection_job(
    InputDataConfig={'S3Uri': input_data_s3_path,
                     'InputFormat': input_data_format},
    OutputDataConfig={'S3Uri': output_data_s3_path},
    DataAccessRoleArn=job_data_access_role,
    JobName=job_name,
    LanguageCode='en'
)

# Get the job ID
kpe_job_id = kpe_response['JobId']


# While that job is running, start the job to detect entities using the same input data.

# In[21]:


job_name = f'entity-job-{job_uuid}'
entity_response = comprehend_client.start_entities_detection_job(
    InputDataConfig={'S3Uri': input_data_s3_path,
                     'InputFormat': input_data_format},
    OutputDataConfig={'S3Uri': output_data_s3_path},
    DataAccessRoleArn=job_data_access_role,
    JobName=job_name,
    LanguageCode='en'
)
# Get the job ID
entity_job_id = entity_response['JobId']


# ## 5. Creating the OpenSearch cluster
# ([Go to top](#Lab-5.2:-Working-with-Entities))
# 
# While the Amazon Comprehend jobs are running, create the Amazon OpenSearch cluster.

# To obtain the IP address for your computer, use the following site. 
# 
# http://checkip.amazonaws.com/
# 
# Copy the IP address value for your computer, and replace the IP address in the following cell.

# In[22]:


my_ip = "49.37.177.177"
print(my_ip)


# Create a boto3 client for the OpenSearch cluster.

# In[23]:


es_client = boto3.client('es')


# Set up an access policy so that only your IP address can access the dashboards.

# In[24]:


access_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "",
                "Effect": "Allow",
                "Principal": {
                    "AWS": "*"
                },
                "Action": "es:*",
                "Resource": "*",
                "Condition": { "IpAddress": { "aws:SourceIp": my_ip } }
            }
        ]
    }


# Create the OpenSearch cluster using the following options:
# 
# 
# - **DomainName** - The name of the cluster
# - **ElasticSearchClusterConfig** - Specifies the instance type, the number of instances, whether a dedicated master is required, and whether the cluster should be multi-zoned
# - **AccessPolicies** - Contains the statement from above that only allows your IP address to access the cluster
# 

# In[25]:


response = es_client.create_elasticsearch_domain(
    DomainName = 'nlp-lab',
    ElasticsearchVersion = '7.9',
    ElasticsearchClusterConfig={
        "InstanceType": 'm3.large.elasticsearch',
        "InstanceCount": 2,
        "DedicatedMasterEnabled": False,
        "ZoneAwarenessEnabled": False
    },
    AccessPolicies = json.dumps(access_policy)
)


# It will take about 10 minutes for the cluster to be created, but you do not need to wait here for it to finish. Continue to the next step.

# ## 6. Using word clouds and ngrams
# ([Go to top](#Lab-5.2:-Working-with-Entities))
# 
# While Amazon Comprehend is processing the data and the OpenSearch cluster is being provisioned, you can explore some other techniques. In this section, you will create a word cloud from the plot text and then look at different ngrams to extract key phrases.  

# First, load the dataset into a new DataFrame and once again randomly sample 50 percent of the records.

# In[26]:


df2 = pd.read_csv('../data/plot_summaries.tsv', sep='\t', names=['movie_id','plot'])
df2 = df2.head(4000)
df2.head()


# Now load the stopwords for the English language.
# 

# In[27]:


stop_words = set(stopwords.words("english"))


# In[28]:


df2.get('plot')


# To normalize the text, use the same cleaning script that you used earlier. A few additional characters are removed in order to improve the word cloud. This cleaning script stores the output in an array.

# In[29]:


corpus = []

for i in range(0, df2.shape[0]):
    # Remove URLs
    text = re.sub(r"http\S+", "", df2['plot'][i])
    # Remove punctuation
    text = re.sub('[^a-zA-Z]', ' ', text)
    
    text = text.lower() # Lowercase all letters
    text = text.strip() # Remove leading/trailing whitespace
    text = re.sub('\s+', ' ', text) # Remove extra spaces and tabs
    text = re.compile('<.*?>').sub('', text) # Remove HTML tags/markup
    
    # Remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    text=re.sub("br","",text)
    
    ##Convert to list from string
    text = text.split()
    
    ##Stemming
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(word) for word in text if not word in stop_words] 
    text = " ".join(text)
    corpus.append(text)


# To make a word cloud, install the wordcloud package. Then, run the corpus through the WordCloud constructor.

# In[30]:


get_ipython().system('pip install wordcloud')


# In[31]:


from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plta

get_ipython().run_line_magic('matplotlib', 'inline')
wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stop_words,
                          max_words=100,
                          max_font_size=50, 
                          random_state=42
                         ).generate(str(corpus))

fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word1.png", dpi=900)


# Another way to evaluate the text is to create a bar chart of the most frequently occuring words. The next several cells will do the following:
#     - create a vector for the terms
#     - create a bag of words model based on the dataset
#     - create an array of the most frequent words
#     - load the most frequent words into a DataFrame
#     - create a bar chart from the DataFrame

# In[33]:


get_ipython().run_cell_magic('time', '', 'from sklearn.feature_extraction.text import CountVectorizer\nimport re\n\ncv=CountVectorizer(max_df=0.8,stop_words=list(stop_words), max_features=10000, ngram_range=(1,3))\n\nX=cv.fit_transform(corpus)\n')


# In[34]:


# Get the most frequently occurring words
def get_top_n_words(corpus, n=None, size=1):
    vec = CountVectorizer(ngram_range=(size,size), max_features=2000).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in      
                   vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                       reverse=True)
    return words_freq[:n]


# In[35]:


get_ipython().run_cell_magic('time', '', '# Convert the most frequent words to a DataFrame\ntop_words = get_top_n_words(corpus, n=20)\ntop_df = pd.DataFrame(top_words)\ntop_df.columns=["Word", "Freq"]\n\n# Create a bar plot\n\nsns.set(rc={\'figure.figsize\':(13,8)})\ng = sns.barplot(x="Word", y="Freq", data=top_df)\ng.set_xticklabels(g.get_xticklabels(), rotation=30)\n')


# In[36]:


# Get the most frequently occurring bi-grams
top2_words = get_top_n_words(corpus, n=20, size=2)
top2_df = pd.DataFrame(top2_words)
top2_df.columns=["Bi-gram", "Freq"]

# Create a bar plot
sns.set(rc={'figure.figsize':(13,8)})
h=sns.barplot(x="Bi-gram", y="Freq", data=top2_df)
h.set_xticklabels(h.get_xticklabels(), rotation=45)


# In[37]:


get_ipython().run_cell_magic('time', '', '\n# Get the most frequently occurring tri-grams\ntop3_words = get_top_n_words(corpus, n=20, size=3)\ntop3_df = pd.DataFrame(top3_words)\ntop3_df.columns=["Tri-gram", "Freq"]\n\n# Create a bar plot\nsns.set(rc={\'figure.figsize\':(13,8)})\nj=sns.barplot(x="Tri-gram", y="Freq", data=top3_df)\nj.set_xticklabels(j.get_xticklabels(), rotation=45)\n')


# In[38]:


get_ipython().run_cell_magic('time', '', '# Get the most frequently occurring 4-grams\ntop4_words = get_top_n_words(corpus, n=20, size=4)\ntop4_df = pd.DataFrame(top4_words)\ntop4_df.columns=["Four-gram", "Freq"]\nprint(top4_df)\n\n# Create a bar plot\nsns.set(rc={\'figure.figsize\':(13,8)})\nj=sns.barplot(x="Four-gram", y="Freq", data=top4_df)\nj.set_xticklabels(j.get_xticklabels(), rotation=45)\n')


# ## 7. Returning to Amazon Comprehend
# ([Go to top](#Lab-5.2:-Working-with-Entities))
# 
# 

# The Amazon Comprehend jobs are probably complete now. To check, use the next two cells.

# In[39]:


# Get current job status
kpe_job = comprehend_client.describe_key_phrases_detection_job(JobId=kpe_job_id)

# Loop until job is completed
waited = 0
timeout_minutes = 30
while kpe_job['KeyPhrasesDetectionJobProperties']['JobStatus'] != 'COMPLETED':
    sleep(10)
    waited += 10
    assert waited//60 < timeout_minutes, "Job timed out after %d seconds." % waited
    print('.', end='')
    kpe_job = comprehend_client.describe_key_phrases_detection_job(JobId=kpe_job_id)

print('Ready')


# In[40]:


# Get current job status
entity_job = comprehend_client.describe_entities_detection_job(JobId=entity_job_id)

# Loop until job is completed
waited = 0
timeout_minutes = 30
while entity_job['EntitiesDetectionJobProperties']['JobStatus'] != 'COMPLETED':
    sleep(10)
    waited += 10
    assert waited//60 < timeout_minutes, "Job timed out after %d seconds." % waited
    print('.', end='')
    entity_job = comprehend_client.describe_entities_detection_job(JobId=entity_job_id)

print('Ready')


# When both cells print "Ready," proceed to the next step.

# To get the output for the key phrases detection job, extract the output location from the job and download it to the file system.
# 

# In[41]:


kpe_comprehend_output_file = kpe_job['KeyPhrasesDetectionJobProperties']['OutputDataConfig']['S3Uri']
print(f'output filename: {kpe_comprehend_output_file}')

kpe_comprehend_bucket, kpe_comprehend_key = kpe_comprehend_output_file.replace("s3://", "").split("/", 1)

s3r = boto3.resource('s3')
s3r.meta.client.download_file(kpe_comprehend_bucket, kpe_comprehend_key, 'output-kpe.tar.gz')


# Next, extract the file and rename the output so you know which file this is.

# In[44]:


# Extract the tar file
import tarfile
tf = tarfile.open('output-kpe.tar.gz')
tf.extractall()
# Rename the output
get_ipython().system("mv 'output' 'kpe_output'")


# Repeat these steps for the entity detection job.

# In[45]:


entity_comprehend_output_file = entity_job['EntitiesDetectionJobProperties']['OutputDataConfig']['S3Uri']
print(f'output filename: {entity_comprehend_output_file}')

entity_comprehend_bucket, entity_comprehend_key = entity_comprehend_output_file.replace("s3://", "").split("/", 1)

s3r = boto3.resource('s3')
s3r.meta.client.download_file(entity_comprehend_bucket, entity_comprehend_key, 'output-entity.tar.gz')

# Extract the tar file
import tarfile
tf = tarfile.open('output-entity.tar.gz')
tf.extractall()
# Rename the output
get_ipython().system("mv 'output' 'entity_output'")


# Read the data from the key phrases file into an array.

# In[46]:


import json
data = []
with open ('kpe_output', "r") as myfile:
    for line in myfile:
        data.append(json.loads(line))


# Load the data array into a DataFrame. There are two columns: **KeyPhrases** and **Line**.

# In[47]:


kpdf = pd.DataFrame(data, columns=['KeyPhrases','Line'])
kpdf.head()


# Repeat these steps for the entity detection data.

# In[48]:


import json
data = []
with open ('entity_output', "r") as myfile:
    for line in myfile:
        data.append(json.loads(line))


# In[49]:


entitydf = pd.DataFrame(data, columns=['Entities','Line'])
entitydf.head()


# Review the entity detection data. Notice that the different detected entities, such as Person or Title, are buried in the same fields.
# 
# Depending on your scenario, you may want to split this out into separate columns for each entity type. To do this, you can write a function.

# In[50]:


def extract_entities(entities, entity_type):
    filtered_entities=[]
    for entity in entities:
        if entity['Type'] == entity_type:
            filtered_entities.append(entity)
    return filtered_entities


# Then apply the function to each of the event types that you want to extract.

# In[51]:


# df['plot_normalized'] = df['plot'].apply(normalize_text)    
entitydf['persons'] = entitydf['Entities'].apply(lambda x: extract_entities(x,'PERSON'))
entitydf['location'] = entitydf['Entities'].apply(lambda x: extract_entities(x, 'LOCATION'))
entitydf['event'] = entitydf['Entities'].apply(lambda x: extract_entities(x, 'EVENT'))
entitydf['organization'] = entitydf['Entities'].apply(lambda x: extract_entities(x, 'ORGANIZATION'))

entitydf.head()


# With the results from Amazon Comprehend loaded into DataFrames, it's time to merge the results with the original DataFrame.
# 
# First, set the index on both results DataFrames to the **Line** column.

# In[52]:


entitydf.set_index('Line', inplace = True)
entitydf.sort_index(inplace=True)
kpdf.set_index('Line', inplace=True)
kpdf.sort_index(inplace=True)
entitydf.head()


# Next, merge the **kpdf** DataFrame with the **entitydf** DataFrame.

# In[53]:


m1 = kpdf.merge(entitydf, left_index=True, right_index=True)
m1.sort_index(inplace=True)
pd.set_option('display.max_colwidth', 200)
m1.head()


# Now merge the **m1** DataFrame with the original **df** DataFrame.

# In[54]:


mergedDf = df.merge(m1, left_index=True, right_index=True)


# In[55]:


mergedDf.head()


# Before you merge with the metadata, set the index to the **movie_id** column.

# In[56]:


mergedDf.set_index('movie_id', inplace=True)


# In[57]:


pd.set_option('display.max_colwidth', 50)
mergedDf.head()


# In[58]:


movie_meta_df.head()


# The final merge is between the metadata **movie_meta_df** and the **mergedDf** DataFrames.

# In[59]:


mergedDf = mergedDf.merge(movie_meta_df, left_index=True, right_index=True)


# In[60]:


pd.set_option('display.max_colwidth', 25)
mergedDf.head()


# Next, load a document into the Amazon OpenSearch Service. If you load a single document, you can create the index pattern that is needed to build the dashboard. You could load all of the data into Amazon OpenSearch then build it, but you would not be able to see the visualizations being updated as the data loads, which can be interesting to watch.
# 
# First, check that the cluster was created.

# In[61]:


from time import sleep
alive = es_client.describe_elasticsearch_domain(DomainName='nlp-lab')
while alive['DomainStatus']['Processing']:
    print('.', end='')
    sleep(10)
    alive = es_client.describe_elasticsearch_domain(DomainName='nlp-lab')
    
print('ready!')


# Get the endpoint from the cluster so that you know where to send requests.

# In[62]:


es_domain = es_client.describe_elasticsearch_domain(DomainName='nlp-lab')
es_endpoint = es_domain['DomainStatus']['Endpoint']


# Import the OpenSearch libraries that are needed, along with AWS4Auth so that you can sign the requests with your AWS credentials.

# In[63]:


from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import requests


# Create an OpenSearch client.

# In[64]:


region= 'us-east-1' # us-east-1
service = 'es' # IMPORTANT: this is key difference while signing the request for proxy endpoint.
credentials = boto3.Session().get_credentials()

awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)
es = OpenSearch(
    hosts = [{'host': es_endpoint, 'port': 443}],
    http_auth = awsauth,
    use_ssl = True,
    verify_certs = True,
    connection_class = RequestsHttpConnection
)


# Index a single document so that you can set up the dashboard.

# In[65]:


mergedDf.head(4)


# Ideally, you would load more data into the dashboard. If you are proceeding this way, it's good to find a row that contains data in all the columns.
# 
# In the output for the previous cell, the movie in the fourth row, "The Lemon Drop Kid," has a value in every column. This maps to position 3 in the DataFrame.
# 
# To extract the information that is needed, use the **iloc** function.

# In[66]:


plot = mergedDf.iloc[3,0]
keyphrases = mergedDf.iloc[3,3]
persons = mergedDf.iloc[3,5]
location = mergedDf.iloc[3,6]
event = mergedDf.iloc[3,7]
organization = mergedDf.iloc[3,8]
movie_name = mergedDf.iloc[3,9]

document = {"name": movie_name, "plot": plot, "keyphrases": keyphrases, "persons":persons, "location":location, "event":event, "organization": organization}
print(document)


# The output from the previous cell displays the document to be indexed.
# 
# Next, to index the document into a new index named **movies**, call the **index** function.

# In[67]:


es.index(index="movies", id=3, body=document)


# To check that this action was completed, get the document from the OpenSearch cluster.

# In[68]:


print(es.get(index="movies", id="3"))


# # 8. Creating the Kibana dashboard
# ([Go to top](#Lab-5.2:-Working-with-Entities))
# 
# In this section, you will create a Kibana Dashboard to display and filter the results.
# 
# First, get the URL for the Kibana dashboard.

# In[69]:


print(f'https://{es_endpoint}/_plugin/kibana')


# 1. Navigate to the Kibana URL that was printed from the previous cell.
# 1. If prompted, choose **Explore on my own**.
# 1. When the page loads, choose **Dashboard**.
# 
#     Because this is the first time the dashboard has been loaded, you must define an index pattern.
# 
# 1. Choose **Create index pattern**. 
# 1. For **Index pattern name**, enter `movie*`
# 
#     A message displays to indicate that your index pattern matches 1 source.
# 
# 1. Choose **Next step**.
# 1. Choose **Create index pattern**.
# 
#     A table of fields displays. If everything is working, you should see 42 fields.
# 
# 1. Choose the hamburger menu, which is in the upper-left corner of the window.
# 1. Choose **Discover**.
# 1. In the **Available fields** list on the left, hover on the **name** field, and choose **Add** when it appears.
# 1. Choose **Save**, which is in the upper left of the window.
# 1. For **Title**, enter `movies`
# 1. Choose **Save**.
# 
# 1. Choose the hamburger menu, and choose **Dashboard**.
# 1. Choose **Create new dashboard**.
# 1. Choose **Add**.
# 1. Select **movies**.
# 1. Close the **Add panels** pane.
# 
# 1. Choose **Create new**.
# 1. From the list of visualizations, choose **Tag Cloud**.
# 1. Choose **movie*** as the source.
# 1. In the **Buckets** section, choose **Add**, **Tags**.
# 1. For **Aggregation**, choose **Terms**.
# 1. For **Field**, choose **keyphrases.Text.keyword**.
# 1. For **Size**, enter `25`
# 1. Choose **Update**.
# 1. Choose **Save**.
# 1. For **Title**, enter `Key Phrases`
# 1. Choose **Save and return**.
# 
# 1. Repeat steps 19-28 for the following fields:
#     - **event.Text.keyword** (Enter `Events` as the title)
#     - **location.Text.keyword** (Enter `Location` as the title)
#     - **organization.Text.keyword** (Enter `Organization` as the title)
#     - **persons.Text.keyword** (Enter `Persons` as the title)
# 
# 1. Choose **Create new**.
# 1. From the list of visualizations, choose **Metric**.
# 1. Choose **movie*** as the source.
# 1. Choose **Save**.
# 1. For **Title**, enter `Total Documents`
# 1. Choose **Save and return**.
# 
# 1. Choose the calendar icon, which is in the upper right of the window.
# 1. From the **Commonly used** list, select **Today**.
# 1. Choose the calendar icon again.
# 1. For **Refresh every**, enter `5` seconds.
# 1. Choose **Start**.
# 
# 1. Choose **Save**.
# 1. For **Title**, enter `Movies`
# 1. Choose **Save**.

# With the dashboard created, you can proceed to upload the remaining documents. A few functions are available to help you to do this quickly.
# 
# First, define a function that will create the document.

# In[70]:


from opensearchpy import helpers

def gendata(start, stop):    
    if stop>mergedDf.shape[0]:
        stop = mergedDf.shape[0]
    for i in range(start, stop):
        yield {
            "_index":'movies',
            "_type": "_doc", 
            "_id":i, 
            "_source": {"name": mergedDf.iloc[i,9], "plot": mergedDf.iloc[i,0], "keyphrases": mergedDf.iloc[i,3], "persons":mergedDf.iloc[i,5], "location":mergedDf.iloc[i,6], "event":mergedDf.iloc[i,7], "organization": mergedDf.iloc[i,8]}
        }


# Next, you need to refresh your credentials for Amazon ES, then call **helpers.bulk** to upload the remaining documents. This will take 3–4 minutes.

# In[71]:


get_ipython().run_cell_magic('time', '', "awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)\nes = OpenSearch(\n    hosts = [{'host': es_endpoint, 'port': 443}],\n    http_auth = awsauth,\n    use_ssl = True,\n    verify_certs = True,\n    connection_class = RequestsHttpConnection\n)\nhelpers.bulk(es, gendata(0,mergedDf.shape[0]))\n")


# While the documents are loading, switch back to the Kibana dashboard. The display updates every 5 seconds to include the new documents.
# 
# When all of the documents have been loaded, you can filter by choosing words in the tag clouds.
# 
# As a challenge, try to filter for James Bond movies. 
# 
# **Hint**: James Bond is likely to be found in London, he works for an organization called MI6, and his code name is 007.

# # 9. Cleaning up
# ([Go to top](#Lab-5.2:-Working-with-Entities))
# 
# When you have finished experimenting with the OpenSearch cluster, shut down the cluster.
# 

# In[72]:


response = es_client.delete_elasticsearch_domain(
    DomainName='nlp-lab'
)


# # Congratulations!
# 
# You have completed this lab, and you can now end the lab by following the lab guide instructions.

# *©2023 Amazon Web Services, Inc. or its affiliates. All rights reserved. This work may not be reproduced or redistributed, in whole or in part, without prior written permission from Amazon Web Services, Inc. Commercial copying, lending, or selling is prohibited. All trademarks are the property of their owners.*
