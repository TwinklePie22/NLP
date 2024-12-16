#!/usr/bin/env python
# coding: utf-8

# Version: 02.14.2023

# # Lab 5.1: Implementing Information Extraction
# 
# In this lab, you will use Amazon Comprehend to extract key entity and event information from financial documents.
# 
# ## About this dataset
# 
# The [sample_finance_dataset.txt](https://github.com/aws-samples/amazon-comprehend-examples/tree/master/amazon_comprehend_events_tutorial) file contains a set of 118 press releases in doclines format from the [Amazon press release archive](https://press.aboutamazon.com/press-releases).
# 
# ## Lab steps
# 
# To complete this lab, you will follow these steps:
# 
# 1. [Installing the packages](#1.-Install-packages)
# 2. [Writing the documents to Amazon S3](#2.-Write-documents-to-S3)
# 3. [Starting an asynchronous events detection job using the SDK](#3.-Start-an-asynchronous-events-detection-job-using-the-SDK)
# 4. [Analyzing the Amazon Comprehend Events output](#4.-Analyzing-Comprehend-Events-output)
# 
# 
# ## Submitting your work
# 
# 1. In the lab console, choose **Submit** to record your progress and when prompted, choose **Yes**.
# 
# 1. If the results don't display after a couple of minutes, return to the top of these instructions and choose **Grades**.
# 
#      **Tip:** You can submit your work multiple times. After you change your work, choose **Submit** again. Your last submission is what will be recorded for this lab.
# 
# 1. To find detailed feedback on your work, choose **Details** followed by **View Submission Report**.

# ## 1. Installing packages
# ([Go to top](#Lab-5.1:-Implementing-Information-Extraction))
# 
# Start by updating and installing the packages that you will use in the notebook. 
# 

# In[1]:


get_ipython().system('pip install smart_open')
get_ipython().system('pip install networkx')
get_ipython().system('pip install pandas')
get_ipython().system('pip install pyvis')
get_ipython().system('pip install spacy')
get_ipython().system('pip install ipywidgets')


#   
# 
# __Note:__ Before you proceed with this lab for the first time, we recommend that you restart the kernel by choosing __Kernel__ > __Restart Kernel__.

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
# !pip install --upgrade pandas
import json
import requests
import uuid

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import boto3
import smart_open

from time import sleep
from matplotlib import cm, colors
from spacy import displacy
from collections import Counter
from pyvis.network import Network


# ## 2. Writing the documents to Amazon S3
# ([Go to top](#Lab-5.1:-Implementing-Information-Extraction))
# 
# The data folder for this lab has a text file containing Amazon press releases as example documents. They have been saved in a single file with one document per line.
# 
# In this section, you will upload the sample_finance_dataset.txt file to an Amazon Simple Storage Service (Amazon S3) bucket for processing. The processing output will be returned to the same bucket.

# In[3]:


bucket = "c137242a3503183l8664525t1w442333327694-labbucket-ew2jaiz4qvwz"


# In[4]:


# Client and session information
session = boto3.Session()
s3_client = session.client(service_name="s3")

# Constants for S3 bucket and input data file

filename = "sample_finance_dataset.txt"
input_data_s3_path = f's3://{bucket}/' + filename
output_data_s3_path = f's3://{bucket}/'

# Upload the local file to S3
s3_client.upload_file("../data/" + filename, bucket, filename)

# Load the documents locally for later analysis
with open("../data/" + filename, "r") as fi:
    raw_texts = [line.strip() for line in fi.readlines()]


# ## 3. Starting an asynchronous events detection job using the SDK
# ([Go to top](#Lab-5.1:-Implementing-Information-Extraction))
# 
# In this section, you will start the events detection job. You will use the `start_events_detection_job` function. 
# 
# Note that the API requires an AWS Identity and Access Management (IAM) role with List, Read, and Write permissions for the S3 bucket that you specified in the previous section.

# In[5]:


# Amazon Comprehend client information
comprehend_client = session.client(service_name="comprehend")

# IAM role with access to Amazon Comprehend and the specified S3 bucket
job_data_access_role = 'arn:aws:iam::442333327694:role/service-role/c137242a3503183l8664525t1w-ComprehendDataAccessRole-yg1kk8z0JUEh'

# Other job parameters
input_data_format = 'ONE_DOC_PER_LINE'
job_uuid = uuid.uuid1()
job_name = f"events-job-{job_uuid}"


# Because you are processing company press releases, you can define the event types that you are most interested in. For the full list of event types, see the [Amazon Comprehend Developer Guide](https://docs.aws.amazon.com/comprehend/latest/dg/how-events.html#events-types).

# In[6]:


event_types = ["BANKRUPTCY", "EMPLOYMENT", "CORPORATE_ACQUISITION", 
               "INVESTMENT_GENERAL", "CORPORATE_MERGER", "IPO",
               "RIGHTS_ISSUE", "SECONDARY_OFFERING", "SHELF_OFFERING",
               "TENDER_OFFERING", "STOCK_SPLIT"]


# Start the Amazon Comprehend job.

# In[7]:


# Begin the inference job
response = comprehend_client.start_events_detection_job(
    InputDataConfig={'S3Uri': input_data_s3_path,
                     'InputFormat': input_data_format},
    OutputDataConfig={'S3Uri': output_data_s3_path},
    DataAccessRoleArn=job_data_access_role,
    JobName=job_name,
    LanguageCode='en',
    TargetEventTypes=event_types
)

# Get the job ID
events_job_id = response['JobId']


# In[8]:


# Get the current job status
job = comprehend_client.describe_events_detection_job(JobId=events_job_id)

# Loop until the job is completed
waited = 0
timeout_minutes = 30
while job['EventsDetectionJobProperties']['JobStatus'] != 'COMPLETED':
    sleep(60)
    waited += 60
    assert waited//60 < timeout_minutes, "Job timed out after %d seconds." % waited
    print('.', end='')
    job = comprehend_client.describe_events_detection_job(JobId=events_job_id)

print('Ready')


# When the job is complete, download the results in memory.

# In[9]:


# The output filename is the input filename + ".out"
output_data_s3_file = job['EventsDetectionJobProperties']['OutputDataConfig']['S3Uri'] + filename + '.out'

# Load the output into a results dictionary
results = []
with smart_open.open(output_data_s3_file) as fi:
    results.extend([json.loads(line) for line in fi.readlines() if line])


# ## 4. Analyzing the Amazon Comprehend Events output
# ([Go to top](#Lab-5.1:-Implementing-Information-Extraction))
# 
# With the data downloaded, it's time to analyze the results.
# 
# You will use the first document in the submitted dataset as an example. The document is typical of what a financial document might consume when projecting market trends. For the full press release, see [Amazon.com Announces Third Quarter Sales up 34% to $43.7 Billion](https://press.aboutamazon.com/news-releases/news-release-details/amazoncom-announces-third-quarter-sales-34-437-billion).
# 
# The following provides the text of the press release:

# > Amazon.com, Inc. (NASDAQ: AMZN) today announced financial results for its third quarter ended September 30, 2017.
# 
# > Operating cash flow increased 14% to \\$17.1 billion for the trailing twelve months, compared with \\$15.0 billion for the trailing twelve months ended September 30, 2016. Free cash flow decreased to \\$8.1 billion for the trailing twelve months, compared with \\$9.0 billion for the trailing twelve months ended September 30, 2016. Free cash flow less lease principal repayments decreased to \\$3.5 billion for the trailing twelve months, compared with \\$5.3 billion for the trailing twelve months ended September 30, 2016. Free cash flow less finance lease principal repayments and assets acquired under capital leases decreased to an outflow of \\$1.0 billion for the trailing twelve months, compared with an inflow of \\$3.8 billion for the trailing twelve months ended September 30, 2016.
# 
# > Common shares outstanding plus shares underlying stock-based awards totaled 503 million on September 30, 2017, compared with 496 million one year ago.
# 
# > Net sales increased 34% to \\$43.7 billion in the third quarter, compared with \\$32.7 billion in third quarter 2016. Net sales includes \\$1.3 billion from Whole Foods Market, which Amazon acquired on August 28, 2017. Excluding Whole Foods Market and the \\$124 million favorable impact from year-over-year changes in foreign exchange rates throughout the quarter, net sales increased 29% compared with third quarter 2016.
# 
# > Operating income decreased 40% to \\$347 million in the third quarter, compared with operating income of \\$575 million in third quarter 2016. Operating income includes income of \\$21 million from Whole Foods Market.
# 
# > Net income was \\$256 million in the third quarter, or \\$0.52 per diluted share, compared with net income of \\$252 million, or \\$0.52 per diluted share, in third quarter 2016.
# 
# > “In the last month alone, we’ve launched five new Alexa-enabled devices, introduced Alexa in India, announced integration with BMW, surpassed 25,000 skills, integrated Alexa with Sonos speakers, taught Alexa to distinguish between two voices, and more. Because Alexa’s brain is in the AWS cloud, her new abilities are available to all Echo customers, not just those who buy a new device,” said Jeff Bezos, Amazon founder and CEO. “And it’s working — customers have purchased tens of millions of Alexa-enabled devices, given Echo devices over 100,000 5-star reviews, and active customers are up more than 5x since the same time last year. With thousands of developers and hardware makers building new Alexa skills and devices, the Alexa experience will continue to get even better.”

# ### Understanding the Amazon Comprehend Events system output
# 
# The system returns JSON output for each submitted document. The following cells show the structure of the response. 
# 
# Note:
# * The system output contains separate objects for `Entities` and `Events`, which are each organized into groups of coreferential object.  
# * Two additional fields, `File` and `Line`, are present to track document provenance.

# In[10]:


# Use the first results document for analysis
result = results[0]
raw_text = raw_texts[0]


# In[11]:


raw_text


# In[12]:


result


# #### Events are groups of triggers
# 
# * The API output includes the text, character offset, and type of each trigger.
# 
# * Confidence scores for classification tasks are given as `Score`. Confidence of event group membership is given as `GroupScore`.

# In[13]:


result['Events'][1]['Triggers']


# #### Arguments are linked to entities by the EntityIndex
# 
# * The API also returns the classification confidence of the role assignment.

# In[14]:


result['Events'][1]['Arguments']


# #### Entities are groups of mentions
# 
# * The API output includes the text, character offset, and type of each mention.  
# 
# * Confidence scores for classification tasks are given as `Score`. Confidence of entity group membership is given as `GroupScore`.  

# In[15]:


result['Entities'][0]['Mentions']


# ### Visualizing the events and entities
# 
# In this section, you explore a number of tabulations and visualizations to help understand what the API is returning.
# 
# First, you consider a visualization of spans, both triggers and entity mentions. One of the most essential visualization tasks for sequence labeling is highlighting tagged text in documents. For demo purposes, you will use [displaCy](https://spacy.io/usage/visualizers).

# In[16]:


# Convert the output to the displaCy format
entities = [
    {'start': m['BeginOffset'], 'end': m['EndOffset'], 'label': m['Type']}
    for e in result['Entities']
    for m in e['Mentions']
]

triggers = [
    {'start': t['BeginOffset'], 'end': t['EndOffset'], 'label': t['Type']}
    for e in result['Events']
    for t in e['Triggers']
]

# Spans need to be sorted for displaCy to process them correctly
spans = sorted(entities + triggers, key=lambda x: x['start'])
tags = [s['label'] for s in spans]

output = [{"text": raw_text, "ents": spans, "title": None, "settings": {}}]


# In[17]:


# Miscellaneous objects for presentation purposes
spectral = cm.get_cmap("Spectral", len(tags))
tag_colors = [colors.rgb2hex(spectral(i)) for i in range(len(tags))]
color_map = dict(zip(*(tags, tag_colors)))


# In[18]:


# Note that only entities participating in events are shown
displacy.render(output, style="ent", options={"colors": color_map}, manual=True)


# ### Rendering as tabular data
# 
# A common use for Amazon Comprehend Events is to create structured data from unstructured text. For this lab, you will do this with pandas.
# 
# First, flatten the hierarchical JSON data into a pandas DataFrame. 

# In[19]:


# Create the entity DataFrame. Entity indices must be explicitly created.
entities_df = pd.DataFrame([
    {"EntityIndex": i, **m}
    for i, e in enumerate(result['Entities'])
    for m in e['Mentions']
])

# Create the events DataFrame. Event indices must be explicitly created.
events_df = pd.DataFrame([
    {"EventIndex": i, **a, **t}
    for i, e in enumerate(result['Events'])
    for a in e['Arguments']
    for t in e['Triggers']
])

# Join the two tables into one flat data structure
events_df = events_df.merge(entities_df, on="EntityIndex", suffixes=('Event', 'Entity'))


# In[20]:


events_df


# ### Creating a more succinct representation
# 
# Because you are primarily interested in the *event structure*, make the data more transparent by creating a new table with Roles as column headers, grouped by Event.

# In[21]:


def format_compact_events(x):
    """Collapse groups of mentions and triggers into a single set."""
    # Take the most commonly occurring EventType and the set of triggers
    d = {"EventType": Counter(x['TypeEvent']).most_common()[0][0],
         "Triggers": set(x['TextEvent'])}
    # For each argument Role, collect the set of mentions in the group
    for role in x['Role']:
        d.update({role: set((x[x['Role']==role]['TextEntity']))})
    return d

# Group data by EventIndex and format
event_analysis_df = pd.DataFrame(
    events_df.groupby("EventIndex").apply(format_compact_events).tolist()
).fillna('')


# In[22]:


event_analysis_df


# ### Graphing event semantics
# 
# A semantic graph often provides the most striking representation of Amazon Comprehend Events output. A semantic graph is a network of the entities and events referenced in a document or documents.
# 
# The following code uses two open-source libraries, NetworkX and pyvis, to render the system output. In the resulting graph, nodes are entity mentions and triggers, while edges are the argument roles that the entities hold in relation to the triggers.

# #### Formatting the data
# ID Comment: Should we separate the words 'network and X' within the first sentence?
# System output must first be conformed to the node (i.e., vertex) and edge list format that networkx requires. This requires iterating over triggers, entities, and argument structural relations. Note that you can use the `GroupScore` and `Score` keys on various objects to prune nodes and edges in which the model has less confidence. You can also use various strategies to pick a "canonical" mention from each mention group to appear in the graph. For this task, you will use the mention with the string-wise longest extent.

# In[23]:


# Entities are associated with events by group, not individual mention
# For simplicity, aassume the canonical mention is the longest one
def get_canonical_mention(mentions):
    extents = enumerate([m['Text'] for m in mentions])
    longest_name = sorted(extents, key=lambda x: len(x[1]))
    return [mentions[longest_name[-1][0]]]

# Set a global confidence threshold
thr = 0.5

# Nodes are (id, type, tag, score, mention_type) tuples
trigger_nodes = [
    ("tr%d" % i, t['Type'], t['Text'], t['Score'], "trigger")
    for i, e in enumerate(result['Events'])
    for t in e['Triggers'][:1]
    if t['GroupScore'] > thr
]
entity_nodes = [
    ("en%d" % i, m['Type'], m['Text'], m['Score'], "entity")
    for i, e in enumerate(result['Entities'])
    for m in get_canonical_mention(e['Mentions'])
    if m['GroupScore'] > thr
]

# Edges are (trigger_id, node_id, role, score) tuples
argument_edges = [
    ("tr%d" % i, "en%d" % a['EntityIndex'], a['Role'], a['Score'])
    for i, e in enumerate(result['Events'])
    for a in e['Arguments']
    if a['Score'] > thr
]    


# #### Creating a compact graph
# 
# Once the nodes and edges are defined, you can create and visualize the graph.

# In[24]:


G = nx.Graph()

# Iterate over triggers and entity mentions
for mention_id, tag, extent, score, mtype in trigger_nodes + entity_nodes:
    label = extent if mtype.startswith("entity") else tag
    G.add_node(mention_id, label=label, size=score*10, color=color_map[tag], tag=tag, group=mtype)
    
# Iterate over argument role assignments
for event_id, entity_id, role, score in argument_edges:
    G.add_edges_from(
        [(event_id, entity_id)],
        label=role,
        weight=score*100,
        color="grey"
    )

# Drop mentions that don't participate in events
G.remove_nodes_from(list(nx.isolates(G)))


# In[25]:


nt = Network("600px", "800px", notebook=True, heading="")
nt.from_nx(G)
nt.show("compact_nx.html")


# #### Creating a more complete graph
# 
# The previous graph is compact, and only relays the essential event type and argument role information. You can use a slightly more complicated set of functions to graph all of the information that the API returns.

# In[26]:


# This function in `events_graph.py` plots a complete graph of the document
# The graph shows all events, triggers, entities, and their groups

import events_graph as evg

evg.plot(result, node_types=['event', 'trigger', 'entity_group', 'entity'], thr=0.5)


# # Congratulations!
# 
# You have completed this lab, and you can now end the lab by following the lab guide instructions.

# *©2023 Amazon Web Services, Inc. or its affiliates. All rights reserved. This work may not be reproduced or redistributed, in whole or in part, without prior written permission from Amazon Web Services, Inc. Commercial copying, lending, or selling is prohibited. All trademarks are the property of their owners.*
# 
