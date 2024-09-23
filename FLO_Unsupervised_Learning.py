###############################################################
# Customer Segmentation with Unsupervised Learning
###############################################################

###############################################################
# Business Problem
###############################################################

# I want to segment customers using unsupervised learning methods (Kmeans, Hierarchical Clustering) and observe their behaviors.

###############################################################
# Dataset Story
###############################################################

# The dataset consists of information obtained from the past shopping behaviors of customers who made their last purchases
# through OmniChannel (both online and offline) in the years 2020 - 2021.

# 20,000 observations, 13 variables

# master_id: Unique customer number
# order_channel: The channel used for the shopping platform (Android, iOS, Desktop, Mobile, Offline)
# last_order_channel: The channel of the last purchase
# first_order_date: The date of the customer's first purchase
# last_order_date: The date of the customer's last purchase
# last_order_date_online: The date of the customer's last online purchase
# last_order_date_offline: The date of the customer's last offline purchase
# order_num_total_ever_online: Total number of purchases made by the customer on the online platform
# order_num_total_ever_offline: Total number of purchases made by the customer offline
# customer_value_total_ever_offline: Total amount paid by the customer for offline purchases
# customer_value_total_ever_online: Total amount paid by the customer for online purchases
# interested_in_categories_12: List of categories the customer shopped in the last 12 months
# store_type: Represents 3 different companies. If a person shopped from company A and company B, it's written as A,B.

###############################################################
# TASKS
###############################################################

# TASK 1: Preparing the Data
           # 1. Read the flo_data_20K.csv file.
           # 2. Select the variables you will use while segmenting the customers. You can create new variables like Tenure (Customer's age), Recency (how many days ago the last purchase was made).

# TASK 2: Customer Segmentation with K-Means
           # 1. Standardize the variables.
           # 2. Determine the optimal number of clusters.
           # 3. Create your model and segment your customers.
           # 4. Statistically examine each segment.

# TASK 3: Customer Segmentation with Hierarchical Clustering
           # 1. Using the dataframe you standardized in Task 2, determine the optimal number of clusters.
           # 2. Create your model and segment your customers.
           # 3. Statistically examine each segment.

###############################################################
# TASK 1: Read the dataset and select the variables you will use to segment the customers.
###############################################################


import pandas as pd
from scipy import stats
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 1000)

df_ = pd.read_csv("C:/Users/Isa Kaan Albayrak/Desktop/Eğitimler/Miuul/MachineLearning/Flo Customer Segmantation/flo_data_20K.csv")
df = df_.copy()
df.head()

# Converting to date
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

df["last_order_date"].max() # 2021-05-30
analysis_date = dt.datetime(2021,6,1)

df["recency"] = (analysis_date - df["last_order_date"]) / np.timedelta64(1, 'D')  # How many days ago the last purchase was made
df["tenure"] = (df["last_order_date"] - df["first_order_date"]) / np.timedelta64(1, 'D')

model_df = df[["order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online","recency","tenure"]]
model_df.head()

###############################################################
# TASK 2: Customer Segmentation with K-Means
###############################################################

# 1. Standardize the variables.
# SKEWNESS
def check_skew(df_skew, column):
    skew = stats.skew(df_skew[column])
    skewtest = stats.skewtest(df_skew[column])
    plt.title('Distribution of ' + column)
    sns.distplot(df_skew[column], color = "g")
    print("{}'s: Skew: {}, : {}".format(column, skew, skewtest))
    return

plt.figure(figsize=(9, 9))
plt.subplot(6, 1, 1)
check_skew(model_df,'order_num_total_ever_online')
plt.subplot(6, 1, 2)
check_skew(model_df,'order_num_total_ever_offline')
plt.subplot(6, 1, 3)
check_skew(model_df,'customer_value_total_ever_offline')
plt.subplot(6, 1, 4)
check_skew(model_df,'customer_value_total_ever_online')
plt.subplot(6, 1, 5)
check_skew(model_df,'recency')
plt.subplot(6, 1, 6)
check_skew(model_df,'tenure')
plt.tight_layout()
plt.savefig('before_transform.png', format='png', dpi=1000)
plt.show()

# Applying Log transformation for normal distribution
model_df['order_num_total_ever_online'] = np.log1p(model_df['order_num_total_ever_online'])
model_df['order_num_total_ever_offline'] = np.log1p(model_df['order_num_total_ever_offline'])
model_df['customer_value_total_ever_offline'] = np.log1p(model_df['customer_value_total_ever_offline'])
model_df['customer_value_total_ever_online'] = np.log1p(model_df['customer_value_total_ever_online'])
model_df['recency'] = np.log1p(model_df['recency'])
model_df['tenure'] = np.log1p(model_df['tenure'])
model_df.head()

# Scaling
sc = MinMaxScaler((0, 1))
model_scaling = sc.fit_transform(model_df)
model_df = pd.DataFrame(model_scaling, columns=model_df.columns)
model_df.head()

# 2. Determine the optimal number of clusters.
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(model_df)
elbow.show()

# 3. Create your model and segment your customers.
k_means = KMeans(n_clusters = 7, random_state= 42).fit(model_df)
segments = k_means.labels_
segments

final_df = df[["master_id","order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online","recency","tenure"]]
final_df["segment"] = segments
final_df.head()

# 4. Statistically examine each segment.
final_df.groupby("segment").agg({"order_num_total_ever_online":["mean","min","max"],
                                  "order_num_total_ever_offline":["mean","min","max"],
                                  "customer_value_total_ever_offline":["mean","min","max"],
                                  "customer_value_total_ever_online":["mean","min","max"],
                                  "recency":["mean","min","max"],
                                  "tenure":["mean","min","max","count"]})

###############################################################
# TASK 3: Customer Segmentation with Hierarchical Clustering
###############################################################

# 1. Using the dataframe standardized in Task 2, determine the optimal number of clusters.
hc_complete = linkage(model_df, 'complete')

plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_complete,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.axhline(y=1.2, color='r', linestyle='--')
plt.show()

# 2. Create your model and segment your customers.
hc = AgglomerativeClustering(n_clusters=5)
segments = hc.fit_predict(model_df)

final_df = df[["master_id","order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online","recency","tenure"]]
final_df["segment"] = segments
final_df.head()
final_df["segment"].value_counts()

# 3. Statistically examine each segment.
final_df.groupby("segment").agg({"order_num_total_ever_online":["mean","min","max"],
                                  "order_num_total_ever_offline":["mean","min","max"],
                                  "customer_value_total_ever_offline":["mean","min","max"],
                                  "customer_value_total_ever_online":["mean","min","max"],
                                  "recency":["mean","min","max"],
                                  "tenure":["mean","min","max","count"]})
