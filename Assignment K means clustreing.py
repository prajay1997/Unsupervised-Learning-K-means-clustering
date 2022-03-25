# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 12:09:34 2022

@author: praja
"""
Q1)
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn.cluster import	KMeans
# from scipy.spatial.distance import cdist 

# Generating random uniform numbers 
X = np.random.uniform(0,1,50)
Y = np.random.uniform(0,1,50)
df_xy = pd.DataFrame(columns=["X","Y"]) #  Creating the dataframe
df_xy.X = X
df_xy.Y = Y

df_xy.plot(x="X", y ="Y", kind = "scatter")

model1 = KMeans(n_clusters = 3).fit(df_xy)

df_xy.plot(x = "X", y = "Y", c = model1.labels_, kind="scatter", s = 10, cmap = plt.cm.coolwarm)

# Kmeans on University Data set 
data1 = pd.read_excel(r"C:\Users\praja\Desktop\EastWestAirlines.xlsx")
data1.info()
data1.describe()

data = data1.drop(['ID#','Award?'], axis=1)

# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(data)

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 10))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
data['clust'] = mb # creating a  new column and assigning it to new column 

data.head()
df_norm.head()

data2 = data.iloc[:,[10,0,1,2,3,4,5,6,7,8,9]]
data2.head()

data2.iloc[:,1:].groupby(data.clust).mean().transpose()

data.to_csv(" Eastwest airline k means.csv", encoding = "utf-8")

import os
os.getcwd()

############################################################################
Q2)
    
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn.cluster import	KMeans
# from scipy.spatial.distance import cdist 

# Generating random uniform numbers 
X = np.random.uniform(0,1,50)
Y = np.random.uniform(0,1,50)
df_xy = pd.DataFrame(columns=["X","Y"]) #  Creating the dataframe
df_xy.X = X
df_xy.Y = Y

df_xy.plot(x="X", y ="Y", kind = "scatter")

model1 = KMeans(n_clusters = 3).fit(df_xy)

df_xy.plot(x = "X", y = "Y", c = model1.labels_, kind="scatter", s = 7, cmap = plt.cm.coolwarm)

# Kmeans on crimedata dataset
data1 = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Data Mining unsupervised learning K -Means\crime_data (1).csv")
data1.info()
data1.describe()

# remove the Unnamed column 

data1 = data.drop(['Unnamed: 0'],axis=1)
# Normalization function 

def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(data1)

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 10))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
data['clust'] = mb # creating a  new column and assigning it to new column 

data.head()
df_norm.head()

data2 = data.iloc[:,[5,0,1,2,3,4,4]]
data2.head()

data2.iloc[:,2:].groupby(data.clust).mean().transpose()

data.to_csv(" crimrate ", encoding = "utf-8")

import os
os.getcwd()
###################################################################

Q3)

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn.cluster import	KMeans
# from scipy.spatial.distance import cdist 

# Generating random uniform numbers 
X = np.random.uniform(0,1,50)
Y = np.random.uniform(0,1,50)
df_xy = pd.DataFrame(columns=["X","Y"]) #  Creating the dataframe
df_xy.X = X
df_xy.Y = Y

df_xy.plot(x="X", y ="Y", kind = "scatter")

model1 = KMeans(n_clusters = 3).fit(df_xy)

df_xy.plot(x = "X", y = "Y", c = model1.labels_, kind="scatter", s = 10, cmap = plt.cm.coolwarm)

# Kmeans on Insurance Dataset
data1 = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Data Mining unsupervised learning K -Means\Insurance Dataset.csv")
data1.info()
data1.describe()

# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(data1)

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 10))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
data1['clust'] = mb # creating a  new column and assigning it to new column 

data1.head()
df_norm.head()

data = data1.iloc[:,[5,0,1,2,3,4]]
data.head()

data.iloc[:,:].groupby(data.clust).mean().transpose()

data.to_csv(" insurance.csv", encoding = "utf-8")

import os
os.getcwd()
#########################################################################

#  Q4)
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn.cluster import	KMeans
# from scipy.spatial.distance import cdist 

# Generating random uniform numbers 
X = np.random.uniform(0,1,50)
Y = np.random.uniform(0,1,50)
df_xy = pd.DataFrame(columns=["X","Y"]) #  Creating the dataframe
df_xy.X = X
df_xy.Y = Y

df_xy.plot(x="X", y ="Y", kind = "scatter")

model1 = KMeans(n_clusters = 3).fit(df_xy)

df_xy.plot(x = "X", y = "Y", c = model1.labels_, kind="scatter", s = 10, cmap = plt.cm.coolwarm)

# Kmeans on Insurance Dataset
data1 = pd.read_excel(r"C:\Users\praja\Desktop\Data Science\Data Mining unsupervised learning K -Means\Telco_customer_churn (1).xlsx")
data1.info()
data1.describe()

# remove the  CustomerID, count, quarter, refered a friend, no of referrals, paperless billing,payment method  as it is not useful for data analysis

data = data1.drop(["Customer ID","Count","Quarter", "Referred a Friend","Number of Referrals","Paperless Billing", "Payment Method"], axis = 1)
data.columns

# dividing the data into numeric and categorical data 

data_num = data1[['Tenure in Months','Avg Monthly Long Distance Charges','Avg Monthly GB Download','Monthly Charge','Total Charges','Total Refunds','Total Extra Data Charges','Total Long Distance Charges','Total Revenue']]
data_num
data_cat = data1[['Offer','Phone Service','Multiple Lines','Internet Service','Internet Type','Online Security','Online Backup','Device Protection Plan','Premium Tech Support','Streaming TV','Streaming Movies','Streaming Music','Unlimited Data','Contract',]]


# converting non numeric data into numeric

a = pd.get_dummies(data_cat, drop_first=True)

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

data_norm = norm_func(data_num)
data_norm.describe()

###### scree plot or elbow curve  for numerical data ############
TWSS = []
k = list(range(2, 15))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(data_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters for num ");plt.ylabel("total_within_SS")

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 5)
model.fit(data_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
data_num['clust'] = mb # creating a  new column and assigning it to new column 

data_num.head

data_num1 = data_num.iloc[:,[9,0,1,2,3,4,5,6,7,8]]
data_num1.head()

mean_numerical = data_num1.iloc[:,:].groupby(data_num1.clust).mean().transpose()

data_num1.to_csv(" telecomunnication_num", encoding = "utf-8")

import os
os.getcwd()

###### scree plot or elbow curve  for categorical data ############
TWSS = []
t = list(range(2, 15))

for i in t:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(a)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(t, TWSS, 'ro-');plt.xlabel("No_of_Clusters for cat");plt.ylabel("total_within_SS")

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model1 = KMeans(n_clusters = 5)
model1.fit(a)

model1.labels_ # getting the labels of clusters assigned to each row 
mb1 = pd.Series(model.labels_)  # converting numpy array into pandas series object 
a['clust1'] = mb1 # creating a  new column and assigning it to new column 

a.head

b = a.iloc[:,[21,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]
b.head()

mean_categorical = b.iloc[:,:].groupby(b.clust1).mean().transpose()

b.to_csv(" telecomunnication_num", encoding = "utf-8")

#########################################################################
# Q5)
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn.cluster import	KMeans
# from scipy.spatial.distance import cdist 

# Generating random uniform numbers 
X = np.random.uniform(0,1,50)
Y = np.random.uniform(0,1,50)
df_xy = pd.DataFrame(columns=["X","Y"]) #  Creating the dataframe
df_xy.X = X
df_xy.Y = Y

df_xy.plot(x="X", y ="Y", kind = "scatter")

model1 = KMeans(n_clusters = 3).fit(df_xy)

df_xy.plot(x = "X", y = "Y", c = model1.labels_, kind="scatter", s = 10, cmap = plt.cm.coolwarm)

# Kmeans on Insurance Dataset
data1 = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Data Mining unsupervised learning K -Means\AutoInsurance (1).csv")
data1.info()
data1.describe()
data1.columns

# remove the  'Customer state, location code, Marital Status, Sales Chaneel  as it is not useful for data analysis

data = data1.drop(["Customer","State",'Location Code','Marital Status', 'Sales Channel'], axis = 1)
data.columns

data.columns

# convert date function to no of days 
import datetime
data['Effective To Date']= pd.to_datetime(data["Effective To Date"])
data.info()

import datetime

today = datetime.datetime.today()
today

data['Effective To Date'] = (today- data["Effective To Date"]).dt.days


# dividing the data into numeric and categorical data 

data_num = data[['Customer Lifetime Value','Effective To Date','Income','Monthly Premium Auto','Months Since Last Claim','Months Since Policy Inception','Number of Open Complaints', 'Number of Policies','Total Claim Amount']]

data_cat = data[['Response','Coverage','Education','EmploymentStatus','Gender','Policy Type','Policy','Renew Offer Type','Vehicle Class', 'Vehicle Size']]


# converting non numeric data into numeric

a = pd.get_dummies(data_cat, drop_first=True)

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

data_norm = norm_func(data_num)
data_norm.describe()

###### scree plot or elbow curve  for numerical data ############
TWSS = []
k = list(range(2, 10))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(data_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters for num ");plt.ylabel("total_within_SS")

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 5)
model.fit(data_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
data_num['clust'] = mb # creating a  new column and assigning it to new column 

data_num.head

data_num1 = data_num.iloc[:,[9,0,1,2,3,4,5,6,7,8]]
data_num1.head()

mean_numerical = data_num1.iloc[:,:].groupby(data_num1.clust).mean().transpose()

data_num1.to_csv(" Autoinsurance num", encoding = "utf-8")

import os
os.getcwd()

###### scree plot or elbow curve  for categorical data ############
TWSS = []
t = list(range(2, 15))

for i in t:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(a)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(t, TWSS, 'ro-');plt.xlabel("No_of_Clusters for cat");plt.ylabel("total_within_SS")

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model1 = KMeans(n_clusters = 5)
model1.fit(a)

model1.labels_ # getting the labels of clusters assigned to each row 
mb1 = pd.Series(model.labels_)  # converting numpy array into pandas series object 
a['clust1'] = mb1 # creating a  new column and assigning it to new column 

a.head

b = a.iloc[:,[32,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]]
b.head()

mean_categorical = b.iloc[:,:].groupby(b.clust1).mean().transpose()

b.to_csv(" Autoinsurance_cat", encoding = "utf-8")

import os
os.getcwd()
#########################################################################













