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
data1 = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Data Mining  Unsupervised Learning Dimension Reduction\heart disease.csv")
data1.info()
data1.describe()
data = data1.drop(["target"], axis = 1)

# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(data)

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 4 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 4)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
data1['clust'] = mb # creating a  new column and assigning it to new column 

data1.head()
df_norm.head()

data2 = data1.iloc[:,[14,13,0,1,2,3,4,5,6,7,8,9,10,11,12]]
data2.head()

mean_heart_disease =data2.iloc[:, 2:].groupby(data2.clust).mean()

data2.to_csv("Kmeans_heart deseaes2.csv", encoding = "utf-8")

import os
os.getcwd()


################### PCA ##########

import pandas as pd
import numpy as np

df1 = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Data Mining  Unsupervised Learning Dimension Reduction\heart disease.csv")
df1.describe()

df1.info()
df = df1.drop(["target"], axis = 1)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 

# Considering only numerical data 
df.data = df.iloc[:, :]

# Normalizing the numerical data 
df_normal = scale(df.data)
df_normal

pca = PCA(n_components = 13)
pca_values = pca.fit_transform(df_normal)

# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var

# PCA weights
pca.components_
pca.components_[0]

# Cumulative variance 
var1 = np.cumsum(np.round(var, decimals = 4) * 100)
var1

# Variance plot for PCA components obtained 
plt.plot(var1, color = "red")

# PCA scores
pca_values

pca_data = pd.DataFrame(pca_values)
pca_data.columns = "comp0", "comp1", "comp2", "comp3", "comp4", "comp5", "comp6","comp7", "comp8","comp9","comp10","comp11","comp12"
final = pd.concat([df1.target, pca_data.iloc[:, 0:10]], axis = 1)

# Scatter diagram
import matplotlib.pylab as plt
ax = final.plot(x='comp0', y='comp1', kind='scatter',figsize=(12,8))
final[['comp0', 'comp1', 'target']].apply(lambda x: ax.text(*x), axis=1)

