from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


dtcl=read_csv("path that contains data")
print(dtcl.isnull().sum())

dtcl=np.array(dtcl)

#k-means clustering algorithm
km=KMeans(n_clusters=3)
km.fit(dtcl)
print(km.cluster_centers_)

#prediction
y_km=km.fit_predict(dtcl)
print(y_km)

#visualization
plt.scatter(dtcl[:,0],dtcl[:,1],c=y_km,cmap='viridis')

plt.scatter(dtcl[y_km==0,0],dtcl[y_km==0,1],s=100,c='red')
plt.scatter(dtcl[y_km==1,0],dtcl[y_km==1,1],s=100,c='black')
plt.scatter(dtcl[y_km==2,0],dtcl[y_km==2,1],s=100,c='blue')

centers = km.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
