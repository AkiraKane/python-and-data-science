import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pdb

def get_random_data():
    x_1 = np.random.normal(loc=0.2, scale=0.2, size=(100,100))
    x_2 = np.random.normal(loc=0.9, scale=0.1, size=(100,100))
    x = np.r_[x_1, x_2]
    return x

x = get_random_data()
plt.cla()
plt.figure(1)
plt.title("Genreated Data")
plt.scatter(x[:,0], x[:,1])
plt.show()


def form_clusters(x,k):
    print "Build Clusters"
    # k = required number of clusters
    no_clusters = k
    model = KMeans(n_clusters=no_clusters, init='random')
    model.fit(x)
    labels = model.labels_
    print labels
    # calculate the silhouette score
    sh_score = silhouette_score(x, labels)
    return sh_score


sh_scores = []
for i in range(1,5):
    sh_score = form_clusters(x, i+1)
    sh_scores.append(sh_score)

no_clusters = [1+1 for i in range(1,5)]

plt.figure(2)
plt.plot(no_clusters, sh_scores)
plt.title("Cluster Quality")
plt.xlabel("No of clusters k")
plt.ylabel("Silhouette Coefficient")
plt.show()
