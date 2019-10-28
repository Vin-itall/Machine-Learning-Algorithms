import random
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

mat = scipy.io.loadmat('Resources/AllSamples.mat')
samples = list(np.array(mat['AllSamples']))
slist = []
sxlist = []
x = []
y = []
objPoints = []
centroids2 = []

'''Preprocessing Dataset'''

for i in range(len(samples)):
    coord = []
    x.append(samples[i][0])
    y.append(samples[i][1])
    coord.append(samples[i][0])
    coord.append(samples[i][1])

    slist.append(coord)

slist = np.array(slist)
sxlist = slist
"""Random seed has been commented, uncomment and insert 
    a same no. across different programs to compare results"""
    #random.seed()
idexes = random.sample(set(range(len(samples))), 1)
centroids2.append(samples[idexes[0]])

'''Generating Centroids'''

def genMaxCentroids(k):
    centroids2.append(getMaxSample(centroids2))
    print("Random Centroids Selected")
    print(np.array(centroids2))
    return centroids2

'''Getting Centroid With Maximum Average Distance From Every Other Centroid'''

def getMaxSample(ce):
    point = -1
    distM = []
    for i in range(len(sxlist)):
        dist = 0
        for j in range(len(ce)):
            cent = np.array(ce[j])
            dist += euclideanDistance(sxlist[i], cent)
        dist = dist/len(ce)
        tup = (i, dist)
        distM.append(tup)
    distM = sorted(distM, key=lambda x: x[1], reverse=True)
    for i in range(len(distM)):
        c = distM[i][0]
        if c not in idexes:
            point = distM[i][0]
            idexes.append(point)
            break
    return sxlist[point]

'''Computes Euclidean Distance Between Two Given Points'''

def euclideanDistance(point, centroid):
    return np.sqrt(np.power(point[0] - centroid[0], 2) + np.power(point[1] - centroid[1], 2))

'''Updating Centroid Value By Taking Average Over All The Points Ih It's Cluster'''

def computeCentroids(samplist):
    x = y = 0
    coord = []
    for i in range(len(samplist)):
        x += samplist[i][0]
        y += samplist[i][1]
    x = x / len(samplist)
    y = y / len(samplist)
    coord.append(x)
    coord.append(y)
    return coord

'''Checking If Previous Centroids Are The Same As Previous For Convergence Of Clusters'''

def converged(cent, new_cent):
    dist = 0
    for i in range(len(cent)):
        dist += euclideanDistance(cent[i], new_cent[i])
    if(dist == 0):
        return True
    else:
        return False

'''Computes Objective Function Value For A Value Of K'''

def objectiveFunction(clust, cent):
    dist = 0
    for i in range(len(cent)):
        if not clust[i]:
            dist = 0
        else:
            for j in range(len(clust[i])):
                dist += (euclideanDistance(cent[i], clust[i][j])) ** 2
    return dist

'''Performs KMeans Clustering Over Given List Of Samples And Centroids'''
def Kmeans(slist, centroids):
    new_centroids = centroids.copy()
    while (True):
        clusters = {}
        for i in range(len(slist)):
            min = 1000000000
            cb = -1
            for j in range(len(centroids)):
                distance = euclideanDistance(slist[i], centroids[j])
                if (distance < min):
                    min = distance
                    cb = j
            if (cb not in clusters.keys()):
                clusters[cb] = []
            clusters[cb].append(list(slist[i]))
        for k in range(len(clusters)):
            if not clusters[k]:
                continue
            else:
                new_centroids[k] = computeCentroids(clusters[k])
        if converged(centroids, new_centroids):
            break
        centroids = new_centroids.copy()
    return clusters, centroids


'''Following is the loop where we call k means for different values of k and plot it's clusters and compute objective function'''

for k in range(2, 11):
    print("Value of K: " + str(k))
    centroids = np.array(genMaxCentroids(k))
    clusters, centroid= Kmeans(slist, centroids)
    print("Converged Centroids")
    print(centroid)
    coord = []
    x = k
    y = objectiveFunction(clusters, centroid)
    coord.append(x)
    coord.append(y)
    objPoints.append(coord)

    """ Following is the code to print cluster plots, Uncomment to plot clusters"""

    # color = ['yellow', 'm', 'tab:olive', 'tab:cyan', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
    #          'tab:brown']
    # for i in range(len(clusters)):
    #     plotter = np.array(clusters[i])
    #     plt.scatter(plotter[:, 0], plotter[:, 1], marker='^', c=color[i])
    #
    # plt.scatter(centroid[:, 0], centroid[:, 1], marker='s', c='black')
    # plt.show()
    print("----------------------------------------------------------")

'''Following code plots the objective function'''

objPoints = np.array(objPoints)
print("Values Of K and Objective Function Values")
print(objPoints)
plt.plot(objPoints[:, 0], objPoints[:, 1], 'k', marker='o')
plt.ylabel('Objective Function Value')
plt.xlabel('K Value')
plt.show()
