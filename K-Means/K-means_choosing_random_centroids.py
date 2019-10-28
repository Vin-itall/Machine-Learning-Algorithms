import random
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

mat = scipy.io.loadmat('Resources/AllSamples.mat')
samples = list(np.array(mat['AllSamples']))
slist = []
x = []
y = []
objPoints = []
centroids = []


'''Preprocessing Dataset'''

for i in range(len(samples)):
    coord = []
    x.append(samples[i][0])
    y.append(samples[i][1])
    coord.append(samples[i][0])
    coord.append(samples[i][1])
    slist.append(coord)
slist = np.array(slist)

'''Generating Centroids'''

def genCentroids(k):
    """Random seed has been commented, uncomment and insert
    a same no. across different programs to compare results"""
    #random.seed()
    centroid = []
    idexes = random.sample(set(range(len(samples))), k)
    for i in idexes:
        centroid.append(samples[i])
    print("Random Centroids Selected")
    print(np.array(centroid))
    return centroid

'''Computes Euclidean Distance Between Two Given Points'''

def euclideanDistance(point, centroid):
    return np.sqrt(np.power(point[0] - centroid[0], 2) + np.power(point[1] - centroid[1], 2))

'''Updating Centroid Value By Taking Average Over All The Points In It's Clusters'''

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
    centroids = np.array(genCentroids(k))
    clusters, centroid = Kmeans(slist, centroids)
    print("Converged Centroids")
    print(centroid)
    coord = []
    x = k
    y = objectiveFunction(clusters, centroid)
    coord.append(x)
    coord.append(y)
    objPoints.append(coord)
    ''' Following is the code to print cluster plots, Uncomment to plot clusters'''

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
