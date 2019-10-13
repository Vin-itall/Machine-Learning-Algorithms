import scipy.io
import numpy as np
from pip._vendor.colorama import Fore

Digits = scipy.io.loadmat('./Resources/mnist_data.mat')
# loadingdata
trX = Digits['trX'];
trY = Digits['trY'];
tsX = Digits['tsX'];
tsY = Digits['tsY'];

mean7 = np.zeros(6265)
std7 = np.zeros(6265)
mean8 = np.zeros(5851)
std8 = np.zeros(5851)
means = np.zeros(12116)
stds = np.zeros(12116)

#Extracting features
for i in range(0,12116):
    img = trX[i].reshape((28, 28))
    means[i] = img.mean()
    stds[i] = img.std()
    if i < 6265:
        mean7[i] = img.mean()
        std7[i] = img.std()
    else:
        mean8[i - 6265] = img.mean()
        std8[i - 6265] = img.std()
m7 = mean7.mean()
sm7 = std7.mean()
means_7 = np.array([m7, sm7])
m8 = mean8.mean()
sm8 = std8.mean()
means_8 = np.array([m8, sm8])
cov_7 = np.cov(mean7, std7)
cov_8 = np.cov(mean8, std8)

#coverting covariance matrix to diagonal covariance matrix.
#since we are assuming the two features to be indpendent.
cov_7 = np.diag(np.diag(cov_7))
cov_8 = np.diag(np.diag(cov_8))

print(Fore.GREEN + "Means of features with label 0 :")
print(Fore.BLUE + str(means_7))
print(Fore.GREEN + "Covariance Matrix for features with label 0 :")
print(Fore.BLUE + str(cov_7))
print(Fore.GREEN + "Means of features with label 1 :")
print(Fore.BLUE + str(means_8))
print(Fore.GREEN + "Covariance Matrix for features with label 1 :")
print(Fore.BLUE + str(cov_8))

#bivariate normal distribution function
def bi_norm_dist(x, mu, cov):
    x_mu = x - mu
    cov_matrix = np.linalg.det(cov)
    solution = np.linalg.solve(cov,x_mu)
    return (1 / (np.sqrt(2 * np.pi) ** 2 * cov_matrix)) * np.exp(-(solution.T.dot(x_mu) / 2))



def accuracy(tY,tX):
    accu_count = 0
    count_7 =0
    count_8 =0
    for i in range(0,2002):
        img = tsX[i].reshape((28, 28))
        mean = img.mean()
        std = img.std()
        fv = np.array([mean, std])
        true_label = tsY.item(i)
        class_label=0
    #Probabilities
        P_C_7 = bi_norm_dist(fv, means_7, cov_7)*(6265/12116)
        P_C_8 = bi_norm_dist(fv, means_8, cov_8)*(5851/12116)
        if P_C_8 > P_C_7:
            class_label=1
        if true_label == class_label and true_label==1.0:
            accu_count +=1
            count_8 += 1
        elif true_label == class_label and true_label == 0.0:
            accu_count +=1
            count_7 += 1
        accu_7 = (count_7 / 1028) * 100
        accu_8 = (count_8 / 974) * 100
        accuracy_w = (accu_count / 2002) * 100
    return accuracy_w,accu_7,accu_8

accuracy_w,accu_7,accu_8 = accuracy(tsY,tsX)

print(Fore.RED + "Total accuracy : " + str(round((accuracy_w),2)) + "%")
print(Fore.RED + "Accuracy for class 7: " + str(round((accu_7),2)) + "%")
print(Fore.RED + "Accuracy for class 8: " + str(round((accu_8),2)) + "%")


