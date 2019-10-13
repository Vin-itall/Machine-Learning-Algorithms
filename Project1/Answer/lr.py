import scipy.io
import numpy as np
from pip._vendor.colorama import Fore

Digits = scipy.io.loadmat('./Resources/mnist_data.mat')
# loadingdata
trX = Digits['trX'];
trY = Digits['trY'];
tsX = Digits['tsX'];
tsY = Digits['tsY'];

means = np.zeros(12116)
stds = np.zeros(12116)

#Extracting features
for i in range(0,12116):
    img = trX[i].reshape((28, 28))
    means[i] = img.mean()
    stds[i] = img.std()

new_col = np.ones(12116)
X_train = np.column_stack((new_col.T, means, stds))
theta = np.zeros((3, 1))
Y_train = trY.T

meanE = np.zeros(2002)
stdE = np.zeros(2002)


for i in range(0,2002):
    img = tsX[i].reshape((28, 28))
    meanE[i] = img.mean()
    stdE[i] = img.std()

new_col2 = np.ones(2002)
Xs = np.column_stack((new_col2.T, meanE, stdE))
Xs = np.array(Xs)
Ys = tsY.T

def sigmoid(x,th):
    z=np.dot(x,th)
    return (1/(1+np.exp(-z)))

def gradient(x,y,th):
    m=x.shape[0]
    h = sigmoid(x,th)
    error = y-h
    dJ = np.dot(x.T, error)
    return dJ

def update_parameters(th,lr,gradient):
    return th + (lr * gradient)


def logReg(x,y,param,num_iter,lr):
    for i in range(iter):
        grad = gradient(X_train, Y_train, param)
        param = update_parameters(param, lr, grad)
    return param


def accuracy_lr(x, th):
    count=0.0
    count_7 =0.0
    count_8 =0.0
    h = sigmoid(x, th)
    for i in range(2002):
        if h.item(i) >= 0.5:
            label = 1.0
        else:
            label = 0.0
        if label == Ys.item(i) and Ys.item(i) == 1.0:
            count += 1
            count_7 +=1
        elif label == Ys.item(i) and Ys.item(i) == 0.0:
            count += 1
            count_8 += 1
        accu_7 = (count_7 / 1028) * 100
        accu_8 = (count_8 / 974) * 100
        Totaccuracy = (count / 2002) * 100
    return accu_7,accu_8,Totaccuracy


iter = 38000
lr = 0.00088

param = logReg(X_train,Y_train,theta,iter,lr)
print(Fore.GREEN + "Learned parameters after " + str(iter) + " iterations and" + "learning rate = " + str(lr))
print(Fore.CYAN + "Parameters :" + str(param))

acc_7,acc_8,totacc = accuracy_lr(Xs,param)

print(Fore.RED + "Total accuracy : " + str(round((totacc),2)) + "%")
print(Fore.RED + "Accuracy for class 7 : " + str(round((acc_7),2)) + "%")
print(Fore.RED + "Accuracy for class 8 : " + str(round((acc_8),2)) + "%")



