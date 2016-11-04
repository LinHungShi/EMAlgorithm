import numpy as np
from numpy import linalg
import sys, os
import pandas as pd
from scipy import stats
import matplotlib 
from matplotlib import pyplot as plt
import copy

NUMS_OF_MIS_PICK = 3
NUMS_OF_AMBIG_PICK = 3
W_PICKUP_ITERATIONS = np.array([0, 4, 9, 24, 49, 99])
NEW_MAX = 256
NEW_MIN = 0
IMG_SIZE = 28
def eStep(X, y, w, w_std, e):
    obs = y.size
    for i in range(obs):
        tmp = np.dot(X[i,:], w)
        ndp = stats.norm.pdf(-tmp/w_std)
        ndc =  stats.norm.cdf(-tmp/w_std)
        if y[i] == 1:
            e[i] = tmp + w_std * (ndp / (1 - ndc))
        else:
            e[i] = tmp + w_std * (-ndp) / ndc

def mStep(X, y_variance, e, w_std, w):
    obs,space = X.shape
    tmp1 = np.zeros((space, space))
    for i in range(obs):
        tmp1 = tmp1 + np.outer(X[i,:], X[i,:]) / (w_std**2)
    tmp2 = np.dot(X.T, e) / (w_std**2)
    w[:] = np.dot(linalg.inv((1 / y_variance) + tmp1), tmp2)

def computeMAP(X, y, w, y_variance, w_std):
    obs, space = X.shape
    tmp1 = (space / 2) * np.log(1 / (y_variance * 2 * np.pi)) - np.dot(w.T, w) / (2 * y_variance)
    tmp2 = np.dot(y.T, np.log(stats.norm.cdf(np.dot(X, w) / w_std)))
    tmp3 = np.dot((1 - y).T, np.log(1 - stats.norm.cdf(np.dot(X, w) / w_std)))
    return tmp1 + tmp2 + tmp3
    print "tmp1: {0},  tmp2: {1}, tmp3: {2}".format(tmp1, tmp2, tmp3)
    
def computeEM(X, y, y_variance, w_std, iteration, w_pickup_iteration):
    
    ## Apply EM algorithm to update w and E(phi_n) alternatively for iteration times
    ## Return the objective function, maximum a posteriori and ws that are updated during the training
    obs,space = X.shape
    wmap = np.zeros(space)
    e = np.zeros(obs)
    maps = np.zeros(iteration)
    num_w = w_pickup_iteration.size
    W = np.zeros((space, num_w))
    index = 0;
    for i in range(iteration):
        
        # In eStep, we calculate E(phi_n) for all n and store values in e
        eStep(X, y, wmap, w_std, e)    
        # In mStep, we update w and store the value in w
        mStep(X, y_variance, e, w_std, wmap)
        maps[i] = computeMAP(X, y, wmap, y_variance, w_std)
        if (i+1) % 100 == 0:
            print "maps: ", maps
        if i in w_pickup_iteration:
            print "store new w in W"
            W[:, index] = wmap
            index = index + 1  
    return (maps, wmap, W)

def normW(W, new_max, new_min):
    space, obs = W.shape
    for i in range(obs):
        W[:,i] = (W[:, i] - np.min(W[:,i])) * (new_max - new_min) / (np.max(W[:, i]) - np.min(W[:, i])) + new_min

def probit(X, w):
    return stats.norm.cdf(np.dot(X, w))

def getConfMat(y, predict):
    mat = np.zeros((2,2))
    obs = y.size
    for i in range(obs):
        if y[i] == predict[i] and y[i] == 1:
            mat[1][1] = mat[1][1] + 1
        elif y[i] == predict[i] and y[i] == 0:
            mat[0][0] = mat[0][0] + 1
        elif y[i] != predict[i] and y[i] == 1:
            mat[1][0] = mat[1][0] + 1
        else:
            mat[0][1] = mat[0][1] + 1
    return mat

def pickMissClassified(y, predict, num):
    obs = y.size
    index = np.empty(0, dtype = np.int32)
    for i in range(obs):
        if index.size == num:
            return index
        elif y[i] != predict[i]:
            index = np.append(index, i)

def pickAmbigProb(prob, num):
    obs = prob.size
    tmp = copy.copy(prob)
    index = np.empty(0, dtype = np.int32)
    for i in range(obs):
        if index.size == num:
            return index
        else:
            ind = np.argmin(abs(tmp - 0.5))
            index = np.append(index, ind)
            tmp[ind] = -1
            
def constructImg(T, X):
    X2 = np.dot(T, X)
    nums = X2.shape[1]
    imgs = np.ndarray((nums, IMG_SIZE, IMG_SIZE)).astype(np.float32)
    for i in range(nums):
        imgs[i,:,:] = X2[:,i].reshape(IMG_SIZE, IMG_SIZE)
    return imgs
def saveImg(X, title, name):
    plt.imshow(X, cmap = 'gray')
    plt.title(title)
    plt.savefig(name)
    
def hw2(y_variance, w_std, iteration):
    ## Read data
    X = pd.read_csv("hw2_data_csv/Xtrain.csv")
    y = pd.read_csv('hw2_data_csv/ytrain.csv')
    X_test = pd.read_csv("hw2_data_csv/Xtest.csv")
    y_test = pd.read_csv('hw2_data_csv/ytest.csv')
    Q = pd.read_csv('hw2_data_csv/Q.csv', header = None)
    T = np.array(Q)
    
    Xtrain = np.array(X).astype(np.float32)
    ytrain = np.array(y).astype(np.float32)
    Xtest = np.array(X_test).astype(np.float32)
    ytest = np.array(y_test).astype(np.float32)
    # Run EM Algorithm
    # maps is the array records ln(p(y,a|w,y))
    # y_variance is 1/lambda and w_std is sigma
    maps, wmap, W = computeEM(Xtrain, ytrain, y_variance, w_std, iteration, W_PICKUP_ITERATIONS)
    
    plt.plot(range(iteration), maps)
    plt.xlabel('Iteration')
    plt.ylabel('ln(p(y,w|X))')
    plt.savefig('Error function.png')
    
    print "before W:\n", W
    normW(W, NEW_MAX, NEW_MIN)
    print "normalized W: \n", W
    
    prob = probit(Xtest, wmap)
    predict = np.int32(prob > 0.5)
    
    ## Get Confusion matrix for new test data
    confusion = getConfMat(ytest, predict)
    print "Confusion Matrix:\n", confusion
    
    error_rate = (confusion[0][0] + confusion[1][1]) / np.sum(confusion)
    print "error_rate = {0}".format(error_rate)
    
    ## Pick up miss classified digits and most ambiguous digits, then reconstruct them
    miss_index = pickMissClassified(ytest, predict, NUMS_OF_MIS_PICK)
    miss_imgs = constructImg(Q, Xtest[miss_index,:].T)
    for i in range(NUMS_OF_MIS_PICK):
        name = "miss_classified-{0}.png".format(i)
        title = "y[{0}] = {1} and Probability = {2}".format(miss_index[i], ytest[miss_index[i]], prob[miss_index[i]])
        saveImg(miss_imgs[i,:,:], title, name)
        
    ambig_index = pickAmbigProb(prob, NUMS_OF_AMBIG_PICK)
    ambig_imgs = constructImg(Q, Xtest[ambig_index,:].T)
    for i in range(NUMS_OF_AMBIG_PICK):
        name = "most_ambig-{0}.png".format(i)
        title = "y[{0}] = {1} and Probability = {2}".format(ambig_index[i], ytest[ambig_index[i]], prob[ambig_index[i]])
        saveImg(ambig_imgs[i,:,:], title,  name)
    W_img = constructImg(T, W)
    for i in range(W_PICKUP_ITERATIONS.size):
        name = "Reconstructed_W({0}).png".format(W_PICKUP_ITERATIONS[i])
        title = "Reconstructed W"
        saveImg(W_img[i,:,:], title, name)
    return (W, W_img)

if __name__ == '__main__':
    print os.getcwd()
    hw2(1, 1.5, 100)