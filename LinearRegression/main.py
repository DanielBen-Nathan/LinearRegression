import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def h(X,theta):
    return np.dot(X,theta)

def cost(X,y,theta):
    m = y.shape[0]#length of y (num training examples)

    J = 0
    J = (1 / (2 * m)) * np.sum(np.square(h(X,theta) - y))
    return J

def gradientDescent(X,y,theta,alpha,its):
    costs=[]


    m = y.shape[0]  # length of y (num training examples)

    for it in range(0,its):


        sumPart = np.sum(np.multiply((h(X,theta) - y),X),axis=0)
        sumPart = np.reshape(sumPart,(1,3))

        theta = theta - (alpha/m) * sumPart.T
        j = cost(X, y, theta)


        costs.append(j)
    return theta,costs

def plotCost(costs):
    plt.plot(range(len(costs)), costs)
    plt.grid(True)
    plt.xlabel("Iteration")
    plt.ylabel("Cost function")
    plt.show()

def featureNormalisation(X):

    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])

    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X = (X - mu) / sigma
    return X

df = pd.read_csv('ex1data2.txt')
X = df.values[:,0:2]
y = df.values[:,2:3]
X = featureNormalisation(X)

#X=np.matrix([np.ones(X.shape[1], dtype='int64'),X])
ones = np.ones((X.shape[0],1), dtype='int64')

X = np.insert(X,0,1,axis=1)
#print(X.shape)
#X = np.concatenate((ones, X))
#X=np.matrix([ones],[X])

#print(X)
theta = np.zeros((X.shape[1],1), dtype='int64')


theta,costs = gradientDescent(X,y,theta,0.01,500)
print("final theta")
print(theta)

plotCost(costs)