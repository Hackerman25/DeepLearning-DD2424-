import numpy as np
from functions import *
import matplotlib.pyplot as plt
import itertools
import numpy as np


def LoadBatch(file):
    import pickle
    with open("Datasets/" + file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def readdata(databatch):  # function 1 (Read in and store the training, validation and test data

    Tdata = LoadBatch(databatch)
    dataTr = Tdata[b'data']
    dataTr = np.flipud(np.rot90(dataTr))

    labelsTr = Tdata[b'labels']

    return dataTr, labelsTr


def fixdata():
    dataTr1, labelsTr1 = readdata("data_batch_1")  # increase training data a)
    dataTr2, labelsTr2 = readdata("data_batch_2")
    dataTr3, labelsTr3 = readdata("data_batch_3")
    dataTr4, labelsTr4 = readdata("data_batch_4")
    dataTr5, labelsTr5 = readdata("data_batch_5")

    dataTr = np.concatenate((dataTr1, dataTr2, dataTr3, dataTr4, dataTr5), axis=1)
    labelsTr = np.concatenate((labelsTr1, labelsTr2, labelsTr3, labelsTr4, labelsTr5))

    return dataTr, labelsTr


def normalizedata(dataTr):  # function 2 (normalize the data by std and mean)
    dataTr = dataTr - np.mean(dataTr, axis=0)
    dataTr = dataTr / np.std(dataTr, axis=0)

    return dataTr


def create_Wb(col, row):  # function 3 initialize the parameters of the model W and b

    W = np.random.normal(0, 1 / np.sqrt(row), size=(col, row))
    b = np.zeros((col, 1))
    print("size w and b", np.shape(W),np.shape(b))
    return W, b





def EvaluateClassifierBonus(X, W, b):  # function 4  evaluates the network function,
    # CORRECT
    k = len(W) + 1
    s = [None] * (k)
    X_batch = [None] * (k-1)
    X_batch[0] = X


    for l in range(1, k - 1):
        s[l-1] = W[l-1] @ X_batch[l-1] + b[l-1]
        X_batch[l] = np.maximum(0, s[l-1])  # ReLu activation function  Xbatch^l gjord på l-1


    s[-1] = W[-1] @ X_batch[-1] + b[-1]
    #X_batch[-1] = np.maximum(0, s[0])  # s[0] ???? ReLu activation function  Xbatch^l gjord på l-1

    p = np.exp(s[-1]) / np.sum(np.exp(s[-1]), axis=0, keepdims=True)  # softmax? could be wrong

    return X_batch, p


def BatchNormalize(s_batch_l,my_l,v_l):
    part1 = np.diag(1 / np.sqrt(v_l + 1e-12))
    part2 = s_batch_l - my_l

    return np.dot(part1, part2)





def EvaluateClassifierBonusBN(X, W, b ,gamma, beta):  # function 4  evaluates the network function,
    # CORRECT
    k = len(W) + 1
    s = [None] * (k)
    X_batch = [None] * (k-1)
    X_batch[0] = X

    n = len(X)
    s_batch = [None] * n
    s_hat = [None] * n
    s_tilde = [None] * n
    my = [None] * (k-1)
    sigma = [None] * (k-1)



    for l in range(1, k - 1):


        s_batch[l-1] = W[l-1] @ X_batch[l-1] + b[l-1]


        my[l-1] = s_batch[l-1].mean(axis=1).reshape(s_batch[l-1].shape[0], 1) #correct
        sigma[l-1] = s_batch[l-1].var(axis=1).reshape(s_batch[l-1].shape[0])  #correct
        s_hat[l-1] = BatchNormalize(s_batch[l-1],my[l-1],sigma[l-1])   #correct

        s_tilde[l-1] = gamma[l-1] * s_hat[l-1]  + beta[l-1]  #correct

        #s[l-1] = W[l-1] @ X_batch[l-1] + b[l-1]
        X_batch[l] = np.maximum(0, s_tilde[l-1])  # ReLu activation function  Xbatch^l gjord på l-1

    #print("bing",np.shape(W[-1]) , np.shape( X_batch[-1] ), np.shape( b[-1]))

    s_batch[-1] = W[-1] @ X_batch[-1] + b[-1]
    #X_batch[-1] = np.maximum(0, s[-1])  # s[0] ???? ReLu activation function  Xbatch^l gjord på l-1

    p = np.exp(s_batch[-1]) / np.sum(np.exp(s_batch[-1]), axis=0, keepdims=True)  # softmax? could be wrong

    return X_batch, p, X_batch, s_batch, s_tilde, s_hat, my, sigma


def clcross(Y, p):  # korrekt confirmed
    return - Y * np.log(p)


def ComputeCostLoss(X, Y, W, b, lambda_):  # function 5    computes cost
    # works
    if BatchNorm:
        _, p,_,_,_,_,_,_= EvaluateClassifierBonusBN(X, W, b,gamma,beta)
    else:
        _, p = EvaluateClassifierBonus(X, W, b)

    lcross = clcross(Y, p)

    Wsum = sum((Wi ** 2).sum() for Wi in W)

    Jloss = (1 / X.shape[1]) * np.sum(lcross)  # cost wrong + cost Weights
    Jcost = Jloss + lambda_ * Wsum

    return Jcost, Jloss


def ComputeGradients(X, Y, W, lambda_):  # P and Y should be batch

    n_batch = X.shape[1]



    # forward pass
    X_batch, Pbatch = EvaluateClassifierBonus(X, W, b)

    # backward pass

    Gbatch = Pbatch - Y

    """
    

    
    gradientW2 = 1 / n_batch * Gbatch @ X_batch[-1].T + 2 * lambda_ * W[1]
    gradientb2 = 1 / n_batch * Gbatch @ np.ones((n_batch, 1))

    Gbatch = W[1].T @ Gbatch
    Gbatch = Gbatch * np.array(X_batch[-1] > 0)  # (Hbatch > 0).astype(float)#[Hbatch > 0]

    gradientW1 = 1 / n_batch * Gbatch @ X.T + lambda_ * W[0]
    gradientb1 = 1 / n_batch * Gbatch @ np.ones((n_batch, 1))

    """

    k = len(W)+1

    gradientWvect = [None] * (k-1)
    gradientbvect = [None] * (k-1)


    for l in range(k,2,-1):

        gradientWvect[l-2] = 1 / n_batch * Gbatch @ X_batch[l-2].T + 2 * lambda_ * W[l-2]


        gradientbvect[l-2] = 1 / n_batch * Gbatch @ np.ones((n_batch, 1))

        Gbatch = W[l-2].T @ Gbatch


        Gbatch = Gbatch * np.array(X_batch[l-2] > 0)  # (Hbatch > 0).astype(float)#[Hbatch > 0]

    gradientWvect[0] = 1 / n_batch * Gbatch @ X_batch[0].T # + lambda_ * W[0]
    gradientbvect[0] = 1 / n_batch * Gbatch @ np.ones((n_batch, 1))

    return gradientWvect, gradientbvect


def BatchNormBackPass(Gbatch,s_batch,my, sigma):
    n_batch = s_batch.shape[1]

    eps = 1e-12



    sigma1 = ((sigma + eps) ** -0.5).T  # ok
    sigma2 = ((sigma + eps) ** -1.5).T  # ok
    G1 = Gbatch * np.reshape(sigma1, (sigma1.shape[0], 1))  # 1_n^T = reshape
    G2 = Gbatch * np.reshape(sigma2, (sigma2.shape[0], 1))  # ok
    Dmine = s_batch - my           #ok

    c = G2 * Dmine

    Gbatch = G1 - 1 / n_batch * np.sum(G1, axis=1, keepdims=True) - \
    1 / n_batch * Dmine * np.sum(c, axis=1, keepdims=True)  # 1_n * 1_n^T = sum



    return Gbatch


def ComputeGradientsBN(Xbatch, Y, W, lambda_, gamma, beta):  # P and Y should be batch

    n_batch = Xbatch.shape[1]



    # forward pass
    X_batch, Pbatch, X_batch, s_batch, s_tilde, s_hat, my, sigma =\
        EvaluateClassifierBonusBN(Xbatch, W, b,gamma, beta)




    # backward pass
    gradientWvect = [None] * (k)
    gradientbvect = [None] * (k)
    gradientGammavect = [None] * (k)
    gradientBetavect  = [None] * (k)

    Gbatch = Pbatch - Y


    # The gradients of J w.r.t. bias vector bk and Wk
    gradientWvect[-1] = 1 / n_batch * Gbatch @ X_batch[-1].T + 2 * lambda_ * W[-1]
    gradientbvect[-1] = 1 / n_batch * Gbatch @ np.ones((n_batch, 1))

    # Propagate Gbatch to the previous layer ... workINprogress

    Gbatch = W[-1].T @ Gbatch
    Gbatch = Gbatch * np.array(X_batch[-1] > 0)





    for l in range(k-1,0,-1):

        #Compute gradient for the scale and offset parameters for layer l
        gradientGammavect[l-1] = 1 / n_batch * (( Gbatch * s_hat[l-1] ) @ np.ones((n_batch, 1))) #correct????
        gradientBetavect[l-1] = 1 / n_batch * Gbatch @ np.ones((n_batch, 1))

        #Propagate the gradients through the scale and shift
        Gbatch = Gbatch * (gamma[l-1] @ np.ones((1, n_batch)))

        #Propagate Gbatch through the batch normalization
        Gbatch = BatchNormBackPass(Gbatch,s_batch[l-1],my[l-1], sigma[l-1])

        #The gradients of J w.r.t. bias vector bl and Wl
        gradientWvect[l-1] = 1 / n_batch * Gbatch @ X_batch[l-1].T + 2* lambda_ * W[l-1]
        gradientbvect[l-1] = 1 / n_batch * Gbatch @ np.ones((n_batch, 1))

        #If l > 1 propagate Gbatch to the previous layer
        if l > 1:
            Gbatch = W[l-1].T @ Gbatch
            Gbatch = Gbatch * np.array(X_batch[l-1] > 0)

    return gradientWvect, gradientbvect, gradientGammavect, gradientBetavect


def compute_grads_num(X, Y, lambda_, h):
    grad_W = [None] * len(W)
    grad_b = [None] * len(W)
    for i in range(len(W)):
        grad_W[i] = np.zeros(W[i].shape, dtype=np.float32)
        grad_b[i] = np.zeros(b[i].shape, dtype=np.float32)

    c, _ = ComputeCostLoss(X, Y, W, b, lambda_);

    for j in range(0, len(W)-1):
        b_try = np.copy(b[j])
        for i in range(len(b[j])):
            b[j] = np.array(b_try)
            b[j][i] += h
            c2, _ = ComputeCostLoss(X, Y, W, b, lambda_)
            grad_b[j][i] = (c2 - c) / h
        b[j] = b_try

        W_try = np.copy(W[j])

        for i in np.ndindex(W[j].shape):
            W[j] = np.array(W_try)
            W[j][i] += h
            c2, _ = ComputeCostLoss(X, Y, W, b, lambda_)
            grad_W[j][i] = (c2 - c) / h
        W[j] = W_try

    gradientWvect = [None] * len(W)
    gradientbvect = [None] * len(W)


    for i in range(len(W)):
        gradientWvect[i] = grad_W[i]
        gradientbvect[i] = grad_b[i]
    return gradientWvect , gradientbvect




def ComputeAccuracy(X, y, W, b,gamma,beta,BatchNorm):  # KORREKT function 6 compute accuracy on evaluation
    """
    s1 = W[0] @ X + b[0]
    h = np.maximum(0, s1)  # ReLu activation function  Xbatch^l gjord på l-1
    s = W[1] @ h + b[1]

    ypredict = np.argmax(s, axis=0)

    acc = np.sum(ypredict == y) / len(y)
    """
    if BatchNorm:
        _, p, _, _, _, _, _, _ = EvaluateClassifierBonusBN(X, W, b,gamma,beta)
    else:
        _, p = EvaluateClassifierBonus(X, W, b)
    ypredict = np.argmax(p, axis=0)
    acc = np.sum(ypredict == y) / len(y)

    return acc


def visualisegraph(trainlossJ, validationlossJ, updatesteps, label):  # plots loss/cost

    plt.plot(updatesteps, trainlossJ, label="train " + label)
    plt.plot(updatesteps, validationlossJ, label="validation " + label)
    plt.legend()

    plt.xlabel('Update step')
    plt.ylabel(label)
    plt.show()


def Cyclicalheta(eta_min, eta_max, ns, t):  # Exercise 3: cyclical learning rates

    t_temp = t % (2 * ns)
    if t_temp <= ns:
        eta = eta_min + t_temp / ns * (eta_max - eta_min)
    elif t_temp <= 2 * ns:
        eta = eta_max - (t_temp - ns) / ns * (eta_max - eta_min)

    t = (t + 1)

    return eta, t


def TrainMiniBatch(y, X, Y, X_val, y_val, W, b, t,gamma, beta, BatchNorm, batch_s, n_epochs, lambda_, eta_min, eta_max, ns, plotpercycle):
    [trainCostJ, trainLossJ] = [], []
    [validationCostJ, validationLossJ] = [], []
    updatesteps = []
    [acctrainlist, accvallist] = [], []

    n_batch = int(X.shape[0] / batch_s)

    eta = eta_min
    for epoch in range(n_epochs): # in range(n_epochs):

        print("Epoch number:", epoch, "  accuracy: ", round(ComputeAccuracy(X, y, W, b,gamma,beta,BatchNorm), 4), "LR eta: ", round(eta, 4))

        for j in range(2, int(X.shape[0] / n_batch)):
            j_start = (j - 1) * n_batch
            j_end = j * n_batch

            Xbatch = X[:, range(j_start, j_end)]                 # @ Sample a batch of the training data
            Ybatch = Y[:, range(j_start, j_end)]

            if BatchNorm:
                [gradientWvect, gradientbvect, gradientGammavect, gradientBetavect] = ComputeGradientsBN(Xbatch, Ybatch, W, lambda_, gamma, beta,)     #Forward propagate and Backward to calc gradient
            else:

                [gradientWvect, gradientbvect] = ComputeGradients(Xbatch, Ybatch, W,lambda_)     #Forward propagate and Backward to calc gradient

            if BatchNorm:
                for i in range(len(W)):
                    gradientWvect[i] = gradientWvect[i].reshape(-1, gradientWvect[i].shape[-1])

                    W[i] = W[i] - eta * gradientWvect[i]  # Update the parameters using the gradient.
                    b[i] = b[i] - eta * gradientbvect[i]
                for i in range(len(W)-1):
                    gamma[i] = gamma[i] - eta * gradientGammavect[i]
                    beta[i] =  beta[i] - eta * gradientBetavect[i]
            else:
                for i in range(len(W)):
                    gradientWvect[i] = gradientWvect[i].reshape(-1, gradientWvect[i].shape[-1])

                    W[i] = W[i] - eta * gradientWvect[i]                     # Update the parameters using the gradient.
                    b[i] = b[i] - eta * gradientbvect[i]

            [eta, t] = Cyclicalheta(eta_min, eta_max, ns, t)

            if t % int(
                    np.floor(ns / plotpercycle * 2)) == 0 or t == 1 or t == batch_s * n_epochs:  # 9 is numb per cycle
                [cost, loss] = ComputeCostLoss(X, Y, W, b, lambda_)
                validationCostJ.append(cost), validationLossJ.append(loss)
                [cost, loss] = ComputeCostLoss(X_val, Y_val, W, b, lambda_)
                trainCostJ.append(cost), trainLossJ.append(loss)
                updatesteps.append(t)

                # print("cost: ", round(trainCostJ[-1], 3), "loss: ", round(trainLossJ[-1], 3))
                acctrainlist.append(ComputeAccuracy(X, y, W, b,gamma,beta,BatchNorm))
                accvallist.append(ComputeAccuracy(X_val, y_val, W, b,gamma,beta,BatchNorm))

            # print("acc on validation", accvallist[-1])

        idx = np.random.permutation(X.shape[1])
        X = X[:, idx]
        Y = Y[:, idx]
        y = y[idx]

    if BatchNorm:
        return W, b,gamma,beta, trainCostJ, validationCostJ, trainLossJ, validationLossJ, updatesteps, acctrainlist, accvallist, eta
    else:
        return W, b,None,None, trainCostJ, validationCostJ, trainLossJ, validationLossJ, updatesteps, acctrainlist, accvallist, eta


def gridsearch(model, parameters):
    global W
    global b

    bestacc = 0
    bestparams = 0

    for combination in itertools.product(*parameters.values()):
        batch_s, n_epochs, lambda_, eta_min, eta_max, ns, plotpercycle = combination
        [W, b, trainCostJ, validationCostJ, trainLossJ, validationLossJ, updatesteps, acctrainlist, accvallist, eta] = \
            model(y, X, Y, X_val, y_val, W, b, t, batch_s, n_epochs, lambda_, eta_min, eta_max, ns, plotpercycle)

        Accuracy = ComputeAccuracy(X_val, y_val, W, b)

        if Accuracy > bestacc:
            bestacc = Accuracy
            bestparams = combination
        print("acc: ", Accuracy, "with parameters: ", combination, "end eta: ", eta)

    return [bestacc, bestparams, W, b, trainCostJ, validationCostJ, trainLossJ, validationLossJ, updatesteps,
            acctrainlist, accvallist]


def randomsearch(model, parameters, randiterations):
    global W
    global b

    bestacc = 0
    bestparams = 0

    for i in range(randiterations):
        randlistparm = []
        for e in range(len(parameters)):
            listofparameters = list(parameters.values())[e]
            if len(listofparameters) > 1:
                randlistparm.append(
                    listofparameters[0] + (listofparameters[1] - listofparameters[0]) * np.random.rand())
            else:
                randlistparm.append(listofparameters[0])

        batch_s, n_epochs, lambda_, eta_min, eta_max, ns, plotpercycle = randlistparm

        [W, b, trainCostJ, validationCostJ, trainLossJ, validationLossJ, updatesteps, acctrainlist, accvallist, eta] = \
            model(y, X, Y, X_val, y_val, W, b, t, batch_s, n_epochs, lambda_, eta_min, eta_max, ns, plotpercycle)

        Accuracy = ComputeAccuracy(X_val, y_val, W, b)
        print("acc: ", Accuracy, "with parameters: ", randlistparm, "end eta: ", eta)
        if Accuracy > bestacc:
            bestacc = Accuracy
            bestparams = randlistparm

    return [bestacc, bestparams, W, b, trainCostJ, validationCostJ, trainLossJ, validationLossJ, updatesteps,
            acctrainlist, accvallist]


# Exercise 1


def testgradients():
    gradientWvect_num, gradientbvect_num= compute_grads_num(X[:,0:1], Y[:,0:1],lambda_=0, h=0.001)
    gradientWvect, gradientbvect = ComputeGradients(X[:,0:1], Y[:,0:1], W, lambda_=0)

    for e in range(0,len(W)):
        print(e,"@@@@@@@@@@@")

        print("Error W compare to numerical",gradientWvect_num[e]-gradientWvect[e])
        print("Error b compare to numerical",gradientbvect_num[e] - gradientbvect[e])


if __name__ == "__main__":
    BatchNorm = True

    [dataTr, labelsTr] = fixdata()

    dataVa = dataTr[:, -1000:]
    labelsVa = labelsTr[-1000:]
    dataTr = dataTr[:, :-1000]
    labelsTr = labelsTr[:-1000]
    dataTrN = normalizedata(dataTr)
    dataVaN = normalizedata(dataVa)

    # creating layers neural network


    m, d = 10, dataTr.shape[0]
    Kend = len(np.unique(np.array(labelsTr)))  # K = probabilities so 10

    K1 = 20
    K2 = 25
    #K3 = 20


    K = [K1,K2]

    W1, b1 = create_Wb(K1, d)  # m = 50 numb hidden layers , d = 3072

    W2, b2 = create_Wb(K2, K1)

    W3, b3 = create_Wb(Kend, K2)

    #W4, b4 = create_Wb(k, K3)

    W = [W1, W2, W3]
    b = [b1, b2, b3]


    k = len(W)
    gamma = []
    beta = []
    if BatchNorm:
        for i in range(k-1):
            gamma.append(np.ones((K[i],1)))
            beta.append(np.zeros((K[i],1)))   #GABAGO









    X = dataTrN
    y = np.array(labelsTr)  # reshape to 1,10000 and make array
    Y = np.eye(len(np.unique(y)))[y].T  # onehotencoded

    X_val = dataVaN
    y_val = np.array(labelsVa)
    Y_val = np.eye(len(np.unique(y_val)))[y_val].T  # onehotencoded

    t = 0
    tcount = 0
    # Exercise 2


    # TEST for if gradients are correct
    #testgradients()


    # gridsearch

    """
    parameters = {"batch_s": [100], "n_epochs": [20],"lambda_": [0.1,0.05,0.01,0.005,0.001,0.0005], "eta_min": [1e-5],"eta_max": [1e-1],"ns": [500], "plotpercycle": [10]}
    [bestacc, bestparams, W, b,trainCostJ,validationCostJ,trainLossJ, validationLossJ,updatesteps,acctrainlist,accvallist] = gridsearch(TrainMiniBatch, parameters)
    print("best acc from gridsearch: ", bestacc, "bestparams: ", bestparams)
    """

    # random search

    """
    parameters = {"batch_s": [100], "n_epochs": [20],
                  "lambda_": [0.005,0.0005], "eta_min": [1e-5],"eta_max": [1e-1],"ns": [500], "plotpercycle": [10]}
    [bestacc, bestparams, W, b,trainCostJ,validationCostJ,trainLossJ, validationLossJ,updatesteps,acctrainlist,accvallist] \
        = randomsearch(TrainMiniBatch, parameters,randiterations = 10)
    print("best acc from randomsearch: ", bestacc, "bestparams: ", bestparams)
    """

    # best value

    [W, b, gamma,beta, trainCostJ, validationCostJ, trainLossJ, validationLossJ, updatesteps, acctrainlist, accvallist,
     eta] = TrainMiniBatch(y, X, Y, X_val, y_val, W, b, t,gamma, beta, BatchNorm,
                           batch_s=100, n_epochs=30, lambda_=0.0045105, eta_min=1e-5, eta_max=1e-1, ns=500,
                           plotpercycle=10)
    Accuracy = ComputeAccuracy(X_val, y_val, W, b,gamma,beta,BatchNorm)
    print("best acc from randomsearch: ", Accuracy)



    # plot
    visualisegraph(trainCostJ, validationCostJ, updatesteps, "cost")
    visualisegraph(trainLossJ, validationLossJ, updatesteps, "loss")
    visualisegraph(acctrainlist, accvallist, updatesteps, "accuracy")