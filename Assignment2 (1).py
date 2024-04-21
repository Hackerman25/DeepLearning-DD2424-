import numpy as np
from functions import *
import matplotlib.pyplot as plt
import itertools
import numpy as np


def LoadBatch(file):
    import pickle
    with open("Datasets/"+ file, 'rb') as fo:
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

def create_Wb(col,row):  # function 3 initialize the parameters of the model W and b

    W = np.random.normal(0, 1/np.sqrt(row), size=(col, row))
    b = np.zeros((col, 1))

    return W, b



def EvaluateClassifierBonus(X, W, b):  # function 4  evaluates the network function,
    #CORRECT
    s1 = W[0] @ X + b[0]
    h = np.maximum(0,s1)   #ReLu activation function  Xbatch^l gjord på l-1
    s = W[1] @ h + b[1]

    p = np.exp(s)/np.sum(np.exp(s),axis =0, keepdims=True) #softmax? could be wrong

    return h,p






def clcross(Y, p):  # korrekt confirmed
    return - Y * np.log(p)

def ComputeCostLoss(X, Y, W, b, lambda_):  # function 5    computes cost
    #works
    _,p = EvaluateClassifierBonus(X, W, b)

    lcross = clcross(Y, p)


    Wsum = sum((Wi**2).sum() for Wi in W)

    Jloss = (1 / X.shape[1]) * np.sum(lcross)  # cost wrong + cost Weights
    Jcost = Jloss + lambda_ * Wsum

    return Jcost, Jloss





def ComputeGradients(X, Y, W, lambda_):  # P and Y should be batch

    n_batch = X.shape[1]

    # forward pass
    Hbatch, Pbatch = EvaluateClassifierBonus(X, W, b)

    # backward pass

    Gbatch = Pbatch - Y
    gradientW2 = 1 / n_batch * Gbatch @ Hbatch.T + 2*lambda_ * W[1]
    gradientb2 = 1 / n_batch * Gbatch @ np.ones((n_batch,1))


    Gbatch = W[1].T @ Gbatch
    Gbatch = Gbatch * np.array(Hbatch > 0)#(Hbatch > 0).astype(float)#[Hbatch > 0]

    gradientW1 = 1 / n_batch * Gbatch @ X.T + lambda_*W[0]
    gradientb1 = 1 / n_batch * Gbatch @ np.ones((n_batch,1))

    return gradientW1,gradientb1,gradientW2,gradientb2










def compute_grads_num(X, Y, lambda_, h):

    grad_W = [np.zeros(W[0].shape, dtype=np.float32), np.zeros(W[1].shape, dtype=np.float32)];
    grad_b = [np.zeros(b[0].shape, dtype=np.float32), np.zeros(b[1].shape, dtype=np.float32)];

    c, _ = ComputeCostLoss(X, Y, W, b, lambda_);

    for j in range(0, 2):
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

    return grad_W[0], grad_b[0], grad_W[1], grad_b[1]






def ComputeAccuracy(X, y, W, b):  #KORREKT function 6 compute accuracy on evaluation
    s1 = W[0] @ X + b[0]
    h = np.maximum(0, s1)  # ReLu activation function  Xbatch^l gjord på l-1
    s = W[1] @ h + b[1]

    ypredict = np.argmax(s, axis=0)

    acc = np.sum(ypredict == y) / len(y)

    return acc


def visualisegraph(trainlossJ,validationlossJ,updatesteps,label):   #plots loss/cost

    plt.plot(updatesteps, trainlossJ, label ="train " + label)
    plt.plot(updatesteps, validationlossJ, label = "validation " + label)
    plt.legend()


    plt.xlabel('Update step')
    plt.ylabel(label)
    plt.show()








def Cyclicalheta(eta_min,eta_max,ns,t): #Exercise 3: cyclical learning rates

    t_temp = t % (2*ns)
    if t_temp <= ns:
        eta = eta_min + t_temp/ns * (eta_max-eta_min)
    elif t_temp <= 2*ns:
        eta = eta_max - (t_temp - ns)/ns * (eta_max-eta_min)

    t = (t + 1)

    return eta,t





def TrainMiniBatch(y, X, Y, X_val, y_val, W, b, t,batch_s, n_epochs, lambda_,eta_min,eta_max,ns,plotpercycle):

    [trainCostJ, trainLossJ] = [],[]
    [validationCostJ, validationLossJ] = [],[]
    updatesteps = []
    [acctrainlist,accvallist] = [],[]

    n_batch =  int(X.shape[0] / batch_s)

    eta = eta_min
    for epoch in range(n_epochs):

        print("Epoch number:", epoch, "  accuracy: ", round(ComputeAccuracy(X, y, W, b), 4), "LR eta: ",round(eta, 4))

        for j in range(2, int(X.shape[0] / n_batch)):
            j_start = (j - 1) * n_batch
            j_end = j * n_batch

            Xbatch = X[:, range(j_start, j_end)]
            Ybatch = Y[:, range(j_start, j_end)]


            [gradientW1,gradientb1,gradientW2,gradientb2] = ComputeGradients(Xbatch, Ybatch, W, lambda_)
            Wgrad = [gradientW1,gradientW2]
            bgrad = [gradientb1,gradientb2]

            for i in range(len(W)):
                Wgrad[i] = Wgrad[i].reshape(-1, Wgrad[i].shape[-1])

                W[i] = W[i] - eta*Wgrad[i]
                b[i] = b[i] - eta*bgrad[i]



            [eta, t] = Cyclicalheta(eta_min, eta_max, ns, t)



            if t % int(np.floor(ns/plotpercycle*2)) == 0 or t == 1 or t == batch_s*n_epochs:  # 9 is numb per cycle
                [cost, loss] = ComputeCostLoss(X, Y, W, b, lambda_)
                validationCostJ.append(cost), validationLossJ.append(loss)
                [cost, loss] = ComputeCostLoss(X_val, Y_val, W, b, lambda_)
                trainCostJ.append(cost),trainLossJ.append(loss)
                updatesteps.append(t)

                #print("cost: ", round(trainCostJ[-1], 3), "loss: ", round(trainLossJ[-1], 3))
                acctrainlist.append(ComputeAccuracy(X, y, W, b))
                accvallist.append(ComputeAccuracy(X_val, y_val, W, b))

                #print("acc on validation", accvallist[-1])


        idx = np.random.permutation(X.shape[1])
        X = X[:, idx]
        Y = Y[:, idx]
        y = y[idx]


    return W, b,trainCostJ,validationCostJ,trainLossJ, validationLossJ,updatesteps,acctrainlist,accvallist,eta




def gridsearch(model, parameters):
    global W
    global b

    bestacc = 0
    bestparams = 0

    for combination in itertools.product(*parameters.values()):
        batch_s, n_epochs, lambda_,eta_min,eta_max,ns,plotpercycle = combination
        [W, b,trainCostJ,validationCostJ,trainLossJ, validationLossJ,updatesteps,acctrainlist,accvallist,eta] = \
            model(y, X, Y, X_val, y_val, W, b, t,batch_s, n_epochs, lambda_,eta_min,eta_max,ns,plotpercycle)

        Accuracy = ComputeAccuracy(X_val, y_val, W, b)

        if Accuracy > bestacc:
            bestacc = Accuracy
            bestparams = combination
        print("acc: ", Accuracy, "with parameters: ", combination, "end eta: ", eta)

    return [bestacc, bestparams, W, b,trainCostJ,validationCostJ,trainLossJ, validationLossJ,updatesteps,acctrainlist,accvallist]


def randomsearch(model, parameters,randiterations):
    global W
    global b

    bestacc = 0
    bestparams = 0



    for i in range(randiterations):
        randlistparm = []
        for e in range(len(parameters)):
            listofparameters = list(parameters.values())[e]
            if len(listofparameters) > 1:
                randlistparm.append( listofparameters[0] + (listofparameters[1]-listofparameters[0])*np.random.rand())
            else:
                randlistparm.append(listofparameters[0])



        batch_s, n_epochs, lambda_, eta_min, eta_max, ns, plotpercycle = randlistparm


        [W, b, trainCostJ, validationCostJ, trainLossJ, validationLossJ, updatesteps, acctrainlist, accvallist,eta] = \
            model(y, X, Y, X_val, y_val, W, b, t, batch_s, n_epochs, lambda_, eta_min, eta_max, ns, plotpercycle)

        Accuracy = ComputeAccuracy(X_val, y_val, W, b)
        print("acc: ", Accuracy, "with parameters: ", randlistparm, "end eta: ", eta)
        if Accuracy > bestacc:
            bestacc = Accuracy
            bestparams = randlistparm


    return [bestacc, bestparams, W, b,trainCostJ,validationCostJ,trainLossJ, validationLossJ,updatesteps,acctrainlist,accvallist]




#Exercise 1


if __name__ == "__main__":

	[dataTr,labelsTr] = fixdata()

	dataVa = dataTr[:, -1000:]
	labelsVa = labelsTr[-1000:]
	dataTr = dataTr[:, :-1000]
	labelsTr = labelsTr[:-1000]
	dataTrN = normalizedata(dataTr)
	dataVaN = normalizedata(dataVa)

	m,d  =  50,dataTr.shape[0]
	W1, b1 = create_Wb(m,d) # m = 50 numb hidden layers , d = 3072


	K = len(np.unique(np.array(labelsTr)))     # K = probabilities so 10
	W2, b2 = create_Wb(K,m)


	W = [W1,W2]
	b = [b1,b2]



	X = dataTrN
	y = np.array(labelsTr)  # reshape to 1,10000 and make array
	Y = np.eye(len(np.unique(y)))[y].T  # onehotencoded

	X_val = dataVaN
	y_val = np.array(labelsVa)
	Y_val = np.eye(len(np.unique(y_val)))[y_val].T  # onehotencoded



	t = 0
	tcount = 0
	#Exercise 2



	#TEST for if gradients are correct
	"""
	gradnum = compute_grads_num(X[:,0:1], Y[:,0:1],lambda_=0, h=0.001)
	grad = ComputeGradients(X[:,0:1], Y[:,0:1], W, lambda_=0)
	
	for e in range(0,4):
	    print("@@@@@@@@@@@")
	    
	    print(gradnum[e]-grad[e])
	"""


	#gridsearch

	"""
	parameters = {"batch_s": [100], "n_epochs": [20],"lambda_": [0.1,0.05,0.01,0.005,0.001,0.0005], "eta_min": [1e-5],"eta_max": [1e-1],"ns": [500], "plotpercycle": [10]}
	[bestacc, bestparams, W, b,trainCostJ,validationCostJ,trainLossJ, validationLossJ,updatesteps,acctrainlist,accvallist] = gridsearch(TrainMiniBatch, parameters)
	print("best acc from gridsearch: ", bestacc, "bestparams: ", bestparams)
	"""



	#random search

	"""
	parameters = {"batch_s": [100], "n_epochs": [20],
	              "lambda_": [0.005,0.0005], "eta_min": [1e-5],"eta_max": [1e-1],"ns": [500], "plotpercycle": [10]}
	[bestacc, bestparams, W, b,trainCostJ,validationCostJ,trainLossJ, validationLossJ,updatesteps,acctrainlist,accvallist] \
	    = randomsearch(TrainMiniBatch, parameters,randiterations = 10)
	print("best acc from randomsearch: ", bestacc, "bestparams: ", bestparams)
	"""

	#best value

	[W, b,trainCostJ,validationCostJ,trainLossJ, validationLossJ,updatesteps,acctrainlist,accvallist,eta] = TrainMiniBatch(y, X, Y, X_val, y_val, W, b,t,
	batch_s=100, n_epochs=30, lambda_=0.0045105,eta_min=1e-5,eta_max = 1e-1,ns = 500, plotpercycle = 10)
	Accuracy = ComputeAccuracy(X_val, y_val, W, b)
	print("best acc from randomsearch: ", Accuracy)


	#plot
	visualisegraph(trainCostJ,validationCostJ,updatesteps,"cost")
	visualisegraph(trainLossJ,validationLossJ,updatesteps,"loss")
	visualisegraph(acctrainlist,accvallist,updatesteps,"accuracy")


