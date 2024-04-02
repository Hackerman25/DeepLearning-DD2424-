import numpy as np
import matplotlib.pyplot as plt



def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)



def LoadBatch(file):
    import pickle
    with open("Datasets/"+ file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def montage(W):
	""" Display the image for each label in W """
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(2,5)
	for i in range(2):
		for j in range(5):
			im  = W[i*5+j,:].reshape(32,32,3, order='F')
			sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
			sim = sim.transpose(1,0,2)
			ax[i][j].imshow(sim, interpolation='nearest')
			ax[i][j].set_title("y="+str(5*i+j))
			ax[i][j].axis('off')
	plt.show()





def readdata(databatch):           #function 1 (Read in and store the training, validation and test data

    Tdata = LoadBatch(databatch)
    dataTr = Tdata[b'data']
    dataTr = np.flipud(np.rot90(dataTr))

    labelsTr = Tdata[b'labels']

    return dataTr, labelsTr

def normalizedata(dataTr):         #function 2 (normalize the data by std and mean)
    dataTr = dataTr - np.mean(dataTr, axis=0)
    dataTr = dataTr / np.std(dataTr, axis=0)

    return dataTr

def create_Wb():                   #function 3 initialize the parameters of the model W and b
    K = 10
    d = dataTr.shape[0]

    W = np.random.normal(0, 0.01, size=(K, d))
    b = np.random.normal(0, 0.01, size=(K,1))

    return W,b


def EvaluateClassifier(X, W, b):   #function 4  evaluates the network function,

    s = W@X + b
    p = softmax(s)

    return p

def clcross(y,p):              #    Cross entropy loss
    return -sum(y*np.log(p))


def ComputeLoss(X, Y, W, b):  #function 5    computes loss
    p = EvaluateClassifier(X,W,b)

    lcross = clcross(Y,p)
    J = 1 / X.shape[1] * sum(lcross)

    return J


def ComputeCost(X, Y, W, b,lambda_):  #function 5    computes cost
    p = EvaluateClassifier(X,W,b)

    lcross = clcross(Y,p)
    J = 1 / X.shape[1] * sum(lcross) + lambda_*sum(sum((W**2)))

    return J


def ComputeAccuracy(x, y, W, b):  #function 6 compute accuracy
    ypredict = W @ x + b
    ypredict = np.argmax(ypredict, axis = 0)

    acc = np.sum(ypredict == y) /len(y)

    return acc


def ComputeGradients(X, Y, P, W,lambda_):   #function 7 computes gradients

    G_batch = P - Y

    grad_W = (1/X.shape[1]) * G_batch @ X.T  +  2*lambda_*W

    grad_b = (1/X.shape[1]) * np.sum(G_batch,axis=1)

    grad_b = grad_b.reshape(len(grad_b),1)



    return grad_W, grad_b


def visualisegraph(trainlossJ,validationlossJ,label):   #plots loss/cost
    plt.plot(range(len(trainlossJ)), trainlossJ, label ="train " + label)
    plt.plot(range(len(validationlossJ)), validationlossJ, label = "validation " + label)
    plt.legend()


    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.show()




def MiniBatch(y,X,Y,X_val,y_val,W,b,n_batch,eta,n_epochs,lambda_):  #Mini batch gradient descent

    [trainCostJ,trainLossJ] = np.empty(n_epochs),np.empty(n_epochs)
    [validationCostJ,validationLossJ] = np.empty(n_epochs),np.empty(n_epochs)

    for epoch in range(n_epochs):

        print("Epoch number:", epoch, "  accuracy: ", ComputeAccuracy(X, y, W, b))
        for j in range(int(X.shape[0]/n_batch)):

            j_start = (j-1*n_batch + 1)
            j_end = j*n_batch

            Xbatch = X[:,range(j_start,j_end)]
            Ybatch = Y[:, range(j_start, j_end)]

            Pbatch = EvaluateClassifier(Xbatch, W, b)

            [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, Pbatch, W, lambda_)

            W = W - eta*grad_W
            b = b - eta*grad_b

        trainCostJ[epoch] = ComputeCost(X, Y, W, b, lambda_)
        validationCostJ[epoch] = ComputeCost(X_val, Y_val, W, b, lambda_)

        trainLossJ[epoch] = ComputeLoss(X, Y, W, b)
        validationLossJ[epoch] = ComputeLoss(X_val, Y_val, W, b)



        idx = np.random.permutation(X.shape[1])   #randomizes training data/labels
        X = X[:,idx]
        Y = Y[:,idx]
        y = y[idx]

    return W, b, trainCostJ,validationCostJ,trainLossJ,validationLossJ





dataTr, labelsTr = readdata("data_batch_1")     #training
dataVa, labelsVa = readdata("data_batch_2")     #validating


dataTrN = normalizedata(dataTr)
dataVaN = normalizedata(dataVa)


W,b = create_Wb()
X = dataTrN
y = np.array(labelsTr)  #reshape to 1,10000 and make array
Y = np.eye(len(np.unique(y)))[y].T   #onehotencoded


X_val = dataVaN
y_val = np.array(labelsVa)
Y_val = np.eye(len(np.unique(y_val)))[y_val].T   #onehotencoded


[Wstar, bstar,trainCostJ,validationCostJ,trainLossJ,validationLossJ] = MiniBatch(y,X,Y,X_val,Y_val,W,b,n_batch=100,eta=0.001,n_epochs=40,lambda_=1)



print("final acc: ", ComputeAccuracy(X_val, y_val, Wstar, bstar))



montage(Wstar)

visualisegraph(trainCostJ,validationCostJ,"cost")

visualisegraph(trainLossJ,validationLossJ,"loss")




