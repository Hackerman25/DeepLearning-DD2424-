import numpy as np

def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)



def LoadBatch(file):
    import pickle
    with open("Datasets/"+ file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict




def ComputeGradsNum(X, Y, P, W, b, lamda, h):
	""" Converted from matlab code """
	no 	= 	W.shape[0]
	d 	= 	X.shape[0]

	grad_W = np.zeros(W.shape);
	grad_b = np.zeros((no, 1));

	c = ComputeCost(X, Y, W, b, lamda);
	
	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] += h
		c2 = ComputeCost(X, Y, W, b_try, lamda)
		grad_b[i] = (c2-c) / h

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i,j] += h
			c2 = ComputeCost(X, Y, W_try, b, lamda)
			grad_W[i,j] = (c2-c) / h

	return [grad_W, grad_b]

def ComputeGradsNumSlow(X, Y, P, W, b, lamda, h):
	""" Converted from matlab code """
	no 	= 	W.shape[0]
	d 	= 	X.shape[0]

	grad_W = np.zeros(W.shape);
	grad_b = np.zeros((no, 1));
	
	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] -= h
		c1 = ComputeCost(X, Y, W, b_try, lamda)

		b_try = np.array(b)
		b_try[i] += h
		c2 = ComputeCost(X, Y, W, b_try, lamda)

		grad_b[i] = (c2-c1) / (2*h)

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i,j] -= h
			c1 = ComputeCost(X, Y, W_try, b, lamda)

			W_try = np.array(W)
			W_try[i,j] += h
			c2 = ComputeCost(X, Y, W_try, b, lamda)

			grad_W[i,j] = (c2-c1) / (2*h)

	return [grad_W, grad_b]

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

def save_as_mat(data, name="model"):
	""" Used to transfer a python model to matlab """
	import scipy.io as sio
	#sio.savemat(name'.mat',{name:b})




def EvaluateClassifier(X, W, b):   #function 4  evaluates the network function,
    #print("dd", W.shape,X.shape, b.shape)

    s = W@X + b
    p = softmax(s)

    return p

def clcross(y,p):  #korrekt confirmed
    return -sum(y*np.log(p))



def ComputeCost(X, Y, W, b, lambda_):  #function 5    computes cost
    p = EvaluateClassifier(X,W,b)

    lcross = clcross(Y,p)
    J = 1 / X.shape[1] * sum(lcross) + lambda_*sum(sum((W**2)))  #cost wrong + cost Weights

    return J















def softmax2(S):
	return np.exp(S) / np.exp(S).sum(axis=0)


def EvaluateClassifier2(X, W_1, W_2, b_1, b_2):
	s_1 = np.dot(W_1, X) + b_1
	h = np.maximum(0, s_1)
	s = np.dot(W_2, h) + b_2
	p = softmax2(s)

	return p, h

def ComputeCost2(X, Y, W_1, W_2, b_1, b_2, λ):
    # Compute network output
    P, H = EvaluateClassifier2(X, W_1, W_2, b_1, b_2)
    # Compute loss function term
    L = sum(-np.log((Y*P).sum(axis=0)))
    # Compute regularization term
    L_λ = λ*((W_1**2).sum() + (W_2**2).sum())
    # Compute total cost
    J = L/X.shape[1] + L_λ
    return J



def ComputeGradsNum(X, Y, W1, b1, W2, b2, lambda_, h=0.00001):
	grad_W2 = np.zeros(shape=W2.shape)
	grad_b2 = np.zeros(shape=b2.shape)
	grad_W1 = np.zeros(shape=W1.shape)
	grad_b1 = np.zeros(shape=b1.shape)
	c = ComputeCost2(X, Y, W1, W2, b1, b2, lambda_)

	for i in range(b1.shape[0]):
		b1_try = b1.copy()
		b1_try[i, 0] = b1_try[i, 0] + h
		c2 = ComputeCost2(X, Y, W1, W2, b1_try, b2, lambda_)
		grad_b1[i, 0] = (c2 - c) / h

	for i in range(W1.shape[0]):
		for j in range(W1.shape[1]):
			W1_try = W1.copy()
			W1_try[i, j] = W1_try[i, j] + h
			c2 = ComputeCost2(X, Y, W1_try, W2, b1, b2, lambda_)
			grad_W1[i, j] = (c2 - c) / h

	for i in range(b2.shape[0]):
		b2_try = b2.copy()
		b2_try[i, 0] = b2_try[i, 0] + h
		c2 = ComputeCost2(X, Y, W1, W2, b1, b2_try, lambda_)
		grad_b2[i, 0] = (c2 - c) / h

	for i in range(W2.shape[0]):
		for j in range(W2.shape[1]):
			W2_try = W2.copy()
			W2_try[i, j] = W2_try[i, j] + h
			c2 = ComputeCost2(X, Y, W1, W2_try, b1, b2, lambda_)
			grad_W2[i, j] = (c2 - c) / h

	return grad_W2, grad_b2, grad_W1, grad_b1