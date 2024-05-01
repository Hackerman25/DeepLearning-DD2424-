import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import *
import random
import itertools
import copy

def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1,1)

def sigmoid(x):
    """ Standard definition of the sigmoid function """
    return np.exp(x)/(1+ np.exp(x))
 
 
def load_batch(filename):
    """ Copied from the dataset website """
    with open('Datasets/' + filename, 'rb') as fo:
        dataset_dict = pickle.load(fo, encoding='bytes')
 
    return dataset_dict

def extract_image(file):
    dic_data=load_batch(file)

    image_data=dic_data[b'data']
    #image_data=np.flipud(np.rot90(image_data))
    labels=dic_data[b'labels']
    label_nr=np.unique(labels)
    data_onehot = np.eye(len(np.unique(labels)))[labels]  

    return image_data, data_onehot, np.array(labels)

def ReLu(x):
    s=x
    s[s<0]=0 #if negative it becomes zero
    return s
    #return np.maximum(0, x)

def normalize(data_mean,data):
    std_X=np.std(data, axis=0)
    norm_data=(data-data_mean)/(std_X)
    return norm_data

def init_weights_bias(d,m):
    """
    Initializes the dimensions of the weights and biases 
    """
    std=1/np.sqrt(d)
    W=np.random.normal(0, std, size=(d,m))  
    b=np.zeros((m,1))
    return np.array(W), np.array(b) 

def weights_bias(X, layers, batch_norm=False):
    """ 
    Calculates the weights and biases of the fully connected layers
    weight dimension= amount of nodes at current layer x amount of nodes in next layer
    """
    n=X.shape[0]
    d=X.shape[1]
    W_arr , b_arr, gamma, beta= [] , [] , [] , []

    if layers == 1:
        W, b = init_weights_bias(d, m[0])
        W_arr.append(W)
        b_arr.append(b)
        W_last, b_last = init_weights_bias(m[-1], K)
        W_arr.append(W_last)
        b_arr.append(b_last)
        if batch_norm:
            gamma=np.ones((m[0],1))
            beta=np.zeros((m[0],1))
    else:
        for k in range(layers):
            W ,b = init_weights_bias(d, m[k])
            W_arr.append(W)
            b_arr.append(b)
            if batch_norm:
                gamma.append(np.ones((d,1)))
                beta.append(np.zeros((d,1)))
            d=m[k]
        # last hidden layer to output layer
        W_last, b_last = init_weights_bias(m[-1], K)
        W_arr.append(W_last)
        b_arr.append(b_last)
        gamma.append(np.ones((m[-1],1)))
        beta.append(np.zeros((m[-1],1)))

        if batch_norm:
            return W_arr , b_arr , gamma , beta
        else:
            return W_arr, b_arr

def Evaluateclassifier(X,W,b, gamma=None, beta=None, batch_norm=False):
    "Computes the score and the non-linear activation of each layer"
    if not batch_norm:
        s= X@W + b.T
        h=ReLu(s)
        return s, h
    else:
        alpha=0.9
        s= X@W + b.T
        mean = np.mean(s, axis=1).reshape(-1,1)  #mean vector   will become nx1
        var= np.var(s, axis=1).reshape(-1,1)     #variance vector  will become nx1
        epsilon= 1e-12
        sigma= np.sqrt(var+epsilon)              #standard deviation +  small
        #print('s', s.shape, 'mean', mean.T.shape, 'sigma', sigma.T.shape)
        S_BN=(s-mean.T)/(sigma.T)                    #implement BN on the score
        S_scaled= gamma.T*S_BN + beta.T              #Scale and shift again
        h=ReLu(S_scaled)

        return s,S_BN, S_scaled, h, mean, var

def forward_pass(X, W_arr, b_arr,gamma=None, beta=None, batch_norm=False):
    "Forward propagation in the neural network"
    S_k , h_k , S_BN, S_hat, mean, var= [] , [] , [] , [] , [] , []
    #mean, var = np.empty(layers-1) , np.empty(layers-1) #layers-1 because you don't need to compute for the input & output
    #S_k , h_k, S_BN, S_hat = np.empty(layers-1) , np.empty(layers-1) ,np.empty(layers-1) , np.empty(layers)

    if batch_norm:  
        for l in range(len(W_arr)):
            s, s_bn , s_scale, h , mu, sigma= Evaluateclassifier(X, W_arr[l], b_arr[l], gamma[l], beta[l], batch_norm=True)
            S_k.append(s), S_BN.append(s_bn) , S_hat.append(s_scale) , h_k.append(h) ,  mean.append(mu) , var.append(sigma)
            X=S_hat[l]
        p=softmax(s_scale)
        return p, S_k,S_BN, S_hat, h_k, mean , var
    else:
        for l in range(len(W_arr)):
            s , h = Evaluateclassifier(X,W_arr[l], b_arr[l])
            S_k.append(s)
            h_k.append(h)
            X=s
        p=softmax(X)
        return p, S_k ,h_k


def backward_pass(X, Y, W_arr, b_arr, lamb):
    """Backward propagation for the network

    states: contains all outputs from each layer starting from the first hidden layer
    P: probabilites calculatede from the forward pass, dimension: nxK

    """ 
    grad_W , grad_b = [] , []
    P, S_k, h_k = forward_pass(X, W_arr, b_arr)
    #W_arr , b_arr =list(reversed(W_arr)) , list(reversed(b_arr))
    W_arr , b_arr =W_arr , b_arr
    N=X.shape[0]
    G=(P-Y)  #propagate this backwards through the layers using chain-rule

    for k in range(len(W_arr),-1, -1):
        d_W=(1/N)*h_k[k].T@G + 2*lamb*W_arr[k]
        d_b=(1/N)*np.sum(G, axis=0, keepdims=True)
        grad_W.append(d_W)
        grad_b.append(d_b)

        #update step
        G=(1/N)*G@W_arr[k].T * (S_k[k]>0)

    d_W1=(1/N)*X.T@ G   #dx50
    d_b1 =(1/N)*np.sum(G, axis=0, keepdims=True)
    grad_W.append(d_W1)
    grad_b.append(d_b1)

    return list(reversed(grad_W)), list(reversed(grad_b))

def backward_pass_batchnorm(X, Y, W_arr, b_arr, gamma, beta, lamb):
    """Backward propagation for the network

    states: contains all outputs from each layer starting from the first hidden layer
    P: probabilites calculatede from the forward pass, dimension: nxK
    
    """ 
    grad_gamma, grad_beta = [None]*(len(W_arr)-1), [None]*(len(W_arr)-1)
    grad_W , grad_b = [] , [] 
    P, S_k, S_BN, S_hat, h_k, mean, var = forward_pass(X, W_arr, b_arr,gamma, beta, batch_norm=True)
    W_arr , b_arr, h_k =list(reversed(W_arr)) , list(reversed(b_arr)) , list(reversed(h_k))
    S_k, S_BN ,gamma, beta= list(reversed(S_k)) , list(reversed(S_BN)) , list(reversed(gamma)) , list(reversed(beta)) 
    mean , var = list(reversed(mean)) , list(reversed(var))
    N=X.shape[0]
    G_batch=(P-Y)  #propagate this backwards through the layers using chain-rule
    epsilone = 1e-12  # to avoid division by zero


    #calculate the gradients for the k:th layer (last layer)
    dW_k=(1/N)*h_k[0].T@G_batch + 2*lamb*W_arr[0]   #10xn
    db_k=(1/N)*np.sum(G_batch, axis=0, keepdims=True) #1x10
    grad_W.append(dW_k)
    grad_b.append(db_k)

    #propagate G_batch to the previous layer         
    G_batch=G_batch@W_arr[0]   
    G_batch=G_batch*h_k[0]  #nx10

    for k in range(0,layers-1):
        grad_gamma[k]=(1/N)*((G_batch.T*S_BN[k].T)@np.ones((N,1)))   
        grad_beta[k]= (1/N)*np.sum(G_batch, axis=0, keepdims=True)#(1/N)*(G_batch@np.ones((N,1)))               

        G_batch = G_batch*(np.ones((N,1))@gamma[k].T)   #nx10
        G_batch=BatchNormBackPass(G_batch, S_k[k], mean[k], var[k], epsilone)  #nx10
        
        d_W=(1/N)*h_k[k+1].T@G_batch + 2*lamb*W_arr[k+1]
        d_b=(1/N)*np.sum(G_batch, axis=0, keepdims=True)
        grad_W.append(d_W)
        grad_b.append(d_b)

        G_batch=G_batch@W_arr[k+1].T  
        G_batch=G_batch*h_k[k+1]

    #calculate gradients for the first layer
    d_W1=X.T@ G_batch + 2*lamb*W_arr[-1]
    d_b1 = np.sum(G_batch, axis=0, keepdims=True)
    grad_W.append(d_W1)
    grad_b.append(d_b1)
    
    return list(reversed(grad_W)), list(reversed(grad_b)) , list(reversed(grad_gamma)) , list(reversed(grad_beta))

def BatchNormBackPass(G_batch, S_batch, mean, var, epsilone):
    N = S_batch.shape[0]
    G_b1 = G_batch*(((var.T+epsilone)**(-0.5))*np.ones((N,1))) 
    G_b2 = G_batch*(((var.T+epsilone)**(-1.5))*np.ones((N,1)))
    D = S_batch-mean.T*np.ones((N,1))
    cons = (G_b2*D)*np.ones((N,1))
    G_new = G_b1-((G_b1*np.ones((N,1))))/N-D*(cons*np.ones((N,1)))/N
    
    return G_new

def ComputeCost(X, Y, lamb, W, b, gamma, beta ,compute_loss=False, batch_norm=False):
    N=X.shape[0]
    if batch_norm:
        P, S_k, S_BN, S_hat, h_k, mean, var = forward_pass(X_train, W, b , gamma, beta,  batch_norm=True)
    else:
        P, S_k, h_k = forward_pass(X, W, b)

    l_cross=sum(-np.log((Y*P).sum(axis=1)))
    if compute_loss:
        J=(1/N)*l_cross
    else:
        regularization=0
        for W_k in W:
            regularization +=((W_k**2).sum())

        J=(1/N)*l_cross + regularization

    return J

def ComputeAccuracy(X,y,W_arr, b_arr,gamma=None, beta=None,batch_norm=False):
    #Input:  X:dnx , y:1xn
    #Output: acc: 1x1
    # for k in range(len(W_arr)):
    #     print('w', W_arr[k].shape)
    #     print('b',b_arr[k].shape)
    if batch_norm:
        P, S_k, S_BN, S_hat, h_k, mean, var = forward_pass(X_train, W_arr, b_arr, gamma, beta,  batch_norm=True)
    else:
        P, S_k, h_k=forward_pass(X,W_arr,b_arr)
    y_pred=np.argmax(P,1)
    acc=np.sum(y==y_pred)/(len(y))
    print(acc)
    return acc

def MiniBatchGD(Xtrain,Xvalid, Ytrain, Yvalid, W_array, b_array, lamb, n_batch,n_epochs, eta_min, eta_max, n_s, 
                gamma, beta, batchnorm=False):
    t=0
    alpha=0.9
    training_loss = []
    validation_loss = []

    update_step = []
    for n in range(n_epochs):
        for i in range(1,int(Xtrain.shape[0]/n_batch)+1):
            t_t=  t% (2*n_s)  # to make it cyclic with cycle 2n_s
            if t_t<= n_s:
                eta=eta_min + t_t/n_s * (eta_max-eta_min)
            elif t_t <= 2*n_s:
                eta= eta_max - (t_t - n_s)/n_s * (eta_max - eta_min)
            t= (t+1)

            j_start=(i-1)*n_batch 
            j_end=i*n_batch
            Xbatch=Xtrain[j_start:j_end,:]
            Ybatch=Ytrain[j_start:j_end,:] 

            if batchnorm:
                p, S_k, S_BN, S_hat, h_k, mean, var =forward_pass(Xbatch, list(reversed(W_array)), list(reversed(b_array)),gamma,beta, batch_norm=True)
                grad_W, grad_b, grad_gamma, grad_beta=backward_pass_batchnorm(Xbatch, Ybatch, W_array, b_array, gamma, beta)

                if t % int((2*n_s)/10)==0 or t== n_epochs*n_batch : 
                    training_loss.append(ComputeCost(Xtrain, Ytrain, lamb, W_array, b_array, batch_norm=True, compute_loss=True))
                    validation_loss.append(ComputeCost(Xvalid, Yvalid, lamb, W_array, b_array, batch_norm=True, compute_loss=True))
                    update_step.append(t)
        
                #update
                for k in range(len(W_array)):
                    W_array[k] =W_array[k] -eta*grad_W[k]
                    b_array[k] =b_array[k] -eta*grad_b[k].T
                    gamma[k] =W_array[k] -eta*grad_gamma[k]
                    beta[k] =W_array[k] -eta*grad_beta[k]
                
                #implement exponentially moving average
                if i==1:
                    mean_av=mean
                    var_av=var
                else:
                    mean_avg = [alpha*mean_avg[l]+(1-alpha)*mean[l] for l in range(len(mean))]
                    var_avg = [alpha*var_avg[l]+(1-alpha)*var[l] for l in range(len(var))]   
            else:
                p, S_k, h_k = forward_pass(Xbatch, W_array, b_array)
                grad_W, grad_b = backward_pass(Xbatch, Ybatch, W_array, b_array, lamb)

                if t % int((2*n_s)/10)==0 or t== n_epochs*n_batch : 
                    training_loss.append(ComputeCost(Xtrain, Ytrain, lamb, W_array, b_array, batch_norm=True, compute_loss=True))
                    validation_loss.append(ComputeCost(Xvalid, Yvalid, lamb, W_array, b_array, batch_norm=True, compute_loss=True))
                    update_step.append(t)

                for k in range(len(W_array)):
                    W_array[k] =W_array[k] -eta*grad_W[k]
                    b_array[k] =b_array[k] -eta*grad_b[k].T

        #Stochastic data points
        idx = np.random.permutation(Xtrain.shape[0])
        Xtrain = Xtrain[idx,:]
        Ytrain = Ytrain[idx,:]
        y = y[idx]

    return W_array, b_array , training_loss, validation_loss, update_step

def coarsesearch(network, parameters):
    #initating accuracy and parameters found
    top_parameters=0
    top_accuracy=0
    batch_norm = True

    for params in itertools.product(*parameters.values()):  #iterating through all possible combination of the parameters
        lamb = params #save current paramters
        
        W_array, b_array , training_loss, validation_loss, update_step = network(X_train, Y, y, X_valid, Y_val, y_val, W_array, b_array, n_batch,n_epochs,eta_min, eta_max, n_s, cycles)

        accuracy=ComputeAccuracy(X_valid, y_val,W_array, b_array, batch_norm) #compute accuracy for current combination

        #if the accuracy has improved, update the top accuracy and parameters
        if accuracy>top_accuracy:

            top_parameters=params
            top_accuracy= accuracy

    return top_accuracy, top_parameters

def finesearch(network, searches, l_min, l_max):
    #have a for loop that checks i random variables between lambda 0.0001 and 0.00005
    top_accuracy=0
    top_parameter=0
    batch_norm = True


    for i in range(searches):
        l = l_min + (l_max - l_min)*np.random.uniform(0, 1)
        W1s, W2s, b1s, b2s, training_cost, validation_cost, update_step,= network(X_train, Y, y, X_valid, Y_val, y_val, W_array, b_array, n_batch,n_epochs,eta_min, eta_max, n_s, cycles)

        accuracy=ComputeAccuracy(X_valid, y_val, W_array, b_array, batch_norm)


        if accuracy>top_accuracy:
            print('accuracy', top_accuracy, 'with lambda', l)
            top_accuracy= accuracy
            top_parameter=l
    print("best accuracy randomsearch: ", top_accuracy, "parameters: ", top_parameter)

    return top_accuracy, top_parameter

def visualize_cost(cost_train,cost_valid, update_step):
    plt.plot(update_step, cost_train, label ="training loss")
    plt.plot(update_step, cost_valid, label = "validation loss")
    plt.ylabel('loss')
    plt.xlabel('Update step')
    plt.legend()
    plt.show()

def ComputeGradsNum(X, Y, lambda_, W, b, gamma, beta, mean, var, batch_normalization, h=0.001):
    
    # Create lists for saving the gradients by layers
    grad_W = [W_l.copy() for W_l in W]
    grad_b = [b_l.copy() for b_l in b]
    if batch_normalization:
        grad_gamma = [gamma_l.copy() for gamma_l in gamma]
        grad_beta = [beta_l.copy() for beta_l in beta]
    
    # Compute initial cost and iterate layers k
    c = ComputeCost(X, Y,lambda_, W, b, gamma, beta, batch_normalization)
    k = len(W)
    for l in range(k):
    # Gradients for bias
        for i in range(b[l].shape[0]):
            b_try = copy.deepcopy(b)
            b_try[l][i] += h
            c2 = ComputeCost(X, Y,lambda_, W, b_try, gamma, beta, batch_normalization)
            print((c2-c)/h)
            grad_b[l][i] = (c2-c)/h



def main():
    global m,d, K, X_train, X_valid, Y, Y_val, y , y_val, layers, W_array, b_array, eta_min, eta_max, n_s, cycles, n_batch, n_epochs

    # Training network with just one batch
    
    X_train ,Y , y = extract_image("data_batch_1")
    X_valid ,Y_val , y_val = extract_image("data_batch_2")

    d=X_train.shape[1]
    #m=[50,30,20,20,10,10,10,10]  #nodes in hidden layers
    m=[50,50]
    #m=[50]
    layers=len(m)
    K=10
    lamb=0.005
    n_batch=100
    n_epochs=50
    eta_min=1e-5
    eta_max=1e-1
    n_s=5*450
    cycles=1
    sigmas = [1e-1, 1e-3, 1e-4]

    #normalizing
    mean=np.mean(X_train,axis=0)
    X_train=normalize(mean,X_train)
    X_valid=normalize(mean,X_valid)

    #Initialize parameters
    W_array, b_array, gamma , beta = weights_bias(X_train, layers, batch_norm=True)
    #p, S_k, S_BN, S_hat, h_k, mean, var = forward_pass(X_train, W_array, b_array,gamma[1:], beta[1:],  batch_norm=False)
    # p, S_k, h_k = forward_pass(X_train, W_array, b_array)
    # backward_pass(X_train, Y, W_array, b_array, lamb)
    
    #gamma, beta= list(reversed(gamma[1:]))  ,list(reversed(beta[1:]))
    grad_W , grad_b , grad_gamma, grad_beta = backward_pass_batchnorm(X_train[1,:][np.newaxis,:], Y[1,:][np.newaxis,:], W_array, b_array, gamma[1:], beta[1:],lamb=lamb)
    # grad_Wa, grad_ba=ComputeGradsNum(X_train[1,:], Y[1,:], lamb, W_array, b_array, gamma=None, beta=None, mean=None, var=None, batch_normalization=False, h=0.001)

    # max_error_W = []
    # max_error_b = []

    # for i in range(len(W_array)):
    #     max_error_W.append(abs(grad_W[i]-grad_Wa[i]).max())
    #     max_error_b.append(abs(grad_b[i]-grad_ba[i]).max())


    # print(max_error_W)

    # print("hello")

    #ComputeCost(X_train, Y, lamb, W_array, b_array, gamma[1:], beta[1:],batch_norm=True)

    # Ws_arr, bs_arr, train_loss, valid_loss, update_values = MiniBatchGD(X_train, X_valid, Y, Y_val, W_array, b_array, lamb, n_batch,n_epochs, eta_min, eta_max, n_s, 
    #             gamma[1:], beta[1:], batchnorm=False)
    # visualize_cost(train_loss, valid_loss, update_values)

    #ComputeAccuracy(X_valid, y_val, Ws_arr, bs_arr)
    
    
main()



#have to fix computecost and everything to include batchnorm case with gamma and beta
#have to implement the exponential moving average
