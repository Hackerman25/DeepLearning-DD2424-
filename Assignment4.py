import numpy as np
import copy

def getdata():
    fileurl = r"C:\Users\Marcus\Documents\GitHub\DeepLearning-DD2424-\Datasets\goblet_book.txt"
    file = open(fileurl, "r")


    return file.read()


def findnumbunique(file):

    output = len(set(file))

    return output

def createmappingfunc(file):

    output1 = sorted(set(file))

    my_map = dict((j,i) for i,j in enumerate(output1))
    inv_map = {v: k for k, v in my_map.items()}

    return my_map, inv_map


def OneHotEncoding(data, mymap,begin, end):
    K = len(mymap)
    N = end - begin

    EncodedData = np.zeros((K,N))


    for i in range(N):
        EncodedData[mymap[data[begin + i]],i] = 1



    return EncodedData


class RNN:
    def __init__(self,m,K,eta= 0.1,seq_length= 25):
        print(K,"ddk")
        self.m = m
        self.K = K
        self.eta = eta
        self.seq_length = seq_length

        self.b = np.zeros((self.m,1))
        self.c = np.zeros((self.K, 1))

        sig = 0.01
        self.U = np.random.normal(0, 1, size = (self.m,self.K)) * sig
        self.W = np.random.normal(0, 1, size=(self.m, self.m)) * sig
        self.V = np.random.normal(0, 1, size=(self.K, self.m)) * sig


    def eval_RNN(self,h0,x0):   #should be correct

        b = RNN.b
        c = RNN.c

        U = RNN.U
        W = RNN.W
        V = RNN.V


        #eq 1-4
        xnext = x0  # 83 x 1


        #xnext = np.reshape(xnext, (xnext.shape[0], 1))

        print(W.shape, h0.shape, "ddda")

        a_t = W @ h0 + U @ xnext[:,np.newaxis] + b  # 100x100  x  100x1  +  100x83  x  83x1       #CORRECT

        h_t = np.tanh(a_t)                                                                        #CORRECT
        o_t = V @ h_t + c                                                                         #CORRECT




        a = np.exp(x0 - np.max(x0, axis=0))
        p_t = a / a.sum(axis=0) #softmax
        p_t = np.reshape(p_t, (p_t.shape[0], 1))


        return a_t,h_t,o_t,p_t

    def synth_text(self,h0, x0, n):

        Y = np.zeros((self.K, n))
        for t in range(n):
            print("x0", x0.shape)
            a_t,h_t,o_t,p_t = RNN.eval_RNN(h0, x0)



            cp = np.cumsum(p_t)
            idx = np.random.choice(self.K, p=p_t.flat)
            x0 = np.zeros(x0.shape)
            x0[idx] = 1
            Y[idx, t] = 1
        return Y

    def CompGrads(self,X_chars,Y_chars,h0):

        seq_length = X_chars.shape[1]

        a = np.zeros((seq_length,RNN.m,1))  #9 x m x m = 100 x 1
        h = np.zeros((seq_length,RNN.m,1))      #9 x m x 1
        o = np.zeros((seq_length,X_chars.shape[0],RNN.m))    # 9 x k x m = 83 x 100
        p = np.zeros((seq_length,X_chars.shape[0],1))          # 9 x k x 1
        h[-1] = h0


        loss = 0

        grad_o = np.zeros((seq_length,1,X_chars.shape[0]))  #9 x 1 x 83
        grad_o2 = np.zeros((seq_length,X_chars.shape[0],RNN.m))

        grad_h = np.zeros_like(h)

        grad_a = np.zeros_like(a)

        grad_U, grad_W, grad_V, grad_b, grad_c = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(self.V), np.zeros_like(self.b), np.zeros_like(self.c)



        #forward pass
        for t in range(seq_length):
            #eq 1-4

            a[t], h[t], o[t], p[t] = self.eval_RNN(h[t-1], X_chars[:,t])
            print(a[t].shape, h[t].shape, o[t].shape, p[t].shape, "HHHHHHHHHHH")
            # 100 x 1

            loss += -np.log(Y_chars.T @ p[t])
        #backward   pass


        for t in reversed(range(seq_length)):

            grad_o[t] = -(Y_chars[:, t].reshape(Y_chars.shape[0], 1) - p[t]).T               # 1 x k   GOOD
            print("grad o @@@@@@@@@@@@@@@@@@@@ ", grad_o.shape)

            #reshaped_array = np.reshape(array, (83, 1))

            grad_V += grad_o[t].T @ h[t].T       #k x m  = 83 x 100                                    GOOD

            grad_c += grad_o[t].T          #k x 1                                                      GOOD

            if t == seq_length-1:

                grad_h[t] = ( grad_o[t] @ RNN.V ).T                  #m x 1 =  (83,)   @   (83, 100)   = 83 x 1 (with added newaxis)
                grad_a[t] = grad_h[t] @ np.diag(1-np.tanh(a[t])**2)[:,np.newaxis]     #m x 1 = 100 x 1 =   100 x 1  @  (1,) (with new axis) =   100 x 1


            else:
                grad_h[t] = (grad_o[t] @ RNN.V).T + RNN.W @ grad_a[t+1]                  #m x 1 =  (83,)  @  (83,100) + (100,1) @ (100,100)
                grad_a[t] = grad_h[t] @ np.diag(1-np.tanh(a[t])**2)[:,np.newaxis]

            print(grad_a[t].T.shape, h[t - 1].T.shape, "ttttttttttttttttttttT")
            grad_W += grad_a[t] @ h[t - 1].T  # (m,m)

            print(grad_a[t].shape, X_chars[:, t].reshape(X_chars.shape[0], 1).T.shape)
            grad_U += grad_a[t] @ X_chars[:, t].reshape(X_chars.shape[0], 1).T  # (m,K)
            grad_b += grad_a[t]  # (m,1)


        #grad_U, grad_W, grad_V, grad_b, grad_c = np.clip(grad_U, -5, 5), np.clip(grad_W, -5, 5) ,np.clip(grad_V, -5, 5), np.clip(grad_b, -5, 5) , np.clip(grad_c, -5, 5)

        hprev = h[seq_length - 1]

        return grad_U, grad_W, grad_V, grad_b, grad_c, loss, hprev

def printsentence(result):
    index = np.where(result == 1)[0]

    print("sentence:")
    for letter in index:

        print(inv_map[letter], end = "")  # ???? dont know this part to get letters
    print("\n")



def computeGrads_Num(rnn, x, y, h = 1e-5):
    res_rnn = copy.deepcopy(rnn)
    m = rnn.m
    h_prev = np.zeros((m,1))
    for idx, att in enumerate(['b', 'c', 'U', 'W', 'V']):
        grad = np.zeros(getattr(rnn, att).shape)
        for i in range(grad.shape[0]):
            for j in range(grad.shape[1]):
                rnn_try = copy.deepcopy(rnn)
                aux = np.copy(getattr(rnn_try, att))
                aux[i, j] -= h
                setattr(rnn_try, att, aux)
                #p, _, _ = rnn_try.forward(h_prev, x)
                #l1 = rnn_try.loss(y, p)
                grad_U, grad_W, grad_V, grad_b, grad_c, loss, hprev = CompGrads(self, X_chars, Y_chars, h0)

                rnn_try = copy.deepcopy(rnn)
                aux = np.copy(getattr(rnn_try, att))
                aux[i, j] += h
                setattr(rnn_try, att, aux)
                p, _, _ = rnn_try.forward(h_prev, x)
                l2 = rnn_try.loss(y, p)
                grad[i, j] = (l2 - l1) / (2 * h)
        setattr(res_rnn, "grad_"+att, grad)

    return res_rnn



if __name__ == "__main__":


    #01

    data = getdata()


    numbunique = findnumbunique(data)
    print(numbunique)

    my_map, inv_map = createmappingfunc(data)

    seq_length = 10
    X_chars = OneHotEncoding(data,my_map,begin = 0, end = seq_length-1)
    Y_chars = OneHotEncoding(data,my_map,begin = 1, end = seq_length)


    print(X_chars.shape)
    print(Y_chars.shape)

    #02

    RNN = RNN(K = X_chars.shape[0], m = 100)

    #03

    #a_t,h_t,o_t,p_t = RNN.eval_RNN(h0, x0, 20)
    h_prev = np.zeros((RNN.m,1))

    print(X_chars.shape, "x shape")
    result = RNN.synth_text(h_prev, X_chars[:, 0], 20)

    printsentence(result)


    print(X_chars.shape,Y_chars.shape)

    grad_U, grad_W, grad_V, grad_b, grad_c, loss, hprev = RNN.CompGrads(X_chars, Y_chars, h_prev)

    computeGrads_Num(RNN, X_chars, Y_chars)