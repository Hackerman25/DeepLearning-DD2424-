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

            # 100 x 1

            loss += -np.log(Y_chars[:,t].T @ p[t])

        #backward   pass


        for t in reversed(range(seq_length)):

            grad_o[t] = -(Y_chars[:, t].reshape(Y_chars.shape[0], 1) - p[t]).T               # 1 x k   GOOD


            #reshaped_array = np.reshape(array, (83, 1))

            grad_V += grad_o[t].T @ h[t].T       #k x m  = 83 x 100                                    GOOD

            grad_c += grad_o[t].T          #k x 1                                                      GOOD

            if t == seq_length-1:

                grad_h[t] = ( grad_o[t] @ RNN.V ).T                  #m x 1 =  (83,)   @   (83, 100)   = 83 x 1 (with added newaxis)
                grad_a[t] = grad_h[t] @ np.diag(1-np.tanh(a[t])**2)[:,np.newaxis]     #m x 1 = 100 x 1 =   100 x 1  @  (1,) (with new axis) =   100 x 1


            else:
                grad_h[t] = (grad_o[t] @ RNN.V).T + RNN.W @ grad_a[t+1]                  #m x 1 =  (83,)  @  (83,100) + (100,1) @ (100,100)
                grad_a[t] = grad_h[t] @ np.diag(1-np.tanh(a[t])**2)[:,np.newaxis]


            grad_W += grad_a[t] @ h[t - 1].T  # (m,m)


            grad_U += grad_a[t] @ X_chars[:, t].reshape(X_chars.shape[0], 1).T  # (m,K)
            grad_b += grad_a[t]  # (m,1)


        grad_U, grad_W, grad_V, grad_b, grad_c = np.clip(grad_U, -5, 5), np.clip(grad_W, -5, 5) ,np.clip(grad_V, -5, 5), np.clip(grad_b, -5, 5) , np.clip(grad_c, -5, 5)

        hprev = h[seq_length - 1]

        return grad_U, grad_W, grad_V, grad_b, grad_c, loss, hprev

    def printsentence(self,result):
        index = np.where(result == 1)[0]

        print("sentence:")
        for letter in index:

            print(inv_map[letter], end = "")  # ???? dont know this part to get letters
        print("\n")


    def compute_grads_num_slow(self, X_chars, Y_chars, h):
        """
        Parameters:
            X (K, n): the input matrix
            Y (K, n): the output matrix
            h: initial hidden states

        Returns:
            grads: gradient values of all hyper-parameters
        """
        grad_U, grad_W, grad_V, grad_b, grad_c = np.zeros_like(self.U), np.zeros_like(
            self.W), np.zeros_like(self.V), np.zeros_like(self.b), np.zeros_like(
            self.c)
        h0 = np.zeros((self.m, 1))

        # Assuming h is defined
        h = 500000#1e-5  # Example value for h, replace with your actual value

        # For attribute U
        U_try = np.copy(self.U)
        grad_U = np.zeros_like(U_try)

        print("length", len(U_try))
        for i in range(len(U_try)):
            self.U = np.array(U_try)
            self.U[i,:] -= h



            _, _, _, _, _, loss1, _ = self.CompGrads(X_chars, Y_chars, h0)


            self.U = np.array(U_try)
            self.U[i,:] += h
            _, _, _, _, _, loss2, _ = self.CompGrads(X_chars, Y_chars, h0)

            print("loss1", loss1, "loss2", loss2)

            grad_U[i] = (loss2 - loss1) / (2 * h)
        self.U = U_try
        self.grad_U = grad_U

        # For attribute W
        W_try = np.copy(self.W)
        grad_W = np.zeros_like(W_try)
        for i in range(len(W_try)):
            self.W = np.array(W_try)
            self.W[i] -= h
            _, _, _, _, _, loss1, _ = self.CompGrads(X_chars, Y_chars, h0)

            self.W = np.array(W_try)
            self.W[i] += h
            _, _, _, _, _, loss2, _ = self.CompGrads(X_chars, Y_chars, h0)

            grad_W[i] = (loss2 - loss1) / (2 * h)
        self.W = W_try
        self.grad_W = grad_W

        # For attribute V
        V_try = np.copy(self.V)
        grad_V = np.zeros_like(V_try)
        for i in range(len(V_try)):
            self.V = np.array(V_try)
            self.V[i] -= h
            _, _, _, _, _, loss1, _ = self.CompGrads(X_chars, Y_chars, h0)

            self.V = np.array(V_try)
            self.V[i] += h
            _, _, _, _, _, loss2, _ = self.CompGrads(X_chars, Y_chars, h0)

            grad_V[i] = (loss2 - loss1) / (2 * h)
        self.V = V_try
        self.grad_V = grad_V

        # For attribute b
        b_try = np.copy(self.b)
        grad_b = np.zeros_like(b_try)
        for i in range(len(b_try)):
            self.b = np.array(b_try)
            self.b[i] -= h
            _, _, _, _, _, loss1, _ = self.CompGrads(X_chars, Y_chars, h0)

            self.b = np.array(b_try)
            self.b[i] += h
            _, _, _, _, _, loss2, _ = self.CompGrads(X_chars, Y_chars, h0)

            grad_b[i] = (loss2 - loss1) / (2 * h)
        self.b = b_try
        self.grad_b = grad_b

        # For attribute c
        c_try = np.copy(self.c)
        grad_c = np.zeros_like(c_try)
        for i in range(len(c_try)):
            self.c = np.array(c_try)
            self.c[i] -= h
            _, _, _, _, _, loss1, _ = self.CompGrads(X_chars, Y_chars, h0)

            self.c = np.array(c_try)
            self.c[i] += h
            _, _, _, _, _, loss2, _ = self.CompGrads(X_chars, Y_chars, h0)

            grad_c[i] = (loss2 - loss1) / (2 * h)
        self.c = c_try
        self.grad_c = grad_c



        grad_U = np.clip(grad_U, -5, 5)
        grad_W = np.clip(grad_W, -5, 5)
        grad_V = np.clip(grad_V, -5, 5)
        grad_b = np.clip(grad_b, -5, 5)
        grad_c = np.clip(grad_c, -5, 5)


        return grad_U, grad_W, grad_V, grad_b, grad_c



    def testgradients(self,X_chars, Y_chars):
        grad_U, grad_W, grad_V, grad_b, grad_c, loss, hprev = self.CompGrads(X_chars, Y_chars, h_prev)
        print(grad_U[0,:])

        grad_U, grad_W, grad_V, grad_b, grad_c = self.compute_grads_num_slow(X_chars, Y_chars, h=1e-5)
        print(grad_U[0,:])

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

    RNN.printsentence(result)


    print(X_chars.shape,Y_chars.shape)

    RNN.testgradients(X_chars,Y_chars)



