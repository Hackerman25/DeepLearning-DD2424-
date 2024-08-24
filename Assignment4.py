import numpy as np
import copy

np.random.seed(0)



def getdata():
    fileurl = r"C:\Users\marcus\Documents\GitHub\DeepLearning-DD2424-\Datasets\goblet_book.txt"
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

        b = self.b
        c = self.c

        U = self.U
        W = self.W
        V = self.V


        #eq 1-4
        xnext = x0  # 83 x 1

        a_t = W @ h0 + U @ xnext[:,np.newaxis] + b  # 100x100  x  100x1  +  100x83  x  83x1       #CORRECT


        h_t = np.tanh(a_t)                                                                        #CORRECT
        o_t = V @ h_t + c                                                                         #CORRECT

        a = np.exp(o_t - np.max(o_t, axis=0))
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

    def CompGradsGITmin(self,X_chars,Y_chars,h0):

        seq_length = X_chars.shape[1]

        a = np.zeros((seq_length,RNN.m,1))  #9 x m x 1 = 100 x 1
        h = np.zeros((seq_length,RNN.m,1))      #9 x m x 1
        o = np.zeros((seq_length,X_chars.shape[0],RNN.m))    # 9 x k x m = 83 x 100
        p = np.zeros((seq_length,X_chars.shape[0],1))          # 9 x k x 1
        h[-1] = h0


        loss = 0

        grad_o = np.zeros((seq_length,1,X_chars.shape[0]))  #9 x 1 x 83
        grad_o2 = np.zeros((seq_length,X_chars.shape[0],RNN.m))

        grad_h = np.zeros((seq_length,1,RNN.m))

        grad_a = np.zeros((seq_length,1,RNN.m))

        grad_U, grad_W, grad_V, grad_b, grad_c = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(self.V), np.zeros_like(self.b), np.zeros_like(self.c)



        #forward pass                                   SAME AS GITHUB
        for t in range(seq_length):                                        #GOOD
            #eq 1-4

            a[t], h[t], o[t], p[t] = self.eval_RNN(h[t-1], X_chars[:,t])

            # 100 x 1

            #loss += -np.log(Y_chars[:,t].T @ p[t])
            loss += -np.log(np.matmul(Y_chars[:, t].reshape(Y_chars.shape[0], 1).T, p[t]))

            #backward   pass


        for t in reversed(range(seq_length)):

            grad_o[t] = -(Y_chars[:, t].reshape(Y_chars.shape[0], 1) - p[t]).T               # 1 x k   GOOD


            #reshaped_array = np.reshape(array, (83, 1))

            grad_V += grad_o[t].T @ h[t].T       #k x m  = 83 x 100                                    GOOD

            grad_c += grad_o[t].T          #k x 1                                                      GOOD

            if t == seq_length-1:

                grad_h[t] =  grad_o[t] @ RNN.V                   #m x 1 =  (83,)   @   (83, 100)   = 83 x 1 (with added newaxis)
                grad_a[t] = np.multiply(grad_h[t], (1 - np.tanh(a[t].T) ** 2))  # (1,m)     #m x 1 = 100 x 1 =   100 x 1  @  (1,) (with new axis) =   100 x 1


            else:
                grad_h[t] = (grad_o[t] @ RNN.V) + grad_a[t+1]  @ RNN.W                #m x 1 =  (83,)  @  (83,100) + (100,1) @ (100,100)
                grad_a[t] = np.multiply(grad_h[t], (1 - np.tanh(a[t].T) ** 2))  # (1,m)


            grad_W += grad_a[t].T @ h[t - 1].T  # (m,m)


            grad_U += grad_a[t].T @ X_chars[:, t].reshape(X_chars.shape[0], 1).T  # (m,K)
            grad_b += grad_a[t].T  # (m,1)

        """
        for t in reversed(range(seq_length)):
            grad_o[t] = -(Y_chars[:, t].reshape(Y_chars.shape[0], 1) - p[t]).T  # (1,K)
            grad_V += np.matmul(grad_o[t].T, h[t].T)  # (K,m)
            grad_c += grad_o[t].T  # (K,1)
            if t == seq_length - 1:

                grad_h[t] = np.matmul(grad_o[t], self.V)  # (1,m)

                grad_a[t] = np.multiply(grad_h[t], (1 - np.tanh(a[t].T) ** 2))  # (1,m)

            else:
                grad_h[t] = np.matmul(grad_o[t], self.V) + np.matmul(grad_a[t + 1], self.W)  # (1,m)
                grad_a[t] = np.multiply(grad_h[t], (1 - np.tanh(a[t].T) ** 2))  # (1,m)
            grad_W += np.matmul(grad_a[t].T, h[t - 1].T)  # (m,m)
            grad_U += np.matmul(grad_a[t].T, X_chars[:, t].reshape(X_chars.shape[0], 1).T)  # (m,K)
            grad_b += grad_a[t].T  # (m,1)

        """





        grad_U, grad_W, grad_V, grad_b, grad_c = np.clip(grad_U, -5, 5), np.clip(grad_W, -5, 5) ,np.clip(grad_V, -5, 5), np.clip(grad_b, -5, 5) , np.clip(grad_c, -5, 5)

        hprev = h[seq_length - 1]

        return grad_U, grad_W, grad_V, grad_b, grad_c, loss, hprev

    def CompGradsGIT(self, X, Y, h0):
        """
        Parameters:
            X (K, n): the input matrix
            Y (K, n): the output matrix
            h0: initial hidden states

        Returns:
            grads: gradient values of all hyper-parameters
            l: loss value
            hprev: previous hidden states
        """
        seq_len = X.shape[1]

        l = 0
        a, h, o, p = {}, {}, {}, {}
        h[-1] = h0
        grad_a, grad_h, grad_o = {}, {}, {}
        grad_U, grad_W, grad_V, grad_b, grad_c = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(self.V), np.zeros_like(self.b), np.zeros_like(self.c)

        # forward-pass
        for t in range(seq_len):
            a[t], h[t], o[t], p[t] = self.eval_RNN(h[t - 1], X[:, t])  # (m,1), (m,1), (K,1), (K,1)
            l += -np.log(np.matmul(Y[:, t].reshape(Y.shape[0], 1).T, p[t]))  # (1,1)
        # backward-pass
        for t in reversed(range(seq_len)):
            grad_o[t] = -(Y[:, t].reshape(Y.shape[0], 1) - p[t]).T  # (1,K)
            grad_V += np.matmul(grad_o[t].T, h[t].T)  # (K,m)
            grad_c += grad_o[t].T  # (K,1)
            if t == seq_len - 1:
                grad_h[t] = np.matmul(grad_o[t], self.V)  # (1,m)
                grad_a[t] = np.multiply(grad_h[t], (1 - np.tanh(a[t].T) ** 2))  # (1,m)
            else:
                grad_h[t] = np.matmul(grad_o[t], self.V) + np.matmul(grad_a[t + 1], self.W)  # (1,m)
                grad_a[t] = np.multiply(grad_h[t], (1 - np.tanh(a[t].T) ** 2))  # (1,m)
            grad_W += np.matmul(grad_a[t].T, h[t - 1].T)  # (m,m)
            grad_U += np.matmul(grad_a[t].T, X[:, t].reshape(X.shape[0], 1).T)  # (m,K)
            grad_b += grad_a[t].T  # (m,1)

        grads = {
            'U': grad_U,
            'W': grad_W,
            'V': grad_V,
            'b': grad_b,
            'c': grad_c,
        }
        for g in grads:
            grads[g] = np.clip(grads[g], -5, 5)

        hprev = h[seq_len - 1]


        grad_U, grad_W, grad_V, grad_b, grad_c = grads["U"], grads['W'], grads['V'], grads['b'], grads['c']

        return grad_U, grad_W, grad_V, grad_b, grad_c, l, hprev








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
        h0 = np.zeros((self.m,1))

        # Assuming h is defined
        h = 999999999999999#1e-5  # Example value for h, replace with your actual value

        # For attribute U

        U_try = np.copy(self.U)
        print("length", len(U_try))
        for i in range(len(U_try)):
            self.U = np.array(U_try)
            self.U[i,:] -= h



            _, _, _, _, _, loss1, _ = self.CompGradsGIT(X_chars, Y_chars, h0)


            self.U = np.array(U_try)
            self.U[i,:] += h
            _, _, _, _, _, loss2, _ = self.CompGradsGIT(X_chars, Y_chars, h0)

            print("loss1", loss1, "loss2", loss2)

            grad_U[i] = (loss2 - loss1) / (2 * h)
        self.U = U_try
        self.grad_U = grad_U

        # For attribute W

        W_try = np.copy(self.W)
        for i in range(len(W_try)):
            self.W = np.array(W_try)
            self.W[i,:] -= h
            _, _, _, _, _, loss1, _ = self.CompGradsGIT(X_chars, Y_chars, h0)

            self.W = np.array(W_try)
            self.W[i,:] += h
            _, _, _, _, _, loss2, _ = self.CompGradsGIT(X_chars, Y_chars, h0)

            grad_W[i] = (loss2 - loss1) / (2 * h)
        self.W = W_try
        self.grad_W = grad_W

        # For attribute V

        V_try = np.copy(self.V)
        for i in range(len(V_try)):
            self.V = np.array(V_try)
            self.V[i] -= h
            _, _, _, _, _, loss1, _ = self.CompGradsGIT(X_chars, Y_chars, h0)

            self.V = np.array(V_try)
            self.V[i] += h
            _, _, _, _, _, loss2, _ = self.CompGradsGIT(X_chars, Y_chars, h0)

            grad_V[i] = (loss2 - loss1) / (2 * h)
        self.V = V_try
        self.grad_V = grad_V

        # For attribute b

        b_try = np.copy(self.b)
        for i in range(len(b_try)):
            self.b = np.array(b_try)
            self.b[i] -= h
            _, _, _, _, _, loss1, _ = self.CompGradsGIT(X_chars, Y_chars, h0)

            self.b = np.array(b_try)
            self.b[i] += h
            _, _, _, _, _, loss2, _ = self.CompGradsGIT(X_chars, Y_chars, h0)

            grad_b[i] = (loss2 - loss1) / (2 * h)
        self.b = b_try
        self.grad_b = grad_b

        # For attribute c

        c_try = np.copy(self.c)
        for i in range(len(c_try)):
            self.c = np.array(c_try)
            self.c[i] -= h
            _, _, _, _, _, loss1, _ = self.CompGradsGIT(X_chars, Y_chars, h0)

            self.c = np.array(c_try)
            self.c[i] += h
            _, _, _, _, _, loss2, _ = self.CompGradsGIT(X_chars, Y_chars, h0)

            grad_c[i] = (loss2 - loss1) / (2 * h)
        self.c = c_try
        self.grad_c = grad_c



        grad_U = np.clip(grad_U, -5, 5)
        grad_W = np.clip(grad_W, -5, 5)
        grad_V = np.clip(grad_V, -5, 5)
        grad_b = np.clip(grad_b, -5, 5)
        grad_c = np.clip(grad_c, -5, 5)


        return grad_U, grad_W, grad_V, grad_b, grad_c

    def compute_grads_num_slowGIT(self, X, Y, h):
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

        b_try = np.copy(self.b)
        for i in range(len(self.b)):
            self.b = np.array(b_try)
            self.b[i] -= h
            _, _, _, _, _, c1, _ = self.CompGradsGIT(X, Y, h0)

            self.b = np.array(b_try)
            self.b[i] += h
            _, _, _, _, _, c2, _ = self.CompGradsGIT(X, Y, h0)

            grad_b[i] = (c2[0,0] - c1[0,0]) / (2 * h)
        self.b = b_try

        c_try = np.copy(self.c)
        for i in range(len(self.c)):
            self.c = np.array(c_try)
            self.c[i] -= h
            _, _, _, _, _, c1, _ = self.CompGradsGIT(X, Y, h0)

            self.c = np.array(c_try)
            self.c[i] += h
            _, _, _, _, _, c2, _ = self.CompGradsGIT(X, Y, h0)

            grad_c[i] = (c2[0,0] - c1[0,0]) / (2 * h)
        self.c = c_try

        U_try = np.copy(self.U)
        for i in np.ndindex(self.U.shape):
            self.U = np.array(U_try)
            self.U[i] -= h
            _, _, _, _, _, c1, _ = self.CompGradsGIT(X, Y, h0)

            self.U = np.array(U_try)
            self.U[i] += h
            _, _, _, _, _, c2, _ = self.CompGradsGIT(X, Y, h0)


            grad_U[i] = (c2[0,0] - c1[0,0]) / (2 * h)
        self.U = U_try

        W_try = np.copy(self.W)
        for i in np.ndindex(self.W.shape):
            self.W = np.array(W_try)
            self.W[i] -= h
            _, _, _, _, _, c1, _ = self.CompGradsGIT(X, Y, h0)

            self.W = np.array(W_try)
            self.W[i] += h
            _, _, _, _, _, c2, _ = self.CompGradsGIT(X, Y, h0)

            grad_W[i] = (c2[0,0] - c1[0,0]) / (2 * h)
        self.W = W_try

        V_try = np.copy(self.V)
        for i in np.ndindex(self.V.shape):
            self.V = np.array(V_try)
            self.V[i] -= h
            _, _, _, _, _, c1, _ = self.CompGradsGIT(X, Y, h0)

            self.V = np.array(V_try)
            self.V[i] += h
            _, _, _, _, _, c2, _ = self.CompGradsGIT(X, Y, h0)

            grad_V[i] = (c2[0,0] - c1[0,0]) / (2 * h)
        self.V = V_try

        grads = {
            'U': grad_U,
            'W': grad_W,
            'V': grad_V,
            'b': grad_b,
            'c': grad_c,
        }
        for g in grads:
            grads[g] = np.clip(grads[g], -5, 5)

        grad_U = np.clip(grad_U, -5, 5)
        grad_W = np.clip(grad_W, -5, 5)
        grad_V = np.clip(grad_V, -5, 5)
        grad_b = np.clip(grad_b, -5, 5)
        grad_c = np.clip(grad_c, -5, 5)

        return grad_U, grad_W, grad_V, grad_b, grad_c




    def testgradients(self,X_chars, Y_chars):

        h = 1e-5
        h0 = np.zeros((self.m, 1))
        grad_U, grad_W, grad_V, grad_b, grad_c, _, _ = self.CompGradsGIT(X_chars, Y_chars, h0)
        print(grad_W)
        epsilon = np.finfo(np.float64).eps

        # ComputeGradsNumSlow
        print('ComputeGradsNumSlow')
        grad_U_slow, grad_W_slow, grad_V_slow, grad_b_slow, grad_c_slow = self.compute_grads_num_slowGIT(X_chars, Y_chars, h=h)
        print(grad_W_slow)

        gap_u = np.divide(np.abs(grad_U - grad_U_slow), np.maximum(epsilon, (np.abs(grad_U)) + (np.abs(grad_U_slow))))
        gap_w = np.divide(np.abs(grad_W - grad_W_slow), np.maximum(epsilon, (np.abs(grad_W)) + (np.abs(grad_W_slow))))
        gap_v = np.divide(np.abs(grad_V - grad_V_slow), np.maximum(epsilon, (np.abs(grad_V)) + (np.abs(grad_V_slow))))
        gap_b = np.divide(np.abs(grad_b - grad_b_slow), np.maximum(epsilon, (np.abs(grad_b)) + (np.abs(grad_b_slow))))
        gap_c = np.divide(np.abs(grad_c - grad_c_slow), np.maximum(epsilon, (np.abs(grad_c)) + (np.abs(grad_c_slow))))
        print("U: max {}, mean {}".format(np.max(gap_u), np.mean(gap_u)))
        print("W: max {}, mean {}".format(np.max(gap_w), np.mean(gap_w)))
        print("V: max {}, mean {}".format(np.max(gap_v), np.mean(gap_v)))
        print("b: max {}, mean {}".format(np.max(gap_b), np.mean(gap_b)))
        print("c: max {}, mean {}".format(np.max(gap_c), np.mean(gap_c)))


        """
        h0 = np.zeros((RNN.m, 1))
        grad_U, grad_W, grad_V, grad_b, grad_c, loss, hprev = self.CompGradsGIT(X_chars, Y_chars, h0)
        print(grad_U[0,:])

        grad_U, grad_W, grad_V, grad_b, grad_c = self.compute_grads_num_slowGIT(X_chars, Y_chars, h=1e-5)
        print(grad_V[0,:])
        
        """

if __name__ == "__main__":


    #01

    data = getdata()


    numbunique = findnumbunique(data)
    print(numbunique)

    my_map, inv_map = createmappingfunc(data)

    seq_length = 25
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


    #model_RNN.fit(data, sig, seq_len, epoch, eta)



