import numpy as np
import copy
import matplotlib.pyplot as plt

np.random.seed(1)



def getdata():
    fileurl = r"C:\Users\Marcus\Documents\GitHub\DeepLearning-DD2424-\Datasets\goblet_book.txt"
    file = open(fileurl, "r")
    data = file.read()

    book_chars = list(set(data))


    return data, book_chars




def createmappingfunc(file):               #CORRECTCONFIRMED

    output1 = sorted(set(file))

    my_map = dict((j,i) for i,j in enumerate(output1))
    inv_map = {v: k for k, v in my_map.items()}

    return my_map, inv_map


def OneHotEncoding(data, mymap,begin, end):              #CORRECTCONFIRMED

    K = len(mymap)
    N = end - begin


    EncodedData = np.zeros((K,N))


    for i in range(N):
        EncodedData[mymap[data[begin + i]],i] = 1



    return EncodedData




class RNN:
    def __init__(self,m,K,seq_length= 25):
        #print(K,"ddk")
        self.m = m
        self.K = K

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


        #softmax
        tmp = np.exp(o_t - np.max(o_t, axis=0))
        p_t = tmp / tmp.sum(axis= 0)




        return a_t,h_t,o_t,p_t

    def synth_text(self,h0, x0, n):

        Y = np.zeros((self.K, n))
        for t in range(n):

            a_t,h_t,o_t,p_t = RNN.eval_RNN(h0, x0)



            cp = np.cumsum(p_t)
            #idx = np.random.choice(self.K, p=p_t.flat)

            unif_sampler = np.random.uniform(0, 1, size=1)  # sample a random value between 0 and 1
            idx = list(unif_sampler <= cp).index(True)


            x0 = np.zeros(x0.shape)
            x0[idx] = 1
            Y[idx, t] = 1


        return Y



    def CompGrads(self, X, Y, h0):

        seq_len = X.shape[1]

        loss = 0
        a = np.zeros((seq_length, RNN.m, 1))  # 9 x m x 1 = 100 x 1
        h = np.zeros((seq_length, RNN.m, 1))  # 9 x m x 1
        o = np.zeros((seq_length, X_chars.shape[0], RNN.m))  # 9 x k x m = 83 x 100
        p = np.zeros((seq_length, X_chars.shape[0], 1))  # 9 x k x 1
        h[-1] = h0

        grad_o = np.zeros((seq_length, 1, X_chars.shape[0]))  # 9 x 1 x 83
        grad_h = np.zeros((seq_length, 1, RNN.m))
        grad_a = np.zeros((seq_length, 1, RNN.m))
        grad_U = np.zeros_like(self.U)
        grad_W = np.zeros_like(self.W)
        grad_V = np.zeros_like(self.V)
        grad_b = np.zeros_like(self.b)
        grad_c = np.zeros_like(self.c)

        # forward pass
        for t in range(seq_len):
            a[t], h[t], o[t], p[t] = self.eval_RNN(h[t - 1], X[:, t])
            loss += -np.log(Y[:, t].reshape(Y.shape[0], 1).T @ p[t])
        # backward pass
        for t in reversed(range(seq_len)):
            grad_o[t] = -(Y[:, t].reshape(Y.shape[0], 1) - p[t]).T
            grad_V += grad_o[t].T @ h[t].T
            grad_c += grad_o[t].T

            if t == seq_len - 1:
                grad_h[t] =   grad_o[t] @ self.V
            else:
                grad_h[t] =  grad_o[t] @ self.V + grad_a[t+1]  @ self.W
            grad_a[t] = grad_h[t] * (1 - np.tanh(a[t].T) ** 2)





            grad_W += grad_a[t].T @ h[t - 1].T

            grad_U += (grad_a[t].T) @ (X[:, t].reshape(X.shape[0], 1).T)
            grad_b += grad_a[t].T

        grad_U, grad_W, grad_V, grad_b, grad_c = np.clip(grad_U, -5, 5), np.clip(grad_W, -5, 5), np.clip(grad_V, -5,5), np.clip(grad_b, -5, 5), np.clip(grad_c, -5, 5)

        hprev = h[-1]



        return grad_U, grad_W, grad_V, grad_b, grad_c, loss, hprev





    def printsentence(self,result):
        index = np.where(result == 1)[0]

        print("sentence:")
        for letter in index:

            print(inv_map[letter], end = "")
        print("\n")

    def compute_grads_num_slowGIT(self, X, Y, h):

        grad_U, grad_W, grad_V = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(self.V)
        grad_b, grad_c = np.zeros_like(self.b), np.zeros_like(self.c)
        h0 = np.zeros((self.m, 1))

        def compute_gradients(param, grad):
            param_try = np.copy(param)
            for i in np.ndindex(param.shape):
                param[i] = param_try[i] - h
                c1 = self.CompGrads(X, Y, h0)[5][0, 0]

                param[i] = param_try[i] + h
                c2 = self.CompGrads(X, Y, h0)[5][0, 0]

                grad[i] = (c2 - c1) / (2 * h)
                param[i] = param_try[i]
            return grad

        grad_b = compute_gradients(self.b, grad_b)
        grad_c = compute_gradients(self.c, grad_c)
        grad_U = compute_gradients(self.U, grad_U)
        grad_W = compute_gradients(self.W, grad_W)
        grad_V = compute_gradients(self.V, grad_V)

        return (np.clip(grad_U, -5, 5),
                np.clip(grad_W, -5, 5),
                np.clip(grad_V, -5, 5),
                np.clip(grad_b, -5, 5),
                np.clip(grad_c, -5, 5))




    def testgradients(self,X_chars, Y_chars):

        h = 1e-5
        h0 = np.zeros((self.m, 1))
        grad_U, grad_W, grad_V, grad_b, grad_c, _, _ = self.CompGrads(X_chars, Y_chars, h0)

        epsilon = np.finfo(np.float64).eps

        # ComputeGradsNumericalySlowly
        print('Compute Grads slowly')
        grad_U_slow, grad_W_slow, grad_V_slow, grad_b_slow, grad_c_slow = self.compute_grads_num_slowGIT(X_chars, Y_chars, h=h)

        grads = [grad_U, grad_W, grad_V, grad_b, grad_c]
        grads_slow = [grad_U_slow, grad_W_slow, grad_V_slow, grad_b_slow, grad_c_slow]
        names = ['U', 'W', 'V', 'b', 'c']

        for name, grad, grad_slow in zip(names, grads, grads_slow):
            gap = np.divide(np.abs(grad - grad_slow), np.maximum(epsilon, np.abs(grad) + np.abs(grad_slow)))
            print(f"{name}: maximum {np.max(gap)}, mean {np.mean(gap)}")


        

    def train(self,epochs,eta,eps):
        print("Training:")

        seq_length = 25

        G = [np.zeros_like(self.U),
            np.zeros_like(self.W),
            np.zeros_like(self.V),
            np.zeros_like(self.b),
            np.zeros_like(self.c)]



        losslist = []
        lendata = len(data)



        h_prev = np.zeros((RNN.m, 1))
        for e in range(0,(lendata-seq_length)*epochs,seq_length):



            X_chars = OneHotEncoding(data, my_map, begin=e%(lendata-seq_length), end=e%(lendata-seq_length) + seq_length)
            Y_chars = OneHotEncoding(data, my_map, begin=e%(lendata-seq_length) + 1, end=e%(lendata-seq_length) + seq_length+1)





            grad_U, grad_W, grad_V, grad_b, grad_c, loss, h = self.CompGrads(X_chars,Y_chars,h_prev)

            #Adagrad updatestep

            gradlist = [grad_U, grad_W, grad_V, grad_b, grad_c]
            #paramlist = [self.U, self.W, self.V, self.b, self.c]





            if e == 0:
                smoothloss = loss
            else:
                smoothloss = .999 * smoothloss + .001 * loss


            for i in range(len(gradlist)):

                G[i] += np.power(gradlist[i], 2)




            self.U -= np.multiply((eta /  np.sqrt(G[0]+eps)) , gradlist[0])
            self.W -= np.multiply((eta / np.sqrt(G[1] + eps)) , gradlist[1])
            self.V -= np.multiply((eta / np.sqrt(G[2] + eps)) , gradlist[2])
            self.b -= np.multiply((eta / np.sqrt(G[3] + eps)) , gradlist[3])
            self.c -= np.multiply((eta / np.sqrt(G[4] + eps)) , gradlist[4])




            h_prev = h

            if (e/(seq_length) % 10000) == 0:
                losslist.append(smoothloss[0,0])
                print("\n")
                print("Epoch:", e // (lendata-seq_length), "iter:", e/(seq_length), "loss", loss, "smoothloss", smoothloss)



                text = self.synth_text(h_prev, X_chars[:,0], 1000) #HERE



                for i in range(0,text.shape[1]):
                    indexone = np.where(text[:, i] == 1)[0][0]
                    print(inv_map[indexone],end ="" )
                #print(inv_map[text[:,0]])



        return losslist








if __name__ == "__main__":



    data, book_chars = getdata()


    my_map, inv_map = createmappingfunc(data)

    seq_length = 25
    X_chars = OneHotEncoding(data,my_map,begin = 0, end = seq_length-1)
    Y_chars = OneHotEncoding(data,my_map,begin = 1, end = seq_length)

    RNN = RNN(K=X_chars.shape[0], m=100)
    m = RNN.m
    n = RNN.seq_length
    K = RNN.K
    h_prev = np.zeros((RNN.m, 1))


    #print(X_chars.shape)
    #print(Y_chars.shape)

    #02

    #rnn, h_prev, smooth_loss, current_update = train_AdaGrad(RNN, data, num_steps=10, eta=0.1, h0=h_prev)



    #03

    #a_t,h_t,o_t,p_t = RNN.eval_RNN(h0, x0, 20)
    #h_prev = np.zeros((RNN.m,1))

    #print(X_chars.shape, "x shape")
    #result = RNN.synth_text(h_prev, X_chars[:, 0], 20)

    #RNN.printsentence(result)


    #print(X_chars.shape,Y_chars.shape)


    #print("testing gradients")
    #RNN.testgradients(X_chars,Y_chars)





    #05

    losslist = RNN.train(epochs= 15,eta=0.005,eps=1e-8)


    plt.plot(losslist)

    plt.title('Smoothed Loss Every 10 000 Iterations')
    plt.xlabel('x 10 000 Iterations')
    plt.ylabel('Smooth Loss')

    plt.show()





