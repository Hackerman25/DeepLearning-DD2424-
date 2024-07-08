import numpy as np

def getdata():
    fileurl = r"C:\Users\marcu\Documents\GitHub\DeepLearning-DD2424-\Datasets\goblet_book.txt"
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


class RNN:
    def __init__(self,m= 100,eta= 0.1,seq_length= 25):
        self.m = m
        self.K = m
        self.eta = eta
        self.seq_length = seq_length

        self.b = np.zeros((self.m,1))
        self.c = np.zeros((self.K, 1))

        sig = 0.01
        self.U = np.random.normal(0, 1, size = (self.m,self.K)) * sig
        self.W = np.random.normal(0, 1, size=(self.m, self.m)) * sig
        self.V = np.random.normal(0, 1, size=(self.K, self.m)) * sig


    def eval_RNN(self,h0,x0,n):
        b = RNN.b
        c = RNN.c

        U = RNN.U
        W = RNN.W
        V = RNN.V

        #eq 1-4
        xnext = x0

        a_t = W @ h0 + U @ xnext + b
        h_t = np.tanh(a_t)
        o_t = V @ h_t + c
        p_t = np.exp(o_t[-1]) / np.sum(np.exp(o_t[-1]), axis=0, keepdims=True) #softmax

        return a_t,h_t,o_t,p_t

    def synth_text(self,h0, x0, n):


        for t in range(n):


if __name__ == "__main__":


    #01

    data = getdata()


    numbunique = findnumbunique(data)
    print(numbunique)

    my_map, inv_map = createmappingfunc(data)
    print(my_map)
    print(inv_map)


    print(my_map[data[0]])
    print(inv_map[0])


    #02

    RNN = RNN()

    #03

    #eval_RNN(h0, x0, n)


