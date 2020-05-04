import numpy as np
from math import exp

class NN_info:
    def __init__(self, nn_size, neuron_f):
        '''
        nn_size =  rozměry objektu  tzv. kolik bude nouronů v jaké vrstvě
            např. (5,4,3,2)
        neufon_f = funkce pouzita pro neuron (sigmoidni nebo jina)
        '''
        self.nn_size = nn_size
        self.biases = [np.random.randn(x, 1) for x in nn_size[1:]]
        self.weights = [np.random.randn(z,y) for y,z in zip(nn_size[:-1], nn_size[1:])]
        self.neuron_f = neuron_f

    def guess(self, input_data):
        '''
        input_data = numpy array with measured values
        result = values in last layer of our network
        '''
        # toto je spatne
        # result = self.neuron_f(self.weights, input_data, self.biases)
        if len(self.nn_size) == 1:
            # print(self.nn_size)
            return input_data
        else:
            result = []
            for i in range(self.nn_size[1]):
                ws = self.weights[i,:]
                xs = input_data
                b = self.biases[:,i]
                ss = simple_sig(ws, xs, b)
                print(ss, type(ss))
                result.append(ss)
        print("jsem tady", result)


        return np.array(result)
        # return np.array(result)

        


def sigmoid_f(ws, xs, b):
    '''
    ws = np.array with weights [1rows x 5columns]matrix for specific neuron
    xs = np.array with values from prevois layer [5rows x 1column]matrix
    b = bias
    returns value of 1-neuron as float

    it is a good idea to optimize this, because it can be performed
    to whole layer of neurons at once
    '''
    return 1/(1 + exp(-np.dot(ws,xs) -b))
    
    
def simple_sig(ws, xs, b):
    # only for testing purposes
    return int(np.dot(ws, xs) + b)




# Mějme neuronovou síť [5,4,3,2]
# 1. funkce která načte data
# 2. prohnat data sítí
# 3. Učení se, porovnání s hodnotou, která by měla vyjít




if __name__ == "__main__":
    test_ws = np.random.randint(1,5, size=(4,5))
    test_x = np.random.randint(1,5, size=(5,1))
    network = NN_info([5,4,3,2], sigmoid_f)
    print(network.weights)


    '''
    ws = np.array with weights [4rows x 5columns]matrix
    xs = np.array with values from prevois layer [5rows x 1column]matrix
    b = bias

    SIGMOID = 1/(1+exp(-∑j*wj*xj -b))
    wj =([
      [ w1,  w2,  w3,  w4,  w5],    
      [ w6,  w7,  w8,  w9, w10],    
      [w11, w12, w13, w14, w15],    
      [w16, w17, w18, w19, w20]     
    ])

    xj = ([
       [x1],
       [x2],
       [x3],
       [x4],
       [x5]
    ])

    ∑j*wj*xj = ([
        [(w1*x1)+(w2*x2)+(w3*x3)+(w4*x4)+(w5*x5)],    
        [(w6*x1)+(w7*x2)+(w8*x3)+(w9*x4)+(w10*x5)],     
        [(w14*x1)+(w12*x2)+(w13*x3)+(w14*x4)+(w15*x5)],     
        [(w16*x1)+(w17*x2)+(w18*x3)+(w19*x4)+(w20*x5)],  
    ])
    '''