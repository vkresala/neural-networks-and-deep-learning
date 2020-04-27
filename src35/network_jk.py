import numpy as np

class NN_info:
    def __init__(self, nn_size, sigmoid_f):
        '''
        nn_size =  rozměry objektu  tzv. kolik bude nouronů v jaké vrstvě
        např. (5,4,3,2)

        [[-1, 1, 0, -0.5], [0.1, 0.7, -0.8], [0, 1.1]]
        '''
        self.nn_size = nn_size
        self.biases = [np.random.randn(x, 1) for x in nn_size[1:]]
        self.weights = [np.random.randn(z,y) for y,z in zip(nn_size[:-1], nn_size[1:])]
        self.sigmoid_f = sigmoid_f

    def guess(self, input_data):
        '''
        input_data = numpy array with measure values
        result = values in last layer of our network
        '''
        # self.sigmoid_f
        return input_data


def sigmoid_f(ws, xs, b):
    '''
    ws = np.array with weights [4rows x 5columns]matrix
    xs = np.array with values from prevois layer [5rows x 1column]matrix
    b = bias
    '''
    return 1/(1 + np.exp(-np.dot(ws,xs) -b))
    
    
def simple_sig(ws, xs, b):
    # only for testing purposes
    return np.dot(ws,xs)+b




# Mějme neuronovou síť [5,4,3,2]
# 1. funkce která načte data
# 2. prohnat data sítí
# 3. Učení se, porovnání s hodnotou, která by měla vyjít




if __name__ == "__main__":
    test_ws = np.random.randint(1,5, size=(4,5))
    test_x = np.random.randint(1,5, size=(5,1))
    network = NN_info([5,4,3,2])
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