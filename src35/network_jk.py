import numpy as np

class NN_info:
    def __init__(self, nn_size):
        '''
        nn_size =  rozměry objektu  tzv. kolik bude nouronů v jaké vrstvě
        např. (5,4,3,2)

        [[-1, 1, 0, -0.5], [0.1, 0.7, -0.8], [0, 1.1]]
        '''
        self.nn_size = nn_size
        self.biases = [np.random.randn(x, 1) for x in nn_size[1:]]
        self.weights = [np.random.randn(z,y) for y,z in zip(nn_size[:-1], nn_size[1:])]
    
def sigmoid_f(ws, xs, b):
    '''
    ws = np.array with weights
    xs = np.array with values from brevois layer
    b = bias
    '''
    # domaci ukol
    pass


network = NN_info([5,4,3,2])
print(network.weights)