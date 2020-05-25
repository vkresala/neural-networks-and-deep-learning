import numpy as np
import mnist_loader
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


    def feed_forward(self, input_data):
        '''
        input_data = numpy array with measured values
        result = values in last layer of our network
        '''
        if len(self.nn_size) == 1:
            return input_data
        else:
            xs = input_data
            for layer in range(len(self.nn_size)-1):
                out_values = []
                for n in range(self.nn_size[layer+1]):
                    ws = self.weights[layer][n,:]
                    b = self.biases[layer][:,n]
                    ss = simple_sig(ws, xs, b)
                    out_values.append(ss)
                xs = out_values

        return np.array(xs)


    def evaluate(self, input_data, expected_value):
        '''
        input_data = np array with measured valeus for one picture  
        expected value = how last layer of neurons should look like
        '''
        # (A==B).all()
        # np.round(data, 2)

        network_output = self.feed_forward(input_data)
        network_output = np.round(network_output, 0)

        if (network_output == expected_value).all():
            return True
        else:
            return False


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

class Evaluator:
    '''
    1. načte data 
    2. zpracuje v siti a u někam si uloží výsledky 
    3.vyhodnotí procent. úspěšnost
    '''

    def __init__(self, data_path, sizes, sigmoid_f):
        self.data_path = data_path
        self.results = []
        self.network = NN_info(sizes, sigmoid_f)
        self.data_loader()


    def data_loader(self):
        '''
        do objektu načte data obrázků z druhe sady
        '''
        all_pics = mnist_loader.load_data_wrapper(self.data_path)
        self.validation_data = all_pics[1]
        del all_pics
        return


    def result_dumper(self):
        '''
        Data prohnat sítí a porovná s očekávaným výsledkem. Zjištěný výsledek 0
        OR 1 uloží do objektu
        '''
        self.dumper_records = []
        for picture in self.validation_data:
            if self.network.evaluate(picture[0], picture[1]):
                self.dumper_records.append(1)
            else:
                self.dumper_records.append(0)
        return


    def get_succes_rate(self):
        '''
        returns success rate (0.0-1.0 float)
        '''
        # print((sum(x))/(len(x)))
        succes_rate = sum(self.dumper_records)/len(self.dumper_records)
        return succes_rate
    


if __name__ == "__main__":
    # test_ws = np.random.randint(1,5, size=(4,5))
    # test_x = np.random.randint(1,5, size=(5,1))
    # network = NN_info([5,4,3,2], sigmoid_f)
    # print(network.weights)

    # p = r'C:\Users\kalina.BUDEJOVICE\Scripts\neural-networks-and-deep-learning\src35\mnist.pkl.gz'
    p = r'C:\Users\vojte\Documents\GitHub\neural-networks-and-deep-learning\src35\mnist.pkl.gz'
    evalator = Evaluator(p, [784, 10], simple_sig)
    evalator.result_dumper()
    print(evalator.get_succes_rate())
    
    
    
    
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