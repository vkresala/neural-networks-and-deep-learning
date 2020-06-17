import numpy as np
import mnist_loader
from math import exp

class NN_info:
    def __init__(self, nn_size, neuron_f='simple_sig', round_f='get_round'):
        '''
        nn_size =  rozměry objektu  tzv. kolik bude nouronů v jaké vrstvě
            např. (5,4,3,2)
        neufon_f = funkce pouzita pro neuron (sigmoidni nebo jina)
        '''
        self.nn_size = nn_size
        self.biases = [np.random.randn(x, 1) for x in nn_size[1:]]
        self.weights = [np.random.randn(z,y) for y,z in zip(nn_size[:-1], nn_size[1:])]
        
        if neuron_f == 'simple_sig':
            self.neuron_f = self.simple_sig
        elif neuron_f == 'sigmoid_f':
            self.neuron_f = self.sigmoid_f
        else:
            print('wrong name of neuron function')

        if round_f == 'get_max':
            self.round_f = self.get_max
        elif round_f == 'get_round':
            self.round_f = self.get_round
        else:
            print('wrong max/round function')


    # def feed_forward(self, input_data):
    #     '''
    #     input_data = numpy array with measured values
    #     result = values in last layer of our network
    #     '''
    #     if len(self.nn_size) == 1:
    #         return input_data
    #     else:
    #         xs = input_data
    #         for layer in range(len(self.nn_size)-1):
    #             out_values = []
    #             for n in range(self.nn_size[layer+1]):
    #                 ws = self.weights[layer][n,:]
    #                 b = self.biases[layer][:,n]
    #                 ss = self.neuron_f(ws, xs, b)
    #                 out_values.append(ss)
    #             xs = out_values

    #     return np.array([xs]).transpose()

    def feed_forward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = self.neuron_f(np.dot(w, a)+b)
        return a


    def evaluate(self, input_data, expected_value):
        '''
        input_data = np array with measured valeus for one picture  
        expected value = how last layer of neurons should look like
        returns something like
        [[4.]
         [5.]]
        '''
        # (A==B).all()
        # np.round(data, 2)
        if type(expected_value) == int:
            expected_value = mnist_loader.vectorized_result(expected_value,
                                                            self.nn_size[-1])
    
        network_output = self.feed_forward(input_data)
        network_output = self.round_f(network_output)
        if (network_output == expected_value).all():
            return True
        else:
            return False

    def backprop(self, x, y):
        pass


    def get_round(self, network_output):
        '''
        network_output = output vercor from neural netowrk
        round all numbers to closest integer
        return the rounded np.array
        '''
        return np.round(network_output, 0)


    # def get_max(self, network_output):
    #     '''
    #     network_output = output vercor from neural netowrk
    #     on max position of vector makes 1, other values to zero
    #     returns the array
    #     '''
    #     i = network_output.argmax()
    #     return mnist_loader.vectorized_result(i, self.nn_size[-1])


    # def sigmoid_f(self, ws, xs, b):
    #     '''
    #     ws = np.array with weights [1rows x 5columns]matrix for specific neuron
    #     xs = np.array with values from prevois layer [5rows x 1column]matrix
    #     b = bias
    #     returns value of 1-neuron as float

    #     it is a good idea to optimize this, because it can be performed
    #     to whole layer of neurons at once
    #     '''
    #     return 1/(1 + exp(-np.dot(ws,xs) -b))
        
        
    # def simple_sig(self, ws, xs, b):
    #     # only for testing purposes
    #     return int(np.dot(ws, xs) + b)

    def sigmoid_f(self, z):
        '''
        ws = np.array with weights [1rows x 5columns]matrix for specific neuron
        xs = np.array with values from prevois layer [5rows x 1column]matrix
        b = bias
        returns value of 1-neuron as float

        it is a good idea to optimize this, because it can be performed
        to whole layer of neurons at once
        '''
        return 1/(1 + exp(-z))
        
        
    def simple_sig(self, z):
        # only for testing purposes
        return int(z)


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

    def __init__(self, data_path, sizes, neuron_f ='simple_sig'):
        self.data_path = data_path
        self.results = []
        self.network = NN_info(sizes, neuron_f=neuron_f)
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
            picture_array = picture[0]
            expected_number = picture[1]

            if self.network.evaluate(picture_array,expected_number):
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

    paths = {'Vojta': r'C:\Users\vojte\Documents\GitHub\neural-networks-and-deep-learning\src35\mnist.pkl.gz',
        'JirkaNB':r'C:\Users\kalina.BUDEJOVICE\Scripts\neural-networks-and-deep-learning\src35\mnist.pkl.gz',
        'JirkaPC': r'C:\Users\krumm\scripts\neural-networks-and-deep-learning\src35\mnist.pkl.gz'}

    p = paths['Vojta']
    evalator = Evaluator(p, [784, 10])
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