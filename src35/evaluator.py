import network
import numpy as np
import mnist_loader


class Evaluator:
    '''
    1. načte data 
    2. zpracuje v siti a u někam si uloží výsledky 
    3.vyhodnotí procent. úspěšnost
    '''

    def __init__(self, data_path, sizes, round_f ='get_max'):
        self.data_path = data_path
        self.results = []
        self.net = network.Network(sizes)
        self.data_loader()

        # if round_f == 'get_max':
        #     self.round_f = self.get_max
        # elif round_f == 'get_round':
        #     self.round_f = self.get_round
        # else:
        #     print('wrong max/round function')


    # def get_round(self, network_output):
    #     '''
    #     network_output = output vercor from neural netowrk
    #     round all numbers to closest integer
    #     return the rounded np.array
    #     '''
    #     return np.round(network_output, 0)


    # def get_max(self, network_output):
    #     '''
    #     network_output = output vercor from neural netowrk
    #     on max position of vector makes 1, other values to zero
    #     returns the array
    #     '''
    #     i = network_output.argmax()
    #     return mnist_loader.vectorized_result(i, self.nn_size[-1])


    def data_loader(self):
        '''
        do objektu načte data obrázků z druhe sady
        '''
        all_pics = mnist_loader.load_data_wrapper(self.data_path)
        self.validation_data = all_pics[1]
        del all_pics
        return


    # def result_dumper(self):
    #     '''
    #     Data prohnat sítí a porovná s očekávaným výsledkem. Zjištěný výsledek 0
    #     OR 1 uloží do objektu
    #     '''
    #     self.dumper_records = []
    #     for picture in self.validation_data:
    #         picture_array = picture[0]
    #         expected_number = picture[1]

    #         if self.network.evaluate(picture_array,expected_number):
    #             self.dumper_records.append(1)
    #         else:
    #             self.dumper_records.append(0)
    #     return


    # def evaluate(self, input_data, expected_value):
    #     '''
    #     input_data = np array with measured valeus for one picture  
    #     expected value = how last layer of neurons should look like
    #     returns something like
    #     [[4.]
    #      [5.]]
    #     '''
    #     # (A==B).all()
    #     # np.round(data, 2)
    #     if type(expected_value) == int:
    #         expected_value = mnist_loader.vectorized_result(expected_value,
    #                                                         self.nn_size[-1])
    
    #     network_output = self.feed_forward(input_data)
    #     network_output = self.round_f(network_output)
    #     if (network_output == expected_value).all():
    #         return True
    #     else:
    #         return False


    # def get_succes_rate(self):
    #     '''
    #     returns success rate (0.0-1.0 float)
    #     '''
    #     # print((sum(x))/(len(x)))
    #     succes_rate = sum(self.dumper_records)/len(self.dumper_records)
    #     return succes_rate
    


if __name__ == "__main__":
    paths = {'Vojta': r'C:\Users\vojte\Documents\GitHub\neural-networks-and-deep-learning\src35\mnist.pkl.gz',
        'JirkaNB':r'C:\Users\kalina.BUDEJOVICE\Scripts\neural-networks-and-deep-learning\src35\mnist.pkl.gz',
        'JirkaPC': r'C:\Users\krumm\scripts\neural-networks-and-deep-learning\src35\mnist.pkl.gz'}

    p = paths['Vojta']
    e = Evaluator(p, [784, 16, 16, 10])

    # vyhodnoti prvni obraze
    for (x, y) in e.validation_data:
        y = mnist_loader.vectorized_result(y)
        ff = e.net.backprop(x, y)
        break
    # print('vysledek')
    # print(ff)
    # print('\nreseni')
    # print(y)
    print('\nbackprop')
    for f in ff:
        for x in f:
            print(x.shape)



    # print(evalator.network.evaluate(evalator.validation_data))

    # evalator.result_dumper()
    # print(evalator.get_succes_rate())