import unittest
import numpy as np
import network_jk

class test_nn(unittest.TestCase):
    # def test_sigmoid(self):
    #     pass
        
    def test_guess_1(self):
        input_layer = np.array([-2,-1,0,1,0])
        network = network_jk.NN_info([5], network_jk.sigmoid_f)
        self.assertEqual(type(network.guess(input_layer)),
                         type(np.array([0.1, 1.1])))
    # ok

    def test_guess_2(self):
        # mockup NN object
        network = network_jk.NN_info([2, 1], network_jk.simple_sig)
        network.weights = [np.array([[2, 3]])]
        network.biases = [np.array([[1]])]

        # input values
        input_layer = np.array([1,2])

        np_eq = np.array_equal(network.guess(input_layer), 9.0)
        self.assertEquals(np_eq, True)

    # def test_guess_1000(self):
    #     input_layer = np.array([-2,-1,0,1,0])
    #     network = network_jk.NN_info([5,4,3,2])
    #     self.assertEqual(type(network.guess(input_layer)),
    #                      type(np.array([0.1, 1.1]))) 



 #  & "C:/Program Files/Python38/python.exe" -m unittest c:/Users/kalina.BUDEJOVICE/Scripts/neural-networks-and-deep-learning/src35/test_network_jk.py