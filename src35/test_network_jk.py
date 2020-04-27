import unittest
import numpy as np
import network_jk

class test_nn(unittest.TestCase):
    def test_sigmoid(self):
        pass
       
    # def test_guess_1(self):
    #     input_layer = np.array([-2,-1,0,1,0])
    #     network = network_jk.NN_info([5], network_jk.sigmoid_f)
    #     self.assertEqual(type(network.guess(input_layer)),
    #                     type(np.array([0.1, 1.1])))
    # ok

    # def test_guess_2(self):
    #     # mockup NN object
    #     network = network_jk.NN_info([2, 1], network_jk.simple_sig)
    #     network.weights = [np.array([[2, 3]])]
    #     network.biases = [np.array([[1]])]

    #     # input values
    #     input_layer = np.array([1,2])

        
    #     np_eq = np.array_equal(network.guess(input_layer), 9.0)
    #     self.assertEquals(np_eq, True)

    
    def test_sigmoid(self):
        ws = np.array([[1., 1., 1., 1., 1.],
                    [2., 1., 1., 1., 1.],
                    [3., 1., 1., 1., 1.],
                    [4., 1., 1., 1., 1.]])

        xs = np.array([[1.],
                        [1.],
                        [1.],
                        [1.],
                        [1.]])

        b = np.array([[-1.],
                        [-2.],
                        [-3.],
                        [-4.]])

        # NUMPY element-wise vector product of sigmoid_f (viz. excel)
        res_sigm_f = np.array([[0.498321169],
                                [0.333058144],
                                [0.24994302],
                                [0.199986582]])
        
        # Actual product of sigmoid_f -- not working element-wise
        res_sigm_f_2 = np.array([[0.98201379],
                                [0.98201379],
                                [0.98201379],
                                [0.98201379]])

        # https://stackoverflow.com/questions/3302949/best-way-to-assert-for-numpy-array-equality
        #  numpy.testing.assert_array_equal(arr1, arr2)
        # nefunguje kvůli floating point toleranci

        # np.testing.assert_allclose(x, y, rtol=1e-5, atol=0)
        # funguje
        try:
            np.testing.assert_allclose(network_jk.sigmoid_f(ws, xs, b), res_sigm_f_2, rtol=1e-5)
            res = True
        except AssertionError as err:
            res = False
            print (err)
        self.assertTrue(res)


    # def test_guess_1000(self):
    #     input_layer = np.array([-2,-1,0,1,0])
    #     network = network_jk.NN_info([5,4,3,2])
    #     self.assertEqual(type(network.guess(input_layer)),
    #                      type(np.array([0.1, 1.1]))) 



 #  & "C:/Program Files/Python38/python.exe" -m unittest c:/Users/kalina.BUDEJOVICE/Scripts/neural-networks-and-deep-learning/src35/test_network_jk.py

if __name__ == "__main__":
    unittest.main()