import unittest
import numpy as np
import network_jk

class test_nn(unittest.TestCase):
    def test_guess_0_return_type(self):
        input_layer = np.array([-2,-1,0,1,0])
        network = network_jk.NN_info([5], network_jk.sigmoid_f)
        self.assertEqual(type(network.guess(input_layer)),
                        type(np.array([0.1, 1.1])))

    def test_guess_1(self):
        # mockup NN object
        network = network_jk.NN_info([2], network_jk.simple_sig)

        # input values
        input_layer = np.array([1,2])
        result = network.guess(input_layer)

        # self.assertEqual(arr1.tolist(), arr2.tolist())
        
        np_eq = np.array_equal(result, np.array([1,2]))
        self.assertEqual(np_eq, True)


    # def test_guess_2(self):
    #     # mockup NN object
    #     network = network_jk.NN_info([2, 1], network_jk.simple_sig)

    #     # hacking part start
    #     network.weights = [np.array([[2], [3]])] # ws
    #     network.biases = [np.array([[1]])] # b
    #     # hacking part end

    #     # input values
    #     input_layer = np.array([[1,2]]) # xs

    #     result = network.guess(input_layer)
    #     print("****************")
    #     print()
    #     print("****************")

    #     np_eq = np.array_equal(result, np.array([9.0]))
    #     self.assertEqual(np_eq, True)

    
    def test_sigmoid_2(self):
        ws = np.array([[1., 1., 1., 1., 1.]])
        xs = np.array([[1.],[1.],[1.],[1.],[1.]])
        b = np.array([[-1.]])

        res_sigm_f = 0.98201379

        self.assertAlmostEqual(network_jk.sigmoid_f(ws, xs, b), res_sigm_f)

    def test_sigmoid_3(self):
        ws = np.array([[2., 1., 1., 1., 1.]])
        xs = np.array([[1.],[1.],[1.],[1.],[1.]])
        b = np.array([[-10.]])

        res_sigm_f = 0.01798621

        self.assertAlmostEqual(network_jk.sigmoid_f(ws, xs, b), res_sigm_f)

    def test_sigmoid_4(self):
        ws = np.array([[3., 1., 1., 1., 1.]])
        xs = np.array([[1.],[1.],[1.],[1.],[1.]])
        b = np.array([[-15.]])

        res_sigm_f = 0.00033535

        self.assertAlmostEqual(network_jk.sigmoid_f(ws, xs, b), res_sigm_f)


    def test_sigmoid_5(self):
        ws = np.array([[4., 1., 1., 1., 1.]])
        xs = np.array([[1.],[1.],[1.],[1.],[1.]])
        b = np.array([[2.]])

        res_sigm_f = 0.999954602

        self.assertAlmostEqual(network_jk.sigmoid_f(ws, xs, b), res_sigm_f)



        # https://stackoverflow.com/questions/3302949/best-way-to-assert-for-numpy-array-equality
        #  numpy.testing.assert_array_equal(arr1, arr2)
        # nefunguje kvůli floating point toleranci
        # np.testing.assert_allclose(x, y, rtol=1e-5, atol=0)
        # try:
        #     np.testing.assert_allclose(network_jk.sigmoid_f(ws, xs, b), res_sigm_f, rtol=1e-5)
        #     res = True
        # except AssertionError as err:
        #     res = False
        #     print (err)
        # self.assertTrue(res)


    # def test_guess_1000(self):
    #     input_layer = np.array([-2,-1,0,1,0])
    #     network = network_jk.NN_info([5,4,3,2])
    #     self.assertEqual(type(network.guess(input_layer)),
    #                      type(np.array([0.1, 1.1]))) 



 #  & "C:/Program Files/Python38/python.exe" -m unittest c:/Users/kalina.BUDEJOVICE/Scripts/neural-networks-and-deep-learning/src35/test_network_jk.py

if __name__ == "__main__":
    unittest.main()
