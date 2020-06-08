import unittest
import numpy as np
import network_jk

class test_nn(unittest.TestCase):
    # @unittest.skip("skip")
    def test_feed_forward_0_return_type(self):
        input_layer = np.array([-2,-1,0,1,0])
        network = network_jk.NN_info([5], neuron_f='sigmoid_f')

        type_network_result = type(network.feed_forward(input_layer))
        type_should_be = type(np.array([0.1, 1.1]))

        self.assertEqual(type_network_result, type_should_be)


    # @unittest.skip("skip")
    def test_feed_forward_1(self):
        # mockup NN object
        network = network_jk.NN_info([2, 2])
        network.weights = [np.array([[1, 1], [1, 1]])]
        network.biases = [np.array([[1, 1]])]

        # input values
        input_layer = np.array([1,2])


        result = network.feed_forward(input_layer).tolist()
        self.assertEqual(result, [[4], [4]])


    # @unittest.skip("skip")
    def test_feed_forward_2(self):
        # mockup NN object
        network = network_jk.NN_info([5, 3])
        network.weights = [np.array([[1,1,1,1,2], [1,1,1,1,3], [1,1,1,1,4],])]
        network.biases = [np.array([[1, 1, 1]])]

        # input values
        input_layer = np.array([1,2,3,4,5])


        result = network.feed_forward(input_layer).tolist()
        self.assertEqual(result, [[21], [26], [31]])


    # @unittest.skip("skip")
    def test_feed_forward_3(self):
        # mockup NN object
        network = network_jk.NN_info([2, 2, 2])
        network.weights = [np.array([[1,1], [1,1]]), np.array([[1,1], [1,2]])]
        network.biases = [np.array([[1, 1]]), np.array([[0, 0]])]

        # input values
        input_layer = np.array([1,2])


        result = network.feed_forward(input_layer).tolist()
        self.assertEqual(result, [[8], [12]])


    # @unittest.skip("skip")
    def test_feed_forward_4(self):
        # mockup NN object
        network = network_jk.NN_info([2, 2, 2])
        network.weights = [np.array([[1,1], [1,1]]), np.array([[1,1], [1,2]])]
        network.biases = [np.array([[1, 1]]), np.array([[0, 0]])]

        # input values
        input_layer = np.array([1,2])


        # result = network.feed_forward(input_layer).tolist()
        self.assertEqual(True, True)

    # @unittest.skip("skip")
    def test_get_max(self):
        network = network_jk.NN_info([2, 5])
        net_out_mock = np.array([[-1.],[0.],[1.],[2.],[1.5]])
        result = network.get_max(net_out_mock).tolist()

        self.assertEqual(result, [[0.],  [0.],  [0.], [1.], [0.]])


    # @unittest.skip("skip")
    def test_evaluate_true(self):
        # mockup NN object
        network = network_jk.NN_info([2, 2])
        network.weights = [np.array([[1, 1], [1, 1]])]
        network.biases = [np.array([[1, 2]])]
        # input values
        input_layer = np.array([1,2])
        expected_value = np.array([[4.], [5]])
        # expected_value[0] = 4.0
        print()
        print(network.feed_forward(input_layer))
        print(expected_value)
        print()

        self.assertTrue(network.evaluate(input_layer, expected_value))


    # @unittest.skip("skip")
    def test_evaluate_true2(self):
        # mockup NN object
        network = network_jk.NN_info([2, 2])
        network.weights = [np.array([[1, 1], [1, 1]])]
        network.biases = [np.array([[-3, -2]])]

        # input values
        input_layer = np.array([1,2])
        # expected_value = np.array([[0, 1]])
        expected_value = 1
        
        # print("##############################")
        # print(network.feed_forward(input_layer, ))
        self.assertTrue(network.evaluate(input_layer, expected_value))


    # @unittest.skip("skip")
    def test_sigmoid_2(self):
        ws = np.array([[1., 1., 1., 1., 1.]])
        xs = np.array([[1.],[1.],[1.],[1.],[1.]])
        b = np.array([[-1.]])

        res_sigm_f = 0.98201379

        self.assertAlmostEqual(network_jk.NN_info.sigmoid_f(1, ws, xs, b),
                               res_sigm_f)


    # @unittest.skip("skip")
    def test_sigmoid_3(self):
        ws = np.array([[2., 1., 1., 1., 1.]])
        xs = np.array([[1.],[1.],[1.],[1.],[1.]])
        b = np.array([[-10.]])

        res_sigm_f = 0.01798621

        self.assertAlmostEqual(network_jk.NN_info.sigmoid_f(1, ws, xs, b),
                               res_sigm_f)


    # @unittest.skip("skip")
    def test_sigmoid_4(self):
        ws = np.array([[3., 1., 1., 1., 1.]])
        xs = np.array([[1.],[1.],[1.],[1.],[1.]])
        b = np.array([[-15.]])

        res_sigm_f = 0.00033535

        self.assertAlmostEqual(network_jk.NN_info.sigmoid_f(1, ws, xs, b),
                               res_sigm_f)


    # @unittest.skip("skip")
    def test_sigmoid_5(self):
        ws = np.array([[4., 1., 1., 1., 1.]])
        xs = np.array([[1.],[1.],[1.],[1.],[1.]])
        b = np.array([[2.]])

        res_sigm_f = 0.999954602

        self.assertAlmostEqual(network_jk.NN_info.sigmoid_f (1, ws, xs, b),
                               res_sigm_f)


    # @unittest.skip("skip")
    def test_simple_sig(self):
        ws = np.array([[4., 1., 1., 1., 1.]])
        xs = np.array([[1.],[1.],[1.],[1.],[1.]])
        b = np.array([[2.]])
        res_simple_sig = 10
        self.assertAlmostEqual(network_jk.NN_info.simple_sig(1, ws, xs, b),
                               res_simple_sig)


    # @unittest.skip("skip")
    def test_simple_sig_type(self):
        ws = np.array([[4., 1., 1., 1., 1.]])
        xs = np.array([[1.],[1.],[1.],[1.],[1.]])
        b = np.array([[2.]])
        result_type = type(network_jk.NN_info.simple_sig(1, ws, xs, b))
        expected_type = type(10)

        self.assertEqual(result_type, expected_type)


'''
4.5.2020
otestovali jsme, ze nam funguje sigmoidni funkce, zjednodusena funkce,
feed_forward funkce funguje na sit o rozmerech 2x2

priste budeme pridavat test case
- sit o rozmerech 5x3
- sit o rozmerech 2x2x2

13.5.2020
dodělali jsem feed forward, otestovali a funguje
vytvorili jsme funkci evaluate, ktera jeste neni otestovana > napsat 2 testy,
pouzit+upravit ty, ktere uz jsou

priste nebo do priste udelame objekt, ktery nacte data a vyhodnoti, jak
se siti dari odhadovat vysledky
pouzit mnist_loader podobne jako v averages

18.5.2020
v classe jsme vytvorili a otestovali metodu evaluate, ktera vyhodnocuje, jestli
sit vyhodnotila data uspesne
vytvorili evaluator tridu, kterou budeme pouzivat pro zjisteni uspesnosti 
vytrenovane neuronove site

DU
dumper, success rate

25.5.2020
prosli jsme opacko objektoveho programovani
uptavovali metody result_dumper a get_succes_rate
konecny stav - error v simple sig, nove metody vypadaji dobre...
TypeError: only size-1 arrays can be converted to Python scalars
line 71, in simple_sig

DU zkusit rozbehat

2.6. 2020
upravena funkce evaluate(), by vracela vektorizovanou expected_value


8.6.2020
fixed tests and bugs
added sigmoid to main class
added two rounding options

we tried running the main script and it doesnt work 
TypeError: only size-1 arrays can be converted to Python scalars

next session - fix this and show results, start with learning of the network
'''

if __name__ == "__main__":
    unittest.main()


