import unittest
import numpy as np
from influence.evo_network import NeuralNetwork

class TestNetwork(unittest.TestCase):
    def test_activations_a(self):
        """Test tanh hidden activation with softmax output activation"""
        nn = NeuralNetwork(
            num_inputs=2,
            num_outputs=2,
            num_hidden=2,
            hidden_activation_func='tanh',
            output_activation_func='softmax'
        )
        # Set weights arbitrarily
        nn.setWeights(list(range(nn.num_weights)))

        # Let's compute this manually outside the network
        input = np.array([4,5])

        # add the bias
        hidden_b = np.hstack((input, [1]))
        # feedforward and sum
        hidden_f = hidden_b.dot(nn.weights[0])
        # Activate the hidden layer
        hidden_a = np.tanh(hidden_f)

        # Add the bias for the output layer
        out_b = np.hstack((hidden_a, [1]))
        # feedforward and sum
        out_f = out_b.dot(nn.weights[1])
        # Activate the output layer
        expected_out = nn.softmax(out_f)

        # Get the output of the network
        out = nn.forward(np.array([4,5]))

        print(expected_out.shape)

        # Check sizing
        self.assertTrue(expected_out.size == out.size)
        self.assertTrue(len(expected_out.shape) == len(out.shape))
        for exp_dim, dim in zip(expected_out.shape, out.shape):
            self.assertTrue(exp_dim == dim)

        # Output values should sum to 1.0 because output layer is softmax
        self.assertTrue(np.isclose(1.0, np.sum(out)))

        # Check values
        for exp_val, val in zip(expected_out, out):
            self.assertTrue(np.isclose(exp_val, val))

        pass

    def test_activations_b(self):
        """Test relu hidden activation with softmax output activation"""
        nn = NeuralNetwork(
            num_inputs=2,
            num_outputs=2,
            num_hidden=2,
            hidden_activation_func='relu',
            output_activation_func='softmax'
        )
        # Set weights arbitrarily
        nn.setWeights(list(range(nn.num_weights)))

        # Let's compute this manually outside the network
        input = np.array([4,5])

        # add the bias
        hidden_b = np.hstack((input, [1]))
        # feedforward and sum
        hidden_f = hidden_b.dot(nn.weights[0])
        # Activate the hidden layer
        hidden_a = nn.relu(hidden_f)
        # self.assertTrue(hidden_a.size == 2)

        # Add the bias for the output layer
        out_b = np.hstack((hidden_a, [1]))
        # feedforward and sum
        out_f = out_b.dot(nn.weights[1])
        # Activate the output layer
        expected_out = nn.softmax(out_f)

        # Get the output of the network
        out = nn.forward(np.array([4,5]))

        # Check sizing
        self.assertTrue(expected_out.size == out.size)
        self.assertTrue(len(expected_out.shape) == len(out.shape))
        for exp_dim, dim in zip(expected_out.shape, out.shape):
            self.assertTrue(exp_dim == dim, msg=f'{exp_dim} != {dim}')

        # Output values should sum to 1.0 because output layer is softmax
        self.assertTrue(np.isclose(1.0, np.sum(out)))

        # Check values
        for exp_val, val in zip(expected_out, out):
            self.assertTrue(np.isclose(exp_val, val))

    def test_activations_c(self):
        """Test tanh hidden activation with tanh output activation"""
        nn = NeuralNetwork(
            num_inputs=2,
            num_outputs=2,
            num_hidden=2,
            hidden_activation_func='tanh',
            output_activation_func='tanh'
        )
        # Set weights arbitrarily
        nn.setWeights(list(range(nn.num_weights)))

        # Let's compute this manually outside the network
        input = np.array([4,5])

        # add the bias
        hidden_b = np.hstack((input, [1]))
        # feedforward and sum
        hidden_f = hidden_b.dot(nn.weights[0])
        # Activate the hidden layer
        hidden_a = np.tanh(hidden_f)

        # Add the bias for the output layer
        out_b = np.hstack((hidden_a, [1]))
        # feedforward and sum
        out_f = out_b.dot(nn.weights[1])
        # Activate the output layer
        expected_out = np.tanh(out_f)

        # Get the output of the network
        out = nn.forward(np.array([4,5]))

        # Check sizing
        self.assertTrue(expected_out.size == out.size)
        self.assertTrue(len(expected_out.shape) == len(out.shape))
        for exp_dim, dim in zip(expected_out.shape, out.shape):
            self.assertTrue(exp_dim == dim, msg=f'{exp_dim} != {dim}')

        # Check values
        for exp_val, val in zip(expected_out, out):
            self.assertTrue(np.isclose(exp_val, val))

    def test_activations_d(self):
        """Test relu hidden activation with tanh output activation"""
        nn = NeuralNetwork(
            num_inputs=2,
            num_outputs=2,
            num_hidden=2,
            hidden_activation_func='relu',
            output_activation_func='tanh'
        )
        # Set weights arbitrarily
        nn.setWeights(list(range(nn.num_weights)))

        # Let's compute this manually outside the network
        input = np.array([4,5])

        # add the bias
        hidden_b = np.hstack((input, [1]))
        # feedforward and sum
        hidden_f = hidden_b.dot(nn.weights[0])
        # Activate the hidden layer
        hidden_a = nn.relu(hidden_f)

        # Add the bias for the output layer
        out_b = np.hstack((hidden_a, [1]))
        # feedforward and sum
        out_f = out_b.dot(nn.weights[1])
        # Activate the output layer
        expected_out = np.tanh(out_f)

        # Get the output of the network
        out = nn.forward(np.array([4,5]))

        # Check sizing
        self.assertTrue(expected_out.size == out.size)
        self.assertTrue(len(expected_out.shape) == len(out.shape))
        for exp_dim, dim in zip(expected_out.shape, out.shape):
            self.assertTrue(exp_dim == dim, msg=f'{exp_dim} != {dim}')

        # Check values
        for exp_val, val in zip(expected_out, out):
            self.assertTrue(np.isclose(exp_val, val))

if __name__=='__main__':
    unittest.main()
