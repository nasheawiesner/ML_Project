import numpy as np
import PreProcess
from RBM import *

class DBN(object):
    def __init__(self, num_visible, num_hidden):
        self.num_hidden = num_hidden
        self.num_visible = num_visible

        np_rng = np.random.RandomState(1234)

        self.weights = np.asarray(np_rng.uniform(
            low=-0.1 * np.sqrt(6.0 / (num_hidden + num_visible)),
            high=0.1 * np.sqrt(6.0 / (num_hidden + num_visible)),
            size=(num_visible, num_hidden)))

        self.weights = np.insert(self.weights, 0, 0, axis=0)
        self.weights = np.insert(self.weights, 0, 0, axis=1)

        #self.weights = self.weights.T

    def train(self, data, max_epochs=1000, learning_rate=0.5):
        num_examples = data.shape[0]

        data = np.insert(data, 0, 1, axis=2)
        data = np.reshape(data,(data.shape[0],data.shape[2]))
        for epoch in range(max_epochs):
            pos_hidden_activations = np.dot(data, self.weights)
            pos_hidden_probs = self._logistic(pos_hidden_activations)
            pos_hidden_probs[:, 0] = 1
            pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
            pos_associations = np.dot(data.T, pos_hidden_probs)

            neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
            neg_visible_probs = self._logistic(neg_visible_activations)
            neg_visible_probs[:, 0] = 1
            neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
            neg_hidden_probs = self._logistic(neg_hidden_activations)

            neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

            self.weights = self.weights + (learning_rate * ((pos_associations - neg_associations) / num_examples))

            error = np.sum((data - neg_visible_probs) ** 2)
            print("Epoch %s: error is %s" % (epoch, error))
        print("Error for dbn is:",error)

    def run_visible(self, data):
        num_examples = data.shape[0]

        hidden_states = np.ones((num_examples, self.num_hidden + 1))

        data = np.insert(data, 0, 1, axis=0)

        hidden_activations = np.dot(data, self.weights)
        hidden_probs = self._logistic(hidden_activations)

        hidden_states[:, :] = hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)

        hidden_states = hidden_states[:, 1:]
        return hidden_states

    def run_hidden(self, data):
        num_examples = data.shape[0]

        visible_states = np.ones((num_examples, self.num_visible + 1))

        data = np.insert(data, 0, 0, axis=1)

        visible_activations = np.dot(data, self.weights.T)
        visible_probs = self._logistic(visible_activations)
        visible_states[:, :] = visible_probs > np.random.rand(num_examples, self.num_visible + 1)

        visible_states = visible_states[:, 1:]

        return visible_states

    def _logistic(self, x):
        return 1.0 / (1 + np.exp(-x))


if __name__ == '__main__':
    r = RBM(num_visible=7, num_hidden=100)
    training_data = PreProcess.parse()
    new_training = []
    r.train(training_data, transform=False, max_epochs=100)
    for example in training_data:
        example = np.array([example])
        new_example = np.asarray(r.run_visible(example))
        new_training.append(new_example)
    new_training = np.asarray(new_training)
    d = DBN(num_visible=100, num_hidden=100)
    d.train(new_training, max_epochs=10000)
    #layer_three = np.asarray(d.run_visible(new_training[-1]))
    #print(d.weights)
    #print(layer_three)
    #three = RBM(num_hidden=7, num_visible=2)
    #three.train(layer_three, transform=True, max_epochs=100)

    #user = np.array([training_data[1]])
    #print(r.run_visible(user))