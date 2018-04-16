import numpy as np
import PreProcess
import process_movement


class RBM:
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

    def train(self, data, max_epochs=1000, learning_rate=0.3):

        num_examples = data.shape[0]

        data = np.insert(data, 0, 1, axis=1)

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

            self.weights += learning_rate * ((pos_associations - neg_associations) / num_examples)

            error = np.sum((data - neg_visible_probs) ** 2)
            print("Epoch %s: error is %s" % (epoch, error))
        print("Error for layer 1 is:", error)

    def run_visible(self, data):

        num_examples = data.shape[0]


        hidden_states = np.ones((num_examples, self.num_hidden + 1))

        data = np.insert(data, 0, 1, axis=1)

        hidden_activations = np.dot(data, self.weights)
        hidden_probs = self._logistic(hidden_activations)

        hidden_states[:, :] = hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)

        hidden_states = hidden_states[:, 1:]
        return hidden_states

    def run_hidden(self, data):

        num_examples = data.shape[0]

        visible_states = np.ones((num_examples, self.num_visible + 1))

        data = np.insert(data, 0, 1, axis=0)

        visible_activations = np.dot(data, self.weights.T)
        visible_probs = self._logistic(visible_activations)
        visible_states[:, :] = visible_probs > np.random.rand(num_examples, self.num_visible + 1)

        visible_states = visible_states[:, 1:]
        print(visible_states)
        return visible_states

    

    def _logistic(self, x):
        return 1.0 / (1 + np.exp(-x))


if __name__ == '__main__':
    r = RBM(num_visible=23, num_hidden=13)
    training_data, labels = PreProcess.parse()
   # print(training_data.shape)
    training_data, labels = process_movement.parse()
    print(training_data.shape)
    r.train(training_data, max_epochs=100)
    #user = np.array([training_data[1]])
    #print(user)
    #print(labels[1])
    total = training_data.shape[0]
    correct = 0
    iter = 0
    for example in training_data:
        example = np.array([example])
        print(r.run_visible(example))
        if labels[iter] == r.run_visible(example).all():
            correct += 1
        iter += 1
    print("The RBM has ", correct, "correct classifications out of", total, " examples")