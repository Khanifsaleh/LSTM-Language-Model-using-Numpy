import numpy as np
from nn import *
import pickle as pkl


class Layers:
    def __init__(
        self,
        vocab_size=None,
        emb_dim=None,
        hidden_size=None,
        emb_matrix=None,
        activation="tanh",
    ):
        self.hyper_prm = {
            "vocab_size": vocab_size,
            "emb_dim": emb_dim,
            "hidden_size": hidden_size,
        }
        self.hidden_size = hidden_size
        self.emb_matrix = emb_matrix
        self.embedding = Embedding(vocab_size, emb_dim, emb_matrix)
        self.lstm = LSTM(emb_dim, hidden_size, activation)
        self.fc = Linear(hidden_size, vocab_size, activation)

    def init_state(self, batch_size):
        h0 = np.zeros((batch_size, self.hidden_size))
        c0 = np.zeros((batch_size, self.hidden_size))
        return h0, c0

    def forward(self, inputs):
        self.inputs = inputs
        batch_size, maxlen = inputs.shape
        h0, c0 = self.init_state(batch_size)
        embedded = self.embedding.to_vector(inputs)
        output, hn = self.lstm.forward(embedded, h0, c0)
        logits = self.fc.forward(output)
        prob = softmax(logits, axis=2)
        return embedded, output, prob

    def backward(self, embedded, output, prob, y_true):
        logits_grad = prob - y_true
        output_grad = self.fc.backward(logits_grad, output)
        self.lstm.backward(embedded, output, output_grad)

    def step(self, lr):
        self.fc.step(lr)
        self.lstm.step(lr)
        if self.emb_matrix is None:
            self.embedding.update(self.inputs, self.lstm.x_grad, lr)

    def save(self, path):
        with open(path, "wb") as f:
            pkl.dump(self, f)
