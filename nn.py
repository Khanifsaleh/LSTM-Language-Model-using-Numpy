import numpy as np
import math as m


def sigmoid(vector, grad=False):
    if grad:
        return vector * (1 - vector)
    else:
        return 1 / (1 + np.exp(-vector))


def tanh(vector, grad=False):
    if grad:
        return 1 - vector ** 2
    else:
        numerator = np.exp(vector) - np.exp(-vector)
        denominator = np.exp(vector) + np.exp(-vector)
        return numerator / denominator


def relu(vector, grad=False):
    if grad:
        return np.where(vector > 0, 1, 0)
    else:
        return np.maximum(vector, 0)


def softmax(inputs, axis):
    return np.exp(inputs) / np.sum(np.exp(inputs), axis=axis, keepdims=True)


def accuracy(y_true, y_hat):
    maxlen, batch_size, vocab_size = y_true.shape
    a = np.argmax(y_hat, 2)
    b = np.argmax(y_true, 2)
    return np.sum(a == b) / (maxlen * batch_size)


def crossentropyloss(y_true, y_hat):
    maxlen, bs, vocab_size = y_true.shape
    return -np.sum(y_true * np.log(y_hat)) / (maxlen * bs)


class Embedding:
    def __init__(self, vocab_size=None, embedding_dim=None, pretrained=None):
        if pretrained is not None:
            self.emb_matrix = pretrained
        else:
            # k = m.sqrt(1/embedding_dim)
            # self.emb_matrix = np.random.uniform(low=-k, high=k,
            #     size=(vocab_size, embedding_dim))
            self.emb_matrix = np.random.randn(vocab_size, embedding_dim)

    def to_vector(self, inputs):
        vector = [[self.emb_matrix[t] for t in seq] for seq in inputs]
        vector = np.array(vector)
        vector = np.transpose(vector, (1, 0, 2))
        return vector

    def update(self, x_inputs, emb_grad, lr):
        bs, maxlen = x_inputs.shape
        id_vec = {}
        for b in range(bs):
            for t in range(maxlen):
                idx = x_inputs[b, t]
                if idx in id_vec:
                    id_vec[idx] = (id_vec[idx] + emb_grad[t, b, :]) / 2
                else:
                    id_vec[idx] = emb_grad[t, b, :]

        for idx in id_vec:
            self.emb_matrix[idx] = self.emb_matrix[idx] - lr * id_vec[idx]


class LSTM:
    def __init__(self, feature_size, hidden_size, activation):
        self.ftr_sz = feature_size
        self.h_sz = hidden_size
        self.activation = activation
        self.init_prm()

    def init_prm(self):
        # k = m.sqrt(1/self.h_sz)
        """
        inisialisasi bobot parameter
        ftr_prm = parameter feature (4, feature_size, hidden_size)
        hs_prm = parameter hidden_state (4, hidden_size, hidden_size)
        Pada ft_prm dan hs_prm, masing2 index pada 0 axis scr 
        berurutan mewakili bobot pada 4 gates, yaitu:
            - ft: forget gate
            - it: ignore gate
            - lt: learning gate
            - ot: output gate
        bias = parameter bias (1, hidden_size)
        """
        # self.ftr_prm = np.array([np.random.uniform(low=-k, high=k,
        #     size=(self.ftr_sz, self.h_sz)) for _ in range(4)])
        # self.hs_prm = np.array([np.random.uniform(low=-k, high=k,
        #     size=(self.h_sz, self.h_sz)) for _ in range(4)])
        # self.ftr_bias = np.array([np.random.uniform(low=-k, high=k,
        #     size=(1, self.h_sz)) for _ in range(4)])
        # self.hs_bias = np.array([np.random.uniform(low=-k, high=k,
        #     size=(1, self.h_sz)) for _ in range(4)])

        self.ftr_prm = np.array(
            [np.random.randn(self.ftr_sz, self.h_sz) for _ in range(4)]
        )
        self.hs_prm = np.array(
            [np.random.randn(self.h_sz, self.h_sz) for _ in range(4)]
        )
        self.ftr_bias = np.array([np.random.randn(1, self.h_sz) for _ in range(4)])
        self.hs_bias = np.array([np.random.randn(1, self.h_sz) for _ in range(4)])

    def forward_gates(self, xt, ht):
        z_ftr = np.einsum("be,neh->nbh", xt, self.ftr_prm)
        z_hs = np.einsum("bh,nhl->nbl", ht, self.hs_prm)
        gates_t = z_ftr + self.ftr_bias + z_hs + self.hs_bias
        gates_t[0] = sigmoid(gates_t[0])
        gates_t[1] = sigmoid(gates_t[1])
        gates_t[2] = tanh(gates_t[2])
        gates_t[3] = sigmoid(gates_t[3])
        return gates_t

    def forward(self, X, h0, c0):
        """
        Proses forward propagation
        X shape (maxlen, batch size, embedding dim)
        t = time step
        t-1 = time step sebelumnya

        xt input feature pada time step ke-t
        h0=c0 shape (maxlen, batch size, hidden size)
        ft = sigmoid(inputs x ftr_prm[0] + ftr_bias[0] + \
            ht-1 x hs_prm[0] + hs_bias[0])
        it = sigmoid(inputs x ftr_prm[1] + ftr_bias[1] + \
            ht-1 x hs_prm[1] + hs_bias[1])
        lt = tanh(inputs x ftr_prm[2] + ftr_bias[2] + \
            ht-1 x hs_prm[2] + hs_bias[2])
        ot = sigmoid(inputs x ftr_prm[3] + ftr_bias[3] + \
            ht-1 x hs_prm[3] + hs_bias[3])
        ct aka cell_state ke-t = (ft * ct-1) + (it * lt)
        ht aka hidden state ke-t = ot * tanh(ct)
        """
        maxlen, bs, embedding_dim = X.shape
        output = np.zeros((maxlen, bs, self.h_sz))
        self.cell_states = []
        self.gates = []
        for t in range(maxlen):
            xt = X[t]
            if t == 0:
                ht = h0
                ct = c0
            gates_t = self.forward_gates(xt, ht)
            ct = gates_t[0] * ct + gates_t[1] * gates_t[2]
            ht = gates_t[3] * tanh(ct)
            output[t] = ht
            self.cell_states.append(ct)
            self.gates.append(gates_t)

        if self.activation == "tanh":
            return tanh(output), tanh(ht)
        elif self.activation == "relu":
            return relu(output), relu(ht)
        elif self.activation == "sigmoid":
            return sigmoid(output), sigmoid(ht)
        else:  # aka linear
            return output, ht

    def gates_gradient(self, gates_t, ht_grad, ct, prev_ct, nx_ct_grad, nx_ft):
        ft, it, lt, ot = gates_t[0], gates_t[1], gates_t[2], gates_t[3]
        ct_grad = ht_grad * ot * (1 - (tanh(ct) ** 2)) + (nx_ct_grad * nx_ft)
        ot_grad = ht_grad * tanh(ct) * sigmoid(ot, grad=True)
        lt_grad = ct_grad * it * tanh(lt, grad=True)
        it_grad = ct_grad * lt * sigmoid(it, grad=True)
        ft_grad = ct_grad * prev_ct * sigmoid(ft, grad=True)
        gates_grad = np.array([ft_grad, it_grad, lt_grad, ot_grad])
        return ct_grad, gates_grad

    def ftr_prm_grad(self, xt, gates_grad_t):
        wxt_grad = np.einsum("eb,nbh->neh", xt.T, gates_grad_t)
        bxt_grad = np.expand_dims(gates_grad_t.sum(1), 1)
        return wxt_grad, bxt_grad

    def ftr_grad_t(self, gates_grad_t):
        ftr_prm = np.transpose(self.ftr_prm, (0, 2, 1))
        return np.einsum("nbh,nhe->be", gates_grad_t, ftr_prm) / 4

    def hs_prm_grad(self, ht, nx_gates_grad_t):
        wht_grad = np.einsum("hb,nbl->nhl", ht.T, nx_gates_grad_t)
        bht_grad = np.expand_dims(nx_gates_grad_t.sum(1), 1)
        return wht_grad, bht_grad

    def hs_grad(self, gates_grad_t):
        hs_prm = np.transpose(self.hs_prm, (0, 2, 1))
        return np.einsum("nbh,nhl->bl", gates_grad_t, hs_prm) / 4

    def backward(self, X, H, output_grad):
        maxlen, bs, hidden_size = output_grad.shape
        wx_grad = np.zeros(self.ftr_prm.shape)
        wh_grad = np.zeros(self.hs_prm.shape)
        bx_grad = np.zeros(self.ftr_bias.shape)
        bh_grad = np.zeros(self.hs_bias.shape)
        self.x_grad = np.zeros(X.shape)

        for t in range(maxlen, 0, -1):
            t -= 1

            if t == maxlen - 1:
                ht_grad = output_grad[t]
                nx_ft, nx_ct_grad = 0, 0
            else:
                ht_grad = output_grad[t] + ht_grad
                nx_ft = self.gates[t + 1][0]
                nx_ct_grad = ct_grad
                wht_grad, bht_grad = self.hs_prm_grad(H[t], gates_grad_t)
                wh_grad += wht_grad
                bh_grad += bht_grad

            if t == 0:
                prev_ct = np.zeros((bs, hidden_size))
            else:
                prev_ct = self.cell_states[t - 1]

            ct = self.cell_states[t]
            gates_t = self.gates[t]
            ct_grad, gates_grad_t = self.gates_gradient(
                gates_t, ht_grad, ct, prev_ct, nx_ct_grad, nx_ft
            )
            wxt_grad, bxt_grad = self.ftr_prm_grad(X[t], gates_grad_t)
            wx_grad += wxt_grad
            bx_grad += bxt_grad
            ht_grad = self.hs_grad(gates_grad_t)
            xt_grad = self.ftr_grad_t(gates_grad_t)
            self.x_grad[t] = xt_grad

        self.wx_grad = wx_grad / (maxlen * bs)
        self.bx_grad = bx_grad / (maxlen * bs)
        self.wh_grad = wh_grad / ((maxlen - 1) * bs)
        self.bh_grad = bh_grad / ((maxlen - 1) * bs)

    def step(self, lr):
        self.ftr_prm = self.ftr_prm - lr * self.wx_grad
        self.hs_prm = self.hs_prm - lr * self.wh_grad
        self.ftr_bias = self.ftr_bias - lr * self.bx_grad
        self.hs_bias = self.hs_bias - lr * self.bh_grad


class Linear:
    def __init__(self, input_sz, output_sz, activation):
        self.activation = activation
        k = m.sqrt(1 / input_sz)
        # self.w = np.random.uniform(low=-k, high=k,
        #     size=(input_sz, output_sz))
        # self.b = np.random.uniform(low=-k, high=k,
        #     size=(1, output_sz))

        self.w = np.random.randn(input_sz, output_sz)
        self.b = np.random.randn(1, output_sz)

    def forward(self, inputs):
        return np.dot(inputs, self.w) + self.b

    def backward(self, logits_grad, output):
        """
        Turunan loss terhadap bobot parameter di linear.
        dL/dw = dL/dlogits * dlogits/dw
        dlogits/db = 1
        dL/db = dL/dlogits
        """
        maxlen, bs, hidden_sz = output.shape
        dlogits_dw = np.transpose(output, (0, 2, 1))
        w_grad = np.einsum("mhb,mbv->mhv", dlogits_dw, logits_grad) / bs
        b_grad = np.expand_dims(logits_grad.sum(0).sum(0), 0) / bs
        self.w_grad = w_grad.sum(0) / (maxlen)
        self.b_grad = b_grad / (maxlen)
        output_grad = np.einsum("mbv,vh->mbh", logits_grad, self.w.T)
        if self.activation == "tanh":
            output_grad = tanh(output_grad, grad=True)
        elif self.activation == "relu":
            output_grad = relu(output_grad, grad=True)
        elif self.activation == "sigmoid":
            output_grad = sigmoid(output_grad, grad=True)
        return output_grad

    def step(self, lr):
        self.w = self.w - (lr * self.w_grad)
        self.b = self.b - (lr * self.b_grad)
