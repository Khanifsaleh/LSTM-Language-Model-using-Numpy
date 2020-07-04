from layers import Layers
import pickle as pkl
import numpy as np
import re


class LanguageModel:
    def __init__(self):
        self.load_bin()

    def load_bin(self):
        with open("./binaries/model.pkl", "rb") as f:
            self.model = pkl.load(f)
        with open("./binaries/vocab.pkl", "rb") as f:
            self.helper = pkl.load(f)

    def generate(self, inputs, topk, len_sent):
        inputs = self.helper.preprocessing(inputs)
        token = inputs.split()
        sequence = self.helper.transform(token)
        for i in range(len_sent):
            tensor = np.array([sequence])
            prob = self.model.forward(tensor)[-1][-1].squeeze()
            idx_candidate = prob.argsort()[::-1][:topk]
            idx = np.random.choice(idx_candidate)
            sequence.append(idx)
        result = " ".join([self.helper.itos[idx] for idx in sequence])
        return re.sub(r" _eos_ | _unk_ ", "\n", result)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-sent', type=str)
    parser.add_argument('-topk', default=1, type=int)
    parser.add_argument('-len_sent', type=int)
    args = vars(parser.parse_args())

    sent = args['sent']
    topk = args['topk']
    len_sent = args['len_sent']

    lm = LanguageModel()
    print(lm.generate(sent, topk, len_sent))
