import os
import re
import numpy as np
from itertools import chain
from gensim.models import Word2Vec
import pickle as pkl
from collections import Counter


class DataHelper:
    def __init__(self):
        pass

    def preprocessing(self, s):
        s = s.lower()
        s = re.sub(r"[^\w\s!?\.,]", " ", s)
        s = re.sub(r"\d+", " ", s)
        s = re.sub(r"([\.,?!])", r" \1 ", s)
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    def load_data(self, path_data):
        if os.path.isfile(path_data):
            with open(path_data) as f:
                raw_data = f.read().splitlines()
        elif os.path.isdir(path_data):
            raw_data = []
            for file in os.listdir(path_data):
                with open(os.path.join(path_data, file)) as f:
                    raw_data.extend(f.read().splitlines())

        raw_data = list(map(lambda s: self.preprocessing(s), raw_data))
        tokens = [[t for t in s.split()] for s in raw_data]
        tokens = list(map(lambda t: t + ["_eos_"], tokens))
        return tokens

    def fit_transform(self, tokens, min_count):
        words = list(chain.from_iterable(tokens))
        w2c = Counter(words)
        w2c_filter = {w: c for w, c in w2c.items() if c >= min_count}
        uniq_wors = list(w2c_filter)
        self.stoi = {w: i for i, w in enumerate(uniq_wors)}
        self.stoi["_pad_"] = len(self.stoi)
        self.stoi["_sos_"] = len(self.stoi)
        self.stoi["_eos_"] = len(self.stoi)
        self.stoi["_unk_"] = len(self.stoi)
        self.itos = {i: t for t, i in self.stoi.items()}

    def filters(self, tokens):
        tokens_filtered = []
        for token in tokens:
            intersect = set(token) & set(self.stoi)
            if len(intersect) == len(token):
                tokens_filtered.append(token)
        return tokens_filtered

    def pretrained_embedding(self, tokens, emb_dim):
        embedding_matrix = np.zeros((len(self.stoi), emb_dim))
        w2v_model = Word2Vec(tokens, min_count=1, 
            workers=4, size=emb_dim, iter=100)
        for w, i in self.stoi.items():
            if w in w2v_model.wv.vocab.keys():
                vector = w2v_model.wv.vocab[w].index
                embedding_matrix[i] = w2v_model.wv.vectors[vector]
            else:
                embedding_matrix[i] = np.random.randn(emb_dim)
        return embedding_matrix

    def transform(self, s):
        seq = [self.stoi.get(w, self.stoi["_unk_"]) for w in s]
        return seq

    def get_input_target(self, sequences, maxlen):
        raw_inputs = []
        raw_targets = []
        sequences = list(chain.from_iterable(sequences))
        input_sequences = sequences[:-1]
        target_sequences = sequences[1:]
        num_seq = (len(sequences) - 1) // maxlen
        input_sequences = input_sequences[: num_seq * maxlen]
        target_sequences = target_sequences[: num_seq * maxlen]
        for i in range(0, len(input_sequences), maxlen):
            raw_inputs.append(input_sequences[i : i + maxlen])
            raw_targets.append(target_sequences[i : i + maxlen])
        return raw_inputs, raw_targets

    def one_hot_encoding(self, sequence, maxlen):
        one_hot = np.zeros((maxlen, len(self.stoi)))
        one_hot[([i for i in range(maxlen)], sequence)] = 1
        return one_hot

    def batching(self, sequences, batch_size):
        batches = []
        num_batch = len(sequences) // batch_size
        sequences = sequences[: num_batch * batch_size]
        for i in range(0, len(sequences), batch_size):
            batches.append(np.array(sequences[i : i + batch_size]))
        return batches

    def trainset_preparation(
        self, path_data, emb_dim, maxlen, batch_size, min_count, pretrained
    ):
        tokens = self.load_data(path_data)
        print("total data:", len(tokens))
        self.fit_transform(tokens, min_count)
        if min_count > 1:
            tokens = self.filters(tokens)
            print("total data after filtered:", len(tokens))
        if pretrained is True:
            emb_matrix = self.pretrained_embedding(tokens, emb_dim)
        else:
            emb_matrix = None
        sequences = list(map(lambda s: self.transform(s), tokens))
        inputs, targets = self.get_input_target(sequences, maxlen)
        targets = list(map(lambda x: self.one_hot_encoding(x, maxlen), targets))
        inputs = self.batching(inputs, batch_size)
        targets = self.batching(targets, batch_size)
        targets = list(map(lambda t: np.transpose(t, (1, 0, 2)), targets))
        return emb_matrix, inputs, targets

    def save(self, path):
        with open(path, "wb") as f:
            pkl.dump(self, f)


if __name__ == "__main__":
    helper = DataHelper()
    path_data = "./data/corona_tweet.txt"
    emb_dim = 50
    batch_size = 16
    maxlen = 20
    emb_matrix, inputs, targets = helper.trainset_preparation(
        path_data, emb_dim, maxlen, batch_size
    )
    print(emb_matrix)
    print(inputs[:2])
    print(targets[:2])
