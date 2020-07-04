import numpy as np
import sys
from helper import DataHelper
from layers import Layers
from nn import crossentropyloss, accuracy

path_data = "./data/all_liric.txt"
EMB_DIM = 100
BATCH_SIZE = 2
MAXLEN = 10
HIDDEN_SIZE = 100
EPOCHS = 500
MIN_COUNT = 1
lr = 0.5
pretrained = False
activation = "linear"

helper = DataHelper()
emb_matrix, inputs, targets = helper.trainset_preparation(
    path_data, EMB_DIM, MAXLEN, BATCH_SIZE, MIN_COUNT, pretrained
)
VOCAB_SIZE = len(helper.stoi)
helper.save("./binaries/vocab.pkl")

itos = {i: t for t, i in helper.stoi.items()}
lm = Layers(VOCAB_SIZE, EMB_DIM, HIDDEN_SIZE, emb_matrix, activation)


def generate(sent_input, len_sent):
    sent_input = helper.preprocessing(sent_input)
    token = sent_input.split()
    sequence = helper.transform(token)
    for i in range(len_sent):
        inputs = np.array([sequence])
        prob = lm.forward(inputs)[-1][-1]
        index = np.argmax(prob)
        sequence.append(index)
    print(" ".join([itos[idx] for idx in sequence]))


for e in range(EPOCHS):
    print("epoch: {}/{}".format(e + 1, EPOCHS))
    total_loss = 0
    total_acc = 0
    for i, (x, y) in enumerate(zip(inputs, targets)):
        embedded, output, prob = lm.forward(x)
        lm.backward(embedded, output, prob, y)
        lm.step(lr)

        loss = crossentropyloss(y, prob)
        total_loss += loss
        avg_loss = total_loss / (i + 1)

        acc = accuracy(y, prob)
        total_acc += acc
        avg_acc = total_acc / (i + 1)

        sys.stdout.write(
            "\riter: {}/{}; loss: {:.3f}; avg_loss: "
            "{:.3f}; acc: {:.3f}; avg_acc: {:.3f}".format(
                i + 1, len(inputs), loss, avg_loss, acc, avg_acc
            )
        )

    if e == 0:
        lowest_loss = avg_loss
    else:
        if lowest_loss > avg_loss:
            lm.save("./binaries/model.pkl")
            lowest_loss = avg_loss

    print()
    generate("cinta", 10)
    print("-" * 50)
