# LSTM-Language-Model-using-Numpy
LSTM Architecture for Language Model case using Numpy only

## preparation
make directory ./binaries <br/>
make directory ./data and put training data into it

## How to train?
run `python train.py`

## How to generate text result?
run `python lm.py -sent 'string inputs' -topk {max k probability (positive int)} -len_sent {length text generated (positive int)}`

