<h1 style="align:center">Language Modeling</h1>

Language modeling using Deep Learning is a way of asking a model to predict which word grammatically (more like logically) comes next. This model is pretty useful in many applications. One such example being speech recognition. Consider asking a speech recgnition system - "Where can I buy a pair of shoes?". How does it know that you meant "pair" and not "pear"? It uses a language model to do it. 

In probabilistic manner we can say that a Language model gives the probability of a sentence to be likely.

## Model Specification

Note: All the specs mentioned below can be configured in `utils/config.py`.

|Property|Value|Comment|
|------:|:---:|:-----|
|Vocab Size|15k|It is a small vocab size increase it if you have the compute power|
|Embedding Size|128|I started with 50 and gradually increased to see which embedding size suits my system's compute power|
|Number of LSTM layers|2|Started with 1 but ran into underfitting with 1 LSTM layer|
|LSTM units|128|-|
|Dropout Rate|keep rate 0.7|I started with 0.5 and after 10k steps increased keep rate to 0.7|
|Learning rate|1e-1|Made many fine tunings and many runs to finally decide that I'll go with this|

## Directory Structure
```
├───dataset
├───langmodel_got
├───model
│    ├───pipeline
├───ppdata
└───utils
```

> There is no need to make `./ppdata` and `./langmodel_got` as they will be created automatically.

## Usage
* Execute command `pip install -r requirements.txt`
* Update [utils/config.py](./utils/config.py) to match your specs
* Run `python train_and_save.py`
* And to see the training logs open `./tensorflow.log` or use tensorboard - run command `tensorboard --logdir=./langmodel_got`
* See results by running `python produce_text.py`

`produce_text.py` takes command line args. To see them run `python produce_text.py -h` which would result in:

```
usage: produce_text.py [-h] [--in_txt IN_TXT] [--seq_len SEQ_LEN]

optional arguments:
  -h, --help         show this help message and exit
  --in_txt IN_TXT    Starting Text Default - "The moon would be black tonight"
  --seq_len SEQ_LEN  Number of words in the output sequence, Default - 100
```

## Results
Running `python produce_text.py` produces result:

> The moon would be black tonight And the old man And the gods are a man And you sunlight And the king s watch was not a <UNK> of the gods And the <UNK> of the great sea And the <UNK> of the red keep And the others had been a warned And the king s watch am not a man Am ll the lord of the king judge And the <UNK>

Running `python produce_text.py --seq_len 300` produces result:

> The moon would be black tonight And <UNK> And <UNK> And <UNK> And <UNK> and <UNK> And <UNK> and the king s watch was not a man And a man You have been a man who had been a king s son Cheese And a man is a man Same and the king s face was a man Than the gods And <UNK> And the king s watch was a man s son of am a man Sunlight And the king s son And <UNK> And <UNK> And <UNK> And <UNK> and <UNK> And <UNK> and the king s watch was a man And a sunlight of the kingsguard of ll the king s watch And the king t be a <UNK> And the king And the old man had been a man s son sunlight sunlight was not a glimpsed sunlight And <UNK> And <UNK> and <UNK> And the <UNK> of the great dogs The king s watch Is a hundred years to the king I am a man and a man Sunlight it was not a man And the king s son And the king of the king s watch And the king s watch tywin said And the king s watch And the king s t sunlight And the gods
