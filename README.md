# tracking-with-rnn

Videos and codebase of [First step toward model-free, anonymous object tracking with recurrent networks](http://arxiv.org/abs/1511.06425).

## Requirements

All models require [Theano](https://github.com/Theano/Theano) to train & test.

Additionally, **ConvTracker** requires [keras](https://github.com/fchollet/keras).

## Files

`conv_base.py` - **ConvTracker**
`recurrent_base.py` - **RecTracker-ID**
`recurrent_att.py` - **RecTracker-Att-N**.  N can be specified in `--grid_size` option.
`data_handler.py` - Moving MNIST dataset generator

## Usage

Run `train.sh` to train the **RecTracker-Att-1** model.

Run `test.sh` to test the **RecTracker-Att-1** model under different configurations.

**RecTracker-ID** and **ConvTracker** accept the same options.
