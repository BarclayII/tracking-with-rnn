import theano as T
import theano.tensor as TT
import theano.tensor.nnet as NN
import theano.tensor.signal as SIG

import numpy as NP
import numpy.random as RNG

from collections import OrderedDict

#####################################################################
# Usage:                                                            #
# python -u recurrent_plain_base.py [opts] [model_name]             #
#                                                                   #
# Options:                                                          #
#       --batch_size=INTEGER                                        #
#       --conv1_nr_filters=INTEGER                                  #
#       --conv1_filter_size=INTEGER                                 #
#       --conv1_stride=INTEGER                                      #
#       --img_size=INTEGER                                          #
#       --gru_dim=INTEGER                                           #
#       --seq_len=INTEGER                                           #
#       --use_cudnn     (Set floatX to float32 if you use this)     #
#       --zero_tail_fc  (Recommended)                               #
#####################################################################

### Utility functions begin
def get_fans(shape):
	'''
	Borrowed from keras
	'''
	fan_in = shape[0] if len(shape) == 2 else NP.prod(shape[1:])
	fan_out = shape[1] if len(shape) == 2 else shape[0]
	return fan_in, fan_out

def glorot_uniform(shape):
	'''
	Borrowed from keras
	'''
	fan_in, fan_out = get_fans(shape)
	s = NP.sqrt(6. / (fan_in + fan_out))
	return NP.cast[T.config.floatX](RNG.uniform(low=-s, high=s, size=shape))

def orthogonal(shape, scale=1.1):
	'''
	Borrowed from keras
	'''
	flat_shape = (shape[0], NP.prod(shape[1:]))
	a = RNG.normal(0, 1, flat_shape)
	u, _, v = NP.linalg.svd(a, full_matrices=False)
	q = u if u.shape == flat_shape else v
	q = q.reshape(shape)
	return NP.cast[T.config.floatX](q)

def tensor5(name=None, dtype=None):
	if dtype == None:
		dtype = T.config.floatX
	return TT.TensorType(dtype, [False] * 5, name=name)()

conv2d = NN.conv2d

### Utility functions end

### CONFIGURATION BEGIN
batch_size = 32
conv1_nr_filters = 32
conv1_filter_row = 10
conv1_filter_col = 10
conv1_stride = 5
img_row = 100
img_col = 100
# attentions are unused yet
attention_row = 25
attention_col = 25
gru_dim = 256
seq_len = 200
model_name = 'model.pkl'
zero_tail_fc = False
variadic_length = False
test = False
acc_scale = 0
zoom_scale = 0
double_mnist = False
dataset_name = "train"
clutter_move = 0.5
with_clutters = 1
nr_objs = 2
### CONFIGURATION END

### getopt begin
from getopt import *
import sys

try:
	opts, args = getopt(sys.argv[1:], "", ["batch_size=", "conv1_nr_filters=", "conv1_filter_size=", "conv1_stride=", "img_size=", "gru_dim=", "seq_len=", "use_cudnn", "zero_tail_fc", "var_len", "test", "acc_scale=",
		"zoom_scale=", "dataset=", "double_mnist", "clutter_move=", "nr_objs="])
	for opt in opts:
		if opt[0] == "--batch_size":
			batch_size = int(opt[1])
		elif opt[0] == "--conv1_nr_filters":
			conv1_nr_filters = int(opt[1])
		elif opt[0] == "--conv1_filter_size":
			conv1_filter_row = conv1_filter_col = int(opt[1])
		elif opt[0] == "--conv1_stride":
			conv1_stride = int(opt[1])
		elif opt[0] == "--img_size":
			img_row = img_col = int(opt[1])
		elif opt[0] == "--gru_dim":
			gru_dim = int(opt[1])
		elif opt[0] == "--seq_len":
			seq_len = int(opt[1])
		elif opt[0] == "--use_cudnn":
			if T.config.device[:3] == 'gpu':
				import theano.sandbox.cuda.dnn as CUDNN
				if CUDNN.dnn_available():
					print 'Using CUDNN instead of Theano conv2d'
					conv2d = CUDNN.dnn_conv
		elif opt[0] == "--zero_tail_fc":
			zero_tail_fc = True
		elif opt[0] == "--var_len":
			variadic_length = True
		elif opt[0] == "--test":
			test = True
		elif opt[0] == "--acc_scale":
			acc_scale = float(opt[1])
		elif opt[0] == "--zoom_scale":
			zoom_scale = float(opt[1])
		elif opt[0] == "--double_mnist":
			double_mnist = True
		elif opt[0] == "--dataset":
			dataset_name = opt[1]
		elif opt[0] == "--clutter_move":
			clutter_move = float(opt[1])
		elif opt[0] == "--nr_objs":
			nr_objs = int(opt[1])
	if len(args) > 0:
		model_name = args[0]
except:
	pass
### getopt end

### Computed hyperparameters begin
conv1_output_dim = ((img_row - conv1_filter_row) / conv1_stride + 1) * \
		((img_col - conv1_filter_col) / conv1_stride + 1) * \
		conv1_nr_filters
gru_input_dim = conv1_output_dim + 4
### Computed hyperparameters end

print 'Initializing parameters'

### NETWORK PARAMETERS BEGIN
conv1_filters = T.shared(glorot_uniform((conv1_nr_filters, 1, conv1_filter_row, conv1_filter_col)), name='conv1_filters')
Wr = T.shared(glorot_uniform((gru_input_dim, gru_dim)), name='Wr')
Ur = T.shared(orthogonal((gru_dim, gru_dim)), name='Ur')
br = T.shared(NP.zeros((gru_dim,), dtype=T.config.floatX), name='br')
Wz = T.shared(glorot_uniform((gru_input_dim, gru_dim)), name='Wz')
Uz = T.shared(orthogonal((gru_dim, gru_dim)), name='Uz')
bz = T.shared(NP.zeros((gru_dim,), dtype=T.config.floatX), name='bz')
Wg = T.shared(glorot_uniform((gru_input_dim, gru_dim)), name='Wg')
Ug = T.shared(orthogonal((gru_dim, gru_dim)), name='Ug')
bg = T.shared(NP.zeros((gru_dim,), dtype=T.config.floatX), name='bg')
W_fc2 = T.shared(glorot_uniform((gru_dim, 4)) if not zero_tail_fc else NP.zeros((gru_dim, 4), dtype=T.config.floatX), name='W_fc2')
b_fc2 = T.shared(NP.zeros((4,), dtype=T.config.floatX), name='b_fc2')
### NETWORK PARAMETERS END

print 'Building network'

### Recurrent step
# img: of shape (batch_size, nr_channels, img_rows, img_cols)
def _step(img, prev_bbox, state):
	# of (batch_size, nr_filters, some_rows, some_cols)
	conv1 = conv2d(img, conv1_filters, subsample=(conv1_stride, conv1_stride))
	act1 = TT.tanh(conv1)
	flat1 = TT.reshape(act1, (batch_size, conv1_output_dim))
	gru_in = TT.concatenate([flat1, prev_bbox], axis=1)
	gru_z = NN.sigmoid(TT.dot(gru_in, Wz) + TT.dot(state, Uz) + bz)
	gru_r = NN.sigmoid(TT.dot(gru_in, Wr) + TT.dot(state, Ur) + br)
	gru_h_ = TT.tanh(TT.dot(gru_in, Wg) + TT.dot(gru_r * state, Ug) + bg)
	gru_h = (1-gru_z) * state + gru_z * gru_h_
	bbox = TT.tanh(TT.dot(gru_h, W_fc2) + b_fc2)
	return bbox, gru_h

# imgs: of shape (batch_size, seq_len, nr_channels, img_rows, img_cols)
imgs = tensor5()
starts = TT.matrix()

# Move the time axis to the top
_imgs = imgs.dimshuffle(1, 0, 2, 3, 4)
sc, _ = T.scan(_step, sequences=[imgs.dimshuffle(1, 0, 2, 3, 4)], outputs_info=[starts, TT.zeros((batch_size, gru_dim))])

bbox_seq = sc[0].dimshuffle(1, 0, 2)

# targets: of shape (batch_size, seq_len, 4)
targets = TT.tensor3()
seq_len_scalar = TT.scalar()

cost = ((targets - bbox_seq) ** 2).sum() / batch_size / seq_len_scalar

print 'Building optimizer'

params = [conv1_filters, Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg, W_fc2, b_fc2]
### RMSProp begin
def rmsprop(cost, params, lr=0.0005, rho=0.9, epsilon=1e-6):
	'''
	Borrowed from keras, no constraints, though
	'''
	updates = OrderedDict()
	grads = T.grad(cost, params)
	acc = [T.shared(NP.zeros(p.get_value().shape, dtype=T.config.floatX)) for p in params]
	for p, g, a in zip(params, grads, acc):
		new_a = rho * a + (1 - rho) * g ** 2
		updates[a] = new_a
		new_p = p - lr * g / TT.sqrt(new_a + epsilon)
		updates[p] = new_p

	return updates

### RMSprop end

train = T.function([seq_len_scalar, imgs, starts, targets], [cost, bbox_seq], updates=rmsprop(cost, params) if not test else None, allow_input_downcast=True)
tester = T.function([seq_len_scalar, imgs, starts, targets], [cost, bbox_seq], allow_input_downcast=True)

import cPickle

try:
    f = open(model_name, "rb")
    param_saved = cPickle.load(f)
    for _p, p in zip(params, param_saved):
        _p.set_value(p)
except IOError:
    pass

print 'Generating dataset'

from data_handler import *

bmnist = BouncingMNIST(nr_objs, seq_len, batch_size, img_row, dataset_name+"/inputs", dataset_name+"/targets", acc=acc_scale, scale_range=zoom_scale, clutter_move = clutter_move, with_clutters = with_clutters)
print 'START'

try:
	for i in range(0, 50):
                train_cost = test_cost = 0
		for j in range(0, 2000):
			data, label = bmnist.GetBatch(count = 2 if double_mnist else 1)
			data = data[:, :, NP.newaxis, :, :] / 255.0
			label = label / (img_row / 2.) - 1.
			cost, bbox_seq = train(seq_len, data, label[:, 0, :], label)
			left = NP.max([bbox_seq[:, :, 0], label[:, :, 0]], axis=0)
			top = NP.max([bbox_seq[:, :, 1], label[:, :, 1]], axis=0)
			right = NP.min([bbox_seq[:, :, 2], label[:, :, 2]], axis=0)
			bottom = NP.min([bbox_seq[:, :, 3], label[:, :, 3]], axis=0)
			intersect = (right - left) * ((right - left) > 0) * (bottom - top) * ((bottom - top) > 0)
			label_area = (label[:, :, 2] - label[:, :, 0]) * (label[:, :, 2] - label[:, :, 0] > 0) * (label[:, :, 3] - label[:, :, 1]) * (label[:, :, 3] - label[:, :, 1] > 0)
			predict_area = (bbox_seq[:, :, 2] - bbox_seq[:, :, 0]) * (bbox_seq[:, :, 2] - bbox_seq[:, :, 0] > 0) * (bbox_seq[:, :, 3] - bbox_seq[:, :, 1]) * (bbox_seq[:, :, 3] - bbox_seq[:, :, 1] > 0)
			union = label_area + predict_area - intersect
			print i, j, cost
                        train_cost += cost
			bmnist = BouncingMNIST(1, seq_len, batch_size, img_row, "test/inputs", "test/targets", acc=acc_scale, scale_range=zoom_scale)
			data, label = bmnist.GetBatch(count = 2 if double_mnist else 1)
			data = data[:, :, NP.newaxis, :, :] / 255.0
			label = label / (img_row / 2.) - 1.
			cost, bbox_seq = tester(seq_len, data, label[:, 0, :], label)
			left = NP.max([bbox_seq[:, :, 0], label[:, :, 0]], axis=0)
			top = NP.max([bbox_seq[:, :, 1], label[:, :, 1]], axis=0)
			right = NP.min([bbox_seq[:, :, 2], label[:, :, 2]], axis=0)
			bottom = NP.min([bbox_seq[:, :, 3], label[:, :, 3]], axis=0)
			intersect = (right - left) * ((right - left) > 0) * (bottom - top) * ((bottom - top) > 0)
			label_area = (label[:, :, 2] - label[:, :, 0]) * (label[:, :, 2] - label[:, :, 0] > 0) * (label[:, :, 3] - label[:, :, 1]) * (label[:, :, 3] - label[:, :, 1] > 0)
			predict_area = (bbox_seq[:, :, 2] - bbox_seq[:, :, 0]) * (bbox_seq[:, :, 2] - bbox_seq[:, :, 0] > 0) * (bbox_seq[:, :, 3] - bbox_seq[:, :, 1]) * (bbox_seq[:, :, 3] - bbox_seq[:, :, 1] > 0)
			union = label_area + predict_area - intersect
                        print i, j, cost
                        test_cost += cost
			iou = intersect / union
			print NP.average(iou, axis=0)       # per frame
			print NP.average(iou, axis=1)       # per batch
                print 'Epoch average loss (train, test)', train_cost / 2000, test_cost / 2000
                f = open(model_name + str(i), "wb")
		cPickle.dump(map(lambda x: x.get_value(), params), f)
                f.close()
except KeyboardInterrupt:
	if not test:
		print 'Saving...'
		f = open(model_name, "wb")
		cPickle.dump(map(lambda x: x.get_value(), params), f)
		f.close()
