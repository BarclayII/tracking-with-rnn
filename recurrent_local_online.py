
import theano as T
import theano.tensor as TT
import theano.tensor.nnet as NN
import theano.tensor.signal as SIG
import theano.gradient as TG

import numpy as NP
import numpy.random as RNG

from collections import OrderedDict

import cvxopt.solvers as CS
from cvxopt import matrix as M

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
conv1_filter_row = 9
conv1_filter_col = 9
conv1_stride = 1
cls1_filter_row = 9
cls1_filter_col = 9
cls1_stride = 1
euc_w = 0
nll_w = 1
img_row = 50
img_col = 50
# attentions are unused yet
attention_row = 25
attention_col = 25
gru_dim = 200
seq_len = 20
model_name = 'model.pkl'
zero_tail_fc = False
variadic_length = False
test = False
acc_scale = 0
zoom_scale = 0
double_mnist = False
NUM_N = 5
dataset_name = "train"
filename = "mnist.h5"

nr_objs = 1
clutter_move = 1
with_clutters = 1
### CONFIGURATION END

### getopt begin
from getopt import *
import sys

try:
	opts, args = getopt(sys.argv[1:], "", ["batch_size=", "conv1_nr_filters=", "conv1_filter_size=", "conv1_stride=", "img_size=", "gru_dim=", "seq_len=", "use_cudnn", "zero_tail_fc", "var_len", "test", "acc_scale=",
		"zoom_scale=", "dataset=", "double_mnist", "nr_objs=", "clutter_static", "without_clutters", "grid_size=", "filename=", "euc_w=", "nll_w="])
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
                elif opt[0] == "--nr_objs":
                        nr_objs = int(opt[1])
                elif opt[0] == "--clutter_static":
                        clutter_move = 0
		elif opt[0] == "--without_clutters":
			with_clutters = 0
		elif opt[0] == "--grid_size":
			NUM_N = int(opt[1])
                elif opt[0] == "--filename":
                        filename = opt[1]
		elif opt[0] == "--nll_w":
			nll_w = float(opt[1])
		elif opt[0] == "--euc_w":
			euc_w = float(opt[1])
	if len(args) > 0:
		model_name = args[0]
except:
	pass
### getopt end

### Computed hyperparameters begin
#conv1_output_dim = ((img_row - conv1_filter_row) / conv1_stride + 1) * \
#		((img_col - conv1_filter_col) / conv1_stride + 1) * \
#		conv1_nr_filters
conv1_output_dim = conv1_nr_filters * img_row * img_col
print conv1_output_dim

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
W_fc3 = T.shared(glorot_uniform((gru_dim, conv1_nr_filters)), name='W_fc3')
b_fc3 = T.shared(NP.zeros((conv1_nr_filters,), dtype=T.config.floatX), name='b_fc3')

### NETWORK PARAMETERS END

print 'Building network'

A = TT.arange(img_col, dtype=T.config.floatX)
B = TT.arange(img_row, dtype=T.config.floatX)
A.name = 'a'
B.name = 'b'

def __filterbank(center_x, center_y, delta, sigma):
	'''
	From Bornschein's DRAW
	cx, cy, delta, sigma are absolute and respective to the whole canvas (in pixels)
	'''
	muX = center_x.dimshuffle(0, 'x') + delta.dimshuffle(0, 'x') * (TT.arange(NUM_N, dtype=T.config.floatX) - (NUM_N - 1) / 2.)
	muY = center_y.dimshuffle(0, 'x') + delta.dimshuffle(0, 'x') * (TT.arange(NUM_N, dtype=T.config.floatX) - (NUM_N - 1) / 2.)

	eps = 1e-8

	FX = TT.exp(-(A - muX.dimshuffle(0, 1, 'x')) ** 2 / 2. / (sigma.dimshuffle(0, 'x', 'x') ** 2 + eps))
	FY = TT.exp(-(B - muY.dimshuffle(0, 1, 'x')) ** 2 / 2. / (sigma.dimshuffle(0, 'x', 'x') ** 2 + eps))

	FX = FX / (FX.sum(axis=-1).dimshuffle(0, 1, 'x') + eps)
	FY = FY / (FY.sum(axis=-1).dimshuffle(0, 1, 'x') + eps)

	return FX, FY
# (batch_size, output_channels, input_channels, filter_row, filter_col)
cls_f = T.shared(NP.zeros((batch_size, conv1_nr_filters, cls1_filter_row, cls1_filter_col), dtype=T.config.floatX), name='cls_f')
cls_b = T.shared(NP.zeros((batch_size,), dtype=T.config.floatX), name='cls_b')
featmaps = T.shared(NP.zeros((batch_size, seq_len, conv1_nr_filters, img_row, img_col), dtype=T.config.floatX), name='featmaps')
probmaps = T.shared(NP.zeros((batch_size, seq_len, img_row, img_col), dtype=T.config.floatX), name='probmaps')
### Recurrent step
# img: of shape (batch_size, nr_channels, img_rows, img_cols)
# featmaps: (batch_size, seq_len, nr_channels, img_rows, img_cols)
# probmaps: (batch_size, seq_len, img_rows, img_cols)
def __step(img, prev_bbox, state, timestep):
	conv1 = conv2d(img, conv1_filters, subsample=(conv1_stride, conv1_stride), border_mode='half')
	act1 = NN.relu(conv1)
	flat1 = TT.reshape(act1, (-1, conv1_output_dim))
	gru_in = TT.concatenate([flat1, prev_bbox], axis=1)
	gru_z = NN.sigmoid(TT.dot(gru_in, Wz) + TT.dot(state, Uz) + bz)
	gru_r = NN.sigmoid(TT.dot(gru_in, Wr) + TT.dot(state, Ur) + br)
	gru_h_ = TT.tanh(TT.dot(gru_in, Wg) + TT.dot(gru_r * state, Ug) + bg)
	gru_h = (1 - gru_z) * state + gru_z * gru_h_
	bbox = TT.tanh(TT.dot(gru_h, W_fc2) + b_fc2)

        bbox_cx = ((bbox[:, 2] + bbox[:, 0]) / 2 + 1) / 2 * img_row
        bbox_cy = ((bbox[:, 3] + bbox[:, 1]) / 2 + 1) / 2 * img_col
        bbox_w = TT.abs_(bbox[:, 2] - bbox[:, 0]) / 2 * img_row
        bbox_h = TT.abs_(bbox[:, 3] - bbox[:, 1]) / 2 * img_col
        x = TT.arange(img_row, dtype=T.config.floatX)
        y = TT.arange(img_col, dtype=T.config.floatX)
	mx = TT.maximum(TT.minimum(-TT.abs_(x.dimshuffle('x', 0) - bbox_cx.dimshuffle(0, 'x')) + bbox_w.dimshuffle(0, 'x') / 2., 1), 1e-4)
	my = TT.maximum(TT.minimum(-TT.abs_(y.dimshuffle('x', 0) - bbox_cy.dimshuffle(0, 'x')) + bbox_h.dimshuffle(0, 'x') / 2., 1), 1e-4)
        bbox_mask = mx.dimshuffle(0, 1, 'x') * my.dimshuffle(0, 'x', 1)

        new_cls1_f = cls_f
        new_cls1_b = cls_b

        mask = act1 * bbox_mask.dimshuffle(0, 'x', 1, 2)

        new_featmaps = TG.disconnected_grad(TT.set_subtensor(featmaps[:, timestep], mask))
	new_featmaps.name = 'new_featmaps'
        new_probmaps = TG.disconnected_grad(TT.set_subtensor(probmaps[:, timestep], bbox_mask))
	new_probmaps.name = 'new_probmaps'

        train_featmaps = TG.disconnected_grad(new_featmaps[:, :timestep+1].reshape(((timestep + 1) * batch_size, conv1_nr_filters, img_row, img_col)))
	train_featmaps.name = 'train_featmaps'
        train_probmaps = TG.disconnected_grad(new_probmaps[:, :timestep+1])
	train_probmaps.name = 'train_probmaps'

        for _ in range(0, 5):
		train_convmaps = conv2d(train_featmaps, new_cls1_f, subsample=(cls1_stride, cls1_stride), border_mode='half').reshape((batch_size, timestep + 1, batch_size, img_row, img_col))
		train_convmaps.name = 'train_convmaps'
		train_convmaps_selected = train_convmaps[TT.arange(batch_size).repeat(timestep+1), TT.tile(TT.arange(timestep+1), batch_size), TT.arange(batch_size).repeat(timestep+1)].reshape((batch_size, timestep+1, img_row, img_col))
		train_convmaps_selected.name = 'train_convmaps_selected'
		train_predmaps = NN.sigmoid(train_convmaps_selected + new_cls1_b.dimshuffle(0, 'x', 'x', 'x'))
		train_loss = NN.binary_crossentropy(train_predmaps, train_probmaps).mean()
                train_grad_cls1_f, train_grad_cls1_b = T.grad(train_loss, [new_cls1_f, new_cls1_b])
                new_cls1_f -= train_grad_cls1_f * 0.1
                new_cls1_b -= train_grad_cls1_b * 0.1

	return (bbox, gru_h, timestep + 1, mask, bbox_mask), {cls_f: TG.disconnected_grad(new_cls1_f), cls_b: TG.disconnected_grad(new_cls1_b), featmaps: TG.disconnected_grad(new_featmaps), probmaps: TG.disconnected_grad(new_probmaps)}

# imgs: of shape (batch_size, seq_len, nr_channels, img_rows, img_cols)
imgs = tensor5()
starts = TT.matrix()

# Move the time axis to the top
_imgs = imgs.dimshuffle(1, 0, 2, 3, 4)
sc,sc_upd = T.scan(__step, sequences=[imgs.dimshuffle(1, 0, 2, 3, 4)], outputs_info=[starts,
                                                                                T.shared(NP.zeros((batch_size, gru_dim), dtype=T.config.floatX)),
                                                                                NP.cast['int32'](0), None, None
                                                                                ])

bbox_seq = sc[0].dimshuffle(1, 0, 2)
# targets: of shape (batch_size, seq_len, 4)
targets = TT.tensor3()
target_masks = TT.tensor4()
seq_len_scalar = TT.scalar()

euc_cost = ((targets - bbox_seq) ** 2).mean()
cost = euc_w * euc_cost

print 'Building optimizer'

params = [conv1_filters, Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg, W_fc2, b_fc2]
### RMSProp begin
def rmsprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
	'''
	Borrowed from keras, no constraints, though
	'''
	updates = OrderedDict()
	grads = T.grad(cost, params, disconnected_inputs='warn')
	acc = [T.shared(NP.zeros(p.get_value().shape, dtype=T.config.floatX)) for p in params]
        grads_norm = sum(map(lambda x: TT.sqr(x).sum(), grads))
	for p, g, a in zip(params, grads, acc):
                g = TT.switch(grads_norm > 1, g / grads_norm, g)
		new_a = rho * a + (1 - rho) * g ** 2
		updates[a] = new_a
		new_p = p - lr * g / TT.sqrt(new_a + epsilon)
		updates[p] = new_p

	return updates

### RMSprop end

train = T.function([imgs, starts, targets], [cost, bbox_seq], updates=OrderedDict((rmsprop(cost, params).items() if not test else []) + sc_upd.items()), allow_input_downcast=True)
import h5py
import os.path

if os.path.isfile(model_name):
        _model = h5py.File(model_name)
        for _p in params:
                _p.set_value(_model[_p.name].value)
        _model.close()
elif os.path.exists(model_name):
        print "Error: cannot read or create file"
        sys.exit(1)
else:
        _model = h5py.File(model_name, "w")
        i = 0
        for _p in params:
                _model[_p.name] = _p.get_value()
                i += 1
        _model.close()

print 'Generating dataset'

from data_handler import *

###### APP-CONV
appconv1_nr_filters = 32
appconv1_filter_row = 8
appconv1_filter_col = 8
appconv1_stride = 4

appconv1_filters = T.shared(glorot_uniform((appconv1_nr_filters, 1, appconv1_filter_row, appconv1_filter_col)), name='conv1_filters')

app = TT.matrix('app')
bbox = TT.vector('bbox')
_appconv = T.function([app], TT.tanh(conv2d(app.dimshuffle('x', 'x', 0, 1), appconv1_filters, subsample=(appconv1_stride, appconv1_stride))), allow_input_downcast=True)
appconv = lambda x: NP.asarray(_appconv(x))
att_row = 48
att_col = 48

def crop_attention_bilinear(bbox, frame):
	att = bbox
	frame_col = img_col
	frame_row = img_row

	_cx = (att[1] + att[3]) / 2; cx = (_cx + 1) / 2. * frame_col
	_cy = (att[0] + att[2]) / 2; cy = (_cy + 1) / 2. * frame_row
	_w = TT.abs_(att[3] - att[1]) / 2; w = _w * frame_col
	_h = TT.abs_(att[2] - att[0]) / 2; h = _h * frame_row

	dx = w / (att_col - 1)
	dy = h / (att_row - 1)

	mx = cx + dx * (TT.arange(att_col, dtype=T.config.floatX) - (att_col - 1) / 2.)
	my = cy + dy * (TT.arange(att_row, dtype=T.config.floatX) - (att_row - 1) / 2.)

	a = TT.arange(frame_col, dtype=T.config.floatX)
	b = TT.arange(frame_row, dtype=T.config.floatX)

	ax = TT.maximum(0, 1 - TT.abs_(a.dimshuffle(0, 'x') - mx.dimshuffle('x', 0)))
	by = TT.maximum(0, 1 - TT.abs_(b.dimshuffle(0, 'x') - my.dimshuffle('x', 0)))

	bilin = TT.dot(by.T, TT.dot(frame, ax))

	return bilin

crop_bilinear = T.function([bbox, app], crop_attention_bilinear(bbox, app), allow_input_downcast=True)
###### APP-CONV
print 'START'

def get_iou(a, b):
	left = NP.max([a[:, 0], b[:, 0]])
	top = NP.max([a[:, 1], b[:, 1]])
	right = NP.min([a[:, 2], b[:, 2]])
	bottom = NP.min([a[:, 3], b[:, 3]])
	intersect = (right - left) * ((right - left) > 0) * (bottom - top) * ((bottom - top) > 0)
	a_area = (a[:, 2] - a[:, 0]) * (a[:, 2] - a[:, 0] > 0) * (a[:, 3] - a[:, 1]) * (a[:, 3] - a[:, 1] > 0)
	b_area = (b[:, 2] - b[:, 0]) * (b[:, 2] - b[:, 0] > 0) * (b[:, 3] - b[:, 1]) * (b[:, 3] - b[:, 1] > 0)
	union = a_area + b_area - intersect
	return intersect / union

def sample_with_iou(bbox, iou_low, iou_high):
	while True:
		left = max(-1, bbox[0, 2] - (bbox[0, 2] - bbox[0, 0]) / iou_low)
		right = min(1, bbox[0, 0] + (bbox[0, 2] - bbox[0, 0]) / iou_low)
		top = max(-1, bbox[0, 3] - (bbox[0, 3] - bbox[0, 1]) / iou_low)
		bottom = min(1, bbox[0, 1] + (bbox[0, 3] - bbox[0, 1]) / iou_low)
		new = RNG.uniform(-1, 1, 4)
		new[0] = RNG.uniform(left, right)
		new[1] = RNG.uniform(top, bottom)
		new[2] = RNG.uniform(left, right)
		new[3] = RNG.uniform(top, bottom)
		if new[0] > new[2]:
			t = new[0]; new[0] = new[2]; new[2] = t
		if new[1] > new[3]:
			t = new[1]; new[1] = new[3]; new[3] = t
		if iou_low <= get_iou(new[NP.newaxis, :], bbox)[0] <= iou_high:
			return new

bmnist = BouncingMNIST(nr_objs, seq_len, batch_size, img_row, dataset_name+"/inputs", dataset_name+"/targets", acc=acc_scale, scale_range=zoom_scale, clutter_move = clutter_move, with_clutters = with_clutters, buff=True, filename=filename)
try:
	for i in range(0, 60):
		for j in range(0, 2000):
                        _len = seq_len
			#_len = int(RNG.exponential(seq_len - 5) + 5) if variadic_length else seq_len	
		        data, label = bmnist.GetBatch(count = 2 if double_mnist else 1)
			tgt_mask = NP.zeros_like(data)
			cls_f.set_value(cls_f.get_value() * 0)
			cls_b.set_value(cls_b.get_value() * 0)
			featmaps.set_value(featmaps.get_value() * 0)
			probmaps.set_value(probmaps.get_value() * 0)
			for b in range(0, batch_size):
				for t in range(0, seq_len):
					tgt_mask[b, t, label[b, t, 0]:label[b, t, 2], label[b, t, 1]:label[b, t, 3]] = 1.
			data = data[:, :, NP.newaxis, :, :] / 255.0
			label = label / (img_row / 2.) - 1.
			cost, bbox_seq = train(data, label[:, 0, :], label)
			left = NP.max([bbox_seq[:, :, 0], label[:, :, 0]], axis=0)
			top = NP.max([bbox_seq[:, :, 1], label[:, :, 1]], axis=0)
			right = NP.min([bbox_seq[:, :, 2], label[:, :, 2]], axis=0)
			bottom = NP.min([bbox_seq[:, :, 3], label[:, :, 3]], axis=0)
			intersect = (right - left) * ((right - left) > 0) * (bottom - top) * ((bottom - top) > 0)
			label_area = (label[:, :, 2] - label[:, :, 0]) * (label[:, :, 2] - label[:, :, 0] > 0) * (label[:, :, 3] - label[:, :, 1]) * (label[:, :, 3] - label[:, :, 1] > 0)
			predict_area = (bbox_seq[:, :, 2] - bbox_seq[:, :, 0]) * (bbox_seq[:, :, 2] - bbox_seq[:, :, 0] > 0) * (bbox_seq[:, :, 3] - bbox_seq[:, :, 1]) * (bbox_seq[:, :, 3] - bbox_seq[:, :, 1] > 0)
			union = label_area + predict_area - intersect
			print i, j, cost
			iou = intersect / union
			print NP.average(iou, axis=1)
                _epoch_model = h5py.File(model_name + str(i), "w")
                for _p in params:
                        _epoch_model[_p.name] = _p.get_value()
                _epoch_model.close()
finally:
	if not test:
                _model = h5py.File(model_name, "w")
                for _p in params:
                        _model[_p.name] = _p.get_value()
                _model.close()
