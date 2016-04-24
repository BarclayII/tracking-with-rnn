
import theano as T
import theano.gradient as TG
import theano.tensor as TT
import theano.tensor.nnet as NN
import theano.tensor.signal as SIG

from theano.compile.nanguardmode import NanGuardMode

import numpy as NP
import numpy.random as RNG

from collections import OrderedDict

TRNG = TT.shared_randomstreams.RandomStreams()

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
batch_size = 16
conv1_nr_filters = 16
conv1_filter_row = 10
conv1_filter_col = 10
conv1_stride = 5
img_row = 100
img_col = 100
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
att_params = 3              # fraction, scale, amplifier
### CONFIGURATION END

### getopt begin
from getopt import *
import sys

try:
	opts, args = getopt(sys.argv[1:], "", ["batch_size=", "conv1_nr_filters=", "conv1_filter_size=", "conv1_stride=", "img_size=", "gru_dim=", "seq_len=", "use_cudnn", "zero_tail_fc", "var_len", "test", "acc_scale=",
		"zoom_scale=", "dataset=", "double_mnist", "nr_objs=", "clutter_static", "without_clutters", "grid_size=", "filename="])
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
	if len(args) > 0:
		model_name = args[0]
except:
	pass
### getopt end

### Computed hyperparameters begin
conv1_output_dim = ((img_row - conv1_filter_row) / conv1_stride + 1) * \
		((img_col - conv1_filter_col) / conv1_stride + 1) * \
		conv1_nr_filters
print conv1_output_dim

#		\phi(x)		   a   c   d
gru_input_dim = conv1_output_dim + 4 + 1 + 4
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
W_fc3 = T.shared(glorot_uniform((gru_dim, att_params)), name='W_fc3')
b_fc3 = T.shared(NP.zeros((att_params,), dtype=T.config.floatX), name='b_fc3')

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

def batch_multicrop(bboxes, frame):
	att_col = img_col
	att_row = img_row

	_cx = (bboxes[:, :, 1] + bboxes[:, :, 3]) / 2; cx = (_cx + 1) / 2. * img_col
	_cy = (bboxes[:, :, 0] + bboxes[:, :, 2]) / 2; cy = (_cy + 1) / 2. * img_row
	_w = TT.abs_(bboxes[:, :, 3] - bboxes[:, :, 1]) / 2; w = _w * img_col
	_h = TT.abs_(bboxes[:, :, 2] - bboxes[:, :, 0]) / 2; h = _h * img_row

	dx = w / (img_col - 1)
	dy = h / (img_row - 1)

	mx = cx.dimshuffle(0, 1, 'x') + dx.dimshuffle(0, 1, 'x') * (TT.arange(att_col, dtype=T.config.floatX).dimshuffle('x', 'x', 0) - (att_col - 1) / 2.)
	my = cy.dimshuffle(0, 1, 'x') + dy.dimshuffle(0, 1, 'x') * (TT.arange(att_row, dtype=T.config.floatX).dimshuffle('x', 'x', 0) - (att_row - 1) / 2.)

	a = TT.arange(img_col, dtype=T.config.floatX)
	b = TT.arange(img_row, dtype=T.config.floatX)

	# (batch_size, nr_samples, channels, frame_size, att_size)
	ax = TT.maximum(0, 1 - TT.abs_(a.dimshuffle('x', 'x', 'x', 0, 'x') - mx.dimshuffle(0, 1, 'x', 'x', 2)))
	by = TT.maximum(0, 1 - TT.abs_(b.dimshuffle('x', 'x', 'x', 0, 'x') - my.dimshuffle(0, 1, 'x', 'x', 2)))

	def __batch_multicrop_dot(a, b):
		return (a.dimshuffle(0, 1, 2, 3, 4, 'x') * b.dimshuffle(0, 1, 2, 'x', 3, 4)).sum(axis=4)

	crop = __batch_multicrop_dot(by.dimshuffle(0, 1, 2, 4, 3), __batch_multicrop_dot(frame.dimshuffle(0, 'x', 1, 2, 3), ax))
	return crop

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



### Recurrent step
# img: of shape (batch_size, nr_channels, img_rows, img_cols)
# prev_bbox: of shape (batch_size, 4)
# prev_att: unused
# state: of shape (batch_size, gru_dim)
# prev_conf: (batch_size, 1, 1)
# prev_sugg: (batch_size, 4)
# prev_W: (batch_size, feat_dim, 1)
# prev_b: (batch_size, 1, 1)
# prev_pos: (batch_size, steps, feat_dim)
# prev_neg: (batch_size, steps, feat_dim)
def __step(img, prev_bbox, prev_att, state, prev_conf, prev_sugg, prev_W, prev_b, prev_pos, prev_neg, timestep, conv1_filters, Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg, W_fc2, b_fc2, W_fc3, b_fc3):
	cx = (prev_bbox[:, 2] + prev_bbox[:, 0]) / 2.
	cy = (prev_bbox[:, 3] + prev_bbox[:, 1]) / 2.
	sigma = TT.exp(prev_att[:, 0]) * (max(img_col, img_row) / 2)
	fract = TT.exp(prev_att[:, 1])
        amplifier = TT.exp(prev_att[:, 2])

        eps = 1e-8

	abs_cx = (cx + 1) / 2. * (img_col - 1)
	abs_cy = (cy + 1) / 2. * (img_row - 1)
	abs_stride = (fract * (max(img_col, img_row) - 1)) * ((1. / (NUM_N - 1.)) if NUM_N > 1 else 0)

	FX, FY = __filterbank(abs_cx, abs_cy, abs_stride, sigma)
	unnormalized_mask = (FX.dimshuffle(0, 'x', 1, 'x', 2) * FY.dimshuffle(0, 1, 'x', 2, 'x')).sum(axis=2).sum(axis=1)
	mask = unnormalized_mask# / (unnormalized_mask.sum(axis=2).sum(axis=1) + eps).dimshuffle(0, 'x', 'x')
	masked_img = img

	conv1 = conv2d(masked_img, conv1_filters, subsample=(conv1_stride, conv1_stride))
	act1 = TT.tanh(conv1)
	flat1 = TT.reshape(act1, (-1, conv1_output_dim))
	gru_in = TT.concatenate([flat1, prev_bbox, prev_conf.reshape((batch_size, 1)), prev_sugg], axis=1)
	gru_z = NN.sigmoid(TT.dot(gru_in, Wz) + TT.dot(state, Uz) + bz)
	gru_r = NN.sigmoid(TT.dot(gru_in, Wr) + TT.dot(state, Ur) + br)
	gru_h_ = TT.tanh(TT.dot(gru_in, Wg) + TT.dot(gru_r * state, Ug) + bg)
	gru_h = (1 - gru_z) * state + gru_z * gru_h_
	bbox = TT.tanh(TT.dot(gru_h, W_fc2) + b_fc2)
	att = TT.dot(gru_h, W_fc3) + b_fc3

	def batch_dot(a, b):
		return (a.dimshuffle(0, 1, 2, 'x') * b.dimshuffle(0, 'x', 1, 2)).sum(axis=2)

	def bounding(bbox):
		return TT.stack([TT.maximum(bbox[:, 0], -1), TT.minimum(bbox[:, 1], 1), TT.maximum(bbox[:, 2], -1), TT.minimum(bbox[:, 3], 1)], axis=1)

	def sample_positives(bbox):
		x0 = bbox[:, 0]
		y0 = bbox[:, 1]
		x1 = bbox[:, 2]
		y1 = bbox[:, 3]
		return TT.stack([bounding(TT.as_tensor([x0, y0, x1, y1]).T),
				 bounding(TT.as_tensor([x0 * 0.75 + x1 * 0.25, y0, x1, y1]).T),
				 bounding(TT.as_tensor([x0, y0 * 0.75 + y1 * 0.25, x1, y1]).T),
				 bounding(TT.as_tensor([x0, y0, x1 * 0.75 + x0 * 0.25, y1]).T),
				 bounding(TT.as_tensor([x0, y0, x1, y1 * 0.75 + y0 * 0.25]).T),
				 bounding(TT.as_tensor([x0 * 1.25 - x1 * 0.25, y0, x1, y1]).T),
				 bounding(TT.as_tensor([x0, y0 * 1.25 - y1 * 0.25, x1, y1]).T),
				 bounding(TT.as_tensor([x0, y0, x1 * 1.25 - x0 * 0.25, y1]).T),
				 bounding(TT.as_tensor([x0, y0, x1, y1 * 1.25 - y0 * 0.25]).T),
				], axis=1)

	def sample_negatives(bbox):
		x0 = bbox[:, 0]
		y0 = bbox[:, 1]
		x1 = bbox[:, 2]
		y1 = bbox[:, 3]
		return TT.stack([bounding(TT.as_tensor([x0 * 0.5 + x1 * 0.5, y0, x1, y1]).T),
				 bounding(TT.as_tensor([x0, y0 * 0.5 + y1 * 0.5, x1, y1]).T),
				 bounding(TT.as_tensor([x0, y0, x1 * 0.5 + x0 * 0.5, y1]).T),
				 bounding(TT.as_tensor([x0, y0, x1, y1 * 0.5 + y0 * 0.5]).T),
				 bounding(TT.as_tensor([x0 * 1.5 - x1 * 0.5, y0, x1 * 0.5 + x0 * 0.5, y1]).T),
				 bounding(TT.as_tensor([x0, y0 * 1.5 - y1 * 0.5, x1, y1 * 0.5 + y0 * 0.5]).T),
				 bounding(TT.as_tensor([x0 * 0.5 + x1 * 0.5, y0, x1 * 1.5 - x0 * 0.5, y1]).T),
				 bounding(TT.as_tensor([x0, y0 * 0.5 + y1 * 0.5, x1, y1 * 1.5 - y0 * 0.5]).T),
				], axis=1)

	def sample_around(bbox):
		return TT.concatenate([sample_positives(bbox), sample_negatives(bbox)], axis=1)

	crop = batch_multicrop(bbox.dimshuffle(0, 'x', 1), img)
	feat = conv2d(crop.reshape((batch_size, 1, img_row, img_col)), conv1_filters, subsample=(conv1_stride, conv1_stride)).reshape((batch_size, 1, -1))
	conf = NN.sigmoid(batch_dot(feat, prev_W) + TT.addbroadcast(prev_b, 1))

	def update_step(W, b, x, y, alpha=1):
		z = batch_dot(x, W) + TT.addbroadcast(b, 1)
		y_hat = NN.sigmoid(z)
		# Deep Learning, p.182
		loss = NN.softplus((1 - 2 * y) * z).mean()
		g = T.grad(loss, [W, b])
		return W - alpha * g[0], b - alpha * g[1], loss

	nr_samples = 9
	pos_bbox = sample_positives(bbox)
	pos_crop = batch_multicrop(pos_bbox, img)
	pos_feat = conv2d(pos_crop.reshape((batch_size * nr_samples, 1, img_row, img_col)), conv1_filters, subsample=(conv1_stride, conv1_stride)).reshape((batch_size, nr_samples, -1))
	pos = TT.set_subtensor(prev_pos[:, (nr_samples*timestep):(nr_samples*(timestep+1))], pos_feat)
	nr_samples = 8
	neg_bbox = sample_negatives(bbox)
	neg_crop = batch_multicrop(neg_bbox, img)
	neg_feat = conv2d(neg_crop.reshape((batch_size * nr_samples, 1, img_row, img_col)), conv1_filters, subsample=(conv1_stride, conv1_stride)).reshape((batch_size, nr_samples, -1))
	neg = TT.set_subtensor(prev_neg[:, (nr_samples*timestep):(nr_samples*(timestep+1))], neg_feat)

	new_W = prev_W
	new_b = prev_b
	mlp_x = TT.concatenate([pos[:, :9*(timestep+1)], neg[:, :8*(timestep+1)]], axis=1)
	mlp_y = TT.concatenate([TT.ones((batch_size, 9*(timestep+1), 1)), TT.zeros((batch_size, 8*(timestep+1), 1))], axis=1)
	for _ in range(0, 10):
		new_W, new_b, _ = update_step(new_W, new_b, mlp_x, mlp_y)
	new_W = TG.disconnected_grad(new_W)
	new_b = TG.disconnected_grad(new_b)

	nr_samples = 17
	sugg_bbox = TT.concatenate([pos_bbox, neg_bbox], axis=1)
	sugg_feat = TT.concatenate([pos_feat, neg_feat], axis=1)
	sugg_conf = batch_dot(sugg_feat, prev_W) + TT.addbroadcast(prev_b, 1)
	print sugg_conf.dtype
	sugg_pos = TT.cast(sugg_conf > 0, T.config.floatX)
	print sugg_pos.dtype
	# TT.maximum(1, *) for avoiding division by zero
	sugg = TG.disconnected_grad((sugg_bbox * TT.patternbroadcast(sugg_pos, [False, False, True])).sum(axis=1) / TT.patternbroadcast(TT.maximum(1, sugg_pos.sum(axis=1)), [False, True]))

	return bbox, att, gru_h, TT.unbroadcast(conf, 1), sugg, new_W, TT.unbroadcast(new_b, 1), pos, neg, timestep + 1

# imgs: of shape (batch_size, seq_len, nr_channels, img_rows, img_cols)
imgs = tensor5()
pos_bboxes = TT.tensor4()
neg_bboxes = TT.tensor4()
starts = TT.matrix()
startAtt = TT.matrix()

params = [conv1_filters, Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg, W_fc2, b_fc2, W_fc3, b_fc3]

# Move the time axis to the top
sc,_ = T.scan(__step, sequences=[imgs.dimshuffle(1, 0, 2, 3, 4)],
		outputs_info=[starts,
			      startAtt,
			      T.shared(NP.zeros((batch_size, gru_dim), dtype=T.config.floatX)),
			      T.shared(NP.ones((batch_size, 1, 1), dtype=T.config.floatX)),
			      T.shared(NP.zeros((batch_size, 4), dtype=T.config.floatX)),
			      T.shared(NP.zeros((batch_size, conv1_output_dim, 1), dtype=T.config.floatX)),
			      T.shared(NP.zeros((batch_size, 1, 1), dtype=T.config.floatX)),
			      T.shared(NP.zeros((batch_size, 9 * seq_len, conv1_output_dim), dtype=T.config.floatX)),
			      T.shared(NP.zeros((batch_size, 8 * seq_len, conv1_output_dim), dtype=T.config.floatX)), NP.cast['int32'](0)],
		non_sequences=params,
		strict=True,
		mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
		)

bbox_seq = sc[0].dimshuffle(1, 0, 2)
att_seq = sc[1].dimshuffle(1, 0, 2)
mask_seq = sc[3].dimshuffle(1, 0, 2, 3)
# targets: of shape (batch_size, seq_len, 4)
targets = TT.tensor3()
seq_len_scalar = TT.scalar()

cost = ((targets - bbox_seq) ** 2).sum() / batch_size / seq_len_scalar

print 'Building optimizer'

### RMSProp begin
def rmsprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
	'''
	Borrowed from keras, no constraints, though
	'''
	updates = OrderedDict()
	grads = T.grad(cost, params, disconnected_inputs='ignore')
	acc = [T.shared(NP.zeros(p.get_value().shape, dtype=T.config.floatX)) for p in params]
	for p, g, a in zip(params, grads, acc):
		new_a = rho * a + (1 - rho) * g ** 2
		updates[a] = new_a
		new_p = p - lr * g / TT.sqrt(new_a + epsilon)
		updates[p] = new_p

	return updates

### RMSprop end

train = T.function([seq_len_scalar, imgs, starts, startAtt, targets], [cost, bbox_seq, att_seq, mask_seq], updates=rmsprop(cost, params) if not test else None, allow_input_downcast=True,
		mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
		)
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

crop_bilinear = T.function([bbox, app], crop_attention_bilinear(bbox, app), allow_input_downcast=True)
###### APP-CONV
print 'START'

bmnist = BouncingMNIST(nr_objs, seq_len, batch_size, img_row, dataset_name+"/inputs", dataset_name+"/targets", acc=acc_scale, scale_range=zoom_scale, clutter_move = clutter_move, with_clutters = with_clutters, buff=True, filename=filename)

import time

try:
	for i in range(0, 60):
		for j in range(0, 2000):
                        _len = seq_len
			#_len = int(RNG.exponential(seq_len - 5) + 5) if variadic_length else seq_len	
			_ts = time.time()
		        data, label = bmnist.GetBatch(count = 2 if double_mnist else 1)
			_tt = time.time()
			print "Batch generation:", _tt - _ts
			data = data[:, :, NP.newaxis, :, :] / 255.0
			label = label / (img_row / 2.) - 1.
			att = NP.zeros((batch_size, att_params))
                        att[:, 0] = 10
                        att[:, 1] = 1
                        att[:, 2] = 0
			_ts = time.time()
			cost, bbox_seq, att_seq, mask = train(_len, data[:, :], label[:, 0, :], att, label[:, :])
			_tt = time.time()
			print "Training:", _tt - _ts
			print bbox_seq.shape, label.shape
                        print 'Mask ', NP.max(mask), NP.min(mask)
			print 'Attention, sigma, strideH, strideW', NP.mean(NP.abs(att_seq), axis=1)
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
                f = open(model_name + str(i), "wb")
		cPickle.dump(map(lambda x: x.get_value(), params), f)
                f.close()
finally:
	if not test:
		f = open(model_name, "wb")
		cPickle.dump(map(lambda x: x.get_value(), params), f)
		f.close()
