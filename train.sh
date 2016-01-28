THEANO_FLAGS='device=gpu0,floatX=float32' python -u recurrent_att.py --nr_objs=1 --seq_len=20 --zero_tail_fc --use_cudnn --acc_scale=0.1 --zoom_scale=0.1 --grid_size=1 model.pkl
