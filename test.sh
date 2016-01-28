echo $1
for seq in 20 40 80 160; do
	for obj in "" "--double_mnist"; do
		for nrobjs in 2 1; do
			for as in 0.1; do
				for zs in 0.1; do
					for cs in ""; do
						python -u recurrent_att_test.py --nr_objs=$nrobjs --seq_len=$seq --zero_tail_fc --use_cudnn --acc_scale=$as --zoom_scale=$zs $cs $obj --grid_size=$2 $1 | tee "$1-$3-$nrobjs-$seq-$obj-$as-$zs-$cs-gr1.log"
					done
				done
			done
		done
	done
done

