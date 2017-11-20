#!/usr/bin/env bash
data_path="$1"
output_path="$2"
peer_path="$3"
wget 'https://www.dropbox.com/s/wnh7b2etns2d8tq/best.h5?dl=1' -O seq2seq.h5
python preprocess.py -data_dir $data_path
python model_seq2seq.py -output_dir $output_path -model_dir './seq2seq.h5'

python preprocess.py -data_dir $data_path -mode 'peer_review'
python model_seq2seq.py -mode 'peer_review' -output_dir $peer_path -model_dir './seq2seq.h5'
