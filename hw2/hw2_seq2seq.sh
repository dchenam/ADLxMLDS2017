#!/usr/bin/env bash
data_path="$1"
output_path="$2"
peer_path="$3"
wget 'https://www.dropbox.com/s/hy3kpi2cm3eg8fi/seq2seq.h5?dl=1' -O seq2seq.h5
python preprocess.py -data_dir $data_path -include 'none'
python seq2seq.py -output_dir $output_path -model_dir './seq2seq.h5'

python preprocess.py -data_dir $data_path -mode 'peer_review'
python seq2seq.py -output_dir $peer_path -model_dir './seq2seq.h5'
