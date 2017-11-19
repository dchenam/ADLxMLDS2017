#!/usr/bin/env bash
data_path="$1"
output_path="$2"
peer_path="$3"
wget 'https://www.dropbox.com/s/hy3kpi2cm3eg8fi/seq2seq.h5?dl=1' -O model.h5
python preprocess.py --data_dir $data_path -mode 'test'
python seq2seq.py --output_dir $output_path --model_dir './model.h5'

python preprocess.py --data_dir $peer_path -mode '' -use_list 'peer_review_id.txt'
python seq2seq.py --output_dir $output_path --model_dir './model.h5'
