#!/usr/bin/env bash
data_path="$1"
output_path="$2"
wget 'https://www.dropbox.com/s/08y4rkzn4ityeae/model2.h5?dl=1' -O model2.h5
python preprocess.py --data_dir $data_path --train_or_test 'test'
python seq2seq.py --output_dir $output_path

-