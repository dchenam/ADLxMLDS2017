#!/usr/bin/env bash
testing_text="$1"
wget 'https://www.dropbox.com/s/jqqok11p8c4vzqx/checkpoints.zip?dl=1' -O checkpoints.zip
mkdir experiments/GAN-CLS/
unzip checkpoints.zip -d experiments/GAN-CLS/
rm checkpoints.zip
python3 generate.py --testing $testing_text


