#!/usr/bin/env bash
testing_text="$1"
wget 'https://www.dropbox.com/s/gyoubzz9zi2s2vn/Vanilla_DQN.zip?dl=1' -O Vanilla_DQN.zip
unzip REINFORCE.zip
rm REINFORCE.zip

python3 generate.py $testing_text


