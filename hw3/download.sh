#!/usr/bin/env bash
wget 'https://www.dropbox.com/s/gyoubzz9zi2s2vn/Vanilla_DQN.zip?dl=1' -O Vanilla_DQN.zip
wget 'https://www.dropbox.com/s/ockjz89j37e0xcu/REINFORCE.zip?dl=1' -O REINFORCE.zip
mkdir saved
unzip REINFORCE.zip -d saved
unzip Vanilla_DQN.zip -d saved
rm REINFORCE.zip
rm Vanilla_DQN.zip
