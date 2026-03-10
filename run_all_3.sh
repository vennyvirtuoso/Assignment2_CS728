#!/bin/bash

if [ "$1" = "cuda" ]; then
    DEVICE="cuda"
else
    DEVICE="cpu"
fi

mkdir -p logs


python3 -m trainingRNNs_torch.train \
    --task torder --model rnn --init smart_tanh \
    --alpha 2.0 --clipstyle rescale --cutoff 0.05 \
    --nhid 50 --lr 0.01 --bs 20 --min_length 50 --max_length 200 \
    --maxiters 100000 --ebs 10000 --cbs 1000 --checkFreq 20 \
    --seed 52 --valid_seed 12345 --collectDiags --diagBins 60 --satThresh 0.05 \
    --name EC_stuck1_torder_alpha2_clip005 --device $DEVICE \
    > logs/EC_stuck1_torder_alpha2_clip005.log 2>&1 &

python3 -m trainingRNNs_torch.train \
    --task torder --model rnn --init smart_tanh \
    --alpha 4.0 --clipstyle rescale --cutoff 0.05 \
    --nhid 50 --lr 0.01 --bs 20 --min_length 50 --max_length 200 \
    --maxiters 100000 --ebs 10000 --cbs 1000 --checkFreq 20 \
    --seed 52 --valid_seed 12345 --collectDiags --diagBins 60 --satThresh 0.05 \
    --name EC_stuck2_torder_alpha4_clip005 --device $DEVICE \
    > logs/EC_stuck2_torder_alpha4_clip005.log 2>&1 &

python3 -m trainingRNNs_torch.train \
    --task torder --model rnn --init smart_tanh \
    --alpha 0.0 --clipstyle rescale --cutoff 0.05 \
    --nhid 50 --lr 0.01 --bs 20 --min_length 50 --max_length 200 \
    --maxiters 100000 --ebs 10000 --cbs 1000 --checkFreq 20 \
    --seed 52 --valid_seed 12345 --collectDiags --diagBins 60 --satThresh 0.05 \
    --name EC_noreg_torder_alpha0_clip005 --device $DEVICE \
    > logs/EC_noreg_torder_alpha0_clip005.log 2>&1 &

python3 -m trainingRNNs_torch.train \
    --task torder --model rnn --init smart_tanh \
    --alpha 2.0 --clipstyle rescale --cutoff 1.0 \
    --nhid 50 --lr 0.01 --bs 20 --min_length 50 --max_length 200 \
    --maxiters 100000 --ebs 10000 --cbs 1000 --checkFreq 20 \
    --seed 52 --valid_seed 12345 --collectDiags --diagBins 60 --satThresh 0.05 \
    --name EC1_torder_alpha2_clip1 --device $DEVICE \
    > logs/EC1_torder_alpha2_clip1.log 2>&1 &

python3 -m trainingRNNs_torch.train \
    --task torder --model rnn --init smart_tanh \
    --alpha 8.0 --clipstyle rescale --cutoff 1.0 \
    --nhid 50 --lr 0.01 --bs 20 --min_length 50 --max_length 200 \
    --maxiters 100000 --ebs 10000 --cbs 1000 --checkFreq 20 \
    --seed 52 --valid_seed 12345 --collectDiags --diagBins 60 --satThresh 0.05 \
    --name EC2_torder_alpha8_clip1 --device $DEVICE \
    > logs/EC2_torder_alpha8_clip1.log 2>&1 &

python3 -m trainingRNNs_torch.train \
    --task torder --model rnn --init smart_tanh \
    --alpha 4.0 --clipstyle nothing \
    --nhid 50 --lr 0.01 --bs 20 --min_length 50 --max_length 200 \
    --maxiters 100000 --ebs 10000 --cbs 1000 --checkFreq 20 \
    --seed 52 --valid_seed 12345 --collectDiags --diagBins 60 --satThresh 0.05 \
    --name EC3_torder_alpha4_noclip --device $DEVICE \
    > logs/EC3_torder_alpha4_noclip.log 2>&1 &

wait
echo "All done."
