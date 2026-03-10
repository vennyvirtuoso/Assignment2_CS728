#!/bin/bash
# ============================================================
# Parallel training script for CS728 PA2
# Usage:
#   bash run_all.sh        # use CPU
#   bash run_all.sh cuda   # use GPU
# ============================================================

if [ "$1" = "cuda" ]; then
    DEVICE="cuda"
else
    DEVICE="cpu"
fi

LOGDIR="logs"
mkdir -p "$LOGDIR"

echo "============================================"
echo "  CS728 PA2 - Running all 7 experiments"
echo "  Device: $DEVICE"
echo "  Logs:   ./$LOGDIR/"
echo "============================================"

echo "[START] A1_mem_rnn_tanh_noclip"
python3 -m trainingRNNs_torch.train \
    --task mem --model rnn --alpha 0.0 --clipstyle nothing \
    --nhid 50 --lr 0.01 --bs 20 --min_length 50 --max_length 200 \
    --maxiters 50000 --ebs 10000 --cbs 1000 --checkFreq 20 \
    --seed 52 --valid_seed 12345 --collectDiags --diagBins 60 --satThresh 0.05 \
    --name A1_mem_rnn_tanh_noclip --device $DEVICE \
    > "$LOGDIR/A1_mem_rnn_tanh_noclip.log" 2>&1 &

echo "[START] A2_mem_rnn_tanh_clip005"
python3 -m trainingRNNs_torch.train \
    --task mem --model rnn --alpha 0.0 --clipstyle rescale --cutoff 0.05 \
    --nhid 50 --lr 0.01 --bs 20 --min_length 50 --max_length 200 \
    --maxiters 50000 --ebs 10000 --cbs 1000 --checkFreq 20 \
    --seed 52 --valid_seed 12345 --collectDiags --diagBins 60 --satThresh 0.05 \
    --name A2_mem_rnn_tanh_clip005 --device $DEVICE \
    > "$LOGDIR/A2_mem_rnn_tanh_clip005.log" 2>&1 &

echo "[START] A3_mem_rnn_tanh_clip001"
python3 -m trainingRNNs_torch.train \
    --task mem --model rnn --alpha 0.0 --clipstyle rescale --cutoff 0.01 \
    --nhid 50 --lr 0.01 --bs 20 --min_length 50 --max_length 200 \
    --maxiters 50000 --ebs 10000 --cbs 1000 --checkFreq 20 \
    --seed 52 --valid_seed 12345 --collectDiags --diagBins 60 --satThresh 0.05 \
    --name A3_mem_rnn_tanh_clip001 --device $DEVICE \
    > "$LOGDIR/A3_mem_rnn_tanh_clip001.log" 2>&1 &

echo "[START] A4_mem_gru_noclip"
python3 -m trainingRNNs_torch.train \
    --task mem --model gru --alpha 0.0 --clipstyle nothing --diagGates \
    --nhid 50 --lr 0.01 --bs 20 --min_length 50 --max_length 200 \
    --maxiters 50000 --ebs 10000 --cbs 1000 --checkFreq 20 \
    --seed 52 --valid_seed 12345 --collectDiags --diagBins 60 --satThresh 0.05 \
    --name A4_mem_gru_noclip --device $DEVICE \
    > "$LOGDIR/A4_mem_gru_noclip.log" 2>&1 &

echo "[START] A5_mem_gru_clip005"
python3 -m trainingRNNs_torch.train \
    --task mem --model gru --alpha 0.0 --clipstyle rescale --cutoff 0.05 --diagGates \
    --nhid 50 --lr 0.01 --bs 20 --min_length 50 --max_length 200 \
    --maxiters 50000 --ebs 10000 --cbs 1000 --checkFreq 20 \
    --seed 52 --valid_seed 12345 --collectDiags --diagBins 60 --satThresh 0.05 \
    --name A5_mem_gru_clip005 --device $DEVICE \
    > "$LOGDIR/A5_mem_gru_clip005.log" 2>&1 &

echo "[START] B1_mul_rnn_tanh_noclip"
python3 -m trainingRNNs_torch.train \
    --task mul --model rnn --alpha 0.0 --clipstyle nothing \
    --nhid 50 --lr 0.01 --bs 20 --min_length 50 --max_length 200 \
    --maxiters 50000 --ebs 10000 --cbs 1000 --checkFreq 20 \
    --seed 52 --valid_seed 12345 --collectDiags --diagBins 60 --satThresh 0.05 \
    --name B1_mul_rnn_tanh_noclip --device $DEVICE \
    > "$LOGDIR/B1_mul_rnn_tanh_noclip.log" 2>&1 &

echo "[START] B2_mul_gru_noclip"
python3 -m trainingRNNs_torch.train \
    --task mul --model gru --alpha 0.0 --clipstyle nothing --diagGates \
    --nhid 50 --lr 0.01 --bs 20 --min_length 50 --max_length 200 \
    --maxiters 50000 --ebs 10000 --cbs 1000 --checkFreq 20 \
    --seed 52 --valid_seed 12345 --collectDiags --diagBins 60 --satThresh 0.05 \
    --name B2_mul_gru_noclip --device $DEVICE \
    > "$LOGDIR/B2_mul_gru_noclip.log" 2>&1 &

echo ""
echo "All 7 experiments launched. Waiting for completion..."
echo "Monitor with: tail -f logs/<NAME>.log"
echo ""

wait

echo ""
echo "============================================"
echo "  All experiments complete!"
echo "============================================"
echo ""
echo "--- Results summary ---"
for NAME in A1_mem_rnn_tanh_noclip A2_mem_rnn_tanh_clip005 A3_mem_rnn_tanh_clip001 \
            A4_mem_gru_noclip A5_mem_gru_clip005 B1_mul_rnn_tanh_noclip B2_mul_gru_noclip; do
    NPZ="${NAME}_final_state.npz"
    if [ -f "$NPZ" ]; then
        echo "  [OK]     $NPZ"
    else
        echo "  [FAILED] $NPZ not found - check logs/${NAME}.log"
    fi
done
