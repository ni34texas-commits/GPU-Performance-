#!/bin/bash
# ============================================
# Lightweight Benchmarker for CUDA kernels
# Tests Device, Host, Unified memory types
# ============================================

KERNEL_DIR=$1          # e.g. histogram-cuda
SRC=~/GPULLM/ParallelCodeEstimation/src/$KERNEL_DIR
ARCH=sm_60
ITERATIONS=100
OUTPUT_CSV=~/GPULLM/benchmarker/results.csv

echo "======================================"
echo "Benchmarking kernel: $KERNEL_DIR"
echo "======================================"

# Go to kernel directory
cd $SRC

# Compile the kernel
echo "Compiling..."
make 2>/dev/null || nvcc -std=c++14 -arch=$ARCH -O3 *.cu -o main

# Run with current memory config and capture output
echo "Running benchmark..."
./main --i=$ITERATIONS > timing_output.txt 2>&1
cat timing_output.txt

# Extract timings using grep
#DEVICE_TIME=$(grep "smem device" timing_output.txt | awk '{print $1}')
#HOST_TIME=$(grep "smem host" timing_output.txt | awk '{print $1}')
#UNIFIED_TIME=$(grep "smem unified" timing_output.txt | awk '{print $1}')
DEVICE_TIME=$(grep "smem device" timing_output.txt | head -1 | awk '{print $1}')
HOST_TIME=$(grep "smem host" timing_output.txt | head -1 | awk '{print $1}')
UNIFIED_TIME=$(grep "smem unified" timing_output.txt | head -1 | awk '{print $1}')

echo ""
echo "======================================"
echo "Extracted Timings (us):"
echo "  Device  : $DEVICE_TIME"
echo "  Host    : $HOST_TIME"
echo "  Unified : $UNIFIED_TIME"
echo "======================================"

# Save to file for Python/Ollama
echo "$KERNEL_DIR,$DEVICE_TIME,$HOST_TIME,$UNIFIED_TIME" >> $OUTPUT_CSV

# Call Python to send to Ollama
python3 ~/GPULLM/benchmarker/ask_ollama.py \
    "$KERNEL_DIR" "$DEVICE_TIME" "$HOST_TIME" "$UNIFIED_TIME"

