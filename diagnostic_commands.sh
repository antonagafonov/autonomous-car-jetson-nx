#!/bin/bash

echo "=== GPU Performance Diagnostics ==="

# 1. Check GPU utilization in real-time
echo "1. Checking GPU utilization (run this in separate terminal):"
echo "nvidia-smi -l 1"

# 2. Check GPU memory and compute utilization
echo -e "\n2. Current GPU status:"
nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv

# 3. Check GPU clocks and power
echo -e "\n3. GPU clocks and power:"
nvidia-smi --query-gpu=clocks.current.graphics,clocks.current.memory,power.draw,power.limit --format=csv

# 4. Check CPU usage and processes
echo -e "\n4. Top CPU-consuming processes:"
ps aux --sort=-%cpu | head -10

# 5. Check memory usage
echo -e "\n5. Memory usage:"
free -h

# 6. Check if any processes are using significant GPU memory
echo -e "\n6. GPU memory usage by process:"
nvidia-smi pmon -c 1

# 7. Check system load
echo -e "\n7. System load:"
uptime

# 8. Check for CPU frequency scaling
echo -e "\n8. CPU frequency info:"
cat /proc/cpuinfo | grep "cpu MHz" | head -4

# 9. Check GPU compute mode
echo -e "\n9. GPU compute mode:"
nvidia-smi --query-gpu=compute_mode --format=csv,noheader

# 10. Monitor inference node specifically
echo -e "\n10. To monitor your inference node performance:"
echo "python3 inference_node.py monitor"

# 11. Check for GPU memory fragmentation
echo -e "\n11. GPU memory fragmentation check:"
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'GPU Memory Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB')
    print(f'GPU Memory Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB')
    print(f'GPU Max Memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB')
    print(f'Memory fragmentation: {(torch.cuda.memory_reserved() - torch.cuda.memory_allocated())/1e9:.2f} GB')
else:
    print('CUDA not available')
"

echo -e "\n=== Optimization Suggestions ==="
echo "1. Run: sudo nvidia-smi -pl <power_limit> (set power limit if needed)"
echo "2. Run: sudo nvidia-smi -lgc <min_clock>,<max_clock> (lock GPU clocks)"
echo "3. Check thermal throttling: sudo dmesg | grep -i thermal"
echo "4. Monitor during inference: nvidia-smi -l 1 & python3 your_inference_node.py"