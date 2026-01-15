#!/bin/bash
# Benchmark runner that tests different rendering configurations

echo "================================================================================"
echo "MuJoCo Rendering Backend Benchmark (10 parallel envs)"
echo "================================================================================"
echo

declare -A results

configs=(
    "egl:64"
    "egl:48"
    "osmesa:64"
    "osmesa:48"
)

for config in "${configs[@]}"; do
    backend="${config%%:*}"
    res="${config##*:}"
    
    echo "Testing: $backend ${res}x${res}..."
    
    output=$(MUJOCO_GL="$backend" RENDER_RES="$res" NUM_ENVS=10 \
             python scripts/test_render_simple.py 2>&1 | grep -v "GLFWError\|glfw\|warning:")
    
    result_line=$(echo "$output" | grep "^RESULT:")
    
    if [[ $result_line == RESULT:ERROR:* ]]; then
        error="${result_line#RESULT:ERROR:}"
        echo "  ✗ Error: ${error:0:80}"
    elif [[ $result_line == RESULT:* ]]; then
        IFS=':' read -r _ avg_ms throughput <<< "$result_line"
        results["$backend:$res"]="$avg_ms:$throughput"
        echo "  ✓ Avg: ${avg_ms}ms, Throughput: ${throughput} steps/s"
    else
        echo "  ✗ Unknown error"
    fi
    echo
done

echo "================================================================================"
echo "SUMMARY (sorted by throughput)"
echo "================================================================================"
echo

# Sort and display results
if [ ${#results[@]} -gt 0 ]; then
    echo "Config                Avg Time (ms)    Throughput (steps/s)"
    echo "--------------------------------------------------------------------------------"
    
    for key in "${!results[@]}"; do
        IFS=':' read -r backend res avg_ms throughput <<< "${key}:${results[$key]}"
        echo "$backend ${res}x${res}    $avg_ms    $throughput"
    done | sort -t' ' -k5 -nr
    
    echo
    echo "Use the configuration with highest throughput for training."
else
    echo "No successful benchmarks completed."
fi
