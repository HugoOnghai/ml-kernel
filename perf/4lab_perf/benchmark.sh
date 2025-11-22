bash <<'EOF'
#!/bin/bash
# filepath: /home/hpo8/CompArch_ECE5755/lab4/ml-kernel-MCSA/perf/4lab_perf/benchmark.sh

SIZES=(10 100 200 500 1000 2000)
TOPLEV_PATH="/classes/ece5755/pmu-tools/toplev.py"
OUTPUT_DIR="./perf/4lab_perf/benchmark_results"

# Create output directory
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

for size in "${SIZES[@]}"; do
    echo "Testing size: $size"
    
    for func in matmul_naive matmul_thread; do
        binary="./perf/4lab_perf/$func"
        output_file="$OUTPUT_DIR/${func}_size${size}_output.txt"
        
        # Write header with implementation and size info
        {
            echo "Implementation: $func"
            echo "Matrix Size: ${size}x${size}"
            echo "================================"
            echo ""
            
            # Run toplev and capture full output
            time python3 "$TOPLEV_PATH" \
                --core C0,C1,C2,C3,C4,C5,C6,C7 \
                -l2 -v --no-desc --force-cpu spr \
                taskset -c 0-7 \
                "$binary" "$size"
        } > "$output_file" 2>&1
        
        echo "  Saved to $output_file"
    done
done

echo "All results saved to $OUTPUT_DIR/"
EOF