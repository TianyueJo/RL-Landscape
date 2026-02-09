#!/bin/bash
# Generate behavior-space graph visualizations for HalfCheetah tasks 16-31

cd /mnt/home/tianyuez/landscape-v2/control

# First check whether the data file exists
DATA_FILE="analysis_outputs/behavior_space_pca_3d_hc.npz"

if [ ! -f "$DATA_FILE" ]; then
    echo "Data file missing. Please run visualization/plot_behavior_space_pca.py first to generate it."
    echo "This may take some time..."
    exit 1
fi

# Create output directory
OUTPUT_DIR="analysis_outputs/hc_behavior_graphs"
mkdir -p "$OUTPUT_DIR"

# Run multi-threshold graph generation
python3 visualization/plot_behavior_graph_multiple_thresholds.py \
    --data-file "$DATA_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --env-name "HalfCheetah-v4" \
    --pca-dims 2 6 10 \
    --layout spring \
    --separate-components

echo "Done! Results saved to: $OUTPUT_DIR"







