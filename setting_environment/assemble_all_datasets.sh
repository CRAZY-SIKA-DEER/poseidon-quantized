#!/bin/bash

# Loop through each dataset subfolder
for dataset_dir in datasets/*; do
    if [ -d "$dataset_dir" ]; then
        echo "Assembling dataset in $dataset_dir/..."
        if [ -f "$dataset_dir/assemble_data.py" ]; then
            # Infer output filename, e.g., datasets/NS-Gauss/NS-Gauss.h5
            dataset_name=$(basename "$dataset_dir")
            python "$dataset_dir/assemble_data.py" \
                --input_dir "$dataset_dir" \
                --output_file "$dataset_dir/${dataset_name}.h5"
        else
            echo "No assemble_data.py in $dataset_dir/, skipping."
        fi
    fi
done

echo "✅ All datasets assembled!"