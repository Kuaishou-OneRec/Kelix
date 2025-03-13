#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <textfile_path>"
    exit 1
fi

# Assign the first argument to textfile_path
textfile_path="/llm_reco/maosiyang/dataset/i2i/part-00001"

# Run the i2iconverter Python script with the provided textfile_path
python3 i2isampler.py "$textfile_path" 