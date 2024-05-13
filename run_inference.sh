#!/bin/bash

rm -f ./Dataset/output_resnet_256x196_fp32/result.json
python3 ./tao_triton/python/entrypoints/tao_client.py \
       ./Dataset/sample_query \
       --test_dir ./Dataset/sample_test \
       -m re_identification_tao \
       -x 1 \
       -b 16 \
       --mode Re_identification \
       -i https \
       -u localhost:8000 \
       --async \
       --output_path ./Dataset/output_resnet_256x196_fp32

# Plot inference results
python3 ./scripts/re_id_e2e_inference/plot_e2e_inference.py \
        /home/minigo/Desktop/tao_inference_run/tao-toolkit-triton-apps/Dataset/output_resnet_256x196_fp32/results.json \
        /home/minigo/Desktop/tao_inference_run/tao-toolkit-triton-apps/Dataset/output_resnet_256x196_fp32
     