#!/bin/bash

trtexec --onnx=/tao_models/re_id_model/resnet50_market1501.onnx \
        --maxShapes=input:16x3x256x128 \
        --minShapes=input:1x3x256x128 \
        --optShapes=input:4x3x256x128 \
        --fp16 \
        --saveEngine=/tao_models/re_id_model/model.plan

