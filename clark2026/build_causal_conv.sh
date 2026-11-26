#!/bin/bash
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export MAX_JOBS=8 TORCH_CUDA_ARCH_LIST=8.9
nvcc --version | tail -1
/opt/tcvenv/bin/pip install -q wheel ninja packaging
/opt/tcvenv/bin/pip install causal-conv1d --no-build-isolation 2>&1 | tail -3
/opt/tcvenv/bin/python -c "import causal_conv1d; print(\"causal_conv1d ok\")"
