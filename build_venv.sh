#!/bin/bash

ml cuda/11.3
ml cudnn
ml nccl

module load intel/19.0.5
module load impi/19.0.5
module load phdf5
module load gcc/9.1.0
module load python3/3.8.2
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

python3 -m virtualenv venv-frontera-gpu
source venv-frontera-gpu/bin/activate

which python
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
