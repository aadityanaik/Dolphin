# Dolphin

Code for Dolphin, a GPU-accelerated neurosymbolic learning framework (https://arxiv.org/abs/2410.03348).
Dolphin is a Python package that enables scalable neurosymbolic learning by performing probabilistic computations over the GPU. It is integrated with PyTorch and provides a set of primitives for writing Pythonic probabilistic programs.

## Installation
To install Dolphin, first clone the repository. Then use pip:
```bash
pip install -e .
```

## Running Experiments

To run the experiments, you must first download the data. You can get it from the following drive link:


### MNIST Sum-N
```bash
cd experiments/mnist
python sum_n.py --sum-n=N --provenance=damp --device cuda 
```

### HWF
```bash
cd experiments/hwf
python hwf.py --device=cuda --l=7 --provenance=dtkp-am --sample-k=7 --top-k=3 --batch-size=64
```

### PathFinder
```bash
cd experiments/path
python run_XYZ.py --n-epochs=10 --difficulty=all --gpu=0 --provenance=dtkp-am --top-k=1 --seed=1234
```

### CLUTRR
```bash
cd experiments/clutrr
python run.py --cuda  --n-epochs=10 --seed 1831 --learning-rate 1e-5
```

### Mugen
```bash
cd experiments/mugen
python run.py --phase=train --train_size=1000 --provenance=damp --seed=1234 --epochs=100 --use_cuda
```