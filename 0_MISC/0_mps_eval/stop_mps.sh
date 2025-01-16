#!/bin/bash

# -- Used by Flyt to stop mps control daemon.
echo quit | nvidia-cuda-mps-control
nvidia-smi -i 2 -c DEFAULT
