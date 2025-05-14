#!/bin/bash
algo=$1
id=$2
scp -r t504-212n5:/home/baxel/Projects/adex-gym/training/wandb/run-*${id}/files/params/ $algo
