#!/bin/bash
#PBS -N DiffusionInference
#PBS -A UCUB0097
#PBS -l walltime=12:00:00
#PBS -o NEWt_TrainDiffusion.out
#PBS -e NEWt_TrainDiffusion.out
#PBS -q casper
#PBS -l select=1:ncpus=64:mem=75GB:ngpus=1 -l gpu_type=a100
#PBS -m a
#PBS -M timothy.higgins@colorado.edu

# qsub -I -q main -A UCUB0097 -l walltime=12:00:00 -l select=1:ncpus=32:mem=75GB:ngpus=1 -l gpu_type=a100
# qsub -I -q casper -A UCUB0097 -l walltime=12:00:00 -l select=1:ncpus=32:mem=75GB:ngpus=1 -l gpu_type=a100

#accelerate config
module load conda
conda activate /glade/work/timothyh/miniconda3/envs/analysis
python Inference.py --lead_time=72
