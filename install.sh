#!/bin/bash

#1) Create and activate environment
ENVS=$(conda info --envs | awk '{print $1}' )
if [[ $ENVS = *"detectron2"* ]]; then
   conda activate detectron2
else
   echo "Creating a new conda environment for Detectron2 project..."
   conda env create -f environment.yml
   conda activate detectron2
   #exit
fi;

