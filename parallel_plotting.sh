#!/bin/bash

output_directory="/home/maqz/Desktop/data_subset/"
data_directory="/home/maqz/Desktop/tmp/"
parallel_plotter_path="/home/maqz/PycharmProjects/Network_Structure_Via_Info_Diffusion"

coupling_start=0
coupling_end=1
coupling_interval="0.05"

adaptation_start=0
adaptation_end=8
adaptation_interval=1

cd $data_directory || exit
for directory in */ ;
do
    python $parallel_plotter_path/parallel_plotter.py $data_directory$directory $coupling_start $coupling_end $coupling_interval $adaptation_start $adaptation_end $adaptation_interval $output_directory
    echo "Started plotting for: $data_directory$directory"
done

