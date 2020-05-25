#!/bin/bash

output_directory="/home/maqz/Desktop/data/"
data_directory="/home/maqz/Desktop/tmp/"
parallel_plotter_path="/home/maqz/PycharmProjects/Network_Structure_Via_Info_Diffusion"

coupling_start=0
coupling_end=1
coupling_interval="0.1"

skew_start=0
skew_end="0.9"
skew_interval="0.1"

cd $data_directory || exit
for directory in */ ;
do
    python $parallel_plotter_path/parallel_plotter.py $data_directory$directory $coupling_start $coupling_end $coupling_interval $skew_start $skew_end $skew_interval $output_directory
    echo "Started plotting for: $data_directory$directory"
done

