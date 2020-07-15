#!/bin/bash

output_directory="/home/maqz/Desktop/data/random_seeding/"
data_directory="/home/maqz/Desktop/data/random_seeding/source_reward_data"
parallel_plotter_path="/home/maqz/PycharmProjects/Network_Structure_Via_Info_Diffusion"

edge_conservation_start="2"
edge_conservation_end="11"
edge_conservation_interval="2"

selectivity_start="2"
selectivity_end="11"
selectivity_interval="2"

cd $data_directory || exit
for directory in */ ;
do
    python $parallel_plotter_path/parallel_plotter.py $data_directory$directory $edge_conservation_start $edge_conservation_end $edge_conservation_interval $selectivity_start $selectivity_end $selectivity_interval $output_directory
    echo "Started plotting for: $data_directory$directory"
done

