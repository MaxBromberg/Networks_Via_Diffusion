#!/bin/bash

output_directory="/home/maqz/Desktop/data/diversity_seeding_beta_1/source_reward_plots/"
data_directory="/home/maqz/Desktop/data/diversity_seeding_beta_1/source_reward_data/"
parallel_plotter_path="/home/maqz/PycharmProjects/Network_Structure_Via_Info_Diffusion"

edge_conservation_start="0"
edge_conservation_end="1.05"
edge_conservation_interval="0.1"

selectivity_start="0"
selectivity_end="1.05"
selectivity_interval="0.1"

cd $data_directory || exit
for directory in */ ;
do
    echo "Started plotting for: $data_directory$directory"
    python $parallel_plotter_path/grid_search_plotter.py $data_directory$directory $edge_conservation_start $edge_conservation_end $edge_conservation_interval $selectivity_start $selectivity_end $selectivity_interval $output_directory
done

