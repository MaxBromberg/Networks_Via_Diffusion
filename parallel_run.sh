#!/bin/bash

# define ranges of nodes / couple_vals / adapt_vals lines {start..end..interval}

output_directory="/home/maqz/Desktop/data/"
run_index=1

for nodes in {10..85..15}
do
  sub_output_directory="$output_directory/node_num_$nodes"
  mkdir -p "$sub_output_directory"
  for couple_val in 0.{0..95..5}
  do
	  for adapt_val in 0.{0..9..1}
	  do
		  python parallel_run.py $sub_output_directory $nodes $run_index $couple_val $adapt_val
		  (( run_index++ ))
	  done
  done
  echo "Run for $nodes complete. Data stored in $sub_output_directory"
  run_index=0
done

echo "Raw graph data recorded at $output_directory"