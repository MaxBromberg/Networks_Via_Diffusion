#!/bin/bash

# define ranges of couple_vals / adapt_vals lines {start..end..interval}

output_directory="/home/maqz/Desktop/data/infoscore_reinforcement_decoupled/50_source_shifts/"
run_index=1
node=50

sub_output_directory="$output_directory/node_num_$node"
mkdir -p "$sub_output_directory"

for coupling_tens in {0..9};
  do
    for coupling_hundreths in {0..9..5};
      do
        for skew_tens in {0..9};
          do
            for skew_hundreths in {0..9..5};
              do
                python parallel_run.py $sub_output_directory $node $run_index 0.$coupling_tens$coupling_hundreths 0.$skew_tens$skew_hundreths
                (( run_index++ ))
              done
          done
      python parallel_run.py $sub_output_directory $node $run_index 0.$coupling_tens$coupling_hundreths 1
      (( run_index++ ))
      done
  done
(( run_index++ ))

for skew_tens in {0..9};
  do
    for skew_hundreths in {0..9..5};
      do
        python parallel_run.py $sub_output_directory $node $run_index 1 0.$skew_tens$skew_hundreths
        (( run_index++ ))
      done
  done
python parallel_run.py $sub_output_directory $node $run_index 1 1 $output_directory



echo "Raw graph data recorded at $output_directory"

python parallel_plotter.py $sub_output_directory 0 "1.05" "0.05" 0 "1.05" "0.05"


